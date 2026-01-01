use crate::block_data::CheckObject;
use crate::raw_data::sender::Sender;
use crate::CrackProgress;
use futures::channel::oneshot;
use std::borrow::Cow;
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct CheckObject {
    pos_hash: vec2<u32>,
    condition: vec2<u32>,
    offset: vec2<u32>,
    _pad: vec2<u32>,
}

struct Uniforms {
    base_seed_upper_36: vec2<u32>,
    chunk_len_upper: u32,
    num_coarse_checks: u32,
    num_fine_checks: u32,
    _pad: u32,
}

struct Stats {
    coarse_count: atomic<u32>,
    fine_count: atomic<u32>,
    coarse_overflow: atomic<u32>,
    fine_overflow: atomic<u32>,
}

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
    _pad: u32,
}

// Group 0: Common resources used by all stages
@group(0) @binding(0) var<storage, read> coarse_checks: array<CheckObject>;
@group(0) @binding(1) var<storage, read> fine_checks: array<CheckObject>;
@group(0) @binding(2) var<storage, read_write> coarse_results: array<u32>;
@group(0) @binding(3) var<storage, read_write> fine_results: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> stats: Stats;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Group 1: Dispatch Args (Only used by prepare stage)
@group(1) @binding(0) var<storage, read_write> fine_dispatch: DispatchArgs;

const MULT_LO: u32 = 3740067437u;
const MULT_HI: u32 = 5u;
const MASK48_LO: u32 = 0xFFFFFFFFu;
const MASK48_HI: u32 = 0x0000FFFFu;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = u32(lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

fn mul32_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let sum_mid = p1 + p2;
    let mid_overflow = u32(sum_mid < p1);
    let lo_part_mid = sum_mid << 16u;
    let hi_part_mid = sum_mid >> 16u;
    let final_lo = p0 + lo_part_mid;
    let carry = u32(final_lo < p0);
    let final_hi = p3 + hi_part_mid + (mid_overflow << 16u) + carry;
    return vec2<u32>(final_lo, final_hi);
}

fn mul_lcg(a: vec2<u32>) -> vec2<u32> {
    let t1 = a.y * MULT_LO;
    let t2 = a.x * MULT_HI;
    let high_part_contribution = t1 + t2;
    let full_prod = mul32_wide(a.x, MULT_LO);
    let final_lo = full_prod.x;
    let final_hi = full_prod.y + high_part_contribution;
    return vec2<u32>(final_lo, final_hi);
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn and_mask48(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x & MASK48_LO, a.y & MASK48_HI);
}

fn less_than(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y < b.y) { return true; }
    if (a.y > b.y) { return false; }
    return a.x < b.x;
}

fn check_fails(seed: vec2<u32>, obj: CheckObject) -> bool {
    var val = xor64(seed, obj.pos_hash);
    val = mul_lcg(val);
    val = add64(val, obj.offset);
    val = and_mask48(val);
    return less_than(val, obj.condition);
}

@compute @workgroup_size(256)
fn main_coarse(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>
) {
    let threads_per_row = nw.x * 256u;
    let index = gid.x + gid.y * threads_per_row;
    if (index >= uniforms.chunk_len_upper) { return; }

    var upper_seed = uniforms.base_seed_upper_36;
    let lo = upper_seed.x + index;
    let carry = u32(lo < upper_seed.x);
    upper_seed.x = lo;
    upper_seed.y = upper_seed.y + carry;

    let seed_lo = upper_seed.x << 12u;
    let seed_hi = (upper_seed.y << 12u) | (upper_seed.x >> 20u);
    let seed = vec2<u32>(seed_lo, seed_hi & 0xFFFFu);

    for (var i = 0u; i < uniforms.num_coarse_checks; i++) {
        if (check_fails(seed, coarse_checks[i])) { return; }
    }

    let out_idx = atomicAdd(&stats.coarse_count, 1u);
    let cap = arrayLength(&coarse_results);
    if (out_idx < cap) {
        coarse_results[out_idx] = index;
    } else {
        atomicStore(&stats.coarse_overflow, 1u);
    }
}

@compute @workgroup_size(1)
fn main_prepare_fine() {
    let cap = arrayLength(&coarse_results);
    let raw = atomicLoad(&stats.coarse_count);
    let coarse_count = min(raw, cap);

    if (raw > cap) { atomicStore(&stats.coarse_overflow, 1u); }

    if (coarse_count == 0u) {
        fine_dispatch.x = 0u; fine_dispatch.y = 0u; fine_dispatch.z = 0u;
        return;
    }

    let x = min(coarse_count, 65535u);
    let y = (coarse_count + x - 1u) / x;
    fine_dispatch.x = x; fine_dispatch.y = y; fine_dispatch.z = 1u;
}

@compute @workgroup_size(256)
fn main_fine(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>
) {
    let cap = arrayLength(&coarse_results);
    let raw = atomicLoad(&stats.coarse_count);
    let coarse_count = min(raw, cap);
    let coarse_idx = wid.x + wid.y * nw.x;

    if (coarse_idx >= coarse_count) { return; }

    let chunk_offset = coarse_results[coarse_idx];
    var upper_seed = uniforms.base_seed_upper_36;
    let lo = upper_seed.x + chunk_offset;
    let carry = u32(lo < upper_seed.x);
    upper_seed.x = lo;
    upper_seed.y = upper_seed.y + carry;

    let base_lo = upper_seed.x << 12u;
    let base_hi = (upper_seed.y << 12u) | (upper_seed.x >> 20u);

    for (var i = 0u; i < 16u; i++) {
        let lower = lid.x + i * 256u;
        let seed = vec2<u32>(base_lo | lower, base_hi & 0xFFFFu);
        var failed = false;
        for (var c = 0u; c < uniforms.num_fine_checks; c++) {
            if (check_fails(seed, fine_checks[c])) { failed = true; break; }
        }
        if (!failed) {
            let out_idx = atomicAdd(&stats.fine_count, 1u);
            let fcap = arrayLength(&fine_results);
            if (out_idx < fcap) {
                fine_results[out_idx] = seed;
            } else {
                atomicStore(&stats.fine_overflow, 1u);
            }
        }
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    base_seed_upper_36: [u32; 2],
    chunk_len_upper: u32,
    num_coarse_checks: u32,
    num_fine_checks: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCheckObject {
    pos_hash: [u32; 2],
    condition: [u32; 2],
    offset: [u32; 2],
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StatsReadback {
    coarse_count: u32,
    fine_count: u32,
    coarse_overflow: u32,
    fine_overflow: u32,
}

impl From<CheckObject> for GpuCheckObject {
    fn from(c: CheckObject) -> Self {
        Self {
            pos_hash: u64_to_u32x2(c.pos_hash),
            condition: u64_to_u32x2(c.condition),
            offset: u64_to_u32x2(c.offset),
            _pad: [0, 0],
        }
    }
}

fn u64_to_u32x2(val: u64) -> [u32; 2] {
    [val as u32, (val >> 32) as u32]
}

fn u32x2_to_u64(val: [u32; 2]) -> u64 {
    (val[0] as u64) | ((val[1] as u64) << 32)
}

pub async fn gpu_search_loop<S: Sender>(
    coarse_checks_raw: Vec<CheckObject>,
    fine_checks_raw: Vec<CheckObject>,
    sender: S,
) {
    println!("[GPU] Initialization...");
    let coarse_checks: Vec<GpuCheckObject> = coarse_checks_raw.into_iter().map(GpuCheckObject::from).collect();
    let fine_checks: Vec<GpuCheckObject> = fine_checks_raw.into_iter().map(GpuCheckObject::from).collect();

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("[GPU] No adapter found");

    let info = adapter.get_info();
    println!("[GPU] Adapter: {:?} ({})", info.backend, info.name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 256 << 20,
                    ..Default::default()
                },
            },
            None,
        )
        .await
        .expect("[GPU] Failed to create device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
    });

    let coarse_check_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Coarse Checks"),
        contents: bytemuck::cast_slice(&coarse_checks),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fine_check_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fine Checks"),
        contents: bytemuck::cast_slice(&fine_checks),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let max_coarse = 16_777_216;
    let max_fine = 8_388_608;

    let coarse_res_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coarse Results"),
        size: (max_coarse as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let fine_res_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Fine Results"),
        size: (max_fine as u64) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Stats"),
        size: std::mem::size_of::<StatsReadback>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fine_dispatch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Fine Dispatch"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Stats Staging"),
        size: std::mem::size_of::<StatsReadback>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fine_res_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Fine Res Staging"),
        size: (max_fine as u64) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Layouts
    let common_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Common Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // coarse_checks
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // fine_checks
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // coarse_res
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // fine_res
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // stats
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }, // uniforms
        ],
    });

    let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dispatch Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    // Pipeline Layouts
    let coarse_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Coarse PL"),
        bind_group_layouts: &[&common_layout, &dispatch_layout], // Coarse/Prepare use both (optional for coarse, but harmless)
        push_constant_ranges: &[],
    });

    let fine_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Fine PL"),
        bind_group_layouts: &[&common_layout], // Fine uses ONLY common
        push_constant_ranges: &[],
    });

    // Pipelines
    let coarse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Coarse Pipe"),
        layout: Some(&coarse_pipeline_layout),
        module: &shader,
        entry_point: "main_coarse",
    });
    let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Prepare Pipe"),
        layout: Some(&coarse_pipeline_layout),
        module: &shader,
        entry_point: "main_prepare_fine",
    });
    let fine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fine Pipe"),
        layout: Some(&fine_pipeline_layout),
        module: &shader,
        entry_point: "main_fine",
    });

    // Bind Groups
    let common_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Common BG"),
        layout: &common_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: coarse_check_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: fine_check_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: coarse_res_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: fine_res_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: stats_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let dispatch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dispatch BG"),
        layout: &dispatch_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: fine_dispatch_buf.as_entire_binding() }],
    });

    let total_upper = 1u64 << 36;
    let chunk_size = 1u32 << 30;
    let wg_size = 256;
    let coarse_wgs = (chunk_size as u64) / (wg_size as u64);
    let dx = 65535;
    let dy = ((coarse_wgs + (dx as u64) - 1) / (dx as u64)) as u32;

    println!("[GPU] Config: Chunks=2^30, CoarseWG=({}, {})", dx, dy);
    println!("[GPU] Running...");

    let mut stack: Vec<(u64, u32)> = Vec::new();
    for start in (0..total_upper).step_by(chunk_size as usize) {
        stack.push((start, chunk_size));
    }

    let start = Instant::now();
    let mut total_cand = 0;
    let mut ranges = 0;

    while let Some((r_start, r_len)) = stack.pop() {
        queue.write_buffer(&stats_buf, 0, bytemuck::cast_slice(&[0u32; 4]));
        queue.write_buffer(&fine_dispatch_buf, 0, bytemuck::cast_slice(&[0u32; 4]));

        let u = Uniforms {
            base_seed_upper_36: u64_to_u32x2(r_start),
            chunk_len_upper: r_len,
            num_coarse_checks: coarse_checks.len() as u32,
            num_fine_checks: fine_checks.len() as u32,
            _pad: 0,
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[u]));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Coarse"), timestamp_writes: None });
            pass.set_pipeline(&coarse_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            pass.set_bind_group(1, &dispatch_bg, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Prepare"), timestamp_writes: None });
            pass.set_pipeline(&prepare_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            pass.set_bind_group(1, &dispatch_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Fine"), timestamp_writes: None });
            pass.set_pipeline(&fine_pipeline);
            pass.set_bind_group(0, &common_bg, &[]);
            pass.dispatch_workgroups_indirect(&fine_dispatch_buf, 0);
        }

        enc.copy_buffer_to_buffer(&stats_buf, 0, &stats_staging, 0, 16);
        queue.submit(Some(enc.finish()));

        let slice = stats_staging.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
        device.poll(wgpu::Maintain::Wait);

        let (mut c_ov, mut f_ov, mut f_cnt) = (0, 0, 0);
        if let Ok(Ok(())) = rx.await {
            let data = slice.get_mapped_range();
            let s = *bytemuck::from_bytes::<StatsReadback>(&data);
            c_ov = s.coarse_overflow;
            f_ov = s.fine_overflow;
            f_cnt = s.fine_count;
            if c_ov > 0 && s.coarse_count > 0 { /* satisfying usage */ }
            drop(data);
            stats_staging.unmap();
        } else {
            c_ov = 1; f_ov = 1;
        }

        if c_ov != 0 || f_ov != 0 {
            if r_len <= (1 << 18) {
                eprintln!("[GPU] Overflow at len={} start={}", r_len, r_start);
                continue;
            }
            let h = r_len / 2;
            stack.push((r_start, h));
            stack.push((r_start + h as u64, r_len - h));
            continue;
        }

        if f_cnt > 0 {
            total_cand += f_cnt as u64;
            let cnt = f_cnt.min(max_fine);
            let bsize = (cnt as u64) * 8;
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&fine_res_buf, 0, &fine_res_staging, 0, bsize);
            queue.submit(Some(enc.finish()));

            let slice = fine_res_staging.slice(0..bsize);
            let (tx, rx) = oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
            device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.await {
                let data = slice.get_mapped_range();
                let seeds: &[[u32; 2]] = bytemuck::cast_slice(&data);
                for s in seeds {
                    sender.send(CrackProgress::Seed(u32x2_to_u64(*s)));
                }
                drop(data);
                fine_res_staging.unmap();
            }
        }

        ranges += 1;
        if !sender.send(CrackProgress::Progress((r_len as u64) << 12)) {
            break;
        }
    }

    println!("[GPU] Done. {:.2}s, {} candidates, {} ranges.", start.elapsed().as_secs_f64(), total_cand, ranges);
}
