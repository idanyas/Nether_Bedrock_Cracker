use std::borrow::Cow;
use crate::block_data::CheckObject;
use crate::CrackProgress;
use crate::raw_data::sender::Sender;
use wgpu::util::DeviceExt;
use tokio::sync::oneshot;

// LCG Constants from java.util.Random (verified from OpenJDK source)
// multiplier = 0x5DEECE66DL = 25214903917
// Low 32 bits:  0xDEECE66D = 3740067437
// High 32 bits: 0x5 = 5
// addend = 0xBL = 11 (not used in check, but documented)
// mask = (1L << 48) - 1

const SHADER: &str = r#"
struct CheckObject {
    pos_hash: vec2<u32>,
    condition: vec2<u32>,
    offset: vec2<u32>,
    _pad: vec2<u32>,
}

struct Uniforms {
    base_seed: vec2<u32>,
    num_checks: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> checks: array<CheckObject>;

@group(0) @binding(1)
var<storage, read_write> results: array<vec2<u32>>;

@group(0) @binding(2)
var<storage, read_write> result_count: atomic<u32>;

@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

// Java LCG multiplier: 0x5DEECE66D
// Split: high=0x5, low=0xDEECE66D (3740067437)
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

// Returns true if seed FAILS this check (should be rejected)
// This matches the Rust CheckObject::check semantics
fn check_fails(seed: vec2<u32>, obj: CheckObject) -> bool {
    var val = xor64(seed, obj.pos_hash);
    val = mul_lcg(val);
    val = add64(val, obj.offset);
    val = and_mask48(val);
    // In Rust: (result & MASK48) < condition means FAIL
    return less_than(val, obj.condition);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    var seed = uniforms.base_seed;
    let lo = seed.x + index;
    let carry = u32(lo < seed.x);
    seed.x = lo;
    seed.y = seed.y + carry;

    // Check if we've exceeded 48-bit space
    if (seed.y > MASK48_HI) {
        return;
    }

    // If ANY check fails, reject this seed
    for (var i = 0u; i < uniforms.num_checks; i++) {
        if (check_fails(seed, checks[i])) {
            return;
        }
    }

    // All checks passed - this seed is a candidate
    let out_idx = atomicAdd(&result_count, 1u);
    if (out_idx < 65536u) {
        results[out_idx] = seed;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    base_seed: [u32; 2],
    num_checks: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCheckObject {
    pos_hash: [u32; 2],
    condition: [u32; 2],
    offset: [u32; 2],
    _pad: [u32; 2],
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

/// Validates that our shader constants match the Rust/Java LCG constants
fn validate_lcg_constants() {
    use java_random::JAVA_LCG;
    let mult = JAVA_LCG.multiplier;
    let mult_lo = mult as u32;
    let mult_hi = (mult >> 32) as u32;
    
    println!("[GPU] LCG Constant Validation:");
    println!("[GPU]   Rust multiplier: 0x{:012X} ({})", mult, mult);
    println!("[GPU]   Expected lo: {} (0x{:08X})", mult_lo, mult_lo);
    println!("[GPU]   Expected hi: {} (0x{:08X})", mult_hi, mult_hi);
    println!("[GPU]   Shader MULT_LO: 3740067437 (0xDEECE66D)");
    println!("[GPU]   Shader MULT_HI: 5 (0x00000005)");
    
    assert_eq!(mult_lo, 3740067437, "MULT_LO mismatch!");
    assert_eq!(mult_hi, 5, "MULT_HI mismatch!");
    println!("[GPU]   ✓ Constants validated successfully");
}

pub async fn gpu_search_loop<S: Sender>(
    checks_raw: Vec<CheckObject>,
    sender: S,
) {
    println!("[GPU] ================================================");
    println!("[GPU]           GPU Search Initialization             ");
    println!("[GPU] ================================================");
    
    validate_lcg_constants();
    
    println!("[GPU] Primary surface checks: {}", checks_raw.len());
    
    let checks: Vec<GpuCheckObject> = checks_raw.into_iter().map(GpuCheckObject::from).collect();

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await;

    if adapter.is_none() {
        eprintln!("[GPU] ERROR: No GPU adapter found! Ensure drivers are installed.");
        return;
    }
    let adapter = adapter.unwrap();

    let info = adapter.get_info();
    println!("[GPU] Adapter: {} ({:?})", info.name, info.backend);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("[GPU] Failed to create device");

    println!("[GPU] Device created. Compiling shader...");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bedrock Crack Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
    });

    let check_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Check Buffer"),
        contents: bytemuck::cast_slice(&checks),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Increased to 64K results per chunk (512KB buffer)
    let max_results: u64 = 65536;
    let results_buffer_size = max_results * 8;
    let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Results Buffer"),
        size: results_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Count Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: results_buffer_size + 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: check_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // Chunk size: 2^24 = 16M seeds per dispatch
    let chunk_size = 1u64 << 24;
    let total_seeds = 1u64 << 48;
    let workgroup_size = 256u32;
    let dispatch_size = (chunk_size as u32) / workgroup_size;

    println!("[GPU] ================================================");
    println!("[GPU] Search Configuration:");
    println!("[GPU]   Total seeds: {} (2^48)", total_seeds);
    println!("[GPU]   Chunk size: {} (2^24)", chunk_size);
    println!("[GPU]   Workgroups per chunk: {}", dispatch_size);
    println!("[GPU]   Max results per chunk: {}", max_results);
    println!("[GPU] ================================================");
    println!("[GPU] Starting search...");

    let mut overflow_warning_shown = false;

    for start_seed in (0..total_seeds).step_by(chunk_size as usize) {
        // Reset count
        let zero = [0u32];
        queue.write_buffer(&count_buffer, 0, bytemuck::cast_slice(&zero));

        let uniforms = Uniforms {
            base_seed: u64_to_u32x2(start_seed),
            num_checks: checks.len() as u32,
            _padding: 0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_buffer, 0, 4);
        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 4, results_buffer_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.await {
            let data = buffer_slice.get_mapped_range();
            let count = *bytemuck::from_bytes::<u32>(&data[0..4]);

            if count > max_results as u32 && !overflow_warning_shown {
                eprintln!("[GPU] WARNING: Result buffer overflow! {} candidates found, max is {}.", count, max_results);
                eprintln!("[GPU] Some candidates may be lost. Consider adding more filter blocks.");
                overflow_warning_shown = true;
            }

            if count > 0 {
                let limit = std::cmp::min(count, max_results as u32);
                let seeds_raw: &[[u32; 2]] = bytemuck::cast_slice(&data[4..]);
                for i in 0..limit as usize {
                    let candidate = u32x2_to_u64(seeds_raw[i]);
                    sender.send(CrackProgress::Seed(candidate));
                }
            }
            drop(data);
            staging_buffer.unmap();
        }

        if !sender.send(CrackProgress::Progress(chunk_size)) {
            println!("[GPU] Search cancelled by user.");
            break;
        }
    }
    println!("[GPU] ================================================");
    println!("[GPU] GPU search loop finished.");
    println!("[GPU] ================================================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_data::{BlockFilter, CheckObject};
    use crate::raw_data::block_type::BlockType;
    use crate::raw_data::modes::BedrockGeneration;
    use crate::MASK48;

    #[test]
    fn test_lcg_constants() {
        validate_lcg_constants();
    }

    #[test]
    fn test_u64_conversion_roundtrip() {
        let values = [0u64, 1, MASK48, 0x123456789ABC, u64::MAX];
        for v in values {
            let arr = u64_to_u32x2(v);
            let back = u32x2_to_u64(arr);
            assert_eq!(v, back, "Roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_gpu_check_object_conversion() {
        let check = CheckObject {
            pos_hash: 0x123456789ABC,
            condition: 0xFEDCBA987654,
            offset: 0x111111111111,
        };
        let gpu_check = GpuCheckObject::from(check.clone());
        
        assert_eq!(u32x2_to_u64(gpu_check.pos_hash), check.pos_hash);
        assert_eq!(u32x2_to_u64(gpu_check.condition), check.condition);
        assert_eq!(u32x2_to_u64(gpu_check.offset), check.offset);
    }
}
