mod block_data;
mod layer;
pub mod raw_data;
mod gpu;

use std::cmp::min;
use std::thread;
use std::sync::Arc;

use next_long_reverser::get_next_long;
use java_random::{JAVA_LCG, Random};

use crate::block_data::{BlockFilter, CheckObject, get_filter_power};
use crate::layer::{create_filter_tree, flat_search};
use crate::raw_data::block::Block;
use crate::raw_data::modes::{BedrockGeneration, OutputMode};
use crate::raw_data::sender::Sender;

const MASK48: u64 = 0xFFFF_FFFF_FFFF;
const ROOF_HASH: u64 = 343340730;
const FLOOR_HASH: u64 = 2042456806;

const CHUNK_SIZE: u64 = 1 << 25;

/// Naive estimate of result count
pub fn estimate_result_amount(blocks: &[Block]) -> u64 {
    let filters: Vec<_> = blocks
        .iter()
        .map(|block| BlockFilter::from(block, BedrockGeneration::Normal))
        .collect();
    get_filter_power(&filters)
}

pub fn search_bedrock_pattern_with_list<S: Sender + 'static>(
    blocks: &[Block],
    thread_count: u64,
    seed_list: &[u64],
    mode: BedrockGeneration,
    sender: S,
) {
    println!(
        "[Logic] Starting List Search. Seeds: {}, Threads: {}, Mode: {}",
        seed_list.len(),
        thread_count,
        mode
    );

    let mut roof_blocks = Vec::new();
    let mut floor_blocks = Vec::new();

    for block in blocks.iter() {
        let check = BlockFilter::from(block, mode).create_check(0);
        if block.y > 5 {
            roof_blocks.push(check);
        } else {
            floor_blocks.push(check);
        }
    }
    let roof_blocks = Arc::new(roof_blocks);
    let floor_blocks = Arc::new(floor_blocks);

    let chunk_size = seed_list.len() as f64 / thread_count as f64;
    let chunks = seed_list.chunks(chunk_size.ceil() as usize);

    for (i, chunk) in chunks.into_iter().enumerate() {
        println!("[Logic] Spawning list-search thread {}", i);
        let chunk = Vec::from(chunk);
        let sender = sender.clone();
        let roof_blocks = roof_blocks.clone();
        let floor_blocks = floor_blocks.clone();

        thread::spawn(move || {
            flat_search(&chunk, roof_blocks, floor_blocks, sender);
        });
    }
}

// Helper functions for seed derivation
fn reverse_next_long(seed: u64) -> Vec<u64> {
    get_next_long(seed)
        .into_iter()
        .map(|seed| seed ^ JAVA_LCG.multiplier)
        .collect()
}

fn next_long(seed: u64) -> u64 {
    Random::with_seed(seed).next_long() as u64
}

/// Splits blocks into floor and roof sets
fn split_floor_roof(blocks: &[Block], mode: BedrockGeneration) -> (Vec<BlockFilter>, Vec<BlockFilter>) {
    let mut floor_blocks = vec![];
    let mut roof_blocks = vec![];

    for block in blocks.iter() {
        let filter = BlockFilter::from(block, mode);
        if block.y < 64 {
            floor_blocks.push(filter);
        } else {
            roof_blocks.push(filter);
        }
    }

    (floor_blocks, roof_blocks)
}

/// GPU Cross-Comparison Sender
///
/// Receives candidate primary-surface seeds from GPU and performs the full CPU
/// cross-comparison to produce structure/world seeds identical to CPU-only mode.
#[derive(Clone)]
struct GpuCrossComparisonSender<S: Sender> {
    inner: S,
    secondary_checks: Arc<Vec<CheckObject>>,
    primary_hash: u64,
    secondary_hash: u64,
    output_mode: OutputMode,
}

impl<S: Sender> GpuCrossComparisonSender<S> {
    fn new(inner: S, secondary_checks: Vec<CheckObject>, is_floor_primary: bool, output_mode: OutputMode) -> Self {
        let (primary_hash, secondary_hash) = if is_floor_primary {
            (FLOOR_HASH, ROOF_HASH)
        } else {
            (ROOF_HASH, FLOOR_HASH)
        };

        Self {
            inner,
            secondary_checks: Arc::new(secondary_checks),
            primary_hash,
            secondary_hash,
            output_mode,
        }
    }

    fn check_secondary(&self, seed: u64) -> bool {
        for check in self.secondary_checks.iter() {
            if check.check(seed) {
                return false;
            }
        }
        true
    }

    fn process_candidate(&self, candidate_seed: u64) {
        for reversed_seed in reverse_next_long(candidate_seed) {
            let bedrock_seed = reversed_seed ^ self.primary_hash;

            let secondary_input = bedrock_seed ^ self.secondary_hash;
            let secondary_seed = next_long(secondary_input) & MASK48;

            if !self.check_secondary(secondary_seed) {
                continue;
            }

            for structure_seed in reverse_next_long(bedrock_seed) {
                if self.output_mode == OutputMode::WorldSeed {
                    for prev_seed in reverse_next_long(structure_seed) {
                        let world_seed = next_long(prev_seed);
                        self.inner.send(CrackProgress::Seed(world_seed));
                    }
                } else {
                    self.inner.send(CrackProgress::Seed(structure_seed));
                }
            }
        }
    }
}

impl<S: Sender> Sender for GpuCrossComparisonSender<S> {
    fn send(&self, progress: CrackProgress) -> bool {
        match progress {
            CrackProgress::Seed(candidate) => {
                self.process_candidate(candidate);
                true
            }
            CrackProgress::Progress(n) => self.inner.send(CrackProgress::Progress(n)),
        }
    }
}

pub fn search_bedrock_pattern<S: Sender + 'static>(
    blocks: &[Block],
    thread_count: u64,
    mode: BedrockGeneration,
    output: OutputMode,
    sender: S,
    use_gpu: bool,
) {
    println!("==================================================");
    println!("           Nether Bedrock Cracker Init            ");
    println!("==================================================");
    println!("[Config] Generation Mode: {}", mode);
    println!("[Config] Output Mode: {}", output);
    println!(
        "[Config] Thread Count: {} {}",
        thread_count,
        if use_gpu { "(ignored for GPU)" } else { "" }
    );
    println!("[Config] Block Count: {}", blocks.len());
    println!("[Config] Use GPU: {}", use_gpu);
    println!("[Config] Estimated Results: {}", estimate_result_amount(blocks));

    for (i, b) in blocks.iter().enumerate() {
        println!("  [{}] {}", i, b);
    }
    println!("==================================================");

    if use_gpu {
        println!("[Logic] GPU mode: Splitting blocks by surface...");

        let (floor_filters, roof_filters) = split_floor_roof(blocks, mode);

        let floor_power = get_filter_power(&floor_filters);
        let roof_power = get_filter_power(&roof_filters);

        println!(
            "[Logic] Floor blocks: {}, Filter power: {}",
            floor_filters.len(),
            floor_power
        );
        println!(
            "[Logic] Roof blocks: {}, Filter power: {}",
            roof_filters.len(),
            roof_power
        );

        let is_floor_primary = floor_power <= roof_power;

        let (primary_filters, secondary_filters) = if is_floor_primary {
            println!("[Logic] Primary surface: FLOOR (better filtering)");
            (floor_filters, roof_filters)
        } else {
            println!("[Logic] Primary surface: ROOF (better filtering)");
            (roof_filters, floor_filters)
        };

        // === Stage-specific pruning and ordering ===
        // Coarse stage uses lower_bits=12 checks, keep only those that actually reject something,
        // and order them by estimated rejection strength to maximize early exits.
        let mut coarse_scored: Vec<(f64, CheckObject)> = primary_filters
            .iter()
            .map(|f| {
                let score = f.discarded_seeds(12);
                let mut ff = f.clone();
                let check = ff.create_check(12);
                (score, check)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        coarse_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let coarse_checks: Vec<CheckObject> = coarse_scored.into_iter().map(|(_, c)| c).collect();

        // Fine stage uses full checks. Ordering still helps; use discarded_seeds(0) as heuristic.
        let mut fine_scored: Vec<(f64, CheckObject)> = primary_filters
            .iter()
            .map(|f| {
                let score = f.discarded_seeds(0);
                let mut ff = f.clone();
                let check = ff.create_check(0);
                (score, check)
            })
            .collect();

        fine_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let fine_checks: Vec<CheckObject> = fine_scored.into_iter().map(|(_, c)| c).collect();

        let secondary_checks: Vec<CheckObject> = secondary_filters
            .into_iter()
            .map(|mut filter| filter.create_check(0))
            .collect();

        println!("[Logic] GPU Coarse Checks kept: {}", coarse_checks.len());
        println!("[Logic] GPU Fine Checks:        {}", fine_checks.len());
        println!(
            "[Logic] Secondary checks for CPU cross-comparison: {}",
            secondary_checks.len()
        );

        let cross_sender = GpuCrossComparisonSender::new(sender, secondary_checks, is_floor_primary, output);

        thread::spawn(move || {
            pollster::block_on(gpu::gpu_search_loop(coarse_checks, fine_checks, cross_sender));
        });

        return;
    }

    // === CPU mode (unchanged) ===
    println!("[Logic] CPU mode enabled. Spawning {} threads...", thread_count);

    let checks = create_filter_tree(blocks, mode, output, sender.clone());

    for thread in 0..thread_count {
        let start_bits = (thread * (1 << 36)) / thread_count;
        let end_bits = ((thread + 1) * (1 << 36)) / thread_count;

        let mut current_bits = start_bits << 12;
        let limit_bits = end_bits << 12;

        println!(
            "[Logic] Thread {} assigned range: {:012X} -> {:012X}",
            thread, current_bits, limit_bits
        );

        let checks = checks.clone();
        let sender = sender.clone();

        thread::spawn(move || {
            while current_bits < limit_bits {
                let chunk_end = min(current_bits + CHUNK_SIZE, limit_bits);
                for upper_bits in (current_bits..chunk_end).step_by(1 << 12) {
                    checks.run_checks(upper_bits);
                }

                if !sender.send(CrackProgress::Progress(chunk_end - current_bits)) {
                    return;
                }

                current_bits = chunk_end;
            }
        });
    }
}

#[derive(Clone, Debug)]
pub enum CrackProgress {
    Seed(u64),
    Progress(u64),
}
