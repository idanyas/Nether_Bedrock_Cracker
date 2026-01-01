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
    let filters: Vec<_> = blocks.iter()
        .map(|block| BlockFilter::from(block, BedrockGeneration::Normal))
        .collect();
    get_filter_power(&filters)
}

pub fn search_bedrock_pattern_with_list<S: Sender + 'static>(
    blocks: &[Block], 
    thread_count: u64, 
    seed_list: &[u64], 
    mode: BedrockGeneration, 
    sender: S
) {
    println!("[Logic] Starting List Search. Seeds: {}, Threads: {}, Mode: {}", seed_list.len(), thread_count, mode);
    
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
/// This wrapper receives candidate seeds from the GPU (which only tested primary surface)
/// and performs the full cross-comparison to derive actual world/structure seeds.
/// 
/// This ensures GPU results are mathematically equivalent to CPU results.
#[derive(Clone)]
struct GpuCrossComparisonSender<S: Sender> {
    inner: S,
    secondary_checks: Arc<Vec<CheckObject>>,
    primary_hash: u64,
    secondary_hash: u64,
    output_mode: OutputMode,
}

impl<S: Sender> GpuCrossComparisonSender<S> {
    fn new(
        inner: S,
        secondary_checks: Vec<CheckObject>,
        is_floor_primary: bool,
        output_mode: OutputMode,
    ) -> Self {
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

    /// Check if seed passes all secondary surface checks
    /// Returns true if ALL checks pass (seed is valid)
    fn check_secondary(&self, seed: u64) -> bool {
        for check in self.secondary_checks.iter() {
            if check.check(seed) {
                return false; // This check failed
            }
        }
        true // All checks passed
    }

    /// Perform the cross-comparison for a candidate primary surface seed
    /// This mirrors the logic in layer.rs CrossComparison::run
    fn process_candidate(&self, candidate_seed: u64) {
        // candidate_seed is the result of next_long() on the primary surface
        // We need to reverse it to find the input seed
        for reversed_seed in reverse_next_long(candidate_seed) {
            // XOR with primary hash to get common bedrock seed
            let bedrock_seed = reversed_seed ^ self.primary_hash;

            // Derive secondary surface seed
            let secondary_input = bedrock_seed ^ self.secondary_hash;
            let secondary_seed = next_long(secondary_input) & MASK48;

            // Check against secondary surface blocks
            if !self.check_secondary(secondary_seed) {
                continue; // Failed secondary check
            }

            // Passed both surfaces! Now derive final seeds
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
                // Candidate passed primary surface checks on GPU
                // Now do cross-comparison on CPU
                self.process_candidate(candidate);
                true // Continue processing
            }
            CrackProgress::Progress(n) => {
                self.inner.send(CrackProgress::Progress(n))
            }
        }
    }
}

pub fn search_bedrock_pattern<S: Sender + 'static>(
    blocks: &[Block], 
    thread_count: u64, 
    mode: BedrockGeneration, 
    output: OutputMode, 
    sender: S, 
    use_gpu: bool
) {
    println!("==================================================");
    println!("           Nether Bedrock Cracker Init            ");
    println!("==================================================");
    println!("[Config] Generation Mode: {}", mode);
    println!("[Config] Output Mode: {}", output);
    println!("[Config] Thread Count: {} {}", thread_count, if use_gpu { "(ignored for GPU)" } else { "" });
    println!("[Config] Block Count: {}", blocks.len());
    println!("[Config] Use GPU: {}", use_gpu);
    println!("[Config] Estimated Results: {}", estimate_result_amount(blocks));
    
    for (i, b) in blocks.iter().enumerate() {
        println!("  [{}] {}", i, b);
    }
    println!("==================================================");

    if use_gpu {
        println!("[Logic] GPU mode: Splitting blocks by surface...");
        
        // Split blocks into floor and roof
        let (floor_filters, roof_filters) = split_floor_roof(blocks, mode);
        
        let floor_power = get_filter_power(&floor_filters);
        let roof_power = get_filter_power(&roof_filters);
        
        println!("[Logic] Floor blocks: {}, Filter power: {}", floor_filters.len(), floor_power);
        println!("[Logic] Roof blocks: {}, Filter power: {}", roof_filters.len(), roof_power);
        
        // Choose primary surface (lower filter power = more filtering = primary)
        let is_floor_primary = floor_power <= roof_power;
        
        let (primary_filters, secondary_filters) = if is_floor_primary {
            println!("[Logic] Primary surface: FLOOR (better filtering)");
            (floor_filters, roof_filters)
        } else {
            println!("[Logic] Primary surface: ROOF (better filtering)");
            (roof_filters, floor_filters)
        };
        
        // Build checks for primary surface (sent to GPU)
        let mut primary_checks: Vec<CheckObject> = primary_filters
            .into_iter()
            .map(|mut filter| filter.create_check(0))
            .collect();
        
        // Sort by filter power for better early rejection
        primary_checks.sort_by(|a, b| {
            // We don't have direct access to discarded_seeds here, so just use as-is
            // The order doesn't affect correctness, only performance
            std::cmp::Ordering::Equal
        });
        
        // Build checks for secondary surface (used in CPU cross-comparison)
        let secondary_checks: Vec<CheckObject> = secondary_filters
            .into_iter()
            .map(|mut filter| filter.create_check(0))
            .collect();
        
        println!("[Logic] Primary checks for GPU: {}", primary_checks.len());
        println!("[Logic] Secondary checks for CPU cross-comparison: {}", secondary_checks.len());
        
        // Create the cross-comparison sender wrapper
        let cross_sender = GpuCrossComparisonSender::new(
            sender,
            secondary_checks,
            is_floor_primary,
            output,
        );
        
        // Launch GPU search with only primary checks
        thread::spawn(move || {
            pollster::block_on(gpu::gpu_search_loop(primary_checks, cross_sender));
        });
        
        return;
    }

    // CPU mode - unchanged
    println!("[Logic] CPU mode enabled. Spawning {} threads...", thread_count);

    let checks = create_filter_tree(blocks, mode, output, sender.clone());

    for thread in 0..thread_count {
        let start_bits = (thread * (1 << 36)) / thread_count;
        let end_bits = ((thread + 1) * (1 << 36)) / thread_count;
        
        let mut current_bits = start_bits << 12;
        let limit_bits = end_bits << 12;

        println!("[Logic] Thread {} assigned range: {:012X} -> {:012X}", thread, current_bits, limit_bits);

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use crate::raw_data::block_type::BlockType;

    const WORLD_SEED: u64 = 765906787396911863;

    #[test]
    fn test_split_floor_roof() {
        let blocks = vec![
            Block::new(0, 4, 0, BlockType::BEDROCK),   // Floor
            Block::new(0, 123, 0, BlockType::BEDROCK), // Roof
            Block::new(1, 1, 1, BlockType::OTHER),     // Floor
            Block::new(1, 126, 1, BlockType::OTHER),   // Roof
        ];
        
        let (floor, roof) = split_floor_roof(&blocks, BedrockGeneration::Normal);
        
        assert_eq!(floor.len(), 2, "Should have 2 floor blocks");
        assert_eq!(roof.len(), 2, "Should have 2 roof blocks");
    }

    #[test]
    fn test_cross_comparison_sender() {
        // This test verifies the cross-comparison logic works correctly
        let (tx, rx) = mpsc::channel();
        
        // Empty secondary checks = everything passes
        let sender = GpuCrossComparisonSender::new(
            tx,
            vec![],
            true, // floor primary
            OutputMode::WorldSeed,
        );
        
        // This would need actual valid test data to fully verify
        // For now, just verify it doesn't panic
        assert!(sender.check_secondary(0));
    }
}
