use crate::raw_data::block::Block;
use crate::raw_data::block_type::BlockType;
use crate::raw_data::modes::BedrockGeneration;
use crate::MASK48;
use bytemuck::{Pod, Zeroable};
use java_random::JAVA_LCG;

#[derive(Clone, Debug)]
pub struct BlockFilter {
    pos_hash: u64,
    lower_bound: u64,
    upper_bound: u64,
    possible_range: u64,
}

impl BlockFilter {
    pub fn from(b: &Block, mode: BedrockGeneration) -> BlockFilter {
        Self::new(b.x, b.y, b.z, b.block_type, mode)
    }

    pub fn new(x: i32, mut y: i32, z: i32, block_type: BlockType, mode: BedrockGeneration) -> Self {
        let (lower_bound, upper_bound) = Self::bounds(y, block_type);
        if mode == BedrockGeneration::Paper1_18 {
            y = if y > 5 { 122 } else { 0 }
        }
        let pos_hash = BlockFilter::hashcode(x, y, z) ^ JAVA_LCG.multiplier;

        Self {
            pos_hash,
            lower_bound,
            upper_bound,
            possible_range: MASK48,
        }
    }

    pub fn create_check(&mut self, lower_bits: u64) -> CheckObject {
        let lower_bits_mask = (1 << lower_bits) - 1;
        self.check_with_bits(lower_bits);
        CheckObject::new(
            self.pos_hash,
            self.lower_bound,
            self.upper_bound,
            lower_bits_mask,
        )
    }

    /// Figure out how many seeds an operation filters
    pub fn discarded_seeds(&self, lower_bits: u64) -> f64 {
        let lower_bits_mask = (1 << lower_bits) - 1;
        let bound = self.bound() + lower_bits_mask * JAVA_LCG.multiplier;
        let success_chance = bound as f64 / self.possible_range as f64;
        let fail_chance = 1.0 - success_chance;
        if fail_chance <= 0.0 {
            return 0.0;
        }
        fail_chance * (1 << lower_bits) as f64
    }

    /// The chance for new info decreases as we use more bits
    fn check_with_bits(&mut self, lower_bits: u64) {
        let lower_bits_mask = (1 << lower_bits) - 1;
        let jiggle_room = lower_bits_mask * JAVA_LCG.multiplier;
        let new_range = jiggle_room + self.bound();
        assert!(new_range < self.possible_range);
        self.possible_range = new_range;
    }

    fn bound(&self) -> u64 {
        assert!(self.upper_bound > self.lower_bound);
        self.upper_bound - self.lower_bound
    }

    fn hashcode(x: i32, y: i32, z: i32) -> u64 {
        let mut pos_hash =
            (x.wrapping_mul(3129871)) as i64 ^ ((z as i64).wrapping_mul(116129781)) ^ y as i64;
        pos_hash = pos_hash
            .wrapping_mul(pos_hash)
            .wrapping_mul(42317861)
            .wrapping_add(pos_hash.wrapping_mul(11));
        let pos_hash = pos_hash as u64;
        pos_hash >> 16
    }

    fn bounds(mut layer: i32, block_type: BlockType) -> (u64, u64) {
        let mut lower_bound = 0.0;
        let mut upper_bound = 1.0;

        if layer > 5 {
            layer -= 122;
            let bound = (5 - layer) as f64 / 5.0;
            match block_type {
                BlockType::BEDROCK => lower_bound = bound,
                BlockType::OTHER => upper_bound = bound,
            }
        } else {
            let bound = (5 - layer) as f64 / 5.0;
            match block_type {
                BlockType::BEDROCK => upper_bound = bound,
                BlockType::OTHER => lower_bound = bound,
            }
        }

        lower_bound *= MASK48 as f64;
        upper_bound *= MASK48 as f64;

        (lower_bound as u64, upper_bound as u64)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Default, Copy, Pod, Zeroable)]
pub struct CheckObject {
    pub pos_hash: u64,
    pub condition: u64,
    pub offset: u64,
}

impl CheckObject {
    fn new(pos_hash: u64, lower_bound: u64, upper_bound: u64, lower_bit_mask: u64) -> Self {
        let offset = MASK48 - upper_bound;
        let pos_hash = pos_hash & (MASK48 - lower_bit_mask);
        let condition = lower_bound
            .wrapping_add(offset)
            .wrapping_sub(lower_bit_mask * JAVA_LCG.multiplier);
        Self {
            pos_hash,
            condition,
            offset,
        }
    }

    /// Check if this seed FAILS the check (should be rejected)
    /// Returns true if seed is INCONSISTENT with this block
    /// Returns false if seed is CONSISTENT with this block (passes)
    #[inline(always)]
    pub fn check(&self, upper_bits: u64) -> bool {
        ((upper_bits ^ self.pos_hash)
            .wrapping_mul(JAVA_LCG.multiplier)
            .wrapping_add(self.offset)
            & MASK48)
            < self.condition
    }
}

/// Calculate the filter power of a set of blocks
/// Lower value = more filtering power = fewer expected results
pub fn get_filter_power(filters: &[BlockFilter]) -> u64 {
    let resulting_seeds: f64 = filters
        .iter()
        .map(|block| 1.0 - block.discarded_seeds(0))
        .product::<f64>()
        * (1u64 << 48) as f64;
    resulting_seeds as u64
}

#[cfg(test)]
mod tests {
    use crate::block_data::{BlockFilter, CheckObject};
    use crate::raw_data::block_type::BlockType;
    use crate::raw_data::modes::BedrockGeneration;
    use crate::MASK48;
    use java_random::JAVA_LCG;

    #[test]
    fn test_hashcode() {
        let block = BlockFilter::new(-98, 4, -469, BlockType::BEDROCK, BedrockGeneration::Normal);
        assert_eq!(block.pos_hash, 99261249361405 ^ JAVA_LCG.multiplier)
    }

    #[test]
    fn test_filler_check() {
        // Default CheckObject should always pass (return false)
        assert!(!CheckObject::default().check(MASK48));
        assert!(!CheckObject::default().check(0));
        assert!(!CheckObject::default().check(12345));
    }

    #[test]
    fn test_check_semantics() {
        // Verify check returns true for FAIL, false for PASS
        // A bedrock block at y=4 should have upper_bound = 0.2 * MASK48
        let mut filter = BlockFilter::new(0, 4, 0, BlockType::BEDROCK, BedrockGeneration::Normal);
        let check = filter.create_check(0);

        // The check is: ((seed ^ pos_hash) * mult + offset) & MASK48 < condition
        // For valid seeds, the result should be >= condition (return false)
        // For invalid seeds, the result should be < condition (return true)

        // We can't easily compute valid seeds here, but we can verify the structure
        assert!(check.condition > 0, "Condition should be positive");
        assert!(check.offset > 0, "Offset should be positive for bedrock at y=4");
    }

    #[test]
    fn test_bounds_floor_bedrock() {
        // For bedrock at y=4 (floor), probability is (5-4)/5 = 0.2
        // So upper_bound = 0.2 * MASK48
        let (lower, upper) = BlockFilter::bounds(4, BlockType::BEDROCK);
        assert_eq!(lower, 0);
        let expected_upper = (0.2 * MASK48 as f64) as u64;
        assert!((upper as i64 - expected_upper as i64).abs() < 2,
                "upper={} expected={}", upper, expected_upper);
    }

    #[test]
    fn test_bounds_floor_other() {
        // For non-bedrock at y=4 (floor), probability is 1 - 0.2 = 0.8
        // So lower_bound = 0.2 * MASK48
        let (lower, upper) = BlockFilter::bounds(4, BlockType::OTHER);
        let expected_lower = (0.2 * MASK48 as f64) as u64;
        assert!((lower as i64 - expected_lower as i64).abs() < 2);
        assert_eq!(upper, MASK48 as u64);
    }

    #[test]
    fn test_bounds_roof_bedrock() {
        // For bedrock at y=123 (roof), layer = 123-122 = 1
        // Probability is (5-1)/5 = 0.8
        // So lower_bound = 0.8 * MASK48
        let (lower, upper) = BlockFilter::bounds(123, BlockType::BEDROCK);
        let expected_lower = (0.8 * MASK48 as f64) as u64;
        assert!((lower as i64 - expected_lower as i64).abs() < 2);
        assert_eq!(upper, MASK48 as u64);
    }
}
