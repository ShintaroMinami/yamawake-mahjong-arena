//! The hashing every hand cache uses: hands packed into a `u128` (3 bits per
//! tile type) and a multiplicative hasher for such keys (FxHash's constant).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative hasher for packed keys (FxHash's constant).
#[derive(Default)]
pub struct PackedHasher(u64);

const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for PackedHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
        }
    }
    fn write_u128(&mut self, v: u128) {
        self.0 = (self.0.rotate_left(5) ^ v as u64).wrapping_mul(K);
        self.0 = (self.0.rotate_left(5) ^ (v >> 64) as u64).wrapping_mul(K);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A `HashMap` over packed keys.
pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<PackedHasher>>;

/// 34 counts (0..=4) in 3 bits each, tile type 0 highest.
pub fn pack(counts: &[u8; 34]) -> u128 {
    counts.iter().fold(0u128, |acc, &n| acc << 3 | u128::from(n))
}
