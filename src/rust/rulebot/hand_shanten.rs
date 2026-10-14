//! The shanten backend the rules call, plus the gates that hold it to them.
//!
//! The rules lean on one non-obvious property: fed a 3n+2 hand, the
//! backend answers with the shanten the best discard leaves, except a
//! complete hand reads -1 rather than 0. riichienv-core's
//! `calc_shanten_from_counts` is the backend; the test below holds it to
//! that property (a deterministic-LCG sweep).

use riichienv_core::shanten::calc_shanten_from_counts;

/// Shanten of `hand` with `n_sets` sets still to form in the concealed part.
#[inline]
pub fn shanten_of(hand: &[u8; 34], n_sets: u8) -> i8 {
    calc_shanten_from_counts(hand, n_sets)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Deterministic xorshift so the sweep needs no rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn shuffle<T>(&mut self, v: &mut [T]) {
            for i in (1..v.len()).rev() {
                let j = (self.next() % (i as u64 + 1)) as usize;
                v.swap(i, j);
            }
        }
    }

    /// A 3n+2 hand must read as its best discard's shanten, and a complete
    /// hand as -1.
    #[test]
    fn accepts_3n2_hands() {
        let mut rng = Lcg(10_000);
        let mut wall: Vec<u8> = (0..34).flat_map(|t| [t; 4]).collect();

        for _ in 0..2000 {
            rng.shuffle(&mut wall);
            for n_sets in 1..=4_u8 {
                let size = (n_sets * 3 + 2) as usize;
                let mut hand = [0_u8; 34];
                for &t in &wall[..size] {
                    hand[t as usize] += 1;
                }
                if hand.iter().any(|&c| c > 4) {
                    continue;
                }

                let direct = shanten_of(&hand, n_sets);
                let mut best = i8::MAX;
                for tile_type in 0..34 {
                    if hand[tile_type] == 0 {
                        continue;
                    }
                    hand[tile_type] -= 1;
                    best = best.min(shanten_of(&hand, n_sets));
                    hand[tile_type] += 1;
                }
                if direct == -1 {
                    assert_eq!(best, 0, "complete hand, n_sets {n_sets}");
                } else {
                    assert_eq!(direct, best, "n_sets {n_sets}, hand {hand:?}");
                }
            }
        }
    }
}
