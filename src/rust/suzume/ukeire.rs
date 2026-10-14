//! Shanten, ukeire and two-step ukeire for one decision.
//!
//! Every count is over the tiles still unseen from this seat; the unseen
//! counts are fixed for one decision (a discard moves a tile from the hand
//! to the river, and both are seen). Which draws improve a hand and which
//! discards keep its shanten depend on the hand alone, so they are kept per
//! thread across decisions (`shanten::facts`); shanten itself is not (see
//! `UkeireTables::shanten`).

use riichienv_core::shanten::calc_shanten_from_counts;

use crate::eval::shanten::{Facts, HandShanten, facts, keeping};

/// The terminal and honor tile types (kokushi).
const ORPHANS: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

/// Tile types whose draw can lower the shanten of a 3n+1 `hand`: same-suit
/// neighbours within two of a held number tile and held honors (a regular
/// hand gains nothing from an isolated tile), every tile type while the
/// hand holds fewer than seven kinds (a new kind helps chiitoitsu), and the
/// orphans when kokushi is at least as close as `base`.
fn probe_mask(hand: &[u8; 34], base: i8) -> u64 {
    let kinds = hand.iter().filter(|&&c| c > 0).count();
    if kinds < 7 {
        return (1u64 << 34) - 1;
    }
    let mut mask = 0u64;
    for t in 0..34 {
        if hand[t] == 0 {
            continue;
        }
        if t >= 27 {
            mask |= 1 << t;
        } else {
            let r = t % 9;
            for x in t - r.min(2)..=t + (8 - r).min(2) {
                mask |= 1 << x;
            }
        }
    }
    let n_orphan_kinds = ORPHANS.iter().filter(|&&t| hand[t] > 0).count() as i8;
    let orphan_pair = ORPHANS.iter().any(|&t| hand[t] >= 2) as i8;
    if 13 - n_orphan_kinds - orphan_pair <= base {
        for t in ORPHANS {
            mask |= 1 << t;
        }
    }
    mask
}

/// Hand metrics for one decision (fixed unseen counts); the improving
/// draws of a hand, which do not depend on the unseen counts, are kept per
/// thread (`shanten::facts`).
pub struct UkeireTables {
    unseen: [u8; 34],
}

impl UkeireTables {
    pub fn new(unseen: [u8; 34]) -> Self {
        Self { unseen }
    }

    /// Shanten of a closed hand (a 3n+2 hand reads as its best discard,
    /// -1 when complete). Not memoized: riichienv's table lookup costs less
    /// than packing and hashing the hand for a memo (three times faster
    /// two-step ukeire without the memo).
    pub fn shanten(&mut self, hand: &[u8; 34]) -> i8 {
        let n: u8 = hand.iter().sum();
        calc_shanten_from_counts(hand, n / 3)
    }

    /// Ukeire of a 3n+1 hand: the unseen copies of every tile type whose
    /// draw lowers the shanten, and those tile types as a bit mask.
    pub fn ukeire(&mut self, hand: &[u8; 34]) -> (u32, u64) {
        let (_, improving) = facts(Facts::Improving, hand, |hand| {
            let hs = HandShanten::new(hand);
            let base = hs.shanten();
            let mut probe = probe_mask(hand, base);
            let mut mask = 0u64;
            while probe != 0 {
                let t = probe.trailing_zeros() as usize;
                probe &= probe - 1;
                if hand[t] < 4 && hs.plus(t) < base {
                    mask |= 1 << t;
                }
            }
            (base, mask)
        });
        let (mut count, mut mask) = (0u32, 0u64);
        let mut m = improving;
        while m != 0 {
            let t = m.trailing_zeros() as usize;
            m &= m - 1;
            if self.unseen[t] > 0 {
                count += u32::from(self.unseen[t]);
                mask |= 1 << t;
            }
        }
        (count, mask)
    }

    /// Two-step ukeire of a 3n+1 hand at shanten `s13`, with its one-step
    /// ukeire: the sum over every advancing draw `t` of
    /// `unseen[t] x` the best ukeire left after drawing `t` and making a
    /// discard that keeps the lowered shanten (counted without the drawn
    /// copy). At tenpai there is no second step: the score is the ukeire.
    pub fn two_step(&mut self, hand: &[u8; 34], s13: i8) -> (u64, u32) {
        let (uk1, mut adv) = self.ukeire(hand);
        if s13 <= 0 {
            return (u64::from(uk1), uk1);
        }
        let mut total = 0u64;
        let mut h = *hand;
        while adv != 0 {
            let t = adv.trailing_zeros() as usize;
            adv &= adv - 1;
            h[t] += 1;
            // The discards that keep the lowered shanten s13 - 1 (that of `h`).
            let (_, keep) = keeping(&h);
            let mut best = 0u32;
            for d in 0..34 {
                // Discarding the drawn tile again gives the hand back.
                if h[d] == 0 || d == t {
                    continue;
                }
                h[d] -= 1;
                if keep >> d & 1 == 1 {
                    let (u, m) = self.ukeire(&h);
                    best = best.max(u - (m >> t & 1) as u32);
                }
                h[d] += 1;
            }
            h[t] -= 1;
            total += u64::from(self.unseen[t]) * u64::from(best);
        }
        (total, uk1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suzume::test_util::counts;

    /// The pruned probe agrees with probing every tile type.
    #[test]
    fn probe_is_exact() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut wall: Vec<u8> = (0..34).flat_map(|t| [t; 4]).collect();
        for _ in 0..3000 {
            for i in (1..wall.len()).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                wall.swap(i, (x % (i as u64 + 1)) as usize);
            }
            let mut hand = [0u8; 34];
            for &t in &wall[..13] {
                hand[t as usize] += 1;
            }
            let mut unseen = [4u8; 34];
            for t in 0..34 {
                unseen[t] -= hand[t];
            }
            let mut tables = UkeireTables::new(unseen);
            let (_, mask) = tables.ukeire(&hand);
            let base = tables.shanten(&hand);
            let mut full = 0u64;
            for t in 0..34 {
                if unseen[t] == 0 {
                    continue;
                }
                let mut h = hand;
                h[t] += 1;
                if tables.shanten(&h) < base {
                    full |= 1 << t;
                }
            }
            assert_eq!(mask, full, "hand {hand:?}");
        }
    }

    #[test]
    fn ukeire_of_a_ryanmen_tenpai() {
        // 23m waits on 1m/4m: 4 + 4 unseen (own 2m3m do not block them).
        let hand = counts("23m456p789s11122z");
        let mut unseen = [4u8; 34];
        for t in 0..34 {
            unseen[t] -= hand[t];
        }
        let mut tables = UkeireTables::new(unseen);
        assert_eq!(tables.shanten(&hand), 0);
        let (n, mask) = tables.ukeire(&hand);
        assert_eq!(mask, 1 << 0 | 1 << 3);
        assert_eq!(n, 8);
        assert_eq!(tables.two_step(&hand, 0), (8, 8));
    }
}
