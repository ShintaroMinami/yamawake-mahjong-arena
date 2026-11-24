//! Shanten, ukeire and two-step ukeire for one decision.
//!
//! Every count is over the tiles still unseen from this seat; the unseen
//! counts are fixed for one decision (a discard moves a tile from the hand
//! to the river, and both are seen). Which draws improve a hand and which
//! discards keep its shanten depend on the hand alone, so they are kept per
//! thread across decisions (`shanten::facts_packed`); shanten itself is not (see
//! `UkeireTables::shanten`).

use riichienv_core::shanten::calc_shanten_from_counts;

use crate::eval::hash::pack;
use crate::eval::shanten::{Facts, HandShanten, facts_packed, facts_peek, keeping_packed};

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
/// thread (`shanten::facts_packed`).
pub struct UkeireTables {
    unseen: [u8; 34],
    /// The bits of the unseen counts (0..=4), one mask per bit, so the unseen
    /// copies in a tile mask are three population counts.
    planes: [u64; 3],
}

/// One copy of tile type `t` in a `hash::pack` key.
fn pack_bit(t: usize) -> u128 {
    1u128 << (99 - 3 * t)
}

/// The shanten and the improving draws of a 3n+1 hand, kept per thread (`key`: its `pack`).
///
/// On a hand of seven kinds or more the DP's improving draws (`dp::improving`) probe the same tile types as
/// `probe_mask`, so they are the same function there: a value the DP already keeps (`Facts::DpImproving`) is
/// read back instead of computed.
fn improving(key: u128, hand: &[u8; 34]) -> (i8, u64) {
    facts_packed(Facts::Improving, key, hand, |hand| {
        if hand.iter().filter(|&&c| c > 0).count() >= 7
            && let Some(v) = facts_peek(Facts::DpImproving, key)
        {
            return v;
        }
        let hs = HandShanten::new(hand);
        let base = hs.shanten();
        (base, hs.lowering_draws(probe_mask(hand, base)))
    })
}

/// The improving draws of a 3n+1 hand, kept per thread (`key`: its `pack`).
fn improving_draws(key: u128, hand: &[u8; 34]) -> u64 {
    improving(key, hand).1
}

impl UkeireTables {
    pub fn new(unseen: [u8; 34]) -> Self {
        let mut planes = [0u64; 3];
        for (t, &c) in unseen.iter().enumerate() {
            for (b, p) in planes.iter_mut().enumerate() {
                *p |= u64::from(c >> b & 1) << t;
            }
        }
        Self { unseen, planes }
    }

    /// The unseen copies of the tile types in `mask`, and those of them that
    /// have any.
    #[inline]
    fn unseen_in(&self, mask: u64) -> (u32, u64) {
        let [p1, p2, p4] = self.planes;
        let count = (mask & p1).count_ones() + 2 * (mask & p2).count_ones() + 4 * (mask & p4).count_ones();
        (count, mask & (p1 | p2 | p4))
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
    /// The waits of a 3n+1 hand (the tile types whose draw completes it, a type held four times excepted) as a bit
    /// mask; 0 when it is not tenpai. At shanten 0 they are its improving draws (kept per thread); above, one tile
    /// cannot complete it.
    pub fn waits(&mut self, hand: &[u8; 34]) -> u64 {
        match improving(pack(hand), hand) {
            (0, draws) => draws,
            _ => 0,
        }
    }

    pub fn ukeire(&mut self, hand: &[u8; 34]) -> (u32, u64) {
        self.unseen_in(improving_draws(pack(hand), hand))
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
        let k13 = pack(hand);
        while adv != 0 {
            let t = adv.trailing_zeros() as usize;
            adv &= adv - 1;
            h[t] += 1;
            let k14 = k13 + pack_bit(t);
            // The discards that keep the lowered shanten s13 - 1 (that of `h`);
            // discarding the drawn tile again gives the hand back.
            let (_, keep) = keeping_packed(k14, &h);
            let mut ds = keep & !(1u64 << t);
            let mut best = 0u32;
            while ds != 0 {
                let d = ds.trailing_zeros() as usize;
                ds &= ds - 1;
                h[d] -= 1;
                let (u, m) = self.unseen_in(improving_draws(k14 - pack_bit(d), &h));
                h[d] += 1;
                best = best.max(u - (m >> t & 1) as u32);
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
    /// `waits` equals its definition (the tile types, held fewer than four times, whose addition completes the hand)
    /// on hands of every 3n+1 size: wins of each form with one tile taken out, some of them changed, and random ones.
    #[test]
    fn waits_are_the_completing_tiles() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let orphans = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        let mut tables = UkeireTables::new([4; 34]);
        let mut tenpai = 0;
        for round in 0..20_000 {
            let n_sets = [4, 4, 4, 3, 2, 1, 0][round % 7];
            let mut h = [0u8; 34];
            let add = |h: &mut [u8; 34], t: usize| h[t] < 4 && { h[t] += 1; true };
            match (round / 7) % 4 {
                0 | 3 => {
                    let p = loop { let t = next(34); if h[t] <= 2 { break t; } };
                    add(&mut h, p);
                    add(&mut h, p);
                    let mut sets = 0;
                    while sets < n_sets {
                        let t = next(34);
                        let set = if t < 27 && t % 9 < 7 && next(2) == 0 { [t, t + 1, t + 2] } else { [t; 3] };
                        if set.iter().all(|&u| h[u] + set.iter().filter(|&&v| v == u).count() as u8 <= 4) {
                            set.iter().for_each(|&u| { add(&mut h, u); });
                            sets += 1;
                        }
                    }
                }
                1 if n_sets == 4 => while h.iter().filter(|&&c| c == 2).count() < 7 {
                    let t = next(34);
                    if h[t] == 0 {
                        h[t] = 2;
                    }
                },
                2 if n_sets == 4 => {
                    orphans.iter().for_each(|&t| h[t] = 1);
                    h[orphans[next(13)]] += 1;
                }
                _ => {
                    let mut n = 0;
                    while n < 3 * n_sets + 2 {
                        n += usize::from(add(&mut h, next(34)));
                    }
                }
            }
            // One tile out, then maybe one swapped (round % 4 == 3: always).
            let held = |h: &[u8; 34], k: usize| (0..34).filter(|&t| h[t] > 0).nth(k).unwrap();
            let kinds = h.iter().filter(|&&c| c > 0).count();
            let out = held(&h, next(kinds));
            h[out] -= 1;
            if round % 4 == 3 || next(3) == 0 {
                let kinds = h.iter().filter(|&&c| c > 0).count();
                let a = held(&h, next(kinds));
                let b = next(34);
                if h[b] < 4 {
                    h[a] -= 1;
                    h[b] += 1;
                }
            }
            let want = (0..34).filter(|&w| h[w] < 4 && { let mut g = h; g[w] += 1; tables.shanten(&g) < 0 }).fold(0u64, |m, w| m | 1 << w);
            assert_eq!(tables.waits(&h), want, "{h:?}");
            tenpai += usize::from(want != 0);
        }
        assert!(tenpai > 5_000, "tenpai {tenpai}");
    }

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
