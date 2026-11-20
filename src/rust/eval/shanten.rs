//! Shanten with one tile added or removed, and the `facts_packed` cache of hand
//! facts (a shanten and a tile mask per hand).
//!
//! `HandShanten` lives in the vendored riichienv-core (a yamawake patch to
//! `src/engine/riichienv-core/src/shanten.rs`, `src/engine/riichienv-core.patch`):
//! it keeps a hand's four Nyanten suit keys and its chiitoitsu / kokushi
//! tallies and answers the shanten with one tile added or removed by
//! rehashing that tile's suit only. Its values equal
//! `calc_shanten_from_counts` (tested below).

use std::cell::RefCell;

use crate::eval::dp::Counts;
use crate::eval::hash::pack;

pub use riichienv_core::shanten::HandShanten;

/// The kinds of hand facts `facts_packed` keeps: a shanten and a tile mask, each a
/// pure function of the counts alone.
#[derive(Clone, Copy)]
pub enum Facts {
    /// `keeping`.
    Keeping,
    /// The DP's improving draws (`dp::effective`).
    DpImproving,
    /// suzume' ukeire draws (`suzume::ukeire`).
    Improving,
}

/// Entries per kind of a thread's table (as a power of two). Small on purpose:
/// most misses are hands never seen before, not hands pushed out (no limit at
/// all gave the same hit rate), so a bigger table only reads memory further
/// away (2^12 was the fastest on an Apple M4 Pro and an AMD EPYC 4564P, by
/// about 1% over 2^14; 2^19 clearly slower).
const FACTS_BITS: u32 = 12;

/// A key no hand has (`pack` fills 102 bits).
const EMPTY: u128 = u128::MAX;

/// One direct-mapped slot: a colliding hand replaces what it finds there, so
/// the table has a fixed size and is never cleared.
#[derive(Clone, Copy)]
struct Slot {
    key: u128,
    mask: u64,
    shanten: i8,
}

thread_local! {
    static FACTS: RefCell<[Box<[Slot]>; 3]> = RefCell::new(std::array::from_fn(|_| {
        vec![Slot { key: EMPTY, mask: 0, shanten: 0 }; 1 << FACTS_BITS].into_boxed_slice()
    }));
}

/// `f(hand)`, kept per thread under `kind` (exact: `f` must depend on the
/// counts alone; a slot holds its whole key). `key`: the hand's `hash::pack`.
#[inline]
pub fn facts_packed(kind: Facts, key: u128, hand: &Counts, f: impl FnOnce(&Counts) -> (i8, u64)) -> (i8, u64) {
    let k = kind as usize;
    let mixed = (key as u64).rotate_left(5) ^ (key >> 64) as u64;
    let i = (mixed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - FACTS_BITS)) as usize;
    let hit = FACTS.with_borrow(|m| {
        let s = &m[k][i];
        (s.key == key).then_some((s.shanten, s.mask))
    });
    if let Some(v) = hit {
        return v;
    }
    let v = f(hand);
    FACTS.with_borrow_mut(|m| m[k][i] = Slot { key, mask: v.1, shanten: v.0 });
    v
}

/// The shanten of a 3n+2 hand and the held tile types whose discard keeps
/// it (bit `d`), kept per thread.
pub fn keeping(hand: &Counts) -> (i8, u64) {
    keeping_packed(pack(hand), hand)
}

/// `keeping` with the hand already packed (`hash::pack`).
pub fn keeping_packed(key: u128, hand: &Counts) -> (i8, u64) {
    facts_packed(Facts::Keeping, key, hand, |hand| {
        let hs = HandShanten::new(hand);
        (hs.shanten(), hs.keeping_discards())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use riichienv_core::shanten::calc_shanten_from_counts;

    fn reference(c: &Counts) -> i8 {
        let n: u8 = c.iter().sum();
        calc_shanten_from_counts(c, n / 3)
    }

    /// Every hand size a seat holds (1..=14 tiles), random hands plus
    /// chiitoitsu- and kokushi-heavy ones: `shanten`, `plus` and `minus`
    /// equal riichienv's value for the changed hand.
    #[test]
    fn matches_riichienv() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let orphans = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        for round in 0..60_000 {
            let n_tiles = [14, 13, 11, 10, 8, 7, 5, 4, 2, 1][round % 10];
            let mut c = [0u8; 34];
            let mut placed = 0;
            while placed < n_tiles {
                let t = match round % 3 {
                    0 => next(34),
                    1 => orphans[next(13)],
                    _ => next(9) + 9 * next(2),
                };
                if c[t] < 4 && (round % 7 != 0 || c[t] < 2) {
                    c[t] += 1;
                    placed += 1;
                }
            }
            let hs = HandShanten::new(&c);
            assert_eq!(hs.shanten(), reference(&c), "{c:?}");
            let base = hs.shanten();
            if n_tiles < 14 {
                let slow = (0..34).filter(|&t| c[t] < 4 && hs.plus(t) < base).fold(0u64, |m, t| m | 1 << t);
                assert_eq!(hs.lowering_draws((1u64 << 34) - 1), slow, "{c:?}");
            }
            let slow = (0..34).filter(|&t| c[t] > 0 && hs.minus(t) == base).fold(0u64, |m, t| m | 1 << t);
            assert_eq!(hs.keeping_discards(), slow, "{c:?}");
            for t in 0..34 {
                if c[t] < 4 && n_tiles < 14 {
                    let mut d = c;
                    d[t] += 1;
                    assert_eq!(hs.plus(t), reference(&d), "{c:?} + {t}");
                }
                if c[t] > 0 {
                    let mut d = c;
                    d[t] -= 1;
                    assert_eq!(hs.minus(t), reference(&d), "{c:?} - {t}");
                }
            }
        }
    }

    /// The riichi check of the engine's legal actions (a yamawake patch to
    /// `state/legal_actions.rs`) skips its per-discard `is_tenpai` tests when
    /// the hand's shanten is 1 or more: on hands of every size a seat holds
    /// after a draw (0..=4 ankan), near wins of every form among them, no
    /// discard of such a hand leaves it tenpai.
    #[test]
    fn far_hands_have_no_tenpai_discard() {
        use riichienv_core::hand_evaluator::HandEvaluator;
        use riichienv_core::types::{Meld, MeldType};
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let orphans = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        let (mut tenpai, mut skipped) = (0, 0);
        for round in 0..40_000 {
            let n_kan = [0, 0, 0, 1, 2, 3, 4][round % 7];
            let n_tiles = 14 - 3 * n_kan;
            let mut c = [0u8; 34];
            let kans: Vec<usize> = (0..n_kan).map(|_| loop {
                let t = next(34);
                if c[t] == 0 {
                    c[t] = 4;
                    break t;
                }
            }).collect();
            // A win of the round's form, then up to two of its tiles swapped (or a random hand).
            let mut hand: Vec<usize> = Vec::new();
            let take = |c: &mut [u8; 34], hand: &mut Vec<usize>, t: usize| c[t] < 4 && { c[t] += 1; hand.push(t); true };
            match (round / 7) % 4 {
                0 => {
                    let pair = loop {
                        let t = next(34);
                        if c[t] <= 2 {
                            break t;
                        }
                    };
                    take(&mut c, &mut hand, pair);
                    take(&mut c, &mut hand, pair);
                    while hand.len() < n_tiles {
                        let t = next(34);
                        let run = t < 27 && t % 9 < 7 && next(2) == 0;
                        let set = if run { [t, t + 1, t + 2] } else { [t; 3] };
                        if set.iter().all(|&u| c[u] + set.iter().filter(|&&v| v == u).count() as u8 <= 4) {
                            set.iter().for_each(|&u| { take(&mut c, &mut hand, u); });
                        }
                    }
                }
                1 if n_kan == 0 => {
                    while hand.len() < 14 {
                        let t = next(34);
                        if c[t] == 0 {
                            take(&mut c, &mut hand, t);
                            take(&mut c, &mut hand, t);
                        }
                    }
                }
                2 if n_kan == 0 => {
                    orphans.iter().for_each(|&t| { take(&mut c, &mut hand, t); });
                    take(&mut c, &mut hand, orphans[next(13)]);
                }
                _ => while hand.len() < n_tiles {
                    take(&mut c, &mut hand, next(34));
                },
            }
            for _ in 0..next(3) {
                let i = next(hand.len());
                let t = next(34);
                if c[t] < 4 && !kans.contains(&t) {
                    c[hand[i]] -= 1;
                    c[t] += 1;
                    hand[i] = t;
                }
            }
            // Tile ids: copy k of type t is 4 t + k (the kans hold all four).
            let mut copy = [0u8; 34];
            let ids: Vec<u8> = hand.iter().map(|&t| { copy[t] += 1; (4 * t) as u8 + copy[t] - 1 }).collect();
            let melds: Vec<Meld> = kans.iter().map(|&t| Meld::new(MeldType::Ankan, (0..4).map(|k| (4 * t + k) as u8).collect(), false, -1, None)).collect();
            let mut counts = [0u8; 34];
            hand.iter().for_each(|&t| counts[t] += 1);
            let far = calc_shanten_from_counts(&counts, (hand.len() / 3) as u8) > 0;
            let any = (0..ids.len()).any(|i| {
                let mut rest = ids.clone();
                rest.remove(i);
                HandEvaluator::new(rest, melds.clone()).is_tenpai()
            });
            assert!(!(far && any), "{hand:?} with ankan {kans:?}");
            tenpai += usize::from(any);
            skipped += usize::from(far);
        }
        // The hands reach both sides of the skip.
        assert!(tenpai > 5_000 && skipped > 5_000, "tenpai {tenpai}, skipped {skipped}");
    }
}
