//! Shanten with one tile added or removed, and the `facts` cache of hand
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
use crate::eval::hash::{FastMap, pack};

pub use riichienv_core::shanten::HandShanten;

/// The kinds of hand facts `facts` keeps: a shanten and a tile mask, each a
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

/// Entries per kind before a thread's table starts over.
const FACTS_CAP: usize = 1 << 18;

thread_local! {
    static FACTS: RefCell<[FastMap<u128, (i8, u64)>; 3]> =
        RefCell::new(Default::default());
}

/// `f(hand)`, kept per thread under `kind` (exact: `f` must depend on the
/// counts alone; the table starts over past `FACTS_CAP` entries).
pub fn facts(kind: Facts, hand: &Counts, f: impl FnOnce(&Counts) -> (i8, u64)) -> (i8, u64) {
    facts_packed(kind, pack(hand), hand, f)
}

/// `facts` with the hand's counts already packed (`key`: 3 bits per tile
/// type, tile type 0 highest).
pub fn facts_packed(kind: Facts, key: u128, hand: &Counts, f: impl FnOnce(&Counts) -> (i8, u64)) -> (i8, u64) {
    let k = kind as usize;
    if let Some(v) = FACTS.with_borrow(|m| m[k].get(&key).copied()) {
        return v;
    }
    let v = f(hand);
    FACTS.with_borrow_mut(|m| {
        if m[k].len() >= FACTS_CAP {
            m[k].clear();
        }
        m[k].insert(key, v);
    });
    v
}

/// The shanten of a 3n+2 hand and the held tile types whose discard keeps
/// it (bit `d`), kept per thread.
pub fn keeping(hand: &Counts) -> (i8, u64) {
    facts(Facts::Keeping, hand, |hand| {
        let hs = HandShanten::new(hand);
        let s = hs.shanten();
        let mask = (0..34).filter(|&d| hand[d] > 0 && hs.minus(d) == s).fold(0u64, |m, d| m | 1 << d);
        (s, mask)
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
}
