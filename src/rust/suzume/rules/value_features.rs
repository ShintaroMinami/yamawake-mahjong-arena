//! The linear value score of a discard over features of the hand it leaves:
//! how the universal features' call options pick the discard after a call
//! when the DP does not value that hand (`calls::after_call_discard`).
//!
//! Every distinct legal discard is a candidate (shanten-breaking ones too),
//! except when the lowest shanten is within `params().dp_max_shanten` (the DP
//! could not value the hand: no draw left): then only the shanten-keeping
//! ones. The score is `sum_k w[k] * f_k` (`params().value_weights`) over the features `f_k` of
//! the 13 tiles left after the discard, read from this seat's view (unseen
//! counts, dora indicators, winds, red fives); the candidates within
//! `params().tie_band` of the best go to the fitted tie score
//! (`tile_features::tie_pick`):
//!
//! | feature | definition |
//! |---|---|
//! | `shanten_loss` | shanten after this discard minus the lowest shanten any legal discard leaves (0, 1, ...) |
//! | `ukeire1` | `ln(1 + n)`, n = unseen copies of the tile types that lower the shanten (`UkeireTables::ukeire`) |
//! | `ukeire2` | `ln(1 + n2)`, n2 = two-step ukeire (`UkeireTables::two_step`); 0 for a shanten-breaking discard |
//! | `dora_discarded` | dora han of the discarded tile (indicator dora, plus 1 for a red five) |
//! | `yakuhai_pairs` | yakuhai tile types held exactly twice; a double wind counts 2 |
//! | `yakuhai_triplets` | yakuhai tile types held three or more times; a double wind counts 2 |
//! | `yakuhai_singles` | yakuhai tile types held once; a double wind counts 2 |
//! | `yaochu` | terminal and honor tiles (copies) among the 13 |
//! | `tanyao_near` | 1 when `yaochu <= 2` (all simples within two swaps), else 0 |
//! | `flush_share` | (copies of the most held suit + honors) / 13 |
//! | `pairs` | tile types held two or more times |
//! | `chiitoi_shanten` | `6 - pairs + max(0, 7 - kinds)` |
//! | `isolated_honors` | honor tile types held exactly once |
//! | `isolated_terminals` | 1/9 tiles held once with no same-suit tile within two ranks |
//!
//! The dora kept plus `dora_discarded` is the same for every candidate of one
//! decision (the 14 tiles' dora), so a dora-kept feature would add nothing
//! (its weight would only shift `dora_discarded`'s); `yaochu` likewise only tells whether the discarded tile
//! is a terminal or honor. Features whose weight is 0 are not computed (the
//! ukeire ones cost most), and two-step ukeire is computed only for the
//! shanten-keeping discards (on every candidate it cost 2.5x the time; a
//! shanten-breaking discard's loss is the `shanten_loss` weight plus its one-step
//! ukeire, at its own higher shanten).

use crate::suzume::params::params;
use crate::suzume::ctx::{Ctx, DiscardCand};
use crate::suzume::rules::tile_features::{TIE_EPS, tie_pick};
use crate::eval::feat::tiles::is_yaochu;

/// Number of value features.
pub const N_VF: usize = 14;

/// Feature names, in feature-vector order.
pub const VF_NAMES: [&str; N_VF] = [
    "shanten_loss",
    "ukeire1",
    "ukeire2",
    "dora_discarded",
    "yakuhai_pairs",
    "yakuhai_triplets",
    "yakuhai_singles",
    "yaochu",
    "tanyao_near",
    "flush_share",
    "pairs",
    "chiitoi_shanten",
    "isolated_honors",
    "isolated_terminals",
];

const SHANTEN_LOSS: usize = 0;
const UKEIRE1: usize = 1;
const UKEIRE2: usize = 2;
const DORA_DISCARDED: usize = 3;
const YAKUHAI_PAIRS: usize = 4;
const YAKUHAI_TRIPLETS: usize = 5;
const YAKUHAI_SINGLES: usize = 6;
const YAOCHU: usize = 7;
const TANYAO_NEAR: usize = 8;
const FLUSH_SHARE: usize = 9;
const PAIRS: usize = 10;
const CHIITOI_SHANTEN: usize = 11;
const ISOLATED_HONORS: usize = 12;
const ISOLATED_TERMINALS: usize = 13;

/// The features with a nonzero weight (the others are not computed).
fn needed() -> [bool; N_VF] {
    std::array::from_fn(|k| params().value_weights[k] != 0.0)
}

/// Per candidate of `ctx.discard_cands()` (same order): the features
/// with a nonzero weight (the others 0) and the score.
fn candidate_features(ctx: &Ctx) -> Vec<([f64; N_VF], f64)> {
    let need = &needed();
    let Some(s) = ctx.min_shanten() else { return Vec::new() };
    let mut out: Vec<([f64; N_VF], f64)> = Vec::with_capacity(14);
    // A red five and a plain one leave the same shape: the ukeire once.
    let mut last: Option<(u8, f64, f64)> = None;
    for c in ctx.discard_cands() {
        let mut f = shape_features(ctx, c, s, need);
        let (u1, u2) = match last {
            Some((t, u1, u2)) if t == c.tile_type => (u1, u2),
            _ => ukeire_features(ctx, c, s, need),
        };
        last = Some((c.tile_type, u1, u2));
        f[UKEIRE1] = u1;
        f[UKEIRE2] = u2;
        let score = (0..N_VF).map(|k| f64::from(params().value_weights[k]) * f[k]).sum();
        out.push((f, score));
    }
    out
}

/// The winner's index in `ctx.discard_cands()` and its score: the candidates
/// within `params().tie_band` below the best score go to the fitted tie score. Within
/// DP range (no draw left) only the shanten-keeping discards compete.
pub fn value_ranked(ctx: &Ctx) -> Option<(usize, f64)> {
    let rows = candidate_features(ctx);
    let s = ctx.min_shanten()?;
    let keep_only = s <= params().dp_max_shanten;
    let cands = ctx.discard_cands();
    let eligible = |i: usize| !(keep_only && cands[i].shanten > s);
    let top = (0..rows.len()).filter(|&i| eligible(i)).map(|i| rows[i].1).fold(f64::NEG_INFINITY, f64::max);
    let floor = top - f64::from(params().tie_band) - TIE_EPS;
    let tied: Vec<usize> = (0..rows.len()).filter(|&i| eligible(i) && rows[i].1 >= floor).collect();
    let rel: Vec<f64> = tied.iter().map(|&i| rows[i].1 - top).collect();
    let picked: Vec<DiscardCand> = tied.iter().map(|&i| cands[i]).collect();
    let i = tied[tie_pick(ctx, &picked, &rel)?];
    Some((i, rows[i].1))
}

/// (ukeire1, ukeire2) of the hand `c` leaves, each only when needed;
/// ukeire2 only when `c` keeps the lowest shanten.
fn ukeire_features(ctx: &Ctx, c: &DiscardCand, min_shanten: i8, need: &[bool; N_VF]) -> (f64, f64) {
    if !(need[UKEIRE1] || need[UKEIRE2]) {
        return (0.0, 0.0);
    }
    let mut h = ctx.view.hand;
    h[c.tile_type as usize] -= 1;
    let mut tables = ctx.tables();
    if need[UKEIRE2] && c.shanten == min_shanten {
        let (two, one) = tables.two_step(&h, c.shanten);
        (f64::from(one).ln_1p(), (two as f64).ln_1p())
    } else {
        (f64::from(tables.ukeire(&h).0).ln_1p(), 0.0)
    }
}

/// Every feature but the ukeire ones (cheap: counts only).
fn shape_features(ctx: &Ctx, c: &DiscardCand, min_shanten: i8, need: &[bool; N_VF]) -> [f64; N_VF] {
    let v = ctx.view;
    let t0 = c.tile_type as usize;
    let mut h = v.hand;
    h[t0] -= 1;
    let mut f = hand_shape_features(ctx, &h, need);
    if need[SHANTEN_LOSS] {
        f[SHANTEN_LOSS] = f64::from(c.shanten - min_shanten);
    }
    f[DORA_DISCARDED] = if need[DORA_DISCARDED] {
        f64::from(v.dora_per_tile_type[t0]) + f64::from(u8::from(c.is_aka))
    } else {
        0.0
    };
    f
}

/// The count features of a 3n+1 concealed hand (`shanten_loss`,
/// `dora_discarded` and the ukeire ones left 0); unneeded ones zeroed.
fn hand_shape_features(ctx: &Ctx, h: &[u8; 34], need: &[bool; N_VF]) -> [f64; N_VF] {
    let v = ctx.view;
    let h = *h;
    let mut f = [0.0f64; N_VF];
    for t in 27..34u8 {
        if !v.is_yakuhai(t) {
            continue;
        }
        let w = if t == v.round_wind && t == v.player_wind { 2.0 } else { 1.0 };
        match h[t as usize] {
            0 => {}
            1 => f[YAKUHAI_SINGLES] += w,
            2 => f[YAKUHAI_PAIRS] += w,
            _ => f[YAKUHAI_TRIPLETS] += w,
        }
    }
    let yaochu: u32 = (0..34).filter(|&t| is_yaochu(t)).map(|t| u32::from(h[t])).sum();
    f[YAOCHU] = f64::from(yaochu);
    f[TANYAO_NEAR] = f64::from(u8::from(yaochu <= 2));
    let honors: u32 = h[27..].iter().map(|&n| u32::from(n)).sum();
    let suit_max = (0..3).map(|s| h[s * 9..s * 9 + 9].iter().map(|&n| u32::from(n)).sum::<u32>()).max();
    let n13: u32 = h.iter().map(|&n| u32::from(n)).sum();
    f[FLUSH_SHARE] = f64::from(suit_max.unwrap_or(0) + honors) / f64::from(n13.max(1));
    let pairs = h.iter().filter(|&&n| n >= 2).count() as i32;
    let kinds = h.iter().filter(|&&n| n >= 1).count() as i32;
    f[PAIRS] = f64::from(pairs);
    f[CHIITOI_SHANTEN] = f64::from(6 - pairs + (7 - kinds).max(0));
    f[ISOLATED_HONORS] = h[27..].iter().filter(|&&n| n == 1).count() as f64;
    f[ISOLATED_TERMINALS] = [0usize, 8, 9, 17, 18, 26]
        .into_iter()
        .filter(|&t| {
            let base = t - t % 9;
            let near = if t % 9 == 0 { base + 1..=base + 2 } else { base + 6..=base + 7 };
            h[t] == 1 && near.into_iter().all(|x| h[x] == 0)
        })
        .count() as f64;
    // Unneeded features are zeroed so the score never reads them.
    for k in 0..N_VF {
        if !need[k] {
            f[k] = 0.0;
        }
    }
    f
}
