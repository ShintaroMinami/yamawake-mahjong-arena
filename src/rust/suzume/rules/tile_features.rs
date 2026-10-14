//! Features of discarding one tile, and the fitted choice among the
//! discards near the top of a linear value score.
//!
//! The universal features read these as the moved tile's block (`B_*`).
//! `tie_pick` chooses among the candidates within `params().tie_band` of
//! the best linear value score (`value_features::value_ranked`): the highest
//! `sum_k w[k] * f_k` (`params().tie_weights`), weights fitted to the teacher's policy
//! renormalized on the band, as a conditional logit; an exact score tie goes
//! to the lower tile type.
//!
//! Features of discarding tile type `t` (a red five is its own candidate),
//! from this seat's view; `hand` is the concealed hand before the discard,
//! "unseen" is `4 - seen_counts` (the own hand counts as seen), "out" is
//! `seen_counts - hand` (copies in rivers, melds and dora indicators):
//!
//! | feature | definition |
//! |---|---|
//! | `unseen` | unseen copies of `t` |
//! | `honor_out1` | `t` is an honor with out >= 1 (one copy or more already cut) |
//! | `honor_out2` | `t` is an honor with out >= 2 (a live honor is the base: both 0) |
//! | `nb_unseen1` | number tile: unseen copies of `t-1` and `t+1` in its suit |
//! | `nb_unseen2` | number tile: unseen copies of `t-2` and `t+2` in its suit |
//! | `guest_wind` | a wind that is neither the round nor the seat wind |
//! | `yakuhai_wind` | the round or the seat wind (a double wind counts 2) |
//! | `dragon` | a dragon |
//! | `terminal` | a 1 or 9 |
//! | `rank_mid` | number tile: `min(rank, 10 - rank)` (1 for 1/9 ... 5 for 5); 0 for honors |
//! | `is_28` | a 2 or 8 |
//! | `dora` | dora han of `t` from the indicators |
//! | `aka` | the candidate is a red five |
//! | `dora_adj1` | number tile: dora han of `t-1` and `t+1` in its suit |
//! | `dora_adj2` | number tile: dora han of `t-2` and `t+2` in its suit |
//! | `yakuhai_pair` | `t` is yakuhai held 2+ times (the discard breaks a yakuhai pair or set) |
//! | `pair_break` | `t` is held exactly twice |
//! | `set_break` | `t` is held 3+ times |
//! | `isolated` | held once, and for a number tile no same-suit tile within 2 ranks |
//! | `flush_share` | (most held suit + honors) / tiles of the concealed hand after the discard |
//! | `tanyao_step` | `t` is a terminal or honor and the hand holds at most 3 of them |
//! | `progress_honor` | `t` is an honor, times `1 - drawable_count / 70` |
//! | `progress_terminal` | `t` is a 1 or 9, times `1 - drawable_count / 70` |
//! | `honor_out1_early` | `honor_out1` times `early = drawable_count / 70` (1 at the deal, 0 at the last draw) |
//! | `honor_out2_early` | `honor_out2` times `early` |
//! | `guest_wind_early` | `guest_wind` times `early` |
//! | `rel_value` | the candidate's value score minus the best (`tie_pick`; 0 in the moved-tile block) |

use crate::suzume::config::DRAWABLE_AT_START;
use crate::suzume::ctx::{Ctx, DiscardCand};
use crate::suzume::params::params;
use crate::eval::feat::tiles::is_yaochu;

/// Score slack below which two candidates count as tied.
pub const TIE_EPS: f64 = 1e-9;

/// Number of tile features.
pub const N_TF: usize = 27;

/// Feature names, in feature-vector order.
pub const TF_NAMES: [&str; N_TF] = [
    "unseen",
    "honor_out1",
    "honor_out2",
    "nb_unseen1",
    "nb_unseen2",
    "guest_wind",
    "yakuhai_wind",
    "dragon",
    "terminal",
    "rank_mid",
    "is_28",
    "dora",
    "aka",
    "dora_adj1",
    "dora_adj2",
    "yakuhai_pair",
    "pair_break",
    "set_break",
    "isolated",
    "flush_share",
    "tanyao_step",
    "progress_honor",
    "progress_terminal",
    "honor_out1_early",
    "honor_out2_early",
    "guest_wind_early",
    "rel_value",
];

pub(crate) const UNSEEN: usize = 0;
pub(crate) const HONOR_OUT1: usize = 1;
const HONOR_OUT2: usize = 2;
pub(crate) const NB_UNSEEN1: usize = 3;
pub(crate) const NB_UNSEEN2: usize = 4;
pub(crate) const GUEST_WIND: usize = 5;
pub(crate) const YAKUHAI_WIND: usize = 6;
pub(crate) const DRAGON: usize = 7;
pub(crate) const TERMINAL: usize = 8;
const RANK_MID: usize = 9;
pub(crate) const IS_28: usize = 10;
pub(crate) const DORA: usize = 11;
pub(crate) const AKA: usize = 12;
pub(crate) const DORA_ADJ1: usize = 13;
pub(crate) const DORA_ADJ2: usize = 14;
const YAKUHAI_PAIR: usize = 15;
pub(crate) const PAIR_BREAK: usize = 16;
pub(crate) const SET_BREAK: usize = 17;
pub(crate) const ISOLATED: usize = 18;
const FLUSH_SHARE: usize = 19;
pub(crate) const TANYAO_STEP: usize = 20;
const PROGRESS_HONOR: usize = 21;
const PROGRESS_TERMINAL: usize = 22;
const HONOR_OUT1_EARLY: usize = 23;
const HONOR_OUT2_EARLY: usize = 24;
const GUEST_WIND_EARLY: usize = 25;
const REL_VALUE: usize = 26;


/// The features of each candidate (`rel`: each one's `rel_value`).
pub fn tie_features(ctx: &Ctx, tied: &[DiscardCand], rel: &[f64]) -> Vec<[f64; N_TF]> {
    let v = ctx.view;
    let h = &v.hand;
    let unseen = |t: usize| f64::from(4u8.saturating_sub(v.seen_counts[t]));
    let progress = 1.0 - f64::from(v.drawable_count) / f64::from(DRAWABLE_AT_START);
    let yaochu_held: u32 = (0..34).filter(|&t| is_yaochu(t)).map(|t| u32::from(h[t])).sum();
    let n_hand: u32 = h.iter().map(|&n| u32::from(n)).sum();
    let honors: u32 = h[27..].iter().map(|&n| u32::from(n)).sum();
    let suit = |s: usize| -> u32 { h[s * 9..s * 9 + 9].iter().map(|&n| u32::from(n)).sum() };
    let suits = [suit(0), suit(1), suit(2)];
    let mut out = Vec::with_capacity(tied.len());
    for (c, &r_value) in tied.iter().zip(rel) {
        let t = c.tile_type as usize;
        let mut f = [0.0f64; N_TF];
        f[UNSEEN] = unseen(t);
        f[AKA] = f64::from(u8::from(c.is_aka));
        f[DORA] = f64::from(v.dora_per_tile_type[t]);
        f[REL_VALUE] = r_value;
        let held = h[t];
        f[PAIR_BREAK] = f64::from(u8::from(held == 2));
        f[SET_BREAK] = f64::from(u8::from(held >= 3));
        if t >= 27 {
            let out_copies = v.seen_counts[t].saturating_sub(held);
            f[HONOR_OUT1] = f64::from(u8::from(out_copies >= 1));
            f[HONOR_OUT2] = f64::from(u8::from(out_copies >= 2));
            let tt = c.tile_type;
            if tt >= 31 {
                f[DRAGON] = 1.0;
            } else {
                let own = u8::from(tt == v.round_wind) + u8::from(tt == v.player_wind);
                f[YAKUHAI_WIND] = f64::from(own);
                f[GUEST_WIND] = f64::from(u8::from(own == 0));
            }
            f[YAKUHAI_PAIR] = f64::from(u8::from(v.is_yakuhai(tt) && held >= 2));
            f[ISOLATED] = f64::from(u8::from(held == 1));
            f[PROGRESS_HONOR] = progress;
            let early = f64::from(v.drawable_count) / f64::from(DRAWABLE_AT_START);
            f[HONOR_OUT1_EARLY] = f[HONOR_OUT1] * early;
            f[HONOR_OUT2_EARLY] = f[HONOR_OUT2] * early;
            f[GUEST_WIND_EARLY] = f[GUEST_WIND] * early;
        } else {
            let (base, r) = (t - t % 9, t % 9);
            let at = |d: isize| -> Option<usize> {
                let x = r as isize + d;
                (0..9).contains(&x).then(|| base + x as usize)
            };
            let sum = |ds: [isize; 2], g: &dyn Fn(usize) -> f64| -> f64 { ds.iter().filter_map(|&d| at(d)).map(g).sum() };
            f[NB_UNSEEN1] = sum([-1, 1], &unseen);
            f[NB_UNSEEN2] = sum([-2, 2], &unseen);
            let dora = |x: usize| f64::from(v.dora_per_tile_type[x]);
            f[DORA_ADJ1] = sum([-1, 1], &dora);
            f[DORA_ADJ2] = sum([-2, 2], &dora);
            let rank = r + 1;
            f[TERMINAL] = f64::from(u8::from(rank == 1 || rank == 9));
            f[RANK_MID] = rank.min(10 - rank) as f64;
            f[IS_28] = f64::from(u8::from(rank == 2 || rank == 8));
            let near = [-2isize, -1, 1, 2].iter().filter_map(|&d| at(d)).any(|x| h[x] > 0);
            f[ISOLATED] = f64::from(u8::from(held == 1 && !near));
            f[PROGRESS_TERMINAL] = progress * f[TERMINAL];
        }
        let yaochu = is_yaochu(t);
        f[TANYAO_STEP] = f64::from(u8::from(yaochu && yaochu_held <= 3));
        let (mut s, mut hon) = (suits, honors);
        if t >= 27 {
            hon -= 1;
        } else {
            s[t / 9] -= 1;
        }
        let n_after = n_hand.saturating_sub(1).max(1);
        f[FLUSH_SHARE] = f64::from(s.iter().max().copied().unwrap_or(0) + hon) / f64::from(n_after);
        out.push(f);
    }
    out
}

/// The index in `tied` to discard: the highest `params().tie_weights` score, then
/// the lower tile type.
pub fn tie_pick(ctx: &Ctx, tied: &[DiscardCand], rel: &[f64]) -> Option<usize> {
    if tied.len() <= 1 {
        return (!tied.is_empty()).then_some(0);
    }
    let feats = tie_features(ctx, tied, rel);
    let mut best: Option<(usize, f64)> = None;
    for (i, f) in feats.iter().enumerate() {
        let s: f64 = (0..N_TF).map(|k| f64::from(params().tie_weights[k]) * f[k]).sum();
        let better = match best {
            None => true,
            Some((b, bs)) => {
                s.total_cmp(&bs).then_with(|| ctx.tie_break(&tied[b], &tied[i])) == std::cmp::Ordering::Greater
            }
        };
        if better {
            best = Some((i, s));
        }
    }
    best.map(|(i, _)| i)
}
