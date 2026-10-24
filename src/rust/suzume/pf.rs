//! Push / fold (`pf=1`; design: docs/suzume.md, "押し引き"): at each own-turn
//! discard decision out of riichi, the state model (`params/legacy/state/<state>.json`,
//! fitted to the pf_hmm posteriors of the
//! teacher's decisions in the hanchan games) predicts the push (P), mawashi
//! (M) or fold (F) state from the fold model's features against every other
//! seat and, in a model file with `sit_inputs`, the situation (`situation`:
//! round, honba, sticks, rank and score gaps); the most likely state picks
//! the models
//! (`rules::fold::PfDiscard`, `universal_mlp`). The hand models are fitted
//! with each row weighted by the posterior of its state, while play takes the
//! one most likely state:
//!
//! | decision | P | M | F |
//! |---|---|---|---|
//! | discard (discard12 / discard3 / tenpai) | hand MLPs | mawashi model (`discard_M`) | fold model (`discard_F`) |
//! | riichi legal | `riichi` | `riichi_M` | `riichi_M` |
//! | chi / pon | `call` | `call_M` | `call_F` |
//! | daiminkan (a call option) | `call` | `call_M` | never |
//! | ankan / kakan | `kan` | `kan` | never |
//!
//! A call decision takes the state of the seat's latest own-turn decision
//! (the kyoku starts in P). In own riichi the state is no longer predicted;
//! its kans are the kan model's.

use super::ctx::Ctx;
use super::fold_model::N_FEATURES;
use super::params::StateMlp;
use super::view::MjView;

/// The push / fold state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PfState {
    #[default]
    Push,
    Mawashi,
    Fold,
}

/// What one seat's state model carries across its decisions of a kyoku
/// (`SeatMemo::pf`).
#[derive(Clone, Debug, Default)]
pub struct PfMemo {
    /// The state of the latest own-turn decision.
    pub state: PfState,
}

/// One seat's fold-model features against each other seat.
pub type SeatFeatures = [(usize, [[f64; N_FEATURES]; 34])];

impl StateMlp {
    /// The P / M / F probabilities from the features against each other seat,
    /// pooled over the `legal` discard types, the experts' per-tile inputs
    /// `tile` (`PMF_TILE`, read by a model with such inputs; zeros else) and the
    /// decision-level inputs `extra` (`sit_inputs`, `n_sit` values).
    pub fn probs(&self, feats: &SeatFeatures, legal: &[usize], tile: &[[f64; N_PMF_TILE]; 34], extra: &[f64]) -> [f64; 3] {
        let cols = &self.cols;
        let (n_in, net) = (self.inputs.len(), &self.net);
        let h = net.out_width();
        debug_assert_eq!(extra.len(), self.n_sit);
        // The extra inputs' share of each first-layer unit, the same for every row.
        let base = net.pre_from(&net.b1, extra, n_in);
        let mut sum = vec![0.0; h];
        let mut max = vec![f64::NEG_INFINITY; h];
        let mut n = 0.0;
        for (_, f) in feats {
            for &t in legal {
                let x: Vec<f64> = cols
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| (if c < N_FEATURES { f[t][c] } else { tile[t][c - N_FEATURES] } - self.mean[j]) / self.std[j])
                    .collect();
                let z = net.pre_from(&base, &x, 0);
                for (k, z) in net.activate(z).into_iter().enumerate() {
                    sum[k] += z;
                    max[k] = max[k].max(z);
                }
                n += 1.0;
            }
        }
        let pooled: Vec<f64> = sum.iter().map(|s| s / n).chain(max.iter().copied()).collect();
        let logits: [f64; 3] = std::array::from_fn(|c| {
            self.b2[c] + self.w2[c * 2 * h..(c + 1) * 2 * h].iter().zip(&pooled).map(|(a, b)| a * b).sum::<f64>()
        });
        softmax(logits)
    }

    /// The decision-level inputs of a decision, standardized (none for a
    /// model without `sit_inputs`): the situation (`situation`), then the
    /// experts' summary columns the model reads (`summary`, `PMF_SUMMARY`
    /// order; ignored by a model without them).
    pub fn sit_inputs(&self, v: &MjView, summary: &[f64; N_PMF_SUMMARY]) -> Vec<f64> {
        if self.n_sit == 0 {
            return Vec::new();
        }
        let raw = situation(v);
        let x = raw.iter().copied().chain(self.summary_cols.iter().map(|&c| summary[c]));
        x.enumerate().map(|(j, x)| (x - self.sit_mean[j]) / self.sit_std[j]).collect()
    }

}

/// The per-row state inputs from the P / M / F experts of the decision's kind (`a1::pmf_tile`; 2026-10-23), after
/// the fold features (`fold_model::FEATURES`) in a state model's `inputs`: the P expert's score of the tile type
/// minus the decision's best, the F expert's the same, the DP's `A_ev` / `A_win` of discarding it, then (v3) the
/// M expert's diff and more universal columns of discarding it (`A_tenpai`, `A_in_dp`, `A_ukeire`, `A_ukeire2`,
/// `R_ev_pct`, `A_dora`, `A_no_yaku`, `A_ron_points`, `O_riichi`); the universal values are those of the type's
/// option the P expert scores highest. Definitions: suzume-train docs/state_predictor_features_v2.md.
pub const PMF_TILE: [&str; N_PMF_TILE] = [
    "pmf:p_diff", "pmf:f_diff", "pmf:a_ev", "pmf:a_win", "pmf:m_diff", "pmf:a_tenpai", "pmf:a_in_dp", "pmf:a_ukeire",
    "pmf:a_ukeire2", "pmf:r_ev_pct", "pmf:a_dora", "pmf:a_no_yaku", "pmf:a_ron_points", "pmf:o_riichi",
];
pub const N_PMF_TILE: usize = 14;
/// The universal columns of `PMF_TILE` from `pmf:a_tenpai` on (after `pmf:m_diff`), by name.
pub const PMF_TILE_UNIVERSAL: [&str; 9] =
    ["A_tenpai", "A_in_dp", "A_ukeire", "A_ukeire2", "R_ev_pct", "A_dora", "A_no_yaku", "A_ron_points", "O_riichi"];

/// The decision-level summary of the experts (2026-10-23), after `SIT_NAMES` in a state model's `sit_inputs`:
/// the top three legal types by the P expert (`top_p<i>`: its F diff, `A_ev`, `A_win`, and from the second on its
/// P diff), the top three by the F expert (`top_f<i>`, the same with P and F swapped), and whether a second /
/// third legal type exists (`has2`, `has3`; a missing slot's values are 0). Ties go to the lower type. Then (v3)
/// the decision's own columns (`dec:*`, `a1::pmf_decision`): wall progress, turn, the reference shanten, riichi
/// legal, a tenpai decision, other seats in riichi, legal types, a closed hand, dealer, leading.
pub const PMF_SUMMARY: [&str; N_PMF_SUMMARY] = [
    "pmf:top_p1:f", "pmf:top_p1:ev", "pmf:top_p1:win",
    "pmf:top_p2:p", "pmf:top_p2:f", "pmf:top_p2:ev", "pmf:top_p2:win",
    "pmf:top_p3:p", "pmf:top_p3:f", "pmf:top_p3:ev", "pmf:top_p3:win",
    "pmf:top_f1:p", "pmf:top_f1:ev", "pmf:top_f1:win",
    "pmf:top_f2:f", "pmf:top_f2:p", "pmf:top_f2:ev", "pmf:top_f2:win",
    "pmf:top_f3:f", "pmf:top_f3:p", "pmf:top_f3:ev", "pmf:top_f3:win",
    "pmf:has2", "pmf:has3",
    "dec:progress", "dec:turn", "dec:shanten_ref", "dec:riichi_legal", "dec:tenpai", "dec:n_opp_riichi",
    "dec:n_legal", "dec:menzen", "dec:oya", "dec:top",
];
pub const N_PMF_SUMMARY: usize = 34;
/// The expert summary's part of `PMF_SUMMARY` (`pmf_summary`); `dec:*` follow.
pub const N_PMF_TOP: usize = 24;

/// Every per-row input a state model may read: `fold_model::FEATURES`, then `PMF_TILE`.
pub fn state_input_names() -> &'static [&'static str] {
    static N: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    N.get_or_init(|| super::fold_model::FEATURES.iter().chain(PMF_TILE.iter()).copied().collect())
}

/// The summary of the per-tile expert inputs `tile` over the `legal` types (the first `N_PMF_TOP` of
/// `PMF_SUMMARY`; the `dec:*` columns left 0).
pub fn pmf_summary(tile: &[[f64; N_PMF_TILE]; 34], legal: &[usize]) -> [f64; N_PMF_SUMMARY] {
    let mut out = [0.0; N_PMF_SUMMARY];
    // (by, other): P first, then F; each slot: [other, ev, win] for the top one, [by, other, ev, win] after.
    for (half, (by, other)) in [(0usize, 1usize), (1, 0)].into_iter().enumerate() {
        let mut order = legal.to_vec();
        order.sort_by(|&a, &b| tile[b][by].total_cmp(&tile[a][by]).then(a.cmp(&b)));
        let base = half * 11;
        for (i, &t) in order.iter().take(3).enumerate() {
            let v = tile[t];
            if i == 0 {
                out[base..base + 3].copy_from_slice(&[v[other], v[2], v[3]]);
            } else {
                let o = base + 3 + (i - 1) * 4;
                out[o..o + 4].copy_from_slice(&[v[by], v[other], v[2], v[3]]);
            }
        }
    }
    out[22] = f64::from(u8::from(legal.len() >= 2));
    out[23] = f64::from(u8::from(legal.len() >= 3));
    out
}

fn softmax<const N: usize>(z: [f64; N]) -> [f64; N] {
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.map(|l| (l - top).exp());
    let s: f64 = e.iter().sum();
    e.map(|x| x / s)
}

/// The legal discard types of a decision, in order.
pub fn legal_types(ctx: &Ctx) -> Vec<usize> {
    let mut legal: Vec<usize> = ctx.discard_cands().iter().map(|c| usize::from(c.tile_type)).collect();
    legal.dedup();
    legal
}

/// The situation names (`StateMlp` `sit_inputs`): the round (East 1 .. South
/// 4, West folded into South 4), honba, riichi sticks, rank (1 .. 4), own
/// score, top - own, own - last, the next rank up - own, own - the next rank
/// down (0 when none) and whether there is one; points / 25,000.
pub const SIT_NAMES: [&str; 21] = [
    "round_E1", "round_E2", "round_E3", "round_E4", "round_S1", "round_S2", "round_S3", "round_S4", "honba", "sticks",
    "rank_1", "rank_2", "rank_3", "rank_4", "score", "to_top", "to_last", "to_up", "to_down", "has_up", "has_down",
];

/// The raw `SIT_NAMES` values of a decision; ranks tie to the seat nearer the
/// first dealer (the lower seat).
pub fn situation(v: &MjView) -> [f64; 21] {
    let mut out = [0.0; 21];
    let round = (usize::from(v.round_wind - 27) * 4 + usize::from(v.kyoku_index)).min(7);
    out[round] = 1.0;
    out[8] = f64::from(v.honba);
    out[9] = f64::from(v.sticks);
    let sc = v.scores.map(|x| f64::from(x) / 25000.0);
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]).then(a.cmp(&b)));
    let seat = usize::from(v.seat);
    let r = order.iter().position(|&s| s == seat).expect("seat");
    out[10 + r] = 1.0;
    let own = sc[seat];
    out[14] = own;
    out[15] = sc[order[0]] - own;
    out[16] = own - sc[order[3]];
    out[17] = if r > 0 { sc[order[r - 1]] - own } else { 0.0 };
    out[18] = if r < 3 { own - sc[order[r + 1]] } else { 0.0 };
    out[19] = f64::from(u8::from(r > 0));
    out[20] = f64::from(u8::from(r < 3));
    out
}

/// The most likely state under the state model `md` of an own-turn discard
/// decision in view `v` (P on a tie, then M).
pub fn predict(md: &StateMlp, v: &MjView, feats: &SeatFeatures, legal: &[usize]) -> PfState {
    let w = md.probs(feats, legal, &[[0.0; N_PMF_TILE]; 34], &md.sit_inputs(v, &[0.0; N_PMF_SUMMARY]));
    if w[0] >= w[1] && w[0] >= w[2] {
        PfState::Push
    } else if w[1] >= w[2] {
        PfState::Mawashi
    } else {
        PfState::Fold
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_ranks_by_p_then_by_f_and_pads_missing_slots() {
        let mut tile = [[0.0; N_PMF_TILE]; 34];
        tile[3][..4].copy_from_slice(&[0.0, -2.0, 1.5, 0.3]);
        tile[7][..4].copy_from_slice(&[-1.0, 0.0, 0.5, 0.1]);
        let s = pmf_summary(&tile, &[3, 7]);
        // top_p1 = 3 (f, ev, win), top_p2 = 7 (p, f, ev, win), top_p3 missing
        assert_eq!(s[..3], [-2.0, 1.5, 0.3]);
        assert_eq!(s[3..7], [-1.0, 0.0, 0.5, 0.1]);
        assert_eq!(s[7..11], [0.0; 4]);
        // top_f1 = 7 (p, ev, win), top_f2 = 3 (f, p, ev, win)
        assert_eq!(s[11..14], [-1.0, 0.5, 0.1]);
        assert_eq!(s[14..18], [-2.0, 0.0, 1.5, 0.3]);
        assert_eq!(s[18..22], [0.0; 4]);
        assert_eq!(s[22..24], [1.0, 0.0]);
        assert_eq!(s[24..], [0.0; N_PMF_SUMMARY - N_PMF_TOP]);
        // ties go to the lower type
        let mut t2 = [[0.0; N_PMF_TILE]; 34];
        t2[9][..4].copy_from_slice(&[0.0, 0.0, 9.0, 0.0]);
        t2[2][..4].copy_from_slice(&[0.0, 0.0, 2.0, 0.0]);
        t2[5][..4].copy_from_slice(&[-1.0, -1.0, 5.0, 0.0]);
        let s = pmf_summary(&t2, &[2, 5, 9]);
        assert_eq!((s[1], s[5], s[9]), (2.0, 9.0, 5.0));
        assert_eq!(s[22..24], [1.0, 1.0]);
        assert_eq!(PMF_SUMMARY.len(), N_PMF_SUMMARY);
        assert_eq!(state_input_names().len(), N_FEATURES + N_PMF_TILE);
    }
}
