//! Push / fold (`pf=1`; design: docs/suzume.md, "押し引き"): at each own-turn
//! discard decision out of riichi, the state model (`params/state/<state>.json`,
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
    /// pooled over the `legal` discard types, and the decision-level inputs
    /// `extra` (the situation inputs, `n_sit` values).
    pub fn probs(&self, feats: &SeatFeatures, legal: &[usize], extra: &[f64]) -> [f64; 3] {
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
                let x: Vec<f64> = cols.iter().enumerate().map(|(j, &c)| (f[t][c] - self.mean[j]) / self.std[j]).collect();
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

    /// The situation inputs of a decision (`situation`, standardized; none
    /// for a model without them).
    pub fn sit_inputs(&self, v: &MjView) -> Vec<f64> {
        if self.n_sit == 0 {
            return Vec::new();
        }
        situation(v).iter().enumerate().map(|(j, x)| (x - self.sit_mean[j]) / self.sit_std[j]).collect()
    }
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
    let w = md.probs(feats, legal, &md.sit_inputs(v));
    if w[0] >= w[1] && w[0] >= w[2] {
        PfState::Push
    } else if w[1] >= w[2] {
        PfState::Mawashi
    } else {
        PfState::Fold
    }
}

