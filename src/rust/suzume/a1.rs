//! A1: suzume without the push / mawashi / fold split (one MLP per decision
//! kind, fitted on every position of the teacher's games), decided in-loop.
//! Since 2026-10-21 the player `suzume` is an A1 set (`BUILTIN`,
//! `params/suzume/`); the earlier suzume plays on as the legacy sets
//! (`legacy-suzume-mlp3` ...).
//!
//! A Rust port of suzume-train's Python `A1Agent` (`suzume_train/a1.py`); it
//! plays the same moves. It is its own bot (`arena::Bot::A1`): nothing here
//! changes the legacy sets.
//!
//! The rows are the universal rows (`universal::universal`, `change` on) and
//! 114 danger columns (`a1_names`): for a discard-like option (discard,
//! dama, break, riichi) the fold features of its tile type against the three
//! other seats from the shimocha (`fold_model::features`, the river applied,
//! `hand_mlp` left out) and `seat:score_diff` (own minus that seat's score,
//! / 25000; 2026-10-19); for another option (pass, chi, pon, daiminkan) only
//! the `seat:*` columns; for a kan decision none. The models read their
//! inputs by name, so a model fitted before `seat:score_diff` plays as
//! before. The decision order:
//!
//! 1. a win (tsumo / ron) or kyushu kyuhai whenever legal;
//! 2. right after its own riichi, the discard the riichi option named;
//! 3. the kan decision (ankan / kakan legal): the kan model compares `keep`
//!    (this decision's own choice) with each kan;
//! 4. the per-decision model's best option (ties keep the earlier);
//! 5. the only legal discard tile, else pass, else the drawn tile, else the
//!    first legal discard, else the first legal action.
//!
//! The context (`cfg`) is a suzume config on the features set (the fold
//! tables, and the push models whose choice the kan rows are first built
//! with, as the Python player's `SuzumeFeatures`).
//!
//! P / M / F mixture (experimental, 2026-10-22): a directory set may add, for
//! a discard kind `k` of `PMF_KINDS`, the experts `{k}_P`, `{k}_M`, `{k}_F`
//! (A1 model files fitted with `--labels --state`), and `state.json`, a P / M / F
//! state model (`params::StateMlp`, suzume-train `sp fit`, `--sit` or not). A
//! decision of such a kind then takes the option of the highest
//! `pi(o) = sum_s p_s softmax_s(o)`, `softmax_s` over the options of expert s's
//! scores and `p` the state model's probabilities on the decision (its inputs
//! as `sp rows` builds them: `fold_model::seat_features` on the view with the
//! river, `hand_mlp` from the features set's push models, over every other
//! seat and the legal discard types, and the situation), or `A1Config::pmf_fixed`.
//! A state model may also read the experts' inputs (2026-10-23; `pmf_inputs`,
//! `pf::PMF_TILE` / `pf::PMF_SUMMARY`): `state_inputs` builds every input of
//! the state model, for play and for suzume-train's rows
//! (`SuzumeFeatures.state_inputs`) alike.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use riichienv_core::action::{Action, ActionType};
use serde_json::Value;

use super::config::SuzumeConfig;
use super::ctx::Ctx;
use super::fold_model::{FEATURES, FoldInputs, N_FEATURES, features, seat_features};
use super::memo::SeatMemo;
use super::mlp::{Hidden, Sit, add_bias, parse_hidden, parse_sit};
use super::params::{StateMlp, num, nums, parse_state, positions, strs};
use super::pf::{N_PMF_SUMMARY, N_PMF_TILE, N_PMF_TOP, PMF_TILE_UNIVERSAL, legal_types, pmf_summary, situation};
use super::rules::universal_mlp::mlp_choice;
use super::universal::{N_U, Universal, kan_universal, kind_of, riichi_keep_row, u_names};
use super::view::{MjView, river_from_mjai};
use super::view_of;

/// The fold features of the danger columns (`foldall.INPUTS` without `hand_mlp`).
const DANGER: [&str; 37] = [
    "patterns_log", "turn", "dshanten", "honor", "riichi_yaku", "safe_sum", "other_wait_est", "ryanmen_log",
    "tanki_unseen", "held", "outside_28", "outside_37", "outside_46", "decl_outside_suji", "last_ago",
    "patterns:kanchan", "patterns:penchan", "patterns:shanpon", "order:others_safe", "supply:safe_left",
    "supply:enough", "tile_kind:dora", "tile_kind:terminal", "tile_kind:two_eight", "genbutsu:genbutsu",
    "shanten:best_shanten", "seat:riichi", "seat:n_melds", "seat:meld_dora", "seat:dealer", "seat:n_discards",
    "seat:yakuhai_melds", "seat:meld_flush", "seat:meld_tanyao", "seat:meld_chanta", "suit:discards", "suit:first",
];
/// The danger columns not among the fold features: own minus that seat's score / 25000 (2026-10-19).
const EXTRA: [&str; 1] = ["seat:score_diff"];
const N_PER: usize = DANGER.len() + EXTRA.len();
const N_D: usize = 3 * N_PER;
const N_ROW: usize = N_U + N_D;
const RED_TIDS: [u8; 3] = [16, 52, 88];

/// Every column an A1 model may read: the universal names, then `o<k>:<feature>` (k = 1..3 from the shimocha).
fn a1_names() -> &'static [String] {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| {
        let mut v = u_names().to_vec();
        for k in 1..=3 {
            v.extend(DANGER.iter().chain(EXTRA.iter()).map(|n| format!("o{k}:{n}")));
        }
        v
    })
}

/// The `FEATURES` index of each danger feature.
fn danger_idx() -> &'static [usize; 37] {
    static I: OnceLock<[usize; 37]> = OnceLock::new();
    I.get_or_init(|| std::array::from_fn(|j| FEATURES.iter().position(|f| *f == DANGER[j]).expect("danger feature")))
}

/// One A1 model file: `w2 . relu(... relu(W1 [x; state one-hot; sit] + b1) ...) + b2`, x (and the
/// situation inputs `sit`, when the file has `sit_inputs`: the hanchan models, 2026-10-19) standardized.
struct Model {
    cols: Vec<usize>,
    states: Vec<String>,
    sit: Option<Sit>,
    mean: Vec<f64>,
    std: Vec<f64>,
    net: Hidden,
    w2: Vec<f64>,
    b2: f64,
}

impl Model {
    /// The model in JSON `text`; `label` names it in errors.
    fn parse(text: &str, label: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("{label}: {e}"))?;
        let at = |e: String| format!("{label}: {e}");
        let (inputs, states) = (strs(&v, &["inputs"]).map_err(at)?, strs(&v, &["state_onehot_order"]).map_err(at)?);
        let n_own = inputs.len() + states.len();
        let sit = parse_sit(&v, n_own).map_err(at)?;
        let n_sit = sit.as_ref().map_or(0, |s| s.mean.len());
        let md = Self {
            cols: positions(&inputs, a1_names()).map_err(at)?,
            net: parse_hidden(&v, n_own + n_sit).map_err(at)?,
            sit,
            states,
            mean: nums(&v, &["mean"]).map_err(at)?,
            std: nums(&v, &["std"]).map_err(at)?,
            w2: nums(&v, &["w2"]).map_err(at)?,
            b2: num(&v, &["b2"]).map_err(at)?,
        };
        if md.w2.len() != md.net.out_width() || md.mean.len() != inputs.len() || md.std.len() != inputs.len() {
            return Err(format!("{label}: w2 / mean / std do not match the network"));
        }
        Ok(md)
    }

    /// The one-hot index of state `name` (`<name>/opp` or `<name>/noopp` first, then `name`; as the Python player).
    fn state_index(&self, name: &str, opp: bool) -> Option<usize> {
        let full = format!("{name}{}", if opp { "/opp" } else { "/noopp" });
        self.states.iter().position(|s| *s == full).or_else(|| self.states.iter().position(|s| s == name))
    }

    fn score(&self, row: &[f64; N_ROW], state: Option<usize>, sit: &[f64]) -> f64 {
        let n_in = self.cols.len();
        let x: Vec<f64> = self.cols.iter().enumerate().map(|(j, &c)| (row[c] - self.mean[j]) / self.std[j]).collect();
        let mut z = self.net.pre_dense(&x);
        add_bias(&mut z, sit);
        if let Some(si) = state {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk += self.net.w1_row(k)[n_in + si];
            }
        }
        self.b2 + self.w2.iter().zip(self.net.activate(z)).map(|(w, h)| w * h).sum::<f64>()
    }

    /// The score of each row, the situation of view `v` added when the model reads it.
    fn scores(&self, rows: &[[f64; N_ROW]], state: &str, opp: bool, v: &MjView) -> Vec<f64> {
        let sit = self.sit.as_ref().map_or_else(Vec::new, |s| s.bias(&self.net, v));
        let si = self.state_index(state, opp);
        rows.iter().map(|r| self.score(r, si, &sit)).collect()
    }

    /// The index of the best row (the first on a tie).
    fn best(&self, rows: &[[f64; N_ROW]], state: &str, opp: bool, v: &MjView) -> usize {
        argmax(&self.scores(rows, state, opp, v))
    }

    /// The index of the best row under the play-style options (`A1Config`): with `bias` (one per row, when given;
    /// the `call_offset` / `riichi_offset`) and, for `aggression` w != 0, `s_end` the scores of the endpoint model of
    /// w's sign (`ends`: fold for w < 0, push for w > 0; `A1Models::ends`), the argmax of
    /// `(1 - |w|) s + |w| s_end + bias`, `s` this model's scores. The softmax normalizers cancel in the argmax, so
    /// this mixes the log policies, w = +-1 plays the endpoint itself, and the bias moves the choice by the same
    /// amount at every w (2026-10-28; it was added to `s` before the mix, so it shrank by 1 - |w| and vanished at
    /// +-1). Without that endpoint, or w = 0: the argmax of `s + bias`. `riichi` (the rows that declare riichi, when
    /// the decision has any): riichi or not stays this model's own choice (from `s + bias`) and the mix picks only
    /// within that group (2026-10-23).
    #[allow(clippy::too_many_arguments)]
    fn best_styled(&self, ends: [Option<&Model>; 2], aggression: f64, rows: &[[f64; N_ROW]], state: &str, opp: bool,
                   v: &MjView, riichi: Option<&[bool]>, bias: Option<&[f64]>) -> usize {
        let s = self.scores(rows, state, opp, v);
        let end = ends[usize::from(aggression > 0.0)].filter(|_| aggression != 0.0).map(|m| m.scores(rows, state, opp, v));
        styled_choice(&s, end.as_deref(), aggression.abs(), riichi, bias)
    }
}

/// `Model::best_styled` on given scores: `s` the model's, `end` the endpoint's (none: w is not used), `w` = |aggression|.
fn styled_choice(s: &[f64], end: Option<&[f64]>, w: f64, riichi: Option<&[bool]>, bias: Option<&[f64]>) -> usize {
    let b = |i: usize| bias.map_or(0.0, |b| b[i]);
    let biased: Vec<f64> = s.iter().enumerate().map(|(i, x)| x + b(i)).collect();
    let own = argmax(&biased);
    let Some(end) = end else { return own };
    let group = riichi.map(|r| r[own]);
    let mut best = (own, f64::NEG_INFINITY);
    for (i, (y, x)) in end.iter().zip(s).enumerate() {
        let t = (1.0 - w) * x + w * y + b(i);
        if group.is_none_or(|g| riichi.unwrap()[i] == g) && t > best.1 {
            best = (i, t);
        }
    }
    best.0
}

/// The index of the highest value (the first on a tie; 0 when none beats -inf).
fn argmax(x: &[f64]) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (i, &s) in x.iter().enumerate() {
        if s > best.1 {
            best = (i, s);
        }
    }
    best.0
}

/// The scores of the P / M / F experts `ex` on `rows`.
fn expert_scores(ex: &[Model; 3], rows: &[[f64; N_ROW]], state: &str, opp: bool, v: &MjView) -> [Vec<f64>; 3] {
    std::array::from_fn(|s| ex[s].scores(rows, state, opp, v))
}

/// The best row of the P / M / F experts' scores `sc` (`expert_scores`) mixed by the state probabilities `p`:
/// the argmax of `sum_s p_s softmax_s` (the first on a tie).
fn mixture_best(sc: &[Vec<f64>; 3], p: &[f64; 3]) -> usize {
    let mut pi = vec![0.0; sc[0].len()];
    for (s, &w) in sc.iter().zip(p) {
        let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - top).exp()).collect();
        let z: f64 = e.iter().sum();
        for (q, x) in pi.iter_mut().zip(e) {
            *q += w * x / z;
        }
    }
    argmax(&pi)
}

/// One decision's A1 rows: its universal options and rows with the danger columns, the model of its kind and the
/// kind's `PMF_KINDS` index (none for discard12 and call).
struct Rows<'m> {
    u: std::rc::Rc<Universal>,
    rows: Vec<[f64; N_ROW]>,
    model: &'m Model,
    /// The kind's endpoint models (`A1Models::ends`).
    ends: [Option<&'m Model>; 2],
    pk: Option<usize>,
    /// `universal::kind_of`: 0 discard, 1 riichi legal, 2 tenpai, 3 call.
    kind: u8,
}

/// The view `view` with the kyoku's river applied (what the fold features read).
fn river_view(view: &MjView, kyoku_events: &[Value]) -> Result<MjView, String> {
    let mut rv = view.clone();
    rv.apply_river(&river_from_mjai(kyoku_events)?);
    Ok(rv)
}

/// The A1 rows of the decision of `ctx` (on the view without the river; `rv` the view with it) under the models
/// `md`; `None` without an option or out of the models' kinds.
fn decision_rows<'m>(ctx: &Ctx, md: &'m A1Models, rv: &MjView) -> Option<Rows<'m>> {
    let u = ctx.universal(true);
    if u.options.is_empty() {
        return None;
    }
    let kind = kind_of(ctx)?;
    // The shanten columns read the hand only, so the decision's own candidates serve.
    let mut shanten_after = [None; 34];
    for c in ctx.discard_cands() {
        shanten_after[usize::from(c.tile_type)] = Some(c.shanten);
    }
    let inp = FoldInputs { shanten_after, hand_mlp: [0.0; 34] };
    let x: [_; 3] = std::array::from_fn(|k| features(ctx.models.tables(), rv, (usize::from(rv.seat) + k + 1) % 4, &inp));
    let diff = score_diffs(ctx.view);
    let rows: Vec<[f64; N_ROW]> = u
        .options
        .iter()
        .zip(&u.rows)
        .map(|(&(k, id), r)| {
            let mut row = [0.0; N_ROW];
            row[..N_U].copy_from_slice(r);
            row[N_U..].copy_from_slice(&danger_row(k, id, &x, &diff));
            row
        })
        .collect();
    let (model, pk, kn) = match kind {
        0 if u.context.shanten_ref >= 3 => (&md.discard3, Some(3), Some(3)),
        0 if u.context.shanten_ref == 1 && md.discard1.is_some() => (md.discard1.as_ref().unwrap(), Some(1), Some(1)),
        0 if u.context.shanten_ref == 2 && md.discard2.is_some() => (md.discard2.as_ref().unwrap(), Some(2), Some(2)),
        0 => match &md.discard12 {
            Some(m) => (m, None, None),
            // A set without discard12 has discard1 (`A1Models::from_reader`); play reaches this only for a discard
            // decision outside shanten 1 / 2 / >= 3.
            None => (md.discard1.as_ref().unwrap(), Some(1), Some(1)),
        },
        1 | 2 => (if kind == 1 { &md.riichi } else { &md.tenpai }, Some(0), Some(0)),
        _ => (&md.call, None, Some(4)),
    };
    let ends = kn.map_or([None, None], |i| md.end_models(i));
    Some(Rows { u, rows, model, ends, pk, kind: kind as u8 })
}

/// The experts' per-tile state inputs (`pf::PMF_TILE`) from their scores `sc` (`expert_scores`) of the rows `r`:
/// for each tile type of a discard-like option (discard, dama, break, riichi), the P expert's highest score over
/// the type's options minus its highest over every such option, the same for the F expert, the universal `A_ev` /
/// `A_win` of the type's option the P expert scores highest (the first on a tie), the M expert's diff and that
/// option's `pf::PMF_TILE_UNIVERSAL` columns; 0 for the other types.
pub fn pmf_tile(options: &[(&str, u16)], rows: &[[f64; N_ROW]], sc: &[Vec<f64>; 3]) -> [[f64; N_PMF_TILE]; 34] {
    static A: OnceLock<[usize; 11]> = OnceLock::new();
    let a = A.get_or_init(|| {
        let at = |n: &str| u_names().iter().position(|x| x == n).unwrap_or_else(|| panic!("universal {n}"));
        let mut a = [at("A_ev"), at("A_win"), 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (j, n) in PMF_TILE_UNIVERSAL.iter().enumerate() {
            a[2 + j] = at(n);
        }
        a
    });
    let ninf = [f64::NEG_INFINITY; 34];
    let (mut best, mut at) = ([ninf; 3], [None; 34]);
    for (i, &(k, id)) in options.iter().enumerate() {
        if !matches!(k, "discard" | "dama" | "break" | "riichi") {
            continue;
        }
        let t = usize::from(tile_of(id).0);
        if sc[0][i] > best[0][t] {
            at[t] = Some(i);
        }
        for s in 0..3 {
            best[s][t] = best[s][t].max(sc[s][i]);
        }
    }
    let top = best.map(|b| b.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    std::array::from_fn(|t| {
        let Some(i) = at[t] else { return [0.0; N_PMF_TILE] };
        let mut x = [0.0; N_PMF_TILE];
        x[0] = best[0][t] - top[0];
        x[1] = best[2][t] - top[2];
        x[2] = rows[i][a[0]];
        x[3] = rows[i][a[1]];
        x[4] = best[1][t] - top[1];
        for j in 0..PMF_TILE_UNIVERSAL.len() {
            x[5 + j] = rows[i][a[2 + j]];
        }
        x
    })
}

/// The decision's own columns of the state summary (`pf::PMF_SUMMARY`'s `dec:*`, after `N_PMF_TOP`): the
/// universal context's progress (`1 - live wall / 70`), turn and reference shanten, riichi legal (kind 1), a
/// tenpai decision (kind 1 or 2), the other seats in riichi, the legal discard types, a closed hand (`A_menzen`),
/// dealer and leading (`UContext::oya` / `top`).
fn pmf_decision(r: &Rows, view: &MjView, legal: &[usize]) -> [f64; N_PMF_SUMMARY - N_PMF_TOP] {
    let c = &r.u.context;
    let menzen = u_names().iter().position(|x| x == "A_menzen").map_or(0.0, |j| r.u.rows[0][j]);
    let b = |x: bool| f64::from(u8::from(x));
    [
        c.progress,
        f64::from(c.turn),
        f64::from(c.shanten_ref),
        b(r.kind == 1),
        b(r.kind == 1 || r.kind == 2),
        view.riichi_threats().count() as f64,
        legal.len() as f64,
        menzen,
        b(c.oya),
        b(c.top),
    ]
}

/// Every input of the P / M / F state model on an own-turn discard decision (`state_inputs`).
pub struct StateInputs {
    /// The fold features against each other seat (`fold_model::seat_features` on the view with the river).
    pub feats: [(usize, [[f64; N_FEATURES]; 34]); 3],
    /// The legal discard types.
    pub legal: Vec<usize>,
    /// The experts' per-tile inputs (`pmf_tile`; zeros without experts).
    pub tile: [[f64; N_PMF_TILE]; 34],
    /// Their summary over the legal types (`pf::pmf_summary`; zeros without experts).
    pub summary: [f64; N_PMF_SUMMARY],
    /// Whether the decision's kind had P / M / F experts (else `tile` and `summary` are zeros).
    pub experts: bool,
    /// The raw situation (`pf::situation`).
    pub sit: [f64; 21],
}

/// The state model's inputs on the decision of `ctx` (the view without the river), `rv` the view with it and
/// `on` the decision's rows and their experts' scores when it has experts; `None` without a legal discard.
fn inputs_on(ctx: &Ctx, rv: &MjView, on: Option<(&Rows, &[Vec<f64>; 3])>) -> Option<StateInputs> {
    let rctx = Ctx::with_memo(rv, ctx.cfg, ctx.take_memo());
    let legal = legal_types(&rctx);
    let out = (!legal.is_empty()).then(|| {
        let tile = on.map_or([[0.0; N_PMF_TILE]; 34], |(r, sc)| pmf_tile(&r.u.options, &r.rows, sc));
        let summary = match on {
            Some((r, _)) => {
                let mut s = pmf_summary(&tile, &legal);
                s[N_PMF_TOP..].copy_from_slice(&pmf_decision(r, ctx.view, &legal));
                s
            }
            None => [0.0; N_PMF_SUMMARY],
        };
        StateInputs { feats: seat_features(&rctx), tile, summary, experts: on.is_some(), sit: situation(rv), legal }
    });
    ctx.put_memo(rctx.into_memo());
    out
}

/// The state model's inputs on an own-turn discard decision of `ctx` (a view without the river) under the A1
/// models `md`, the experts those of the decision's kind (`decision_rows`); `None` without a legal discard. The
/// one code path of play (`decide`, `state_probs`) and of suzume-train's rows.
pub fn state_inputs(ctx: &Ctx, md: &A1Models, kyoku_events: &[Value]) -> Result<Option<StateInputs>, String> {
    let rv = river_view(ctx.view, kyoku_events)?;
    let opp = ctx.view.riichi_threats().next().is_some();
    let r = decision_rows(ctx, md, &rv);
    let sc = r.as_ref().and_then(|r| {
        let ex = r.pk.and_then(|k| md.pmf[k].as_deref())?;
        Some(expert_scores(ex, &r.rows, &r.u.state.0, opp, ctx.view))
    });
    Ok(inputs_on(ctx, &rv, r.as_ref().zip(sc.as_ref())))
}

/// The P / M / F probabilities of state model `st` on inputs `x`.
fn probs_of(st: &StateMlp, x: &StateInputs, rv: &MjView) -> [f64; 3] {
    st.probs(&x.feats, &x.legal, &x.tile, &st.sit_inputs(rv, &x.summary))
}

/// The discard kinds that may have P / M / F experts (`{k}_P`, `{k}_M`, `{k}_F`): discard0 serves both the
/// riichi-legal and the tenpai decisions.
pub const PMF_KINDS: [&str; 4] = ["discard0", "discard1", "discard2", "discard3"];

/// The models of an A1 directory (`discard0`, `discard12`, `discard3`, `call`, `kan`; or the first A1's names:
/// `discard_P_t` + `riichi_P` for discard0, `discard_P_12`, `discard_P_3`, `call_P`, `kan`). Optional `discard1` /
/// `discard2` (2026-10-19, the split of discard12) take the 1- / 2-shanten discards when present; a set with both
/// needs no `discard12` (2026-10-24: the split sets never reach it).
pub struct A1Models {
    dir: String,
    discard12: Option<Model>,
    discard1: Option<Model>,
    discard2: Option<Model>,
    discard3: Model,
    tenpai: Model,
    riichi: Model,
    call: Model,
    kan: Model,
    /// The P / M / F experts of each `PMF_KINDS` kind, when the set has all three files.
    pmf: [Option<Box<[Model; 3]>>; 4],
    /// The P / M / F state model (`state.json`), when present.
    state: Option<StateMlp>,
    /// The endpoint models `{k}_fold.json` / `{k}_push.json` of `STYLE_KINDS` (2026-10-24: plain models fitted on the
    /// teacher's pi_maxthr / pi_push, suzume-train `fit --endpoint`), when present; read only under
    /// `A1Config::aggression`.
    ends: [[Option<Model>; 2]; 6],
}

/// The kinds a set may give endpoint models (`A1Models::ends`).
pub const STYLE_KINDS: [&str; 6] = ["discard0", "discard1", "discard2", "discard3", "call", "kan"];

impl std::fmt::Debug for A1Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A1Models({})", self.dir)
    }
}

impl A1Models {
    /// The models of directory `dir`, read on first use in this process (the same path again gives the same set).
    pub fn load(dir: &str) -> Result<&'static Self, String> {
        static SETS: OnceLock<Mutex<Vec<&'static A1Models>>> = OnceLock::new();
        let mut sets = SETS.get_or_init(|| Mutex::new(Vec::new())).lock().expect("A1 sets");
        if let Some(m) = sets.iter().find(|m| m.dir == dir) {
            return Ok(m);
        }
        let p = Path::new(dir);
        let read = |name: &str| -> Result<Option<String>, String> {
            let f = p.join(format!("{name}.json"));
            match std::fs::read_to_string(&f) {
                Ok(t) => Ok(Some(t)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(format!("{}: {e}", f.display())),
            }
        };
        let md: &'static Self = Box::leak(Box::new(Self::from_reader(dir, read)?));
        sets.push(md);
        Ok(md)
    }

    /// The models of set `label`, each file's JSON text read by `read` (`None` when absent).
    fn from_reader(label: &str, read: impl Fn(&str) -> Result<Option<String>, String>) -> Result<Self, String> {
        let file = |name: &str| format!("{label}/{name}.json");
        let opt = |name: &str| read(name)?.map(|t| Model::parse(&t, &file(name))).transpose();
        // The A1 names (2026-10-19: discard0 for every tenpai decision, riichi legal or not; discard12, discard3,
        // call, kan), else the first A1's (discard_P_t + riichi_P, discard_P_12, discard_P_3, call_P, kan).
        let m = |names: &[&str]| -> Result<Model, String> {
            for n in names {
                if let Some(md) = opt(n)? {
                    return Ok(md);
                }
            }
            Err(format!("{}: missing", file(names[0])))
        };
        let mut pmf: [Option<Box<[Model; 3]>>; 4] = Default::default();
        for (slot, k) in pmf.iter_mut().zip(PMF_KINDS) {
            let ex = ["P", "M", "F"].map(|s| opt(&format!("{k}_{s}")));
            if ex.iter().all(|e| matches!(e, Ok(None))) {
                continue;
            }
            let [p, m, f] = ex;
            match (p?, m?, f?) {
                (Some(p), Some(m), Some(f)) => *slot = Some(Box::new([p, m, f])),
                _ => return Err(format!("{label}: {k}_P / {k}_M / {k}_F: all three or none")),
            }
        }
        let state = read("state")?.map(|t| parse_state(&t).map_err(|e| format!("{}: {e}", file("state")))).transpose()?;
        let mut ends: [[Option<Model>; 2]; 6] = Default::default();
        for (slot, k) in ends.iter_mut().zip(STYLE_KINDS) {
            for (e, s) in slot.iter_mut().zip(["fold", "push"]) {
                let name = format!("{k}_{s}");
                if let Some(md) = opt(&name)? {
                    *e = Some(md);
                }
            }
        }
        let (discard1, discard2) = (opt("discard1")?, opt("discard2")?);
        let discard12 = match opt("discard12")? {
            Some(md) => Some(md),
            None if discard1.is_some() && discard2.is_some() => opt("discard_P_12")?,
            None => Some(m(&["discard12", "discard_P_12"])?),
        };
        Ok(Self {
            dir: label.to_string(),
            discard12,
            discard1,
            discard2,
            discard3: m(&["discard3", "discard_P_3"])?,
            tenpai: m(&["discard0", "discard_P_t"])?,
            riichi: m(&["discard0", "riichi_P"])?,
            call: m(&["call", "call_P"])?,
            kan: m(&["kan"])?,
            pmf,
            state,
            ends,
        })
    }

    /// The endpoint models (a = 0, a = 1) of `STYLE_KINDS` kind `i`.
    fn end_models(&self, i: usize) -> [Option<&Model>; 2] {
        [self.ends[i][0].as_ref(), self.ends[i][1].as_ref()]
    }

    /// The built-in set of player name `name` (`BUILTIN`), parsed on first use.
    pub fn builtin(name: &str) -> Option<&'static Self> {
        static M: [OnceLock<A1Models>; BUILTIN.len()] = [const { OnceLock::new() }; BUILTIN.len()];
        let i = BUILTIN.iter().position(|b| b.name == name)?;
        let b = &BUILTIN[i];
        Some(M[i].get_or_init(|| {
            let read = |n: &str| Ok(b.files.iter().find(|(f, _)| *f == n).map(|(_, t)| t.to_string()));
            Self::from_reader(&format!("params/suzume/{}", b.dir), read).unwrap_or_else(|e| panic!("{e}"))
        }))
    }
}

/// One built-in A1 set: its player name, its directory under `params/suzume/` and its files (name, JSON text).
pub struct Builtin {
    pub name: &'static str,
    dir: &'static str,
    files: &'static [(&'static str, &'static str)],
}

macro_rules! a1_files {
    ($dir:literal, [$($f:literal),*]) => {
        &[$(($f, include_str!(concat!("../../../params/suzume/", $dir, "/", $f, ".json")))),*]
    };
}

/// The built-in A1 sets, the default (`suzume`) first: the discard1 / discard2 split sets fitted on the
/// Mortal-teacher games (`suzume`) and the rlactor-225-teacher games (`suzume-rla225`), without discard12
/// (2026-10-24: the discard12 counterparts `-d12` and the sets' unused discard12 models dropped).
/// `suzume` also has the push / fold endpoint models of `aggression` (`A1Models::ends`, 2026-10-24: discard0 3x48,
/// discard1 / discard2 3x88, discard3 3x64, call 3x40, kan 3x16).
pub static BUILTIN: [Builtin; 2] = [
    Builtin {
        name: "suzume",
        dir: "suzume",
        files: a1_files!("suzume", ["discard0", "discard1", "discard2", "discard3", "call", "kan",
                                    "discard0_push", "discard0_fold", "discard1_push", "discard1_fold", "discard2_push",
                                    "discard2_fold", "discard3_push", "discard3_fold", "call_push", "call_fold", "kan_push",
                                    "kan_fold"]),
    },
    Builtin {
        name: "suzume-rla225",
        dir: "rla225",
        files: a1_files!("rla225", ["discard0", "discard1", "discard2", "discard3", "call", "kan"]),
    },
];

/// An in-loop A1: its models and the suzume config of its features set.
#[derive(Clone, Copy, Debug)]
pub struct A1Config {
    pub cfg: SuzumeConfig,
    pub models: &'static A1Models,
    /// The P / M / F weights of the experts' mixture in place of the state model's (an ablation; `None`: the
    /// state model's).
    pub pmf_fixed: Option<[f64; 3]>,
    /// Push / fold in [-1, 1] (`Model::best_styled`; 2026-10-23, named `knob_delta` until 2026-10-24): > 0 toward
    /// the push endpoint, < 0 toward the fold endpoint, applied to the kinds with endpoint models (`{k}_fold.json` /
    /// `{k}_push.json`); 0 plays the set's own models as they are.
    pub aggression: f64,
    /// How readily it calls, in [-1, 1] (2026-10-24, named `call_bias` at first; scaled 2026-10-25): `call_offset`
    /// (`STYLE_SCALES`) is added to the score of every chi / pon / daiminkan option of a call decision; +-1 moves the
    /// log odds of calling, over the decisions with a call legal, by about +-1.5. 0 plays as before.
    pub call_shift: f64,
    /// How readily it declares riichi, in [-1, 1] (2026-10-24, named `riichi_bias` at first; scaled 2026-10-25):
    /// `riichi_offset` is added to the score of every riichi option of a riichi-legal decision (the riichi lock of
    /// `Model::best_styled` then keeps that choice); +-1 moves the log odds of declaring, over the riichi-legal
    /// decisions, by about +-3. 0 plays as before.
    pub riichi_shift: f64,
    /// The score offsets of `call_shift` / `riichi_shift` (`style_offset`).
    pub call_offset: f64,
    pub riichi_offset: f64,
}

/// Per shift (call, riichi): the score offset per unit below and above 0 (2026-10-25, from a sweep of the raw
/// offset vs the default suzume on standard walls 0-1999: the offsets that moved the log odds of taking the option,
/// over the decisions with it legal, by -1.5 / +1.5 for calls and -3 / +3 for riichi, interpolated).
pub const STYLE_SCALES: [(f64, f64); 2] = [(2.5, 3.4), (4.5, 3.5)];

/// The score offset of a style shift `v` in [-1, 1] with `scales` (below, above 0).
pub fn style_offset(v: f64, (below, above): (f64, f64)) -> f64 {
    v * if v < 0.0 { below } else { above }
}

impl A1Config {
    /// The bot of built-in set `name` (`BUILTIN`): its models with the built-in features set
    /// (`params::features_set`, `pf=0`, `fold=0`, as suzume-train's `A1RustAgent` plays a directory set).
    pub fn builtin(name: &str) -> Option<Self> {
        let models = A1Models::builtin(name)?;
        let cfg = SuzumeConfig { pf: false, fold: false, models: super::params::features_set(), ..SuzumeConfig::default() };
        Some(Self { cfg, models, pmf_fixed: None, aggression: 0.0, call_shift: 0.0, riichi_shift: 0.0, call_offset: 0.0,
                    riichi_offset: 0.0 })
    }

    /// The bot of a directory set's `models` on the features-set config `cfg`, the mixture weights fixed by
    /// `pmf_fixed` (`parse_pmf`) when given. A set with P / M / F experts needs `state.json` or `pmf_fixed`;
    /// `pmf_fixed` needs experts. A nonzero `aggression` needs at least one endpoint model.
    pub fn new(cfg: SuzumeConfig, models: &'static A1Models, pmf_fixed: Option<&str>, aggression: f64,
               call_shift: f64, riichi_shift: f64) -> Result<Self, String> {
        if ![aggression, call_shift, riichi_shift].iter().all(|x| x.is_finite()) {
            return Err(format!("aggression {aggression} / call_shift {call_shift} / riichi_shift {riichi_shift}: not finite"));
        }
        for (name, x) in [("aggression", aggression), ("call_shift", call_shift), ("riichi_shift", riichi_shift)] {
            if x.abs() > 1.0 {
                return Err(format!("{name} {x}: not in [-1, 1]"));
            }
        }
        if aggression != 0.0 && models.ends.iter().flatten().all(Option::is_none) {
            return Err(format!("{}: aggression without push / fold models ({{k}}_push / _fold.json)", models.dir));
        }
        let pmf_fixed = pmf_fixed.map(parse_pmf).transpose()?;
        let has = models.pmf.iter().any(Option::is_some);
        if has && models.state.is_none() && pmf_fixed.is_none() {
            return Err(format!("{}: P / M / F experts without state.json (add it, or pass pmf_fixed)", models.dir));
        }
        if !has && pmf_fixed.is_some() {
            return Err(format!("{}: pmf_fixed without P / M / F experts ({{k}}_P / _M / _F)", models.dir));
        }
        let (call_offset, riichi_offset) = (style_offset(call_shift, STYLE_SCALES[0]), style_offset(riichi_shift, STYLE_SCALES[1]));
        Ok(Self { cfg, models, pmf_fixed, aggression, call_shift, riichi_shift, call_offset, riichi_offset })
    }
}

/// Mixture weights from `"P"`, `"M"` or `"F"` (one-hot) or three non-negative numbers `"p,m,f"` (brackets
/// allowed; normalized to sum 1).
pub fn parse_pmf(s: &str) -> Result<[f64; 3], String> {
    let t = s.trim();
    if let Some(i) = ["P", "M", "F"].iter().position(|x| t.eq_ignore_ascii_case(x)) {
        let mut p = [0.0; 3];
        p[i] = 1.0;
        return Ok(p);
    }
    let bad = || format!("pmf_fixed {s:?}: P, M, F or three non-negative numbers p,m,f");
    let v: Vec<f64> = t
        .trim_start_matches(['[', '(']).trim_end_matches([']', ')'])
        .split(',')
        .map(|x| x.trim().parse::<f64>().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    let z: f64 = v.iter().sum();
    if v.len() != 3 || v.iter().any(|&x| !(x >= 0.0) || !x.is_finite()) || !(z > 0.0) {
        return Err(bad());
    }
    Ok([v[0] / z, v[1] / z, v[2] / z])
}

/// The tile type and red flag a discard-like option id names.
fn tile_of(id: u16) -> (u8, bool) {
    if id >= 34 { ([4, 13, 22][usize::from(id - 34)], true) } else { (id as u8, false) }
}

/// The legal action of option `(kind, id)`, the first in legal order (as the Python player's `option_action`).
fn option_action(legal: &[Action], kind: &str, id: u16) -> Option<Action> {
    let red = |t: u8| RED_TIDS.contains(&t);
    let of = |at: ActionType| legal.iter().filter(move |a| a.action_type == at);
    match kind {
        "discard" | "dama" | "break" => {
            let (t, r) = tile_of(id);
            of(ActionType::Discard).find(|a| a.tile.is_some_and(|x| x / 4 == t && red(x) == r)).cloned()
        }
        "riichi" => of(ActionType::Riichi).next().cloned(),
        "pass" => of(ActionType::Pass).next().cloned(),
        "chi" | "pon" => {
            let r = id >= 76;
            let at = if kind == "chi" { ActionType::Chi } else { ActionType::Pon };
            of(at).find(|a| {
                if a.consume_tiles.iter().any(|&c| red(c)) != r {
                    return false;
                }
                if kind == "pon" {
                    return true;
                }
                let called = a.tile.map_or(0, |t| t / 4);
                let pos = a.consume_tiles.iter().filter(|&&c| c / 4 < called).count() as u16;
                id - if r { 76 } else { 38 } == pos
            })
            .cloned()
        }
        "daiminkan" | "ankan" | "kakan" => {
            let at = match kind {
                "daiminkan" => ActionType::Daiminkan,
                "ankan" => ActionType::Ankan,
                _ => ActionType::Kakan,
            };
            of(at).find(|a| a.tile.is_some_and(|t| u16::from(t / 4) + 42 == id)).cloned()
        }
        _ => None,
    }
}

/// Own score minus each other seat's (from the shimocha), / 25000.
pub fn score_diffs(v: &MjView) -> [f64; 3] {
    let s = usize::from(v.seat);
    std::array::from_fn(|k| (f64::from(v.scores[s]) - f64::from(v.scores[(s + k + 1) % 4])) / 25000.0)
}

/// The danger columns of one option from the features `x` [seat][tile type][feature] and the score
/// differences `diff` (`score_diffs`).
fn danger_row(kind: &str, id: u16, x: &[[[f64; super::fold_model::N_FEATURES]; 34]; 3], diff: &[f64; 3]) -> [f64; N_D] {
    let mut out = [0.0; N_D];
    let di = danger_idx();
    let n = DANGER.len();
    match kind {
        "discard" | "dama" | "break" | "riichi" => {
            let t = usize::from(tile_of(id).0);
            for s in 0..3 {
                for j in 0..n {
                    out[s * N_PER + j] = x[s][t][di[j]];
                }
                out[s * N_PER + n] = diff[s];
            }
        }
        "keep" | "ankan" | "kakan" => {}
        _ => {
            for s in 0..3 {
                for j in 0..n {
                    if DANGER[j].starts_with("seat:") {
                        out[s * N_PER + j] = x[s][0][di[j]];
                    }
                }
                out[s * N_PER + n] = diff[s];
            }
        }
    }
    out
}

/// The state model's P / M / F probabilities on the seat's decision, as the mixture reads them; `None` for a
/// set without `state.json` or a decision that is not an own-turn discard out of riichi (for checks against
/// suzume-train's `sp` model).
pub fn state_probs(
    obs: &riichienv_core::observation::Observation,
    kyoku_events: &[Value],
    a1: &A1Config,
    memo: &mut SeatMemo,
) -> Result<Option<[f64; 3]>, String> {
    let Some(st) = &a1.models.state else { return Ok(None) };
    let view = view_of(obs, kyoku_events, &a1.cfg, false)?;
    if view.in_riichi || !view.legal.iter().any(|a| a.action_type == ActionType::Discard) {
        return Ok(None);
    }
    let ctx = Ctx::with_memo(&view, &a1.cfg, std::mem::take(memo));
    let x = state_inputs(&ctx, a1.models, kyoku_events);
    *memo = ctx.into_memo();
    let rv = river_view(&view, kyoku_events)?;
    Ok(x?.map(|x| probs_of(st, &x, &rv)))
}

/// One A1 decision on the seat's observation, the kyoku's events and the seat's kyoku memo.
pub fn decide(
    obs: &riichienv_core::observation::Observation,
    kyoku_events: &[Value],
    a1: &A1Config,
    memo: &mut SeatMemo,
) -> Result<Action, String> {
    let legal = obs.legal_actions_method();
    if legal.is_empty() {
        return Err(format!("A1 at seat {} has no legal actions", obs.player_id));
    }
    for at in [ActionType::Tsumo, ActionType::Ron, ActionType::KyushuKyuhai] {
        if let Some(a) = legal.iter().find(|a| a.action_type == at) {
            return Ok(a.clone());
        }
    }
    if let Some((t, r)) = memo.riichi_tile.take() {
        let id = if r { 34 + u16::from(t / 9) } else { u16::from(t) };
        if let Some(a) = option_action(&legal, "discard", id) {
            return Ok(a);
        }
    }
    let cfg = &a1.cfg;
    let md = a1.models;
    let view = view_of(obs, kyoku_events, cfg, false)?;
    let ctx = Ctx::with_memo(&view, cfg, std::mem::take(memo));
    let opp = view.riichi_threats().next().is_some();
    // This decision's universal options, rows (with the danger columns) and its model's choice.
    let uni = (|| {
        let rv = river_view(&view, kyoku_events)?;
        let Some(r) = decision_rows(&ctx, md, &rv) else { return Ok(None) };
        let ex = r.pk.and_then(|k| md.pmf[k].as_deref());
        let best = match ex {
            Some(ex) if a1.pmf_fixed.is_some() || md.state.is_some() => {
                let sc = expert_scores(ex, &r.rows, &r.u.state.0, opp, &view);
                let p = match (a1.pmf_fixed, &md.state) {
                    (Some(p), _) => Some(p),
                    (None, Some(st)) => inputs_on(&ctx, &rv, Some((&r, &sc))).map(|x| probs_of(st, &x, &rv)),
                    (None, None) => None,
                };
                match p {
                    Some(p) => mixture_best(&sc, &p),
                    None => r.model.best(&r.rows, &r.u.state.0, opp, &view),
                }
            }
            _ => {
                let riichi: Vec<bool> = r.u.options.iter().map(|(k, _)| *k == "riichi").collect();
                let calls = |k: &str| matches!(k, "chi" | "pon" | "daiminkan");
                let bias: Vec<f64> = r
                    .u
                    .options
                    .iter()
                    .map(|(k, _)| match () {
                        _ if r.kind == 3 && calls(k) => a1.call_offset,
                        _ if *k == "riichi" => a1.riichi_offset,
                        _ => 0.0,
                    })
                    .collect();
                r.model.best_styled(r.ends, a1.aggression, &r.rows, &r.u.state.0, opp, &view,
                                    riichi.contains(&true).then_some(&riichi[..]),
                                    (a1.call_offset != 0.0 || a1.riichi_offset != 0.0).then_some(&bias[..]))
            }
        };
        Ok::<_, String>(Some((r.u, best)))
    })();
    let out = (|| {
        let uni = uni?;
        if legal.iter().any(|a| matches!(a.action_type, ActionType::Ankan | ActionType::Kakan)) {
            // The kan rows are built with the features set's keep, then keep becomes A1's own choice (as Python).
            let base = if view.in_riichi {
                riichi_keep_row(&ctx)
            } else {
                mlp_choice(&ctx).map(|(_, u, _, best)| (u.options[best], u.rows[best]))
            };
            if let Some(mut k) = base.and_then(|(b, row)| kan_universal(&ctx, b, row)) {
                if let Some((u, j)) = &uni {
                    k.rows[0] = u.rows[*j];
                    k.options[0] = u.options[*j];
                }
                let rows: Vec<[f64; N_ROW]> = k
                    .rows
                    .iter()
                    .map(|r| {
                        let mut row = [0.0; N_ROW];
                        row[..N_U].copy_from_slice(r);
                        row
                    })
                    .collect();
                let best = md.kan.best_styled(md.end_models(5), a1.aggression, &rows, &k.state.0, opp, &view, None, None);
                if best > 0 {
                    let (kind, id) = k.options[best];
                    if let Some(a) = option_action(&legal, kind, id) {
                        return Ok(Some(a));
                    }
                }
            }
        }
        if let Some((u, j)) = &uni {
            let (kind, id) = u.options[*j];
            if let Some(a) = option_action(&legal, kind, id) {
                ctx.memo().riichi_tile = (kind == "riichi").then(|| tile_of(id));
                return Ok(Some(a));
            }
        }
        Ok::<_, String>(None)
    })();
    *memo = ctx.into_memo();
    if let Some(a) = out? {
        return Ok(a);
    }
    let discards: Vec<&Action> = legal.iter().filter(|a| a.action_type == ActionType::Discard).collect();
    let first = discards.first().map(|a| a.tile);
    if !discards.is_empty() && discards.iter().all(|a| a.tile == first.flatten()) {
        return Ok(discards[0].clone());
    }
    if let Some(a) = legal.iter().find(|a| a.action_type == ActionType::Pass) {
        return Ok(a.clone());
    }
    if let Some(a) = discards.iter().find(|a| a.tile == obs.drawn_tile) {
        return Ok((*a).clone());
    }
    Ok(discards.first().map_or_else(|| legal[0].clone(), |a| (*a).clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_set_parses_with_its_features_set() {
        for b in &BUILTIN {
            let c = A1Config::builtin(b.name).expect("a built-in set");
            assert!(c.models.discard1.is_some() && c.models.discard2.is_some(), "{}", b.name);
            assert!(c.models.discard12.is_none(), "{}", b.name);
            assert!(!c.cfg.pf && !c.cfg.fold);
            assert!(c.cfg.check_models().is_ok());
            assert!(super::super::params::set_models(c.cfg.models).tables.is_some());
        }
        assert_eq!(BUILTIN[0].name, "suzume");
        assert!(A1Config::builtin("legacy-suzume-mlp3").is_none());
    }

    #[test]
    fn builtin_suzume_has_the_endpoint_models() {
        let c = A1Config::builtin("suzume").unwrap();
        assert_eq!((c.aggression, c.call_shift, c.riichi_shift), (0.0, 0.0, 0.0));
        for (i, k) in STYLE_KINDS.iter().enumerate() {
            assert!(c.models.ends[i].iter().all(Option::is_some), "{k}");
        }
        assert!(A1Config::new(c.cfg, c.models, None, -1.0, 0.0, 0.0).is_ok());
        let r = A1Config::builtin("suzume-rla225").unwrap();
        assert!(A1Config::new(r.cfg, r.models, None, 0.5, 0.0, 0.0).is_err());
        assert!(A1Config::new(c.cfg, c.models, None, 0.0, 1.5, 0.0).is_err());
        let s = A1Config::new(c.cfg, c.models, None, 0.0, -1.0, 0.5).unwrap();
        assert_eq!((s.call_offset, s.riichi_offset), (-2.5, 1.75));
    }

    #[test]
    fn the_tile_inputs_are_the_experts_scores_minus_their_best() {
        let a_ev = u_names().iter().position(|x| x == "A_ev").unwrap();
        let a_win = u_names().iter().position(|x| x == "A_win").unwrap();
        // a riichi decision: dama 3, riichi 3, break 4 (the red five 34 too), and a non-discard option
        let options = [("dama", 3u16), ("riichi", 3), ("break", 4), ("discard", 34), ("pass", 82)];
        let mut rows = vec![[0.0; N_ROW]; options.len()];
        for (i, r) in rows.iter_mut().enumerate() {
            r[a_ev] = 10.0 + i as f64;
            r[a_win] = 0.1 * i as f64;
        }
        let p = vec![1.0, 2.5, 0.5, 0.7, 9.0];
        let f = vec![-1.0, -3.0, 0.2, 0.1, 9.0];
        let sc = [p.clone(), vec![0.0; 5], f.clone()];
        let x = pmf_tile(&options, &rows, &sc);
        // type 3: P best is the riichi option (2.5, the decision's best), F's best over its options -1.0
        assert_eq!(x[3][..4], [0.0, -1.0 - 0.2, 11.0, 0.1]);
        assert_eq!(x[3][4], 0.0);
        // type 4: P max(0.5, 0.7) - 2.5, A_* of the red-five discard (row 3); F max(0.2, 0.1) - 0.2
        assert_eq!(x[4][0], 0.7 - 2.5);
        assert_eq!(x[4][1], 0.0);
        assert_eq!(x[4][2..4], [13.0, 0.1 * 3.0]);
        assert_eq!(x[5], [0.0; N_PMF_TILE]);
    }

    #[test]
    fn a_state_model_reads_expert_inputs_by_name() {
        let text = |inputs: &[&str], sit: &[&str]| {
            let n = inputs.len() + sit.len();
            serde_json::json!({
                "inputs": inputs, "mean": vec![0.0; inputs.len()], "std": vec![1.0; inputs.len()],
                "sit_inputs": sit, "sit_mean": vec![0.0; sit.len()], "sit_std": vec![1.0; sit.len()],
                "hidden": 2, "layers": 1, "W1": vec![vec![0.0; n]; 2], "b1": [0.0, 0.0], "Wh": [], "bh": [],
                "W2": vec![vec![0.0; 4]; 3], "b2": [0.0, 0.0, 0.0],
            })
            .to_string()
        };
        let sit: Vec<&str> = super::super::pf::SIT_NAMES.iter().copied().chain(["pmf:top_p2:p", "pmf:has3"]).collect();
        let st = parse_state(&text(&["patterns_log", "pmf:f_diff"], &sit)).unwrap();
        assert_eq!(st.cols, [0, N_FEATURES + 1]);
        assert_eq!(st.summary_cols, [3, 23]);
        assert_eq!(st.n_sit, 23);
        assert!(parse_state(&text(&["patterns_log", "pmf:nope"], &sit)).is_err());
        assert!(parse_state(&text(&["patterns_log"], &["pmf:has2"])).is_err());
        let old = parse_state(&text(&["patterns_log"], &super::super::pf::SIT_NAMES)).unwrap();
        assert!(old.summary_cols.is_empty() && old.n_sit == 21);
    }

    #[test]
    fn the_style_bias_moves_the_choice_at_every_aggression() {
        // options: pass, call. The model prefers pass by 1, the endpoint by 4; a call bias of 2 is added after the mix.
        let (s, end, bias) = ([1.0, 0.0], [4.0, 0.0], [0.0, 2.0]);
        assert_eq!(styled_choice(&s, None, 0.0, None, Some(&bias)), 1);
        assert_eq!(styled_choice(&s, Some(&end), 0.5, None, None), 0);
        // w = 0.5: mixed pass 2.5, call 0 + 2; w = 0.25: pass 1.75, call 2 (the bias is not scaled by 1 - w)
        assert_eq!(styled_choice(&s, Some(&end), 0.5, None, Some(&bias)), 0);
        assert_eq!(styled_choice(&s, Some(&end), 0.25, None, Some(&bias)), 1);
        // w = 1 plays the endpoint plus the bias: pass 4 vs call 2, then a bias of 5 calls
        assert_eq!(styled_choice(&s, Some(&end), 1.0, None, Some(&bias)), 0);
        assert_eq!(styled_choice(&s, Some(&end), 1.0, None, Some(&[0.0, 5.0])), 1);
        // riichi: the group comes from s + bias; the mix picks within it
        let riichi = [false, true, true];
        let (s, end) = ([0.0, 0.5, 0.4], [9.0, 0.0, 1.0]);
        assert_eq!(styled_choice(&s, Some(&end), 1.0, Some(&riichi), None), 2);
        assert_eq!(styled_choice(&s, Some(&end), 1.0, Some(&riichi), Some(&[0.0, -1.0, -1.0])), 0);
    }

    #[test]
    fn pmf_weights_parse() {
        assert_eq!(parse_pmf("P").unwrap(), [1.0, 0.0, 0.0]);
        assert_eq!(parse_pmf(" f ").unwrap(), [0.0, 0.0, 1.0]);
        assert_eq!(parse_pmf("[1, 2, 1]").unwrap(), [0.25, 0.5, 0.25]);
        for bad in ["", "X", "1,2", "1,-1,1", "0,0,0", "a,b,c", "1,nan,1"] {
            assert!(parse_pmf(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn pmf_experts_need_all_three_files() {
        let b = &BUILTIN[0];
        // The base models without the built-in endpoint models.
        let base = |n: &str| {
            let ends = n.ends_with("_push") || n.ends_with("_fold");
            b.files.iter().find(|(f, _)| *f == n && !ends).map(|(_, t)| t.to_string())
        };
        let with = |extra: &'static [&'static str]| {
            move |n: &str| -> Result<Option<String>, String> {
                if extra.contains(&n) {
                    return Ok(base("discard3"));
                }
                Ok(base(n))
            }
        };
        let full = A1Models::from_reader("t", with(&["discard3_P", "discard3_M", "discard3_F"])).unwrap();
        assert!(full.pmf[3].is_some() && full.pmf[..3].iter().all(Option::is_none) && full.state.is_none());
        assert!(A1Models::from_reader("t", with(&["discard3_P", "discard3_M"])).is_err());
        assert!(A1Models::from_reader("t", with(&[])).unwrap().pmf.iter().all(Option::is_none));
    }

    #[test]
    fn discard12_is_needed_only_without_the_split() {
        let b = &BUILTIN[0];
        let base = |n: &str| b.files.iter().find(|(f, _)| *f == n).map(|(_, t)| t.to_string());
        assert!(A1Models::from_reader("t", |n: &str| Ok(base(n))).unwrap().discard12.is_none());
        let no_d2 = |n: &str| Ok(if n == "discard2" { None } else { base(n) });
        assert!(A1Models::from_reader("t", no_d2).is_err());
        let d12 = |n: &str| Ok(base(match n { "discard2" => "x", "discard12" => "discard1", _ => n }));
        assert!(A1Models::from_reader("t", d12).unwrap().discard12.is_some());
    }

    #[test]
    fn the_style_options_are_checked() {
        let b = &BUILTIN[0];
        // The base models without the built-in endpoint models.
        let base = |n: &str| {
            let ends = n.ends_with("_push") || n.ends_with("_fold");
            b.files.iter().find(|(f, _)| *f == n && !ends).map(|(_, t)| t.to_string())
        };
        let plain = A1Models::from_reader("t", |n: &str| Ok(base(n))).unwrap();
        assert!(plain.ends.iter().flatten().all(Option::is_none));
        let md: &'static A1Models = Box::leak(Box::new(plain));
        let cfg = A1Config::builtin("suzume").unwrap().cfg;
        assert!(A1Config::new(cfg, md, None, 0.0, 0.0, 0.0).is_ok());
        assert!(A1Config::new(cfg, md, None, 0.25, 0.0, 0.0).is_err()); // no endpoint models
        assert!(A1Config::new(cfg, md, None, f64::NAN, 0.0, 0.0).is_err());
        assert!(A1Config::new(cfg, md, None, 0.0, 1.0, 0.0).is_ok());
        assert!(A1Config::new(cfg, md, None, 0.0, 0.0, f64::INFINITY).is_err());
        // discard3 itself as both endpoints of discard3 only.
        let ends = A1Models::from_reader("t", |n: &str| {
            Ok(base(if n == "discard3_fold" || n == "discard3_push" { "discard3" } else { n }))
        })
        .unwrap();
        assert!(ends.ends[3].iter().all(Option::is_some) && ends.ends[..3].iter().flatten().all(Option::is_none));
        let md: &'static A1Models = Box::leak(Box::new(ends));
        assert!(A1Config::new(cfg, md, None, -0.5, 0.0, 0.0).is_ok());
        assert!(A1Config::new(cfg, md, None, 1.5, 0.0, 0.0).is_err());
    }
}
