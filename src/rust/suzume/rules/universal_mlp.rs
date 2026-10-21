//! The per-decision MLPs over the universal features (`suzume::universal`)
//! choose the option; the kan MLP decides kans against that choice.
//!
//! One model per decision kind (`params::models`, from
//! `params/legacy/hand/<hand>/<file>.json`, `params::MODEL_FILES`; docs/suzume.md):
//!
//! | model (file) | scope |
//! |---|---|
//! | `discard12` (`discard_P_12`) | `discard` decisions, lowest after-discard shanten 1-2 |
//! | `discard3` (`discard_P_3`) | `discard` decisions, lowest after-discard shanten 3+ |
//! | `tenpai` (`discard_P_t`) | after-discard tenpai without riichi legal |
//! | `riichi` (`riichi_P`) | riichi legal: dama / break per discard, riichi per tenpai-keeping discard |
//! | `call` (`call_P`) | chi or pon legal: pass and each chi / pon |
//! | `kan` (`kan`) | an ankan / kakan legal (`KanMlp`): keep vs each kan (a daiminkan is a `call` option) |
//!
//! With `pf=1` the riichi and call decisions take the models of the seat's
//! push / fold state (`suzume::pf`: `riichi_M` in M and F, `call_M`,
//! `call_F`; `params::pf_models`).
//!
//! A hand the DP does not value (no draw left, a yakuless tenpai) is decided
//! too: its rows carry `A_in_dp` = 0, as in the training data. Decisions
//! under an opponent riichi are in the fits (weighted by the push state).
//!
//! An option scores `w2 . h_L + b2`, `h_1 = relu(W1 [(x - mean) / std,
//! onehot(state), sit] + b1)` and any further layers (`mlp::Hidden`), `x` the
//! model's input columns of the option's universal row and `sit` the
//! decision's standardized situation (`pf::situation`; only in a model file
//! with `sit_inputs`, the same for every option); the
//! decision's state outside the model's one-hot list gets a zero one-hot. A
//! model whose states carry `/opp` / `/noopp` splits each state by whether
//! another seat has declared riichi.
//! A file's outside bias (`outside_bias`; only the 1-layer set has one) is
//! added to the riichi options (`riichi`: round bin + unique top) and to the
//! chi / pon options (`call`: dealer + unique top).
//! The highest score is played, the first in option order on a tie (pass
//! first in a call, dama and break before riichi, discards in tile order);
//! `riichi=0` / `call=0` / `kan=0` take the riichi / chi and pon / daiminkan
//! options out. A riichi option declares; its tile is kept in the seat memo
//! for the declaring discard.

use std::rc::Rc;

use riichienv_core::action::{Action, ActionType};

use crate::suzume::ctx::Ctx;
use crate::suzume::mlp::add_bias;
use crate::suzume::pf::PfState;
use crate::suzume::registry::{DecisionKind, Mode, Rule, RuleInfo};
use crate::suzume::view::MjView;
use crate::suzume::universal::{
    N_U, Universal, kan_universal, kind_of, option_action, riichi_keep_row, u_names,
};

pub use crate::suzume::params::Umlp;

impl Umlp {
    /// The network's score of one universal row in state `state`, with the
    /// decision's situation share `sit` (`sit_bias`).
    pub fn score(&self, row: &[f64; N_U], state: Option<usize>, sit: &[f64]) -> f64 {
        let n_in = self.inputs.len();
        let x: Vec<f64> = self.cols.iter().enumerate().map(|(j, &c)| (row[c] - self.mean[j]) / self.std[j]).collect();
        let mut z = self.net.pre_dense(&x);
        if let Some(si) = state {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk += self.net.w1_row(k)[n_in + si];
            }
        }
        add_bias(&mut z, sit);
        let mut s = self.b2;
        for (w, h) in self.w2.iter().zip(self.net.activate(z)) {
            s += w * h;
        }
        s
    }

    /// The situation share of a decision in view `v` (empty without
    /// `sit_inputs`).
    pub fn sit_bias(&self, v: &MjView) -> Vec<f64> {
        self.sit.as_ref().map_or_else(Vec::new, |s| s.bias(&self.net, v))
    }

    /// The one-hot index of state `name`. A model fitted with the opponent
    /// riichi in its states (`<name>/opp`, `<name>/noopp`) takes the entry
    /// that `opp` (some other seat has declared riichi) picks.
    pub fn state_index(&self, name: &str, opp: bool) -> Option<usize> {
        let suf = if opp { "/opp" } else { "/noopp" };
        self.states.iter().position(|s| s == name || s.strip_prefix(name) == Some(suf))
    }
}

/// Some other seat has declared riichi.
fn opp_riichi(ctx: &Ctx) -> bool {
    ctx.view.threats.iter().any(|t| t.riichi)
}

/// The push (P) model a decision of this kind uses.
fn push_model(ctx: &Ctx) -> Option<&'static Umlp> {
    let m = ctx.models;
    Some(match kind_of(ctx)? {
        0 if ctx.min_shanten()? >= 3 => &m.discard3,
        0 => match &m.discard_split {
            Some((one, two)) => if ctx.min_shanten()? <= 1 { one } else { two },
            None => &m.discard12,
        },
        1 => &m.riichi,
        2 => &m.tenpai,
        _ => {
            ctx.view.find(ActionType::Pass)?;
            &m.call
        }
    })
}

/// The push models' decision of this context (`mlp_choice_in` in P): (model
/// index, the decision's universal rows, every option's score, the chosen
/// index); `None` when no model applies. A hand the DP does not value (no
/// draw left, a yakuless tenpai) is decided too: its rows carry
/// `A_in_dp` = 0, as in the training data.
pub fn mlp_choice(ctx: &Ctx) -> Option<(&'static Umlp, Rc<Universal>, Vec<f64>, usize)> {
    mlp_choice_in(ctx, PfState::Push)
}

/// The model a decision of this kind uses in push / fold state `state`:
/// riichi in M or F takes `riichi_m`, call in M `call_m`, in F `call_f`;
/// everything else its push model.
fn model_in_state(ctx: &Ctx, state: PfState) -> Option<&'static Umlp> {
    let (p, m) = (push_model(ctx)?, ctx.models);
    Some(match state {
        PfState::Push => p,
        _ if std::ptr::eq(p, &m.riichi) => &m.pf().riichi_m,
        PfState::Mawashi if std::ptr::eq(p, &m.call) => &m.pf().call_m,
        PfState::Fold if std::ptr::eq(p, &m.call) => &m.pf().call_f,
        _ => p,
    })
}

/// `mlp_choice` with the models of push / fold state `state` (the model
/// used is returned). In F a daiminkan is never chosen.
pub fn mlp_choice_in(ctx: &Ctx, state: PfState) -> Option<(&'static Umlp, Rc<Universal>, Vec<f64>, usize)> {
    let kan_ok = ctx.cfg.kan && state != PfState::Fold;
    let md = model_in_state(ctx, state)?;
    let change = md.inputs.iter().any(|n| n.starts_with("A_change"));
    let u = ctx.universal(change);
    if u.options.is_empty() {
        return None;
    }
    let state = md.state_index(&u.state.0, opp_riichi(ctx));
    let c = &u.context;
    let names = u_names();
    let target = if md.kind == "riichi" { "O_riichi" } else { "O_call" };
    let tcol = names.iter().position(|n| n == target).expect("option column");
    let bias = md.out_round[c.round_bin] + if c.top { md.out_top } else { 0.0 } + if c.oya { md.out_dealer } else { 0.0 };
    // The scores are kept for the decision: the fold features, the P-state
    // choice and the kan decision's keep row all score the same rows.
    let scores = ctx.mlp_scores(std::ptr::from_ref(md) as usize, &u, || {
        let sit = md.sit_bias(ctx.view);
        u.rows.iter().map(|r| md.score(r, state, &sit) + if r[tcol] > 0.0 { bias } else { 0.0 }).collect()
    });
    // The switches: `riichi=0` never picks a riichi option, `call=0` never a
    // chi or pon, `kan=0` (or the F state) never a daiminkan. Ties keep the earlier option (discards are in tile
    // order).
    let allowed = |i: usize| match u.options[i].0 {
        "riichi" => ctx.cfg.riichi,
        "chi" | "pon" => ctx.cfg.call,
        "daiminkan" => kan_ok,
        _ => true,
    };
    let mut best = (0..scores.len()).find(|&i| allowed(i))?;
    for i in best + 1..scores.len() {
        if allowed(i) && scores[i] > scores[best] {
            best = i;
        }
    }
    Some((md, u, scores, best))
}

/// The kan decision (`kan=1`): when an ankan or kakan is
/// legal, the kan MLP compares `keep` (what the decision's own MLP would
/// play; in riichi the tsumogiri) with each kan (`universal::kan_universal`)
/// and returns the best kan when it beats keep; otherwise the rules after it
/// decide as before. With `pf=1` a seat out of riichi whose latest state is F
/// never kans (in P and M the kan MLP decides).
pub struct KanMlp;

static KAN_MLP: RuleInfo = RuleInfo {
    name: "kan_mlp",
    mode: Mode::Push,
    decision: DecisionKind::Discard,
    description: "kan=1: the kan MLP (universal features) takes an ankan / kakan when it beats \
                  the decision's own choice (in riichi: the tsumogiri); kan=0, or pf=1 in the F state out of \
                  riichi: never kan",
};

/// The kan decision's MLP choice: the legal kan to make, or `None` (keep).
pub fn kan_choice(ctx: &Ctx) -> Option<Action> {
    let v = ctx.view;
    if !v.legal.iter().any(|a| matches!(a.action_type, ActionType::Ankan | ActionType::Kakan)) {
        return None;
    }
    let (base, row) = if v.in_riichi {
        riichi_keep_row(ctx)?
    } else {
        let (_, u, _, best) = mlp_choice(ctx)?;
        (u.options[best], u.rows[best])
    };
    let u = kan_universal(ctx, base, row)?;
    let md = &ctx.models.kan;
    let state = md.state_index(&u.state.0, opp_riichi(ctx));
    let sit = md.sit_bias(v);
    let scores: Vec<f64> = u.rows.iter().map(|r| md.score(r, state, &sit)).collect();
    // Keep (index 0) wins ties.
    let best = (1..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
    if best == 0 {
        return None;
    }
    let (kind, id) = u.options[best];
    let t = (id - 42) as u8;
    v.legal
        .iter()
        .find(|a| {
            let at = if kind == "ankan" { ActionType::Ankan } else { ActionType::Kakan };
            a.action_type == at && a.tile.is_some_and(|x| x / 4 == t)
        })
        .cloned()
}

impl Rule for KanMlp {
    fn info(&self) -> &'static RuleInfo {
        &KAN_MLP
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        if !ctx.cfg.kan || (ctx.cfg.pf && !ctx.view.in_riichi && ctx.memo().pf.state == PfState::Fold) {
            return None;
        }
        kan_choice(ctx)
    }
}

/// The universal MLPs choose the discard, riichi or dama, and call.
pub struct UniversalMlp;

static UNIVERSAL_MLP: RuleInfo = RuleInfo {
    name: "universal_mlp",
    mode: Mode::Push,
    decision: DecisionKind::Discard,
    description: "per-decision MLPs over the universal features choose the discard, riichi / dama or call \
                  (discard12, discard3, tenpai, riichi, call)",
};

impl Rule for UniversalMlp {
    fn info(&self) -> &'static RuleInfo {
        &UNIVERSAL_MLP
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        let v = ctx.view;
        // The declaring discard of a riichi this rule chose.
        if v.find(ActionType::Riichi).is_none() && v.find(ActionType::Discard).is_some() {
            if let Some((t, aka)) = ctx.memo().riichi_tile.take() {
                return ctx.discard_action(t, aka);
            }
        }
        if ctx.discard_cands().len() == 1 && kind_of(ctx).is_some_and(|k| k != 3) {
            return None;
        }
        let state = if ctx.cfg.pf { ctx.memo().pf.state } else { PfState::Push };
        let (_, u, _, best) = mlp_choice_in(ctx, state)?;
        let (kind, id) = u.options[best];
        if v.find(ActionType::Riichi).is_some() {
            ctx.memo().riichi_tile = None;
        }
        let action = option_action(ctx, kind, id)?;
        if kind == "riichi" {
            let (t, aka) = if id >= 34 { (4 + 9 * (id - 34) as u8, true) } else { (id as u8, false) };
            ctx.memo().riichi_tile = Some((t, aka));
        }
        Some(action)
    }
}

#[cfg(test)]
mod tests {
    use riichienv_core::action::ActionType;

    use super::*;
    use crate::suzume::test_util::{own_turn_view, react_view};
    use crate::suzume::{SuzumeConfig, SeatMemo, decide, decide_with_memo, decider_name};


    #[test]
    fn models_are_consistent() {
        for md in crate::suzume::params::set_models(Default::default()).push_models() {
            let (n_in, n_st) = (md.inputs.len(), md.states.len());
            assert_eq!(md.cols.len(), n_in, "{}", md.kind);
            assert_eq!((md.mean.len(), md.std.len()), (n_in, n_in));
            assert_eq!(md.net.n_cols, n_in + n_st);
            assert_eq!(md.w2.len(), md.net.out_width());
            assert!(md.std.iter().all(|&s| s > 0.0));
            assert!(md.states.iter().all(|s| s.starts_with(match md.kind.as_str() {
                "discard12" | "discard3" => "discard/",
                k => k,
            })));
        }
    }

    /// Each model decides its decision kind with finite scores and a legal
    /// action, by the rule `universal_mlp`.
    #[test]
    fn models_decide_their_kinds() {
        let def = SuzumeConfig::default();
        let cases = [
            (own_turn_view("13m456p789s11z2457p"), "discard12"),
            (own_turn_view("1479m258p369s1357z"), "discard3"),
            (own_turn_view("123m456p789s1122z5m"), "riichi"),
            (react_view("2344m567p789s115z", "4m", 0), "call"),
        ];
        for (view, kind) in &cases {
            let ctx = Ctx::new(view, &def);
            let (md, _, scores, best) = mlp_choice(&ctx).unwrap();
            assert_eq!(md.kind, *kind);
            assert!(scores.iter().all(|s| s.is_finite()) && best < scores.len());
            let (a, r) = decide(view, &def).unwrap();
            assert_eq!(decider_name(r), "universal_mlp", "{kind}");
            assert!(view.legal.contains(&a));
        }
        let mut tenpai = own_turn_view("123m456p789s1122z5m");
        tenpai.legal.retain(|a| a.action_type != ActionType::Riichi);
        let ctx = Ctx::new(&tenpai, &def);
        assert_eq!(mlp_choice(&ctx).unwrap().0.kind, "tenpai");
    }

    /// `riichi=0` never picks a riichi option; `call=0` always passes.
    #[test]
    fn switches_remove_their_options() {
        let no_riichi = SuzumeConfig { riichi: false, ..SuzumeConfig::default() };
        let view = own_turn_view("123m456p789s1122z5m");
        assert_ne!(decide(&view, &no_riichi).unwrap().0.action_type, ActionType::Riichi);
        let no_call = SuzumeConfig { call: false, ..SuzumeConfig::default() };
        let view = react_view("2344m567p789s115z", "4m", 0);
        assert_eq!(decide(&view, &no_call).unwrap().0.action_type, ActionType::Pass);
    }

    /// A riichi this rule declares keeps its tile for the declaring discard.
    #[test]
    fn riichi_keeps_its_tile() {
        let view = own_turn_view("123m456p789s1122z5m");
        let mut memo = SeatMemo::default();
        let (a, _) = decide_with_memo(&view, &SuzumeConfig::default(), &mut memo).unwrap();
        if a.action_type == ActionType::Riichi {
            let tile = memo.riichi_tile.expect("kept");
            let mut next = view.clone();
            next.legal.retain(|x| x.action_type == ActionType::Discard);
            let (d, r) = decide_with_memo(&next, &SuzumeConfig::default(), &mut memo).unwrap();
            assert_eq!(decider_name(r), "universal_mlp");
            assert_eq!((d.tile.unwrap() / 4, crate::eval::feat::tiles::is_aka_tid(d.tile.unwrap())), tile);
        } else {
            assert_eq!(a.action_type, ActionType::Discard);
            assert!(memo.riichi_tile.is_none());
        }
    }
}
