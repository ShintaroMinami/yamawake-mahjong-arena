//! suzume: a bot decided by per-decision MLPs over universal features,
//! fitted to a reinforcement-learned teacher (design: docs/suzume.md).
//!
//! A decision tries the rules of `registry::RULES` in order; the first that
//! answers decides:
//!
//! | rule | decides |
//! |---|---|
//! | `win_always` | tsumo / ron whenever offered |
//! | `kyushu_always` | kyushu kyuhai whenever offered |
//! | `pf_discard` | push / fold: the predicted state; the mawashi or fold model's discard (`pf=1`, the default) |
//! | `fold_discard` | betaori against a riichi (`fold=1`, only with `pf=0`) |
//! | `kan_mlp` | a kan when the kan MLP prefers it to the decision's own choice (`kan=1`) |
//! | `universal_mlp` | the discard, riichi or dama, and chi / pon / pass by the per-decision MLPs |
//! | `discard_forced` | the only legal discard (after riichi: tsumogiri) |
//!
//! When no rule decides, the fallback is: pass if passing is legal, else
//! the drawn tile, else the first legal discard, else the first legal
//! action.
//!
//! The options (`config::SuzumeConfig`: kan, call, riichi, fold,
//! fold_shanten, pf) are all a caller may vary; the fixed parameters (the DP's
//! constants, the weights the universal features read, the MLPs) come from
//! `params/` (`params.rs`).
//!
//! # The rule contract
//!
//! A rule (`registry::Rule`) is a unit struct with static metadata
//! (`RuleInfo`: stable `name`, `mode`, `decision`, one-line
//! `description`) and `decide(&Ctx) -> Option<Action>`. A rule returns
//! only actions taken from `ctx.view.legal`, so the bot cannot pick an
//! illegal action (and `decide` checks it). Rules share one `ctx::Ctx` per
//! decision, whose memo tables (shanten, ukeire, the DP per riichi
//! assumption) are filled on first use.
//!
//! Adding a rule is one unit struct in `rules/` and one line in
//! `registry::RULES`; removing one is deleting its line.
//!
//! # Traceability
//!
//! `decide` returns the deciding rule's registry index (`FALLBACK` for the
//! fallback); the arena records it per decision (`GameSlot::suzume_trace`).

pub mod config;
pub mod ctx;
pub mod fold_model;
pub mod memo;
pub mod mlp;
pub mod params;
pub mod pf;
pub mod registry;
pub mod rules;
pub mod sp;
pub mod ukeire;
pub mod universal;
pub mod view;

use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use serde_json::Value;

pub use config::SuzumeConfig;
use ctx::Ctx;
pub use memo::SeatMemo;
use registry::RULES;
pub use view::MjView;

/// The decider id of the fallback (no rule decided).
pub const FALLBACK: u8 = u8::MAX;

/// The name of a decider id (`RULES` index or `FALLBACK`).
pub fn decider_name(id: u8) -> &'static str {
    RULES.get(usize::from(id)).map_or("fallback", |r| r.info().name)
}

/// One decision with an empty memo: the action and the id of the rule that
/// chose it.
#[cfg(test)]
pub fn decide(view: &MjView, cfg: &SuzumeConfig) -> Result<(Action, u8), String> {
    decide_with_memo(view, cfg, &mut SeatMemo::default())
}

/// `decide` with the seat's per-kyoku memo (`memo::SeatMemo`): the same
/// decision, faster when the memo is warm.
pub fn decide_with_memo(view: &MjView, cfg: &SuzumeConfig, memo: &mut SeatMemo) -> Result<(Action, u8), String> {
    if view.legal.is_empty() {
        return Err("suzume: no legal action".into());
    }
    let ctx = Ctx::with_memo(view, cfg, std::mem::take(memo));
    let out = decide_in(&ctx);
    *memo = ctx.into_memo();
    out
}

/// `decide` in a given context (its memo tables are left filled, e.g. for
/// reading the DP the decision used).
pub(crate) fn decide_in(ctx: &Ctx) -> Result<(Action, u8), String> {
    let view = ctx.view;
    let mut chosen = None;
    for (i, rule) in RULES.iter().enumerate() {
        if let Some(a) = rule.decide(&ctx) {
            chosen = Some((a, i as u8));
            break;
        }
    }
    let (action, id) = chosen.unwrap_or_else(|| (fallback(&ctx), FALLBACK));
    if !view.legal.contains(&action) {
        return Err(format!(
            "suzume rule {} chose an illegal action {action:?}",
            decider_name(id)
        ));
    }
    Ok((action, id))
}

fn fallback(ctx: &Ctx) -> Action {
    let v = ctx.view;
    if let Some(a) = v.find(ActionType::Pass) {
        return a.clone();
    }
    if let Some(a) = v
        .legal
        .iter()
        .find(|a| a.action_type == ActionType::Discard && a.tile == v.drawn_tile && a.tile.is_some())
    {
        return a.clone();
    }
    v.find(ActionType::Discard).unwrap_or(&v.legal[0]).clone()
}

#[cfg(test)]
pub(crate) mod test_util;
#[cfg(test)]
mod tests;

/// One suzume decision: the seat's observation `obs`, the mjai events of
/// the kyoku so far (the river the fold and push / fold rules read) and the
/// live wall count, with the seat's kyoku memo `memo` (a fresh one for a
/// decision on its own). Returns the action and the deciding rule's id.
pub fn decide_observation(
    obs: &Observation,
    kyoku_events: &[Value],
    cfg: &SuzumeConfig,
    memo: &mut SeatMemo,
) -> Result<(Action, u8), String> {
    let view = view_of(obs, kyoku_events, cfg, false)?;
    decide_with_memo(&view, cfg, memo)
}

/// The view a decision of `decide_observation` plays on (also what the
/// feature dumps read, so they see the same view): the river record is
/// applied when a rule of `cfg` reads it, or always with `river`.
pub fn view_of(obs: &Observation, kyoku_events: &[Value], cfg: &SuzumeConfig, river: bool) -> Result<MjView, String> {
    let legal = obs.legal_actions_method();
    if legal.is_empty() {
        return Err(format!("suzume at seat {} has no legal actions", obs.player_id));
    }
    let mut view = MjView::from_observation(obs, legal);
    // Only the fold and push / fold rules read the river record
    // (genbutsu, early cuts, red fives, recency).
    let own_discard = !view.in_riichi && view.find(ActionType::Discard).is_some();
    if river || (cfg.fold && !cfg.pf && view.riichi_threats().next().is_some()) || (cfg.pf && own_discard) {
        view.apply_river(&view::river_from_mjai(kyoku_events)?);
    }
    Ok(view)
}
