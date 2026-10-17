//! Folding against a riichi (`fold=1`), and push / fold by the predicted
//! state (`pf=1`) (Mode::Fold; design: docs/suzume.md,
//! "押し引き").

use riichienv_core::action::{Action, ActionType};

use crate::suzume::ctx::Ctx;
use crate::suzume::fold_model::{choice_by, fold_choice, seat_features};
use crate::suzume::pf::{PfState, legal_types, predict};
use crate::suzume::registry::{DecisionKind, Mode, Rule, RuleInfo};
use crate::suzume::rules::universal_mlp::kan_choice;
use crate::suzume::universal::kind_of;

/// Push / fold (`pf=1`): at an own-turn discard decision out of riichi (two
/// or more legal types, not the declaring discard of a riichi), predict the
/// state (`pf::predict`) and keep it in the seat memo for the decisions after
/// it. In the mawashi or fold state a discard12 / discard3 / tenpai decision
/// is the mawashi or fold model's discard, in M after a kan when the kan MLP
/// prefers one (`kan=1`; F never kans); a riichi-legal decision and the push
/// state are left to the rules after this one (`universal_mlp` takes the
/// state's models).
pub struct PfDiscard;

static PF_DISCARD: RuleInfo = RuleInfo {
    name: "pf_discard",
    mode: Mode::Fold,
    decision: DecisionKind::Discard,
    description: "pf=1: predict the push / mawashi / fold state; in M or F discard by the mawashi or fold model \
                  (params/state, params/hand discard_M / discard_F)",
};

impl Rule for PfDiscard {
    fn info(&self) -> &'static RuleInfo {
        &PF_DISCARD
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        let v = ctx.view;
        if !ctx.cfg.pf || v.in_riichi || v.find(ActionType::Discard).is_none() || ctx.memo().riichi_tile.is_some() {
            return None;
        }
        if ctx.discard_cands().len() < 2 {
            return None;
        }
        let feats = seat_features(ctx);
        let legal = legal_types(ctx);
        let state = predict(&ctx.models.pf().state, ctx.view, &feats, &legal);
        ctx.memo().pf.state = state;
        if state == PfState::Push || !matches!(kind_of(ctx)?, 0 | 2) {
            return None;
        }
        if ctx.cfg.kan && state == PfState::Mawashi {
            if let Some(a) = kan_choice(ctx) {
                return Some(a);
            }
        }
        let md = if state == PfState::Mawashi { &ctx.models.pf().mawashi } else { ctx.models.fold() };
        let (t, aka) = choice_by(ctx, md, &feats)?;
        ctx.discard_action(t, aka)
    }
}

/// Betaori: with another seat in riichi, no own riichi and the best
/// after-discard shanten at least `fold_shanten`, discard what the fold
/// model (`fold_model`, `params/hand/<hand>/discard_F.json`) scores highest: each
/// tile type's scores against every other seat combined by their soft
/// minimum (`FoldMlp::combine`), the lower type on a tie, the plain copy
/// before the red five.
pub struct FoldDiscard;

static FOLD_DISCARD: RuleInfo = RuleInfo {
    name: "fold_discard",
    mode: Mode::Fold,
    decision: DecisionKind::Discard,
    description: "fold=1 (not with pf=1): betaori against a riichi from fold_shanten out (the fold MLP, params/hand discard_F)",
};

impl Rule for FoldDiscard {
    fn info(&self) -> &'static RuleInfo {
        &FOLD_DISCARD
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        let v = ctx.view;
        if !ctx.cfg.fold || ctx.cfg.pf || v.in_riichi || v.riichi_threats().next().is_none() {
            return None;
        }
        if ctx.min_shanten()? < ctx.cfg.fold_shanten {
            return None;
        }
        let (t, aka) = fold_choice(ctx)?;
        ctx.discard_action(t, aka)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suzume::SuzumeConfig;
    use crate::suzume::test_util::own_turn_view;

    fn decide(hand: &str, safe: &[usize], fold_mode: u8) -> Option<u8> {
        let mut v = own_turn_view(hand);
        v.threats[1].riichi = true;
        v.threats[1].wind = 28;
        safe.iter().for_each(|&t| v.threats[1].safe[t] = true);
        let cfg = SuzumeConfig { fold: fold_mode == 1, pf: false, ..SuzumeConfig::default() };
        FoldDiscard.decide(&Ctx::new(&v, &cfg)).and_then(|a| a.tile).map(|t| t / 4)
    }

    #[test]
    fn folds_the_genbutsu_from_two_shanten() {
        // 3 shanten, no pair broken by 9p; 5s is the genbutsu.
        assert_eq!(decide("13579m2468p159s12z", &[22], 1), Some(22));
        assert_eq!(decide("13579m2468p159s12z", &[22], 0), None);
    }

    #[test]
    fn pushes_below_fold_shanten() {
        // 1 shanten: folds from fold_shanten = 1 (the default), pushes at 2.
        let mut v = own_turn_view("123m456p789s1135z7z");
        v.threats[1].riichi = true;
        v.threats[1].wind = 28;
        v.threats[1].safe[33] = true;
        let at = |fs: i8| {
            let cfg = SuzumeConfig { fold: true, fold_shanten: fs, pf: false, ..SuzumeConfig::default() };
            FoldDiscard.decide(&Ctx::new(&v, &cfg)).and_then(|a| a.tile).map(|t| t / 4)
        };
        assert_eq!(at(2), None);
        assert_eq!(at(1), Some(33));
    }

    #[test]
    fn an_honor_beats_a_middle_tile_without_genbutsu() {
        // Nothing safe: the fold model avoids the live middle tiles.
        let t = decide("13579m2468p159s12z", &[], 1).unwrap();
        assert!(crate::eval::feat::tiles::is_yaochu(usize::from(t)), "{t}");
    }
}
