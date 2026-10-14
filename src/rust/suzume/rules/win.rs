//! Wins.

use riichienv_core::action::{Action, ActionType};

use crate::suzume::ctx::Ctx;
use crate::suzume::registry::{DecisionKind, Mode, Rule, RuleInfo};

/// Take every win offered (tsumo or ron).
pub struct WinAlways;

static WIN_ALWAYS: RuleInfo = RuleInfo {
    name: "win_always",
    mode: Mode::Push,
    decision: DecisionKind::Win,
    description: "always declare tsumo/ron when the engine offers it",
};

impl Rule for WinAlways {
    fn info(&self) -> &'static RuleInfo {
        &WIN_ALWAYS
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        let v = ctx.view;
        v.find(ActionType::Tsumo).or_else(|| v.find(ActionType::Ron)).cloned()
    }
}
