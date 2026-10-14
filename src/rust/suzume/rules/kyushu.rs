//! Kyushu kyuhai.

use riichienv_core::action::{Action, ActionType};

use crate::suzume::ctx::Ctx;
use crate::suzume::registry::{DecisionKind, Mode, Rule, RuleInfo};

/// Declare kyushu kyuhai (an abortive draw) whenever the engine offers it.
pub struct KyushuAlways;

static KYUSHU_ALWAYS: RuleInfo = RuleInfo {
    name: "kyushu_always",
    mode: Mode::Push,
    decision: DecisionKind::Kyushu,
    description: "always declare kyushu kyuhai when the engine offers it",
};

impl Rule for KyushuAlways {
    fn info(&self) -> &'static RuleInfo {
        &KYUSHU_ALWAYS
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        ctx.view.find(ActionType::KyushuKyuhai).cloned()
    }
}
