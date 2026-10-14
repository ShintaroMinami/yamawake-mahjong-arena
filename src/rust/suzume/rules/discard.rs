//! The forced discard.

use riichienv_core::action::Action;

use crate::suzume::ctx::Ctx;
use crate::suzume::registry::{DecisionKind, Mode, Rule, RuleInfo};

/// Only one tile may go (e.g. the drawn tile after riichi): discard it
/// without valuing anything.
pub struct DiscardForced;

static DISCARD_FORCED: RuleInfo = RuleInfo {
    name: "discard_forced",
    mode: Mode::Push,
    decision: DecisionKind::Discard,
    description: "discard the only legal tile (after riichi: tsumogiri)",
};

impl Rule for DiscardForced {
    fn info(&self) -> &'static RuleInfo {
        &DISCARD_FORCED
    }

    fn decide(&self, ctx: &Ctx) -> Option<Action> {
        match ctx.discard_cands() {
            [only] => ctx.discard_action(only.tile_type, only.is_aka),
            _ => None,
        }
    }
}
