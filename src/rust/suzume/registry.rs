//! The rule registry: every suzume rule, in evaluation order.
//!
//! Adding a rule is one new struct (in `rules/`) and one line in `RULES`;
//! removing one is deleting its line. The index in `RULES` is the rule's id
//! inside a process (the decision trace uses it); the `name` is the stable
//! id everything outside the crate uses.

use riichienv_core::action::Action;

use super::ctx::Ctx;
use super::rules::*;

/// Which mode a rule belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Maximise the own hand's value.
    Push,
    /// Avoid dealing in.
    Fold,
}

/// The decision a rule acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionKind {
    Win,
    Discard,
    Kyushu,
}

/// What a rule is, for listing and tracing.
pub struct RuleInfo {
    /// Stable snake_case id.
    pub name: &'static str,
    pub mode: Mode,
    pub decision: DecisionKind,
    /// One line.
    pub description: &'static str,
}

/// One rule.
pub trait Rule: Sync {
    fn info(&self) -> &'static RuleInfo;

    /// The action to take, always one of `ctx.view.legal`, or `None` to
    /// leave the decision to the rules after this one.
    fn decide(&self, _ctx: &Ctx) -> Option<Action> {
        None
    }
}

/// Every rule, in evaluation order: the first that answers decides.
pub static RULES: &[&dyn Rule] = &[&WinAlways, &KyushuAlways, &PfDiscard, &FoldDiscard, &KanMlp, &UniversalMlp, &DiscardForced];

/// The registry index of a rule name.
#[cfg(test)]
pub fn index_of(name: &str) -> Option<usize> {
    RULES.iter().position(|r| r.info().name == name)
}

impl Mode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Mode::Push => "push",
            Mode::Fold => "fold",
        }
    }
}

impl DecisionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            DecisionKind::Win => "win",
            DecisionKind::Discard => "discard",
            DecisionKind::Kyushu => "kyushu",
        }
    }
}

