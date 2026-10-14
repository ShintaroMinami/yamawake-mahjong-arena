//! One turn's view of the table: everything a decision is allowed to read.
//!
//! Pure functions of the state and the config, no history of its own, so
//! replaying a state always yields the same play.

use std::cell::OnceCell;

use super::RulebotConfig;
use super::sp_adapter::DiscardValue;
use super::view::BotView;

pub struct Decision<'a> {
    pub state: &'a BotView,
    pub config: &'a RulebotConfig,
    /// Per-decision memo of `hand_ev` (the single-player DP): the rungs
    /// of one decision ask for the same table repeatedly, and the DP is
    /// 60% of the bot's whole runtime when recomputed each time.
    pub(super) sp_cache: OnceCell<Option<(f32, Vec<DiscardValue>)>>,
}

impl<'a> Decision<'a> {
    pub fn new(state: &'a BotView, config: &'a RulebotConfig) -> Self {
        Self { state, config, sp_cache: OnceCell::new() }
    }
}
