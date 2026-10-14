//! The rulebot's knobs and the fixed values its play is built on.
//!
//! Only what tells the two shipped configs apart (`STANDARD` and `LATEST` in
//! `src/python/yamawake/rulebot.py`) is a knob; everything both share is a
//! constant here. `Default` is `LATEST`, and the golden tests pin the play.

/// Fold once shanten reaches this, on the shanten number alone.
pub(super) const FOLD_SHANTEN: i8 = 2;
/// Points of one-shanten hand value that buy one unit of danger.
pub(super) const EV_PER_DANGER: f32 = 50.;
/// The same trade-off at tenpai.
pub(super) const TENPAI_EV_PER_DANGER: f32 = 5.;
/// Danger multiplier per threat past the first.
pub(super) const DANGER_PER_THREAT: f32 = 1.6;
/// Least visible value (dora plus yakuhai pons, at least 1) before a melded
/// hand counts as a threat.
pub(super) const MELD_MIN_VALUE: u32 = 3;
/// Silent ron value at or above which a tenpai hand stays dama.
pub(super) const RIICHI_DAMA_PT: u32 = 6400;
/// Build the hand on the single-player DP's value up to this shanten.
pub(super) const BUILD_EV_SHANTEN: i8 = 2;
/// Least a call must be worth into a threat without reaching tenpai.
pub(super) const CALL_EV_PUSH: f32 = 750.;
/// Least a call must be worth into a threat when it reaches tenpai.
pub(super) const CALL_EV_TENPAI: f32 = 250.;

/// The knobs of the rulebot. `Default` is `LATEST`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RulebotConfig {
    /// Price calls with the single-player DP instead of kobalab_ev. The
    /// same threatened floors apply, but on the DP's point scale.
    pub call_ev_sp: bool,
    /// Value the pass hand without the riichi assumption, so the call
    /// comparison is not skewed by the DP's riichi+ura upside.
    pub call_sp_flat_pass: bool,
    /// Declare ankan / kakan by kobalab's `select_gang` (`kan.rs`); never
    /// daiminkan.
    pub kan: bool,
}
