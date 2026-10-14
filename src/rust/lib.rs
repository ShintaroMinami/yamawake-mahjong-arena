//! yamawake: a duplicate riichi mahjong arena over riichienv-core.
//!
//! - `arena`: the N-game lockstep loop (split fixed walls, in-loop rulebot
//!   and suzume bots, observation rows for everyone else);
//! - `rulebot`: the author's own rule-based bot, ported to riichienv;
//! - `suzume`: a bot decided by per-decision MLPs, run through a registry of
//!   rules (the frame);
//! - `eval`: hand evaluation the bots share (the single-player DP, its
//!   scoring, shanten; on riichienv-core, the fork in `src/engine/`).
//!
//! The modules are crate-private: the Python extension (`py`) is the only
//! way in, so the compiler reports any code it no longer reaches. Without
//! the `extension-module` feature (plain `cargo test`) nothing is reachable
//! from outside, so dead code is only reported with the feature on.
#![cfg_attr(not(feature = "extension-module"), allow(dead_code))]

pub(crate) mod eval;
pub(crate) mod arena;
pub(crate) mod rulebot;
pub(crate) mod suzume;
#[cfg(feature = "extension-module")]
mod py;
