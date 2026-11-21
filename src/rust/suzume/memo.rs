//! What one suzume bot keeps across its decisions within one kyoku.
//!
//! Owned by the game that drives the bot (`arena::GameSlot`, one per
//! seat), so a game advanced by one worker thread at a time needs no lock;
//! cleared at every kyoku start. The caches (`eval`, `solves`) are exact: a
//! decision made with them warm is the decision made with them empty. The
//! other two fields are part of the decisions: `riichi_tile` carries the tile
//! an A1 bot chose with its riichi to the discard right after the declaration
//! (without it that discard is chosen anew and can differ), and `pf` carries
//! the push / fold state model's history (`pf=1` sets; A1 sets do not use it).

use crate::eval::score::EvalCache;

use super::pf::PfMemo;
use super::sp::SolveCache;

/// Per-seat, per-kyoku memo tables.
#[derive(Default)]
pub struct SeatMemo {
    /// The DP scorer's evaluator verdicts (`score::EvalCache`).
    pub eval: EvalCache,
    /// The DP solves of the kyoku, keyed by their full input (`sp::SolveCache`).
    pub solves: SolveCache,
    /// The tile of the riichi the riichi MLP just declared, for the
    /// declaring discard.
    pub riichi_tile: Option<(u8, bool)>,
    /// The push / fold state model's history (`pf=1`).
    pub pf: PfMemo,
}

impl SeatMemo {
    /// Start a new kyoku: forget everything.
    pub fn clear(&mut self) {
        self.eval.clear();
        self.solves.clear();
        self.riichi_tile = None;
        self.pf = PfMemo::default();
    }

    /// Keep only the exact caches (`eval`, `solves`), for another play of the same seat in the same kyoku
    /// (`playout` reuses them across the play-outs from one position).
    pub fn keep_caches(&mut self) {
        self.riichi_tile = None;
        self.pf = PfMemo::default();
    }
}
