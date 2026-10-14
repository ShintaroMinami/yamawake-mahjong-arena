//! The single-player DP (`dp.rs`) as suzume asks it.
//!
//! What the solver is told:
//!
//! * horizon: on the own turn the other three seats draw before each of our
//!   draws, so `drawable_count / 4` draws are left (capped by the DP at
//!   `dp::MAX_TURNS`), the last one haitei when the count divides exactly.
//!   While reacting to a discard (the no-discard root, `pass_value`), the
//!   `draws_before_us` seats draw first, so with `n = drawable_count -
//!   draws_before_us` live tiles our draws are `ceil(n / 4)`, the last one
//!   haitei when `(n - 1) % 4 == 0`;
//! * unseen tiles: 4 minus the copies visible from this seat (hand, rivers,
//!   melds, dora indicators), red fives likewise;
//! * riichi: assumed once tenpai or not, as the caller asks (the riichi
//!   MLP reads both DP solves), only for a closed hand (an ankan keeps it
//!   closed); double riichi while nobody has called and this seat has not
//!   discarded;
//! * ron: the DP's ron model (`dp::RonModel`) at `params().ron_rate`, the
//!   seat's river as furiten, and the opponent discards after the last own
//!   draw (`drawable_count % 4` on the own turn); a wait whose ron has no
//!   yaku (dama without one, or an open hand without one) does not ron. The
//!   no-discard root counts three opponent discards before its first draw,
//!   as every DP step does (in fact `draws_before_us` of them come first);
//! * melds: the seat's own melds go to the scorer (their dora and red fives
//!   count, an open hand scores only with a yaku: a yakuless open hand is
//!   worth 0); the concealed part is the hand the DP plays; the DP does not
//!   call.
//!
//! Only discards that keep the hand's lowest shanten are valued.
//!
//! Solves go through the seat's `SolveCache` (`SeatMemo::solves`): equal
//! inputs give equal outputs, so a solve asked again within the kyoku (the
//! same hand valued by two rules, the discard after a call valued at the
//! call decision) is read back instead of solved.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::memo::SeatMemo;
use super::params::params;
use super::view::MjView;
use crate::eval::dp::{Candidate, Input, RonModel, Solver};
use crate::eval::score::EvalCache;

/// One discard as the DP values it, over the whole horizon.
#[derive(Clone, Copy, Debug)]
pub struct DpRow {
    pub tile_type: u8,
    pub is_aka: bool,
    /// Expected points (tsumo and ron).
    pub ev: f64,
    /// Win probability (tsumo and ron).
    pub win: f64,
    pub tenpai: f64,
    pub ukeire: u32,
}

/// The DP input of the view's hand: the own-turn hand (3n+2) when
/// `own_turn`, else the hand held while reacting (3n+1, no-discard root).
/// `None` when no draw is left.
fn dp_input_at(view: &MjView, own_turn: bool, assume_riichi: bool) -> Option<Input> {
    // Live tiles from our next draw on, and how many of our draws they hold.
    let (draws_left, tail) = if own_turn {
        (usize::from(view.drawable_count / 4), view.drawable_count % 4)
    } else {
        let n = view.drawable_count.saturating_sub(view.draws_before_us);
        (usize::from(n.div_ceil(4)), n.saturating_sub(1) % 4)
    };
    if draws_left == 0 {
        return None;
    }
    let is_menzen = !view.is_open();
    let ron = Some(RonModel { rate: f64::from(params().ron_rate), furiten: view.own_discards, tail_discards: tail });
    Some(Input {
        hand: view.hand,
        akas_in_hand: view.akas_in_hand,
        seen_counts: view.seen_counts,
        akas_seen: view.akas_seen,
        draws_left,
        own_turn,
        last_draw_is_haitei: tail == 0,
        double_riichi_possible: view.double_riichi_possible,
        riichi_when_tenpai: assume_riichi && is_menzen,
        is_menzen,
        round_wind: view.round_wind,
        player_wind: view.player_wind,
        dora_indicators: view.dora_indicator_types.clone(),
        meld_dora_han: 0,
        melds: view.own_melds.iter().map(|(t, ids, opened)| (*t, ids.clone(), *opened, 0)).collect(),
        collect_tiles: Vec::new(),
        ron,
    })
}

/// The DP input of the own-turn hand, or `None` when no draw is left.
pub fn dp_input(view: &MjView, assume_riichi: bool) -> Option<Input> {
    dp_input_at(view, true, assume_riichi)
}

/// Run the solver with `eval` lent to its scorer.
fn solve(input: &Input, eval: &mut EvalCache) -> Vec<Candidate> {
    Solver::run_cached(input, eval)
}

/// A candidate's row (curves at the full horizon); the no-discard root's
/// tile type and aka are meaningless (0, false).
fn row_of(c: &Candidate) -> DpRow {
    DpRow {
        tile_type: c.tile_type.max(0) as u8,
        is_aka: c.discard_is_aka,
        ev: c.ev.first().copied().unwrap_or(0.),
        win: c.win.first().copied().unwrap_or(0.),
        tenpai: c.tenpai.first().copied().unwrap_or(0.),
        ukeire: c.effective_tiles.iter().map(|&(_, n)| n).sum(),
    }
}

/// One seat's solves within a kyoku, keyed by their full input: an own-turn
/// input keeps its discard rows, a reacting input its no-discard root.
#[derive(Default)]
pub struct SolveCache {
    map: HashMap<u64, Vec<(Input, Vec<DpRow>)>>,
}

impl SolveCache {
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// The rows of `input`, solved on the first ask.
    fn rows(&mut self, input: &Input, eval: &mut EvalCache) -> &[DpRow] {
        let bucket = self.map.entry(input_hash(input)).or_default();
        let i = match bucket.iter().position(|(k, _)| k == input) {
            Some(i) => i,
            None => {
                let cands = solve(input, eval);
                let rows = if input.own_turn {
                    cands.iter().filter(|c| c.tile_type >= 0).map(row_of).collect()
                } else {
                    cands.first().map(row_of).into_iter().collect()
                };
                bucket.push((input.clone(), rows));
                bucket.len() - 1
            }
        };
        &bucket[i].1
    }
}

/// A hash of every field of `input` (the ron rate by its bits).
fn input_hash(i: &Input) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (i.hand, i.akas_in_hand, i.seen_counts, i.akas_seen, i.draws_left).hash(&mut h);
    (i.own_turn, i.last_draw_is_haitei, i.double_riichi_possible, i.riichi_when_tenpai, i.is_menzen).hash(&mut h);
    (i.player_wind, i.round_wind, &i.dora_indicators, i.meld_dora_han, &i.melds, &i.collect_tiles).hash(&mut h);
    if let Some(r) = &i.ron {
        (r.rate.to_bits(), r.furiten, r.tail_discards).hash(&mut h);
    }
    h.finish()
}

/// Every shanten-keeping discard of an own-turn DP input (`dp_input`)
/// valued by the DP, through the seat's solve cache.
pub fn dp_rows_of(input: &Input, memo: &mut SeatMemo) -> Vec<DpRow> {
    let SeatMemo { solves, eval, .. } = memo;
    solves.rows(input, eval).to_vec()
}

/// The value of the hand held while reacting to a discard, if we pass: the
/// DP's no-discard root (tile type and aka of the row are meaningless).
/// `None` when no draw is left.
pub fn pass_value(view: &MjView, assume_riichi: bool, memo: &mut SeatMemo) -> Option<DpRow> {
    let input = dp_input_at(view, false, assume_riichi)?;
    let SeatMemo { solves, eval, .. } = memo;
    solves.rows(&input, eval).first().copied()
}

/// The riichi stick cost in the DP's value unit (points).
pub fn deposit_value() -> f64 {
    f64::from(params().riichi_deposit)
}
