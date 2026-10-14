//! The single-player DP table the rulebot reads: every discard of the hand,
//! valued by `dp::Solver` and ranked.
//!
//! What the solver is told follows from the rules and the visible state:
//!
//! * how many draws are left: on our own turn, a quarter of the live wall;
//!   while reacting to a discard, the seats between the discarder and us
//!   draw first, so a quarter of what remains after them. The last of those
//!   draws is haitei when the count divides exactly.
//! * which tiles and red fives are visible from this seat (hand, every
//!   seat's discards, every meld), so the solver counts what is left to draw;
//! * whether riichi is assumed once tenpai (it needs the 1000-point stick,
//!   and a closed hand), and whether a double riichi is still possible
//!   (nobody has called and this seat has not discarded).
//!
//! Meld dora are not added as a han bonus: the scorer reads the meld tiles
//! (indicator dora and red fives) itself, so a bonus would count them twice.
//!
//! Values are f32, the precision the rulebot's thresholds are written in.
//! Ranking: higher EV first; ties go to the higher win chance, then tenpai
//! chance, then wider ukeire, then the tile more worth giving up
//! (`discard_priority`), then the lower tile type.

use std::cmp::Ordering;

use super::decision::Decision;
use super::tile::Tile;
use crate::eval::dp::{Input, Solver};
use crate::eval::feat::tiles::AKA_TIDS;

thread_local! {
    /// Hand verdicts the rulebot's DP runs share on this thread (`EvalCache`
    /// keys hold the hand, its context and the winning tile, so sharing
    /// across kyoku and games is exact); bounded (`EVAL_CAP`).
    static EVAL: std::cell::RefCell<crate::eval::score::EvalCache> = Default::default();
}

/// Verdicts past which the shared cache starts over.
const EVAL_CAP: usize = 1 << 20;

/// The DP's candidates for `input`, through this thread's verdict cache.
fn solve(input: &Input) -> Vec<crate::eval::dp::Candidate> {
    EVAL.with_borrow_mut(|eval| {
        eval.bound(EVAL_CAP);
        Solver::run_cached(input, eval)
    })
}

/// One discard and what the DP says it is worth.
pub struct DiscardValue {
    pub tile: Tile,
    /// Expected points per extra-han column; `[0]` is the hand as it stands.
    pub ev: Vec<f32>,
    win_prob: f32,
    tenpai_prob: f32,
    ukeire: u32,
}

impl DiscardValue {
    fn headline(&self) -> f32 {
        self.ev.first().copied().unwrap_or(0.)
    }

    /// Best first; see the module notes for the tie order.
    fn rank(a: &Self, b: &Self) -> Ordering {
        b.headline()
            .total_cmp(&a.headline())
            .then(b.win_prob.total_cmp(&a.win_prob))
            .then(b.tenpai_prob.total_cmp(&a.tenpai_prob))
            .then(b.ukeire.cmp(&a.ukeire))
            .then(discard_priority(b.tile).cmp(&discard_priority(a.tile)))
            .then(a.tile.tile_type.cmp(&b.tile.tile_type))
    }
}

/// How readily a tile is given up: honors 7, terminals 6, down to 2 for a
/// plain five and 1 for a red five (kept longest).
fn discard_priority(t: Tile) -> u8 {
    if t.is_aka {
        return 1;
    }
    if t.tile_type >= 27 {
        return 7;
    }
    let rank = t.tile_type % 9 + 1;
    7 - rank.min(10 - rank)
}

/// Draws left before the wall runs out, and whether the last one is haitei.
/// `drawn_before_us` is how many tiles other seats draw before our next one.
fn draw_horizon(drawable_count: u8, drawn_before_us: u8) -> (usize, bool) {
    let left = drawable_count.saturating_sub(drawn_before_us);
    (usize::from(left / 4), left % 4 == 0)
}

impl Decision<'_> {
    /// The hand's expected value and its ranked discards, or `None` when
    /// there is nothing to value (fewer than four tiles in the wall, no draw
    /// left for us, or a hand already complete). Computed once per decision.
    pub(super) fn hand_ev(&self) -> Option<(f32, &[DiscardValue])> {
        self.sp_cache
            .get_or_init(|| self.hand_ev_with(true))
            .as_ref()
            .map(|(ev, table)| (*ev, table.as_slice()))
    }

    /// `hand_ev`, optionally without assuming riichi
    /// (`call_sp_flat_pass` prices passing without the riichi upside).
    pub(super) fn hand_ev_with(&self, allow_riichi: bool) -> Option<(f32, Vec<DiscardValue>)> {
        let state = self.state;
        if state.drawable_count() < 4 || state.shanten() < 0 {
            return None;
        }
        let own_turn = state.allowed().discard;
        let before_us = if own_turn { 0 } else { self.seats_drawing_before_us() };
        let (draws_left, last_draw_is_haitei) = draw_horizon(state.drawable_count(), before_us);
        if draws_left < 1 {
            return None;
        }
        let is_menzen = state.chi_runs().is_empty() && state.pon_sets().is_empty() && state.open_kans().is_empty();
        let input = Input {
            hand: state.hand(),
            akas_in_hand: state.akas_in_hand(),
            seen_counts: state.seen_counts(),
            akas_seen: self.akas_seen_with(state.akas_in_hand()),
            draws_left,
            own_turn: own_turn,
            last_draw_is_haitei,
            double_riichi_possible: own_turn && self.double_riichi_possible(),
            riichi_when_tenpai: allow_riichi && self.can_afford_riichi(),
            is_menzen,
            round_wind: state.round_wind(),
            player_wind: state.player_wind(),
            dora_indicators: state.dora_indicator_types(),
            meld_dora_han: 0,
            melds: self.dp_melds(),
            collect_tiles: Vec::new(),
            ron: None,
        };
        let mut table: Vec<DiscardValue> = solve(&input)
            .into_iter()
            .map(|c| DiscardValue {
                tile: if c.tile_type < 0 { Tile::new(0, false) } else { Tile::new(c.tile_type as u8, c.discard_is_aka) },
                ev: c.ev.iter().map(|&v| v as f32).collect(),
                win_prob: c.win.first().copied().unwrap_or(0.) as f32,
                tenpai_prob: c.tenpai.first().copied().unwrap_or(0.) as f32,
                ukeire: c.effective_tiles.iter().map(|&(_, n)| n).sum(),
            })
            .collect();
        table.sort_by(DiscardValue::rank);
        let ev = *table.first()?.ev.first()?;
        Some((ev, table))
    }

    /// DP value of the hand after a hypothetical call (the `call_ev_sp` pricing).
    ///
    /// `hand_after` is the concealed hand minus the tiles the meld uses
    /// (3n+2: the call takes the place of a draw), `spent_aka` the red-five
    /// slot the meld used (if any) and `meld_ids` the new meld's physical ids.
    /// Priced over the same draws as passing, so the two compare directly.
    pub(super) fn hand_ev_after_call(
        &self,
        hand_after: &[u8; 34],
        spent_aka: Option<usize>,
        meld_type: u8,
        meld_ids: Vec<u8>,
    ) -> Option<f32> {
        let state = self.state;
        if state.drawable_count() < 4 {
            return None;
        }
        let (draws_left, last_draw_is_haitei) =
            draw_horizon(state.drawable_count(), self.seats_drawing_before_us());
        if draws_left < 1 {
            return None;
        }
        let mut akas_in_hand = state.akas_in_hand();
        if let Some(slot) = spent_aka {
            akas_in_hand[slot] = false;
        }
        let mut akas_seen = self.akas_seen_with(akas_in_hand);
        for (slot, seen) in akas_seen.iter_mut().enumerate() {
            *seen |= meld_ids.contains(&AKA_TIDS[slot]);
        }
        let mut melds = self.dp_melds();
        melds.push((meld_type, meld_ids, true, 0));
        let input = Input {
            hand: *hand_after,
            akas_in_hand: akas_in_hand,
            seen_counts: state.seen_counts(),
            akas_seen: akas_seen,
            draws_left,
            own_turn: true,
            last_draw_is_haitei,
            double_riichi_possible: false,
            riichi_when_tenpai: self.can_afford_riichi(),
            is_menzen: false,
            round_wind: state.round_wind(),
            player_wind: state.player_wind(),
            dora_indicators: state.dora_indicator_types(),
            meld_dora_han: 0,
            melds,
            collect_tiles: Vec::new(),
            ron: None,
        };
        solve(&input)
            .iter()
            .map(|c| c.ev.first().copied().unwrap_or(0.) as f32)
            .reduce(f32::max)
    }

    /// Tiles drawn by other seats before our next draw, while reacting to a
    /// discard (`discarder` is relative to us).
    fn seats_drawing_before_us(&self) -> u8 {
        4 - self.state.allowed().discarder
    }

    fn can_afford_riichi(&self) -> bool {
        self.state.scores()[0] >= 1000
    }

    /// Nobody has called and this seat has not discarded yet.
    fn double_riichi_possible(&self) -> bool {
        self.state.discard_count(0) == 0 && (0..4).all(|r| self.state.meld_count(r) == 0)
    }

    /// `in_hand` plus every red five visible in a seat's discards or a meld.
    fn akas_seen_with(&self, in_hand: [bool; 3]) -> [bool; 3] {
        let mut seen = in_hand;
        for seat in 0..4 {
            let melds = self.state.meld_tiles(seat);
            let tiles = self.state.discards_of(seat).into_iter().chain(melds.iter().flatten().copied());
            for t in tiles.filter(|t| t.is_aka) {
                seen[super::aka_slot(t.tile_type as usize).unwrap()] = true;
            }
        }
        seen
    }

    /// Own melds as physical ids: a red five keeps its red id; plain copies
    /// take ids from the top of their tile type (never an aka id).
    fn dp_melds(&self) -> Vec<(u8, Vec<u8>, bool, i8)> {
        let mut used = [0u8; 34];
        self.state
            .own_meld_tiles()
            .iter()
            .map(|(meld_type, tiles)| {
                let ids = tiles
                    .iter()
                    .map(|t| {
                        if t.is_aka {
                            AKA_TIDS[super::aka_slot(t.tile_type as usize).unwrap()]
                        } else {
                            let k = used[t.tile_type as usize].min(2);
                            used[t.tile_type as usize] += 1;
                            t.tile_type * 4 + 3 - k
                        }
                    })
                    .collect();
                (*meld_type, ids, *meld_type != 3, 0)
            })
            .collect()
    }
}
