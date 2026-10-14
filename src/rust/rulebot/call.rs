//! Whether to call on the tile just discarded, and for what.
//!
//! kobalab's `select_fulou`: within three of tenpai, call when the hand is
//! worth more called than passed (priced by `kobalab_ev`, or with
//! `call_ev_sp` by the single-player DP); further out, call when it lowers
//! the yaku-aware shanten (`yaku_shanten`).

use super::config::{CALL_EV_PUSH, CALL_EV_TENPAI};
use super::decision::Decision;
use super::action::BotAction;
use super::hand_shanten;
use super::tile::Tile;
use super::view::AllowedActions;
use super::{aka_slot, kobalab_ev};

impl Decision<'_> {
    /// Every call the rules allow here, paired with the tiles it gives up.
    ///
    /// Daiminkan is left out on purpose: it opens a new dora and shortens the
    /// wall for everyone on another seat's tile, a bet this bot does not price
    /// (ankan / kakan are `kan.rs`, on the own turn).
    pub(super) fn call_candidates(&self, allowed: &AllowedActions) -> Vec<([usize; 2], BotAction)> {
        let Some(called_tile) = self.state.last_discard() else {
            return vec![];
        };
        let called_tile_type = called_tile.plain().tile_type_index();
        let mut out: Vec<([usize; 2], BotAction)> = Vec::new();
        if allowed.pon {
            out.push((
                [called_tile_type; 2],
                BotAction::Pon {
                    target: allowed.discarder,
                    tile: called_tile,
                    consume_tiles: self.pon_consume_tiles(called_tile),
                },
            ));
        }
        let chi = [
            (allowed.chi_low, called_tile.next_tile_type(), called_tile.next_tile_type().next_tile_type()),
            (allowed.chi_mid, called_tile.prev_tile_type(), called_tile.next_tile_type()),
            (allowed.chi_high, called_tile.prev_tile_type().prev_tile_type(), called_tile.prev_tile_type()),
        ];
        for (offered, lo, hi) in chi {
            if offered {
                out.push((
                    [lo.tile_type_index(), hi.tile_type_index()],
                    BotAction::Chi {
                        target: allowed.discarder,
                        tile: called_tile,
                        consume_tiles: self.chi_consume_tiles(lo, hi),
                    },
                ));
            }
        }
        out
    }

    /// Which call to make on the tile just discarded, if any: kobalab's
    /// `select_fulou`, calling when the hand is worth more called.
    ///
    /// What the hand is worth, not whether the call advances it, is what
    /// decides -- a call that gains a step and loses the yaku scores zero, and
    /// one that gains nothing but opens a bigger hand can still be right.
    ///
    /// Only within three of tenpai. Further out `eval_shoupai` changes unit
    /// from points to ukeire, so neither the comparison against passing nor
    /// the thresholds below would mean anything; kobalab switches to a
    /// yaku-aware shanten there (`select_call_far`).
    pub(super) fn select_call(&self, allowed: &AllowedActions) -> Option<BotAction> {
        let hand = self.state.hand();
        let n_sets = self.state.n_sets();
        let now = hand_shanten::shanten_of(&hand, n_sets);
        let threatened = self.threat_danger().is_some();

        // Calling into a riichi from this far out buys a step and gives up the
        // safe tiles needed to get out of the way.
        if threatened && now >= 3 {
            return None;
        }
        if now >= 3 {
            return self.select_call_far(allowed);
        }
        if self.config.call_ev_sp {
            return self.select_call_sp(allowed, threatened);
        }

        let mut evaluator = self.kobalab_ev();
        let chi_runs = self.state.chi_runs();
        let pon_sets = self.state.pon_sets();
        // What the hand is worth left alone, which a call has to beat.
        let mut best = evaluator.value(&hand, chi_runs, pon_sets, n_sets, kobalab_ev::NO_TILE);
        let mut chosen = None;

        let called_tile_type = self.state.last_discard()?.plain().tile_type_index();
        for (consume_tiles, action) in self.call_candidates(allowed) {
            // The called tile joins the meld, so the concealed hand only loses
            // what the meld consumed: 3n+1 becomes 3n+2, one block lighter.
            let mut after = hand;
            for tile_type in consume_tiles {
                after[tile_type] -= 1;
            }
            let n_sets_after = n_sets - 1;
            // `shanten_of` reads a 3n+2 hand as the shanten its best discard leaves,
            // which is the shape a call leaves behind. Checked in `kobalab_ev`.
            let shanten_after = hand_shanten::shanten_of(&after, n_sets_after);
            if shanten_after >= 3 {
                continue;
            }
            let (chi_runs_after, pon_sets_after) = self.melds_after(&action, chi_runs, pon_sets, called_tile_type);
            let value = evaluator.value(&after, &chi_runs_after, &pon_sets_after, n_sets_after, called_tile_type as u8);

            if threatened {
                let floor = if shanten_after > 0 { CALL_EV_PUSH } else { CALL_EV_TENPAI };
                if value < floor {
                    continue;
                }
            }
            if value - best > f32::EPSILON {
                best = value;
                chosen = Some(action);
            }
        }
        chosen
    }

    /// DP pricing (`call_ev_sp`): the single-player DP values both futures directly.
    ///
    /// Passing is worth `hand_ev` (the DP already prices the draw about to
    /// happen elsewhere going to us later); calling is worth the DP value of
    /// the melded hand over the same horizon. Both are point expectations
    /// from one solver, so the comparison is like with like -- the
    /// thing kobalab's mixed-unit valuation could not offer.
    fn select_call_sp(&self, allowed: &AllowedActions, threatened: bool) -> Option<BotAction> {
        let hand = self.state.hand();
        let n_sets = self.state.n_sets();
        let pass = self
            .hand_ev_with(!self.config.call_sp_flat_pass)
            .map_or(0., |(ev, _)| ev);
        // Best call by value first; whether calling at all beats passing is
        // a separate question, asked once at the end.
        let mut best = f32::NEG_INFINITY;
        let mut chosen = None;

        for (consume_tiles, action) in self.call_candidates(allowed) {
            let mut after = hand;
            for tile_type in consume_tiles {
                after[tile_type] -= 1;
            }
            let n_sets_after = n_sets - 1;
            let shanten_after = hand_shanten::shanten_of(&after, n_sets_after);
            if shanten_after >= 3 {
                continue;
            }
            let (meld_type, ids, spent_aka) = self.sp_meld_of(&action)?;
            let Some(value) = self.hand_ev_after_call(&after, spent_aka, meld_type, ids)
            else {
                continue;
            };

            if threatened {
                let floor = if shanten_after > 0 { CALL_EV_PUSH } else { CALL_EV_TENPAI };
                if value < floor {
                    continue;
                }
            }
            if value > best {
                best = value;
                chosen = Some(action);
            }
        }
        if best > pass { chosen } else { None }
    }

    /// The physical meld a call action makes: (scorer meld type, real ids,
    /// the aka slot it spends from the hand). Reds keep their red id so the
    /// DP scorer credits their han; plain ids never collide with a red one.
    fn sp_meld_of(&self, action: &BotAction) -> Option<(u8, Vec<u8>, Option<usize>)> {
        let called_tile = self.state.last_discard()?;
        let (meld_type, consume_tiles) = match action {
            BotAction::Pon { consume_tiles, .. } => (1u8, consume_tiles),
            BotAction::Chi { consume_tiles, .. } => (0u8, consume_tiles),
            _ => return None,
        };
        let mut counter = [0u8; 34];
        let mut spent_aka = None;
        let mut id_of = |t: &Tile, ours: bool| -> u8 {
            if t.is_aka {
                let slot = super::aka_slot(t.tile_type as usize).unwrap();
                if ours {
                    spent_aka = Some(slot);
                }
                [16u8, 52, 88][slot]
            } else {
                let k = counter[t.tile_type as usize].min(2);
                counter[t.tile_type as usize] += 1;
                t.tile_type * 4 + 3 - k
            }
        };
        let ids = vec![
            id_of(&consume_tiles[0], true),
            id_of(&consume_tiles[1], true),
            id_of(&called_tile, false),
        ];
        Some((meld_type, ids, spent_aka))
    }

    /// kobalab's far-from-tenpai branch: call if it buys a step towards a yaku.
    ///
    /// Points mean nothing this far out -- `eval_shoupai` is counting
    /// ukeire rather than value there, and the two are not comparable. What
    /// is left is the shanten number, but the plain one would happily open a
    /// hand that can no longer win. `yaku_shanten` is the number that answers
    /// instead, and any call that lowers it is taken, first one found.
    fn select_call_far(&self, allowed: &AllowedActions) -> Option<BotAction> {
        let hand = self.state.hand();
        let n_sets = self.state.n_sets();
        let chi_runs = self.state.chi_runs();
        let pon_sets = self.state.pon_sets();
        let now = self.yaku_shanten(&hand, chi_runs, pon_sets, n_sets);
        let called_tile_type = self.state.last_discard()?.plain().tile_type_index();

        for (consume_tiles, action) in self.call_candidates(allowed) {
            let mut after = hand;
            for tile_type in consume_tiles {
                after[tile_type] -= 1;
            }
            let (chi_runs_after, pon_sets_after) = self.melds_after(&action, chi_runs, pon_sets, called_tile_type);
            if self.yaku_shanten(&after, &chi_runs_after, &pon_sets_after, n_sets - 1) < now {
                return Some(action);
            }
        }
        None
    }

    /// The meld lists the hand would hold after this call.
    pub(super) fn melds_after(
        &self,
        action: &BotAction,
        chi_runs: &[u8],
        pon_sets: &[u8],
        called_tile_type: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut chi_runs_after = chi_runs.to_vec();
        let mut pon_sets_after = pon_sets.to_vec();
        match action {
            BotAction::Pon { .. } => pon_sets_after.push(called_tile_type as u8),
            BotAction::Chi { consume_tiles, .. } => {
                let low = consume_tiles
                    .iter()
                    .map(|t| t.plain().tile_type_index())
                    .chain([called_tile_type])
                    .min()
                    .unwrap_or(called_tile_type);
                chi_runs_after.push(low as u8);
            }
            _ => {}
        }
        (chi_runs_after, pon_sets_after)
    }

    /// The two tiles a pon gives up, spending an owned red five first.
    ///
    /// A red five is worth more in a meld than in the hand -- both count the
    /// same han, and the meld is the copy that cannot be forced out later.
    fn pon_consume_tiles(&self, called_tile: Tile) -> [Tile; 2] {
        let plain = called_tile.plain();
        match aka_slot(plain.tile_type_index()) {
            Some(slot) if self.state.akas_in_hand()[slot] => [plain.as_aka(), plain],
            _ => [plain; 2],
        }
    }

    /// The two tiles a chi gives up, marking whichever is an owned red five.
    fn chi_consume_tiles(&self, lo: Tile, hi: Tile) -> [Tile; 2] {
        let akas = self.state.akas_in_hand();
        let owned = |t: Tile| aka_slot(t.tile_type_index()).is_some_and(|slot| akas[slot]);
        if owned(lo) || owned(hi) {
            [lo.as_aka(), hi.as_aka()]
        } else {
            [lo, hi]
        }
    }
}
