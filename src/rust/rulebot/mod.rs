//! The rulebot: the author's own rule-based bot, ported to riichienv.
//!
//! It runs on this repo's stack: riichienv-core state, shanten and hand
//! scoring, and this crate's single-player DP. The config holds only the
//! knobs that tell the shipped configs apart; the rest is fixed
//! (`config.rs`).
//!
//! The state view (`view.rs`) is everything a decision may read, seat 0
//! being this hand.

pub mod config;
pub mod hand_shanten;
pub mod tile;
pub mod view;
pub mod agari;
pub mod call;
pub mod decision;
pub mod discard;
pub mod action;
pub mod kobalab_ev;
pub mod kan;
pub mod sp_adapter;
pub mod yaku_shanten;
pub mod danger;
pub mod decide;

pub use config::RulebotConfig;

pub use crate::eval::feat::tiles::{AKA_TILE_TYPES, aka_slot};

use decision::Decision;
use action::BotAction;
use view::BotView;

impl Decision<'_> {
    /// The discard to make, red fives and a declared riichi accounted for.
    fn discard_action(&self) -> Result<BotAction, String> {
        if self.state.self_in_riichi() {
            let tile = self
                .state
                .drawn_tile()
                .ok_or("riichi with no tsumo")?;
            return Ok(BotAction::Discard { tile, tsumogiri: true });
        }
        let tile_type = self.choose_discard().ok_or("no tile to discard")?;
        let tile = self.emit_tile(tile_type);
        Ok(BotAction::Discard {
            tile,
            tsumogiri: self.state.drawn_tile().is_some_and(|t| t == tile),
        })
    }
}

/// One decision for the seat the view belongs to.
///
/// Kan: with `config.kan`, ankan and kakan by kobalab's `select_gang`
/// (`kan.rs`), checked after a win and before riichi or the discard, as in
/// majiang-ai; daiminkan never (it hands the table a new dora and a shorter
/// wall on another seat's tile, a bet this bot does not price).
pub fn react(state: &BotView, config: &RulebotConfig) -> Result<BotAction, String> {
    let allowed = state.allowed();
    let decision = Decision::new(state, config);

    let kan = || if config.kan && allowed.discard { decision.select_kan(&allowed) } else { None };
    let action = if allowed.win {
        BotAction::Win
    } else if let Some(k) = kan() {
        k
    } else if allowed.riichi && decision.should_riichi() {
        // The engine asks for the discard separately, right after this.
        BotAction::Riichi
    } else if allowed.discard {
        return decision.discard_action();
    } else if allowed.pon || allowed.any_chi() {
        decision.select_call(&allowed).unwrap_or(BotAction::Pass)
    } else {
        BotAction::Pass
    };
    Ok(action)
}

/// Diagnostic string: the branch inputs a decision turns on. Test tooling
/// only.
pub fn debug_info(state: &BotView, config: &RulebotConfig) -> String {
    let d = Decision::new(state, config);
    let threat = d.threat_danger().is_some();
    let hand_ev = d.hand_ev();
    let mut out = format!(
        "shanten={} threat={threat} n_sets={} drawable_count={} \
         in_riichi={:?} ",
        state.shanten(),
        state.n_sets(),
        state.drawable_count(),
        state.in_riichi(),
    );
    if !state.allowed().discard && (state.allowed().pon || state.allowed().any_chi()) {
        let hand = state.hand();
        let n_sets = state.n_sets();
        let mut evaluator = d.kobalab_ev();
        let pass_ev = evaluator.value(&hand, state.chi_runs(), state.pon_sets(), n_sets,
                               kobalab_ev::NO_TILE);
        out.push_str(&format!("call_pass_ev={pass_ev:.3} "));
        let shanten = hand_shanten::shanten_of(&hand, n_sets);
        out.push_str(&format!("call_now_shanten={shanten} cands=["));
        if let Some(called_tile) = state.last_discard() {
            let called_tile_type = called_tile.plain().tile_type_index();
            for (consume_tiles, action) in d.call_candidates(&state.allowed()) {
                let mut after = hand;
                for t in consume_tiles {
                    after[t] -= 1;
                }
                let n_sets_after = n_sets - 1;
                let shanten_after = hand_shanten::shanten_of(&after, n_sets_after);
                let (ca, pa) = d.melds_after(&action, state.chi_runs(), state.pon_sets(),
                                             called_tile_type);
                let v = evaluator.value(&after, &ca, &pa, n_sets_after, called_tile_type as u8);
                out.push_str(&format!(
                    "({consume_tiles:?} shanten_after={shanten_after} v={v:.3}) "
                ));
            }
        }
        out.push(']');
        // Wait-by-wait tsumo scores of the PASS hand (tenpai only).
        let melds0 = evaluator.melds_of(state.chi_runs(), state.pon_sets());
        let seen = state.seen_counts();
        out.push_str(" pass_waits=[");
        let mut probe = hand;
        for t in 0..34 {
            if seen[t] >= 4 {
                continue;
            }
            probe[t] += 1;
            if hand_shanten::shanten_of(&probe, n_sets) == -1 {
                let sc = evaluator.score(&probe, &melds0, t as u8);
                out.push_str(&format!("({t},left={},sc={sc:.0}) ",
                                      4 - seen[t]));
            }
            probe[t] -= 1;
        }
        out.push(']');
    }
    if let Some(danger) = d.threat_danger() {
        let hand = state.hand();
        out.push_str("danger=[");
        for t in 0..34 {
            if hand[t] > 0 {
                out.push_str(&format!("({t},{}) ", danger[t]));
            }
        }
        out.push_str("] ");
        for p in 1..4 {
            if state.in_riichi()[p] {
                let discarded: Vec<usize> =
                    (0..34).filter(|&t| state.discard_mask(p)[t]).collect();
                let passed: Vec<usize> = (0..34)
                    .filter(|&t| state.passed_since_riichi(p)[t])
                    .collect();
                out.push_str(&format!(
                    "p{p}: discarded={discarded:?} passed={passed:?} discard_count={} ",
                    state.discard_count(p)
                ));
            }
        }
    }
    match &hand_ev {
        Some((ev, table)) => {
            out.push_str(&format!("ev={ev} table=["));
            for c in table.iter().take(6) {
                out.push_str(&format!(
                    "({},{},{:.1}) ",
                    c.tile.tile_type,
                    c.tile.is_aka,
                    c.ev.first().copied().unwrap_or(0.)
                ));
            }
            out.push(']');
        }
        None => out.push_str("ev=None"),
    }
    out
}
