//! The rulebot on the engine: one decision from a seat's observation and the
//! kyoku's mjai events (the loop and `decide` both come here), its answer
//! turned into the legal engine action that realises it.

use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use serde_json::Value;

use super::action::BotAction;
use super::config::RulebotConfig;
use super::view::{AllowedActions, BotView};
use crate::eval::feat::tiles::is_aka_tid;

/// One rulebot decision: the seat's observation `obs`, the mjai events of
/// the kyoku so far (from its `start_kyoku`) and the live wall count. The
/// BotView is rebuilt from them and the engine's action candidates; when
/// reacting, history is cut AT the row being reacted to -- a reach_accepted
/// the core may have already logged has not happened yet from this
/// decision's view. `debug` also returns the branch inputs
/// (`rulebot::debug_info`).
pub fn decide(
    obs: &Observation,
    kyoku_events: &[Value],
    cfg: &RulebotConfig,
    debug: bool,
) -> Result<(Action, Option<String>), String> {
    let seat = obs.player_id;
    let legal = obs.legal_actions_method();
    if legal.is_empty() {
        return Err(format!("rulebot at seat {seat} has no legal actions"));
    }
    let player_id = seat as usize;
    let last = kyoku_events.last().map(|e| {
        let t = e.get("type").and_then(Value::as_str).unwrap_or("");
        let actor = e.get("actor").and_then(Value::as_u64).map_or(usize::MAX, |a| a as usize);
        (t, actor)
    });
    let reacting_to = match last {
        Some((t, actor)) if (t == "dahai" || t == "kakan") && actor != player_id => Some(actor),
        _ => None,
    };
    let allowed = AllowedActions::from_legal(&legal, reacting_to.map_or(0, |a| ((a + 4 - player_id) % 4) as u8));
    let events: &[Value] = if reacting_to.is_some() {
        let cut = kyoku_events
            .iter()
            .rposition(|e| matches!(e.get("type").and_then(Value::as_str), Some("dahai" | "kakan")))
            .map_or(kyoku_events.len(), |i| i + 1);
        &kyoku_events[..cut]
    } else {
        kyoku_events
    };
    let mut view = BotView::build(obs, events, allowed)?;
    if allowed.discard {
        // riichienv forbids the kuikae discard; ban the tile types held in
        // hand that no legal action offers.
        let mut in_hand = [false; 34];
        for &tid in &obs.hands[player_id] {
            in_hand[(tid / 4) as usize] = true;
        }
        let mut discardable = [false; 34];
        for a in &legal {
            if a.action_type == ActionType::Discard
                && let Some(t) = a.tile
            {
                discardable[(t / 4) as usize] = true;
            }
        }
        let mut forbidden = [false; 34];
        for k in 0..34 {
            forbidden[k] = in_hand[k] && !discardable[k];
        }
        view.set_forbidden_discards(forbidden);
    }
    let dbg = debug.then(|| crate::rulebot::debug_info(&view, cfg));
    let action = crate::rulebot::react(&view, cfg)?;
    let engine = to_engine_action(&action, &legal, obs.drawn_tile).ok_or_else(|| {
        let acts: Vec<String> =
            legal.iter().map(|a| format!("{:?}/{:?}/{:?}", a.action_type, a.tile, a.consume_tiles)).collect();
        format!(
            "rulebot chose {action:?} but no legal action matches; seat {seat} \
             drawn={:?} legal=[{}] last3={:?}",
            obs.drawn_tile,
            acts.join(" "),
            kyoku_events.iter().rev().take(3).map(|e| e.to_string()).collect::<Vec<_>>(),
        )
    })?;
    Ok((engine, dbg))
}

/// The legal riichienv action realising a rulebot action.
///
/// Tile identity: the bot names (tile type, aka); the physical id is resolved
/// here -- tsumogiri prefers the drawn tile, tedashi avoids it (D17's
/// convention, in reverse). Calls match on the consumed pair's tile types and
/// reds; the called tile is fixed by the engine.
fn to_engine_action(
    action: &BotAction,
    legal: &[Action],
    drawn_tile: Option<u8>,
) -> Option<Action> {
    let type_and_aka = |id: u8| ((id / 4), is_aka_tid(id));
    match action {
        BotAction::Discard { tile: chosen, tsumogiri } => {
            let matches_tile = |a: &&Action| {
                a.action_type == ActionType::Discard
                    && a.tile.is_some_and(|t| type_and_aka(t) == (chosen.tile_type, chosen.is_aka))
            };
            let is_drawn_tile = |a: &&Action| a.tile == drawn_tile;
            let picks: Vec<&Action> = legal.iter().filter(matches_tile).collect();
            picks
                .iter()
                .find(|a| if *tsumogiri { is_drawn_tile(a) } else { !is_drawn_tile(a) })
                .or_else(|| picks.first())
                .map(|a| (*a).clone())
        }
        BotAction::Riichi => legal
            .iter()
            .find(|a| a.action_type == ActionType::Riichi)
            .cloned(),
        BotAction::Win => legal
            .iter()
            .find(|a| {
                a.action_type == ActionType::Tsumo || a.action_type == ActionType::Ron
            })
            .cloned(),
        BotAction::Pass => legal
            .iter()
            .find(|a| a.action_type == ActionType::Pass)
            .cloned(),
        // A kakan adds the red five when one is held, as majiang-ai does.
        BotAction::Ankan { tile_type } | BotAction::Kakan { tile_type } => {
            let kind = if matches!(action, BotAction::Ankan { .. }) { ActionType::Ankan } else { ActionType::Kakan };
            let picks: Vec<&Action> = legal
                .iter()
                .filter(|a| a.action_type == kind && a.tile.is_some_and(|t| t / 4 == *tile_type))
                .collect();
            picks
                .iter()
                .find(|a| a.tile.is_some_and(is_aka_tid))
                .or_else(|| picks.first())
                .map(|a| (*a).clone())
        }
        BotAction::Pon { consume_tiles, .. } | BotAction::Chi { consume_tiles, .. } => {
            let want_type = if matches!(action, BotAction::Pon { .. }) {
                ActionType::Pon
            } else {
                ActionType::Chi
            };
            let mut want = [
                (consume_tiles[0].tile_type, consume_tiles[0].is_aka),
                (consume_tiles[1].tile_type, consume_tiles[1].is_aka),
            ];
            want.sort_unstable();
            legal
                .iter()
                .find(|a| {
                    if a.action_type != want_type || a.consume_tiles.len() != 2 {
                        return false;
                    }
                    let mut got = [
                        type_and_aka(a.consume_tiles[0]),
                        type_and_aka(a.consume_tiles[1]),
                    ];
                    got.sort_unstable();
                    got == want
                })
                .cloned()
        }
    }
}
