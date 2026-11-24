//! Forks of the lockstep loop (`LockstepLoop::fork`): a fork that plays the parent's own move replays the parent's
//! kyoku, sibling forks share every seat's draws under `split_wall`, simultaneous deciders are asked again, and
//! released indices are reused.

use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::rule::GameRule;
use serde_json::Value;

use super::{ActionSpec, Batch, Bot, LockstepLoop, tsumogiri};
use crate::rulebot::config::RulebotConfig;

/// The exact spec of an action.
fn spec(a: &Action) -> ActionSpec {
    let mut c = a.consume_tiles.clone();
    c.sort_unstable();
    (a.action_type as u8, a.tile, c)
}

/// A move that depends on the observation alone: win when possible, else the tsumogiri bot's move (no calls).
fn policy(b64: &str) -> Action {
    let obs = Observation::deserialize_from_base64(b64).unwrap();
    let legal = obs.legal_actions_method();
    legal
        .iter()
        .find(|a| matches!(a.action_type, ActionType::Tsumo | ActionType::Ron))
        .cloned()
        .unwrap_or_else(|| tsumogiri(&obs).unwrap())
}

fn legal_of(b64: &str) -> Vec<Action> {
    Observation::deserialize_from_base64(b64).unwrap().legal_actions_method()
}

fn new_loop(seeds: &[u64], split: bool, bots: Vec<(u8, Bot)>) -> LockstepLoop {
    let bot_seats = seeds.iter().map(|_| bots.clone()).collect();
    LockstepLoop::new(seeds, 2, bot_seats, Vec::new(), split, GameRule::default_tenhou()).unwrap()
}

/// Apply `policy` to every row of `batch`.
fn apply_policy(lp: &mut LockstepLoop, batch: &Batch) {
    let specs: Vec<ActionSpec> = batch.obs.iter().map(|o| spec(&policy(o))).collect();
    lp.apply(&specs).unwrap();
}

/// Play until every fork in `forks` has a result; returns them (events recorded).
fn finish(lp: &mut LockstepLoop, forks: &[usize]) -> Vec<Value> {
    loop {
        if let Some(r) = forks.iter().map(|&f| lp.fork_result(f, true).unwrap()).collect::<Option<Vec<_>>>() {
            return r;
        }
        let batch = lp.collect().unwrap().expect("the forks still play");
        apply_policy(lp, &batch);
    }
}

/// The game's mjai events from line `from` to the end of that kyoku (the hora / ryukyoku events included; the
/// end_kyoku and what follows left out).
fn kyoku_tail(lp: &LockstepLoop, game: usize, from: usize) -> Vec<Value> {
    lp.slots[game].state.mjai_log[from..]
        .iter()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .take_while(|e| !matches!(e["type"].as_str(), Some("end_kyoku" | "start_kyoku" | "end_game")))
        .collect()
}

/// A fork's events without the `rinshan` marks and the end_kyoku event (the parent's raw log has neither).
fn plain(events: &Value) -> Vec<Value> {
    events
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["type"] != "end_kyoku")
        .map(|e| {
            let mut e = e.clone();
            e.as_object_mut().unwrap().remove("rinshan");
            e
        })
        .collect()
}

/// Collect until a row satisfies `want` (row index, batch); applies the policy to the batches before it.
fn collect_until(lp: &mut LockstepLoop, mut want: impl FnMut(usize, &Batch, &LockstepLoop) -> bool) -> (Batch, usize) {
    loop {
        let batch = lp.collect().unwrap().expect("a matching decision before the games end");
        if let Some(r) = (0..batch.games.len()).find(|&r| want(r, &batch, lp)) {
            return (batch, r);
        }
        apply_policy(lp, &batch);
    }
}

#[test]
fn a_fork_of_the_parents_own_move_replays_its_kyoku() {
    // Seat 1 is a rulebot and seat 3 suzume (their decisions of the fork's tick carry over); 0 and 2 play outside.
    let a1 = crate::suzume::a1::A1Config::builtin("suzume").unwrap();
    let with_bots = vec![(1, Bot::Rulebot(RulebotConfig::default())), (3, Bot::A1(a1))];
    for (split, bots) in [(false, with_bots.clone()), (true, with_bots), (true, Vec::new())] {
        let mut lp = new_loop(&[5, 6], split, bots.clone());
        // A discard of game 1 some way into its first kyoku, after its first call or riichi if any.
        let mut n = 0;
        let (batch, r) = collect_until(&mut lp, |r, b, _| {
            n += 1;
            b.games[r] == 1 && n > 40
        });
        let (g, seat) = (batch.games[r], batch.seats[r]);
        let own = spec(&policy(&batch.obs[r]));
        let from = lp.slots[g].state.mjai_log.len();
        let forks = lp.fork(g, seat, &[own.clone(), own]).unwrap();
        assert_eq!(forks, [2, 3]);
        // A fork holds no log before its fork point.
        assert!(lp.slots[forks[0]].state.mjai_log.is_empty());
        apply_policy(&mut lp, &batch);
        let res = finish(&mut lp, &forks);
        assert_eq!(res[0], res[1], "split {split}: the same move twice");
        assert_eq!(plain(&res[0]["events"]), kyoku_tail(&lp, g, from), "split {split}: the parent's kyoku");
        // The parent's next kyoku starts from the fork's final scores.
        let next = lp.slots[g].state.mjai_log[from..]
            .iter()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .find(|e| e["type"] == "start_kyoku" || e["type"] == "end_game")
            .expect("the parent's kyoku ends");
        if next["type"] == "start_kyoku" {
            assert_eq!(next["scores"], res[0]["scores"], "split {split}");
        }
        assert_eq!((res[0]["parent"].clone(), res[0]["seat"].clone()), (Value::from(g), Value::from(seat)));
    }
}

/// Each seat's draws after the fork point (rinshan draws left out).
fn draws(res: &Value) -> [Vec<String>; 4] {
    let mut out: [Vec<String>; 4] = Default::default();
    for e in res["events"].as_array().unwrap() {
        if e["type"] == "tsumo" && e.get("rinshan").is_none() {
            out[e["actor"].as_u64().unwrap() as usize].push(e["pai"].as_str().unwrap().to_string());
        }
    }
    out
}

fn prefix_of_each_other(a: &[String], b: &[String]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}

#[test]
fn sibling_forks_share_each_seats_draws_under_split_walls() {
    // The first pon offered to a seat that plays outside: fork it as pon and as pass.
    let mut shifted = false;
    for split in [true, false] {
        let mut lp = new_loop(&[21, 22, 23], split, Vec::new());
        let (batch, r) =
            collect_until(&mut lp, |r, b, _| legal_of(&b.obs[r]).iter().any(|a| a.action_type == ActionType::Pon));
        let legal = legal_of(&batch.obs[r]);
        let pon = legal.iter().find(|a| a.action_type == ActionType::Pon).unwrap();
        let pass = legal.iter().find(|a| a.action_type == ActionType::Pass).unwrap();
        let forks = lp.fork(batch.games[r], batch.seats[r], &[spec(pon), spec(pass)]).unwrap();
        apply_policy(&mut lp, &batch);
        let res = finish(&mut lp, &forks);
        let (a, b) = (draws(&res[0]), draws(&res[1]));
        let same = (0..4).all(|s| prefix_of_each_other(&a[s], &b[s]));
        if split {
            assert!(same, "split walls: {a:?} vs {b:?}");
        } else {
            shifted = !same;
        }
    }
    assert!(shifted, "without split walls the pon should shift the draws (else this test shows nothing)");
}

#[test]
fn the_other_deciders_of_the_tick_are_asked_again() {
    let mut lp = new_loop(&[31, 32, 33, 34], false, Vec::new());
    // A tick where two seats of one game decide on the same discard.
    let (batch, r) = collect_until(&mut lp, |r, b, _| b.games.iter().filter(|&&g| g == b.games[r]).count() >= 2);
    let g = batch.games[r];
    let rows: Vec<usize> = (0..batch.games.len()).filter(|&i| batch.games[i] == g).collect();
    let (seat, other) = (batch.seats[rows[0]], batch.seats[rows[1]]);
    let fork = lp.fork(g, seat, &[spec(&policy(&batch.obs[rows[0]]))]).unwrap()[0];
    apply_policy(&mut lp, &batch);
    let next = lp.collect().unwrap().unwrap();
    let mine: Vec<usize> = (0..next.games.len()).filter(|&i| next.games[i] == fork).collect();
    assert_eq!(mine.iter().map(|&i| next.seats[i]).collect::<Vec<_>>(), [other], "only the other decider");
    assert_eq!(legal_of(&next.obs[mine[0]]), legal_of(&batch.obs[rows[1]]));
    apply_policy(&mut lp, &next);
}

#[test]
fn a_fork_of_a_fork_replays_it() {
    for bots in [Vec::new(), vec![(2, Bot::Rulebot(RulebotConfig::default()))]] {
        let mut lp = new_loop(&[51], true, bots);
        let mut n = 0;
        let (batch, r) = collect_until(&mut lp, |_, _, _| {
            n += 1;
            n > 20
        });
        let first = lp.fork(batch.games[r], batch.seats[r], &[spec(&policy(&batch.obs[r]))]).unwrap()[0];
        apply_policy(&mut lp, &batch);
        // Some ticks into the fork, fork it with its own move.
        let mut n = 0;
        let (batch, r) = collect_until(&mut lp, |r, b, _| {
            n += usize::from(b.games[r] == first);
            b.games[r] == first && n > 8
        });
        let second = lp.fork(first, batch.seats[r], &[spec(&policy(&batch.obs[r]))]).unwrap()[0];
        apply_policy(&mut lp, &batch);
        let res = finish(&mut lp, &[first, second]);
        assert_eq!(res[0]["scores"], res[1]["scores"]);
        assert_eq!(res[0]["end"], res[1]["end"]);
        let (a, b) = (res[0]["events"].as_array().unwrap(), res[1]["events"].as_array().unwrap());
        assert_eq!(a[a.len() - b.len()..], b[..], "the second fork's events end the first's");
        assert_eq!(res[1]["parent"], first);
    }
}

#[test]
fn released_forks_free_their_index_and_take_no_decisions() {
    let mut lp = new_loop(&[41], false, Vec::new());
    let batch = lp.collect().unwrap().unwrap();
    let (g, seat) = (batch.games[0], batch.seats[0]);
    let own = spec(&policy(&batch.obs[0]));
    assert!(lp.fork(g, seat + 1, std::slice::from_ref(&own)).is_err(), "no decision of that seat");
    let forks = lp.fork(g, seat, &[own.clone(), own.clone()]).unwrap();
    assert_eq!(forks, [1, 2]);
    assert!(lp.release(&[0]).is_err(), "a main game");
    assert!(lp.fork_result(0, false).is_err(), "not a fork");
    apply_policy(&mut lp, &batch);
    assert_eq!(lp.fork_result(1, false).unwrap(), None, "still playing");
    lp.release(&[1]).unwrap();
    assert!(lp.release(&[1]).is_err() && lp.fork_result(1, false).is_err());
    let batch = lp.collect().unwrap().unwrap();
    assert!(!batch.games.contains(&1), "a released fork takes no decisions");
    assert!(batch.games.contains(&2));
    let (r, g) = (0, batch.games[0]);
    assert!(lp.release(&[2]).is_err(), "a fork with a decision pending");
    let again = lp.fork(g, batch.seats[r], &[spec(&policy(&batch.obs[r]))]).unwrap();
    assert_eq!(again, [1], "the freed index serves the next fork");
    apply_policy(&mut lp, &batch);
    let res = finish(&mut lp, &[1, 2]);
    assert!(res.iter().all(|x| x["end"] == "hora" || x["end"] == "ryukyoku"));
}
