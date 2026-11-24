//! The Rust entry (`yamawake::external`), used as another crate uses it: it gives the Python player's recorded
//! answers (`tests/external/choose_cases.json`, from `choose_cases.py`), its riichi tile is the discard a seat plays
//! after the declaration, and one bot serves many threads.

use std::collections::HashMap;

use yamawake::external::{Choice, Suzume, SuzumeSeat, observation_from_base64};
use yamawake::riichienv_core::action::{Action, ActionType, Phase};
use yamawake::riichienv_core::rule::GameRule;
use yamawake::riichienv_core::state::GameState;
use yamawake::serde_json::{self, Value, json};

fn cases() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/external/choose_cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("choose_cases.json")).expect("cases JSON")
}

/// `[action_type, tile, sorted consume_tiles]`, as the Python recorder writes an action.
fn act_of(a: &Action) -> Value {
    let mut c = a.consume_tiles.clone();
    c.sort_unstable();
    json!([a.action_type as u8, a.tile, c])
}

/// Equal JSON, the scores to 1e-9 (the models' sums may round differently on another platform).
fn same_choice(got: &Value, want: &Value, i: usize) {
    let strip = |v: &Value| {
        let mut v = v.clone();
        v.as_object_mut().unwrap().remove("scores");
        v
    };
    assert_eq!(strip(got), strip(want), "case {i}");
    match (got["scores"].as_array(), want["scores"].as_array()) {
        (None, None) => {}
        (Some(g), Some(w)) => {
            assert_eq!(g.len(), w.len(), "case {i}: scores");
            for (x, y) in g.iter().zip(w) {
                match (x.as_f64(), y.as_f64()) {
                    (None, None) => {}
                    (Some(x), Some(y)) => assert!((x - y).abs() < 1e-9, "case {i}: score {x} vs {y}"),
                    _ => panic!("case {i}: scores {g:?} vs {w:?}"),
                }
            }
        }
        _ => panic!("case {i}: scores {} vs {}", got["scores"], want["scores"]),
    }
}

/// The typed fields agree with each other and with the JSON.
fn consistent(c: &Choice, i: usize) {
    let k = c.choice_index.unwrap_or_else(|| panic!("case {i}: no choice index"));
    let m = &c.legal[k];
    assert_eq!(m.mjai, c.choice, "case {i}");
    assert_eq!(m.riichi, c.riichi_tile.is_some(), "case {i}");
    match c.riichi_tile {
        Some(t) => {
            assert_eq!(c.action.action_type, ActionType::Riichi, "case {i}");
            assert_eq!((m.action.action_type, m.action.tile), (ActionType::Discard, Some(t)), "case {i}");
        }
        None => assert_eq!(m.action, c.action, "case {i}"),
    }
    if c.basis == "model" {
        let s = c.scores.as_ref().unwrap_or_else(|| panic!("case {i}: a model's choice without scores"));
        let top = s.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(s[k], Some(top), "case {i}: the choice is not the best score");
    }
}

/// Each case: (spec, observation base64, its events, the recorded case).
fn each_case(data: &Value) -> Vec<(&str, &str, &[Value], &Value)> {
    let kyokus = data["kyokus"].as_array().unwrap();
    data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let events = kyokus[c["kyoku"].as_u64().unwrap() as usize].as_array().unwrap();
            (c["spec"].as_str().unwrap(), c["obs"].as_str().unwrap(), &events[..c["n_events"].as_u64().unwrap() as usize], c)
        })
        .collect()
}

#[test]
fn the_rust_entry_gives_the_python_players_answers() {
    let data = cases();
    let cases = each_case(&data);
    assert!(cases.len() >= 40);
    for (i, &(spec, obs, events, case)) in cases.iter().enumerate() {
        let bot = Suzume::from_spec(spec).unwrap();
        let obs = observation_from_base64(obs, events).unwrap();
        let c = bot.choose(&obs, events).unwrap();
        same_choice(&c.to_json(), &case["choose"], i);
        consistent(&c, i);
        assert_eq!(act_of(&c.action), case["act"], "case {i}: the choice's action");
        assert_eq!(act_of(&bot.decide(&obs, events).unwrap()), case["act"], "case {i}: decide");
        assert_eq!(json!(c.riichi_tile), case["riichi_tile"], "case {i}: the riichi tile");
        assert_eq!(SuzumeSeat::new(bot).choose(&obs, events).unwrap(), c, "case {i}: a new seat");
    }
}

#[test]
fn one_bot_serves_many_threads() {
    fn shared<T: Send + Sync + Copy>() {}
    fn sent<T: Send>() {}
    shared::<Suzume>();
    sent::<SuzumeSeat>();
    sent::<Choice>();
    let data = cases();
    let cases = each_case(&data);
    let bot = Suzume::from_spec("suzume").unwrap();
    let mine: Vec<_> = cases.iter().filter(|c| c.0 == "suzume").collect();
    let alone: Vec<Choice> =
        mine.iter().map(|&&(_, obs, ev, _)| bot.choose(&observation_from_base64(obs, ev).unwrap(), ev).unwrap()).collect();
    let together: Vec<Choice> = std::thread::scope(|s| {
        let handles: Vec<_> = mine
            .iter()
            .map(|&&(_, obs, ev, _)| s.spawn(move || bot.choose(&observation_from_base64(obs, ev).unwrap(), ev).unwrap()))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(alone, together);
}

/// The kyoku's events from the engine's log (from its last `start_kyoku`).
fn kyoku_events(st: &GameState) -> Vec<Value> {
    let start = st.mjai_log.iter().rposition(|l| l.contains("\"start_kyoku\"")).unwrap_or(0);
    st.mjai_log[start..].iter().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// East-only games played by fresh choices (a riichi: the declaration, then `riichi_tile`), each seat also kept as a
/// `SuzumeSeat`: the seat decides as the fresh choice does, and after its riichi discards `riichi_tile`.
#[test]
fn a_riichi_tile_is_the_discard_after_the_declaration() {
    let bot = Suzume::from_spec("suzume").unwrap();
    let (mut riichi, mut decisions) = (0, 0);
    for seed in 0..3 {
        let mut st = GameState::new(1, false, Some(seed), 0, GameRule::default_tenhou());
        let mut seats: Vec<SuzumeSeat> = (0..4).map(|_| SuzumeSeat::new(bot)).collect();
        let mut declared: [Option<u8>; 4] = [None; 4];
        while !st.is_done {
            let actors = match st.phase {
                Phase::WaitAct => vec![st.current_player],
                Phase::WaitResponse => st.active_players.clone(),
            };
            let events = kyoku_events(&st);
            let mut acts = HashMap::new();
            for p in actors {
                let obs = st.get_observation(p);
                let seat = &mut seats[usize::from(p)];
                let a = match declared[usize::from(p)].take() {
                    Some(t) => {
                        let a = seat.decide(&obs, &events).unwrap();
                        assert_eq!((a.action_type, a.tile), (ActionType::Discard, Some(t)), "seed {seed}");
                        a
                    }
                    None => {
                        let c = bot.choose(&obs, &events).unwrap();
                        assert_eq!(seat.choose(&obs, &events).unwrap(), c, "seed {seed}");
                        if c.riichi_tile.is_some() {
                            riichi += 1;
                        }
                        declared[usize::from(p)] = c.riichi_tile;
                        c.action
                    }
                };
                decisions += 1;
                acts.insert(p, a);
            }
            st.step(&acts);
            assert!(st.last_error.is_none(), "seed {seed}: {:?}", st.last_error);
        }
    }
    assert!(riichi >= 3, "{riichi} riichi in {decisions} decisions");
}

#[test]
fn specs_take_the_python_players_options() {
    assert_eq!(Suzume::builtin_sets().collect::<Vec<_>>(), ["suzume", "suzume-rla225"]);
    assert!(Suzume::from_spec("suzume:aggression=0.5:call_shift=0.3:riichi_shift=-0.2:win=0:call=1").is_ok());
    assert!(Suzume::new("suzume", [("aggression", "-1"), ("call", "false")]).is_ok());
    for (spec, err) in [
        ("suzume:aggression=2", "not in [-1, 1]"),
        ("suzume:aggression=x", "not a number"),
        ("suzume:call=2", "not 0 or 1"),
        ("suzume:kan=0", "takes only"),
        ("suzume:aggression", "not KEY=VALUE"),
        ("suzume-rla225:aggression=0.5", "without push / fold"),
    ] {
        let e = Suzume::from_spec(spec).unwrap_err();
        assert!(e.contains(err), "{spec}: {e}");
    }
}
