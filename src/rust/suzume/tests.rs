//! suzume unit tests: options, the parts the universal features read
//! (ukeire, tile features, call options, the DP of open hands), legality,
//! the option switches, and whole games in the lockstep loop.

use std::collections::HashSet;

use riichienv_core::action::{Action, ActionType};

use super::ctx::Ctx;
use super::rules::calls::{CallKind, call_options};
use super::test_util::{counts, own_turn_view, react_view, tids};
use super::*;

/// The default options.
fn linear() -> SuzumeConfig {
    SuzumeConfig::default()
}

/// The default options with `pairs` applied.
fn cfg(pairs: &[(&str, &str)]) -> SuzumeConfig {
    SuzumeConfig::from_pairs(pairs.iter().copied()).unwrap()
}

#[test]
fn options_parse_and_reject() {
    let d = SuzumeConfig::default();
    assert!(d.kan && d.call && d.riichi && d.fold && d.fold_shanten == 1 && d.pf);
    let c = cfg(&[("kan", "0"), ("call", "false"), ("riichi", "0"), ("fold", "0"), ("fold_shanten", "3"), ("pf", "0")]);
    assert!(!c.kan && !c.call && !c.riichi && !c.fold && c.fold_shanten == 3 && !c.pf);
    for (k, v) in [("kan", "2"), ("call", "yes"), ("fold_shanten", "9"), ("fold_shanten", "-1"), ("ron_rate", "0.3"),
                   ("dp_max_shanten", "2"), ("universal_mode", "1"), ("disable", "universal_mlp"), ("vf_pairs", "1")] {
        assert!(SuzumeConfig::from_pairs([(k, v)]).is_err(), "{k}={v}");
    }
}

#[test]
fn registry_is_the_mlp_bot() {
    let names: Vec<&str> = registry::RULES.iter().map(|r| r.info().name).collect();
    assert_eq!(names, ["win_always", "kyushu_always", "pf_discard", "fold_discard", "kan_mlp", "universal_mlp", "discard_forced"]);
}

/// (tile type, aka) of a chosen discard.
fn discarded(a: &Action) -> (u8, bool) {
    assert_eq!(a.action_type, ActionType::Discard, "{a:?}");
    let t = a.tile.unwrap();
    (t / 4, crate::eval::feat::tiles::is_aka_tid(t))
}
fn tile(s: &str) -> u8 {
    tids(s)[0] / 4
}
#[test]
fn call_options_map_red_fives() {
    // Plain and red 5p in hand, 4p from the left: chi 23p (high) has one
    // option; chi 35p (mid) and 56p (low) each come plain and red.
    let view = react_view("2305p56p123m789s1z", "4p", 0);
    let opts: Vec<(CallKind, u8, bool)> = call_options(&view).iter().map(|o| (o.kind, o.chi_pos, o.red)).collect();
    assert_eq!(
        opts,
        vec![
            (CallKind::Pass, 0, false),
            (CallKind::Chi, 0, false),
            (CallKind::Chi, 0, true),
            (CallKind::Chi, 1, false),
            (CallKind::Chi, 1, true),
            (CallKind::Chi, 2, false),
        ]
    );
}

#[test]
fn call_kuikae_is_respected() {
    use rules::calls::{kuikae_forbidden, view_after_call};
    let (s3, s4, s5, s2, s6) = (tile("3s"), tile("4s"), tile("5s"), tile("2s"), tile("6s"));
    // 5s called with 34s: 5s and the suji 2s may not go.
    assert_eq!(kuikae_forbidden(s5, &[s3 * 4, s4 * 4], true), 1 << s5 | 1 << s2);
    // With 46s (mid): only 5s. Pon: only the called type.
    assert_eq!(kuikae_forbidden(s5, &[s4 * 4, s6 * 4], true), 1 << s5);
    assert_eq!(kuikae_forbidden(s5, &[s5 * 4, s5 * 4 + 1], false), 1 << s5);
    // 1s called with 23s: the suji 4s; 9s called with 78s: the suji 6s.
    assert_eq!(kuikae_forbidden(tile("1s"), &[tile("2s") * 4, s3 * 4], true), 1 << tile("1s") | 1 << s4);
    assert_eq!(kuikae_forbidden(tile("9s"), &[tile("7s") * 4, tile("8s") * 4], true), 1 << tile("9s") | 1 << s6);
    // 7s called with 89s: no suji (it would be 10s).
    assert_eq!(kuikae_forbidden(tile("7s"), &[tile("8s") * 4, tile("9s") * 4], true), 1 << tile("7s"));

    // After chi 345s the hand holds 2s and 5s: neither may go.
    let view = react_view("234m567p234s5s88s9m", "5s", 0);
    let chi = call_options(&view)
        .into_iter()
        .find(|o| o.kind == CallKind::Chi && o.chi_pos == 2)
        .unwrap()
        .action;
    let after = view_after_call(&view, &chi);
    assert_eq!(after.own_melds.len(), 1);
    assert!(after.is_open());
    let legal: HashSet<u8> = after.legal.iter().map(|a| a.tile.unwrap() / 4).collect();
    assert!(!legal.contains(&s5) && !legal.contains(&s2) && legal.contains(&tile("9m")));
    // The discard MLP plays the open hand and stays legal.
    let (a, rule) = decide(&after, &linear()).unwrap();
    assert!(after.legal.contains(&a));
    assert!(![s5, s2].contains(&discarded(&a).0));
    assert_eq!(decider_name(rule), "universal_mlp");
}

#[test]
fn open_hand_dp_values_yaku_only() {
    use rules::calls::view_after_call;
    // Open tanyao tenpai after a discard: the DP values it; the yakuless one is 0.
    let tanyao = react_view("234m567p34s6688s9m", "5s", 0);
    let chi = call_options(&tanyao).into_iter().find(|o| o.chi_pos == 2 && o.kind == CallKind::Chi).unwrap();
    let after = view_after_call(&tanyao, &chi.action);
    let config = linear();
    let ctx = Ctx::new(&after, &config);
    let best = ctx.dp(false).unwrap().iter().map(|r| r.ev).fold(0.0, f64::max);
    assert!(best > 100.0, "{best}");
    let plain = react_view("123m456p78s11s99m1z", "9s", 0);
    let chi = call_options(&plain).into_iter().nth(1).unwrap();
    let after = view_after_call(&plain, &chi.action);
    let ctx = Ctx::new(&after, &config);
    assert!(ctx.dp(true).unwrap().iter().all(|r| r.ev == 0.0 && r.win == 0.0));
}

#[test]
fn wins_and_passes() {
    let view = own_turn_view("123m456p789s11p678s");
    let (a, rule) = decide(&view, &linear()).unwrap();
    assert_eq!(a.action_type, ActionType::Tsumo);
    assert_eq!(decider_name(rule), "win_always");

    // A 4m to pon: call=0 passes (the call MLP decides, its call options
    // removed).
    let react = react_view("2344m567p789s115z", "4m", 0);
    assert!(react.legal.iter().any(|a| a.action_type == ActionType::Pon));
    let (a, rule) = decide(&react, &cfg(&[("call", "0")])).unwrap();
    assert_eq!(a.action_type, ActionType::Pass);
    assert_eq!(decider_name(rule), "universal_mlp");
    // Daiminkan and pass alone are no call decision: with kan=0 the fallback passes.
    let mut kan_only = react.clone();
    kan_only.legal.retain(|a| a.action_type == ActionType::Pass);
    kan_only.legal.push(Action::new(ActionType::Daiminkan, Some(12), vec![13, 14, 15], Some(0)));
    let (a, rule) = decide(&kan_only, &cfg(&[("kan", "0")])).unwrap();
    assert_eq!((a.action_type, rule), (ActionType::Pass, FALLBACK));
}

/// Random hands with random legal subsets (as riichi or kuikae restrict
/// them): every decision is one of the legal actions.
#[test]
fn decisions_are_always_legal() {
    let mut x = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut wall: Vec<u8> = (0..136).collect();
    for round in 0..400 {
        for i in (1..wall.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            wall.swap(i, j);
        }
        let ids = &wall[..14];
        let mut view = own_turn_view("123m456p789s12345z");
        view.hand = [0; 34];
        view.akas_in_hand = [false; 3];
        for &t in ids {
            view.hand[(t / 4) as usize] += 1;
            if let Some(s) = crate::eval::feat::tiles::aka_slot_of_tid(u32::from(t)) {
                view.akas_in_hand[s] = true;
            }
        }
        view.seen_counts = view.hand;
        view.akas_seen = view.akas_in_hand;
        view.drawn_tile = Some(ids[13]);
        view.drawable_count = (next() % 70) as u8;
        let keep = next();
        view.legal = ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| keep >> i & 1 == 1 || i == 13)
            .map(|(_, &t)| Action::new(ActionType::Discard, Some(t), vec![], Some(0)))
            .collect();
        if round % 3 == 0 {
            view.legal.push(Action::new(ActionType::Riichi, None, vec![], Some(0)));
        }
        let config = if round % 2 == 0 { linear() } else { cfg(&[("kan", "0"), ("call", "0")]) };
        let (a, _) = decide(&view, &config).unwrap();
        assert!(view.legal.contains(&a));
    }
}

/// Whole east-only games of four suzume bots run to the end inside the
/// lockstep loop (an illegal action would stop it with an error), and the
/// switches hold: a bot with `call=0` never chis or pons, `riichi=0` never
/// declares, `kan=0` never kans; the default bot's `universal_mlp` and
/// `pf_discard` both decide.
#[test]
fn full_games_complete_and_switches_hold() {
    let seeds: Vec<u64> = (0..8).collect();
    let configs = [
        SuzumeConfig::default(),
        cfg(&[("call", "0")]),
        cfg(&[("riichi", "0")]),
        cfg(&[("kan", "0")]),
    ];
    let seats: Vec<Vec<(u8, crate::arena::Bot)>> = (0..seeds.len())
        .map(|_| (0..4).map(|s| (s, crate::arena::Bot::Suzume(configs[usize::from(s)]))).collect())
        .collect();
    let mut lp = crate::arena::LockstepLoop::new(&seeds, 1, seats, Vec::new(), false).unwrap();
    assert!(lp.collect().unwrap().is_none(), "every seat is in-loop");
    let mut seen: HashSet<(&str, u64)> = HashSet::new();
    for slot in &lp.slots {
        assert!(slot.state.is_done);
        assert!(slot.suzume_trace.iter().any(|&(_, _, r)| decider_name(r) == "universal_mlp"));
        for line in &slot.state.mjai_log {
            let e: serde_json::Value = serde_json::from_str(line).unwrap();
            let (Some(t), Some(a)) = (e["type"].as_str(), e["actor"].as_u64()) else { continue };
            let t = match t {
                "chi" | "pon" => "call",
                "reach" => "riichi",
                "ankan" | "kakan" | "daiminkan" => "kan",
                _ => continue,
            };
            seen.insert((t, a));
        }
    }
    assert!(!seen.contains(&("call", 1)) && !seen.contains(&("riichi", 2)) && !seen.contains(&("kan", 3)), "{seen:?}");
    assert!(seen.contains(&("call", 0)) && seen.contains(&("riichi", 0)), "the default bot calls and declares: {seen:?}");
    let pf = crate::suzume::registry::index_of("pf_discard").unwrap() as u8;
    assert!(lp.slots.iter().flat_map(|s| &s.suzume_trace).any(|&(_, _, r)| r == pf), "the default pf=1 decides some discards");
}

#[test]
fn counts_notation() {
    assert_eq!(counts("055m")[4], 3);
    assert_eq!(tids("0p"), vec![52]);
}
