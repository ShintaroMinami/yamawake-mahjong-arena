//! Hand notation and hand-made views for the suzume tests.

use riichienv_core::action::{Action, ActionType};

use super::view::MjView;
use crate::eval::feat::tiles::{AKA_TIDS, aka_slot_of_tid};

/// "123m406p789s11z" -> 136-format ids (`0` is the red five; plain copies
/// take the ids from the top of their tile type down).
pub fn tids(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut used = [0u8; 34];
    let mut digits: Vec<u8> = Vec::new();
    for ch in s.chars() {
        if let Some(d) = ch.to_digit(10) {
            digits.push(d as u8);
            continue;
        }
        let base = match ch {
            'm' => 0,
            'p' => 9,
            's' => 18,
            'z' => 27,
            _ => panic!("bad suit {ch}"),
        };
        for d in digits.drain(..) {
            if d == 0 {
                out.push(AKA_TIDS[(base / 9) as usize]);
                continue;
            }
            let t = base + d - 1;
            out.push(t * 4 + 3 - used[t as usize]);
            used[t as usize] += 1;
        }
    }
    out
}

/// 34-format counts of a hand string.
pub fn counts(s: &str) -> [u8; 34] {
    let mut c = [0u8; 34];
    for t in tids(s) {
        c[(t / 4) as usize] += 1;
    }
    c
}

/// A view reacting to `called` (one tile, e.g. "5s" or "0s") discarded by
/// the seat `draws_before_us + 1` seats before us (0: kamicha, chi
/// allowed): `hand` holds 13 tiles; seen are the hand, the called tile and
/// the dora indicator 9m. Legal: pass, every pon pair and chi pair of
/// physical ids (as riichienv lists them, kuikae not checked), at 25000
/// points with 60 live tiles.
pub fn react_view(hand: &str, called: &str, draws_before_us: u8) -> MjView {
    let mut view = own_turn_view(&format!("{hand}{called}"));
    let ids = tids(hand);
    let mut c = tids(&format!("{hand}{called}"));
    let called_id = c.pop().unwrap();
    assert_eq!(c, ids);
    let t = called_id / 4;
    view.hand[t as usize] -= 1;
    if let Some(s) = aka_slot_of_tid(u32::from(called_id)) {
        view.akas_in_hand[s] = false;
    }
    view.drawn_tile = None;
    view.draws_before_us = draws_before_us;
    let mut legal = vec![Action::new(ActionType::Pass, None, vec![], Some(0))];
    let of = |tt: u8| ids.iter().copied().filter(move |&x| x / 4 == tt);
    let same: Vec<u8> = of(t).collect();
    for a in 0..same.len() {
        for b in a + 1..same.len() {
            legal.push(Action::new(ActionType::Pon, Some(called_id), vec![same[a], same[b]], Some(0)));
        }
    }
    if draws_before_us == 0 && t < 27 {
        let r = t % 9;
        let mut pairs: Vec<(u8, u8)> = Vec::new();
        if r >= 2 {
            pairs.push((t - 2, t - 1));
        }
        if (1..=7).contains(&r) {
            pairs.push((t - 1, t + 1));
        }
        if r <= 6 {
            pairs.push((t + 1, t + 2));
        }
        for (x, y) in pairs {
            for a in of(x) {
                for b in of(y) {
                    legal.push(Action::new(ActionType::Chi, Some(called_id), vec![a, b], Some(0)));
                }
            }
        }
    }
    view.legal = legal;
    view
}

/// An own-turn view: `hand` holds 14 tiles, the last one the drawn tile;
/// nothing else is visible but one dora indicator (9m: dora 1m). The legal
/// actions are what riichienv would offer a closed hand at 25000 points:
/// one discard per tile, riichi when a discard keeps tenpai, tsumo when
/// complete.
pub fn own_turn_view(hand: &str) -> MjView {
    let ids = tids(hand);
    assert_eq!(ids.len(), 14, "{hand}");
    let mut counts = [0u8; 34];
    let mut akas_in_hand = [false; 3];
    for &t in &ids {
        counts[(t / 4) as usize] += 1;
        if let Some(slot) = aka_slot_of_tid(u32::from(t)) {
            akas_in_hand[slot] = true;
        }
    }
    let mut seen_counts = counts;
    seen_counts[8] += 1;
    let mut dora_per_tile_type = [0u8; 34];
    dora_per_tile_type[0] = 1;
    let mut legal: Vec<Action> =
        ids.iter().map(|&t| Action::new(ActionType::Discard, Some(t), vec![], Some(0))).collect();
    let shanten = |h: &[u8; 34]| riichienv_core::shanten::calc_shanten_from_counts(h, 4);
    if shanten(&counts) == -1 {
        legal.push(Action::new(ActionType::Tsumo, None, vec![], Some(0)));
    }
    let tenpai_after_discard = (0..34).any(|t| {
        let mut h = counts;
        h[t] > 0 && {
            h[t] -= 1;
            shanten(&h) == 0
        }
    });
    if tenpai_after_discard {
        legal.push(Action::new(ActionType::Riichi, None, vec![], Some(0)));
    }
    MjView {
        seat: 0,
        hand: counts,
        akas_in_hand,
        seen_counts,
        akas_seen: akas_in_hand,
        dora_indicator_types: vec![8],
        dora_per_tile_type,
        round_wind: 27,
        player_wind: 28,
        score: 25000,
        kyoku_index: 0,
        scores: [25000; 4],
        honba: 0,
        sticks: 0,
        lead: 0,
        drawable_count: 60,
        drawn_tile: ids.last().copied(),
        in_riichi: false,
        own_melds: Vec::new(),
        draws_before_us: 0,
        double_riichi_possible: false,
        own_discards: [false; 34],
        n_own_discards: 0,
        last_ago: [crate::suzume::view::NEVER; 34],
        threats: Default::default(),
        legal,
    }
}
