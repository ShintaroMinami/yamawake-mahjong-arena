//! A fingerprint of the vendored engine's yaku evaluation (`yaku::calculate_yaku`, through
//! `HandEvaluator::verdict`, the DP's entry, and `HandEvaluator::calc`, which scores the games' wins), over
//! generated winning hands of every form: with calls and kans or closed, flushes, terminal and honor hands,
//! seven pairs and thirteen orphans, any winning tile, tsumo or ron, riichi, the other win conditions, both winds,
//! dora, ura and red fives. The expected digest was taken from the engine before its yaku evaluation was sped up
//! (2026-11-20), so a change there must give every result exactly as before (yaku ids and their order included).

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};

/// A deterministic xorshift draw in `0..n`.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }

    fn chance(&mut self, k: usize) -> bool {
        self.below(k) == 0
    }
}

/// The tile types a hand of `style` draws from: 0 any, 1 one suit and honors, 2 terminals and honors, 3 simples,
/// 4 honors and one suit's terminals, 5 one suit.
fn allowed(style: usize, suit: usize, t: usize) -> bool {
    let term = t >= 27 || t.is_multiple_of(9) || t % 9 == 8;
    match style {
        1 => t >= 27 || t / 9 == suit,
        2 => term,
        3 => !term,
        4 => t >= 27 || (t / 9 == suit && (t % 9).is_multiple_of(8)),
        5 => t < 27 && t / 9 == suit,
        _ => true,
    }
}

/// One generated winning hand: the concealed 14 tile ids (the winning tile last) and the melds.
pub(super) fn winning_hand(r: &mut Rng) -> Option<(Vec<u8>, Vec<Meld>)> {
    let mut count = [0u8; 34];
    let (style, suit) = (r.below(6), r.below(3));
    let form = r.below(10);
    // (tile types of each group, the group's kind as a meld: 0 chi, 1 pon, 2 daiminkan, 3 ankan, 4 kakan; None: concealed)
    let mut groups: Vec<(Vec<usize>, Option<u8>)> = Vec::new();
    let take = |count: &mut [u8; 34], ts: &[usize]| {
        let mut c = *count;
        for &t in ts {
            c[t] += 1;
            if c[t] > 4 {
                return false;
            }
        }
        *count = c;
        true
    };
    if form == 0 {
        // Seven pairs.
        while groups.len() < 7 {
            let t = r.below(34);
            if allowed(style, suit, t) && count[t] == 0 && take(&mut count, &[t, t]) {
                groups.push((vec![t, t], None));
            }
        }
    } else if form == 1 {
        // Thirteen orphans.
        let orphans = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        let mut ts: Vec<usize> = orphans.to_vec();
        ts.push(orphans[r.below(13)]);
        take(&mut count, &ts);
        groups.push((ts, None));
    } else {
        let n_melds = [0, 0, 0, 0, 1, 1, 2, 2, 3, 4][r.below(10)];
        let pair = (0..200).map(|_| r.below(34)).find(|&t| allowed(style, suit, t))?;
        take(&mut count, &[pair, pair]);
        groups.push((vec![pair, pair], None));
        let mut tries = 0;
        while groups.len() < 5 {
            tries += 1;
            if tries > 400 {
                return None;
            }
            let t = r.below(34);
            let meld = groups.len() <= n_melds;
            let run = t < 27 && t % 9 < 7 && r.chance(2) && (0..3).all(|k| allowed(style, suit, t + k));
            let kind = if run { 0 } else { [1, 1, 1, 2, 3, 4][r.below(6)] };
            let ts: Vec<usize> = if run {
                vec![t, t + 1, t + 2]
            } else if meld && kind >= 2 {
                vec![t; 4]
            } else {
                vec![t; 3]
            };
            if (run || allowed(style, suit, t)) && take(&mut count, &ts) {
                groups.push((ts, meld.then_some(kind)));
            }
        }
    }
    // Ids: each type's copies in a shuffled order (the red copy 4t of a five among them).
    let mut copies: Vec<Vec<u8>> = (0..34u8).map(|t| (0..4).map(|k| 4 * t + k).collect()).collect();
    for c in &mut copies {
        for i in (1..4).rev() {
            c.swap(i, r.below(i + 1));
        }
    }
    let mut id = |t: usize| copies[t].pop().expect("at most four copies");
    let mut melds = Vec::new();
    let mut concealed = Vec::new();
    for (ts, kind) in &groups {
        let ids: Vec<u8> = ts.iter().map(|&t| id(t)).collect();
        match kind {
            Some(k) => {
                let mt = [MeldType::Chi, MeldType::Pon, MeldType::Daiminkan, MeldType::Ankan, MeldType::Kakan][usize::from(*k)];
                let from = if mt == MeldType::Ankan { -1 } else { 1 + r.below(3) as i8 };
                let called = (mt != MeldType::Ankan).then(|| ids[0]);
                melds.push(Meld::new(mt, ids, mt != MeldType::Ankan, from, called));
            }
            None => concealed.extend(ids),
        }
    }
    let w = r.below(concealed.len());
    let win = concealed.remove(w);
    concealed.push(win);
    Some((concealed, melds))
}

/// FNV-1a over `bytes`, continuing from `h`.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The digest of every verdict and every full evaluation over the generated hands, and the number of wins and of
/// yakuman among the verdicts.
fn fingerprint(n_hands: usize) -> (u64, usize, usize) {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut h, mut wins, mut yakuman) = (0xcbf2_9ce4_8422_2325u64, 0, 0);
    let mut made = 0;
    while made < n_hands {
        let Some((tiles, melds)) = winning_hand(&mut r) else { continue };
        made += 1;
        let (win, hand13) = (tiles[tiles.len() - 1], &tiles[..tiles.len() - 1]);
        let mut counts = [0u8; 34];
        hand13.iter().for_each(|&t| counts[usize::from(t / 4)] += 1);
        let akas = hand13.iter().filter(|&&t| matches!(t, 16 | 52 | 88)).count() as u8;
        let dora: Vec<u8> = (0..r.below(4)).map(|_| r.below(136) as u8).collect();
        let ura: Vec<u8> = (0..dora.len()).map(|_| r.below(136) as u8).collect();
        let closed = melds.iter().all(|m| !m.opened);
        let (player_wind, round_wind) = (Wind::from(r.below(4) as u8), Wind::from(r.below(2) as u8));
        let verdict_entry = HandEvaluator::new(Vec::new(), melds.clone());
        let calc_entry = HandEvaluator::new(hand13.to_vec(), melds.clone());
        for (tsumo, riichi) in [(true, false), (false, false), (true, true), (false, true)] {
            let riichi = riichi && closed;
            let cond = Conditions { tsumo, riichi, player_wind, round_wind, ..Default::default() };
            let v = verdict_entry.verdict(&counts, akas, win, &dora, &cond);
            wins += usize::from(v.is_some());
            yakuman += usize::from(v.is_some_and(|x| x.2));
            h = fnv(h, format!("{v:?}").as_bytes());
            // The full evaluation, with the rarer conditions now and then.
            let cond = Conditions {
                double_riichi: riichi && r.chance(8),
                ippatsu: riichi && r.chance(6),
                haitei: tsumo && r.chance(10),
                houtei: !tsumo && r.chance(10),
                rinshan: tsumo && !closed && r.chance(6),
                chankan: !tsumo && r.chance(12),
                tsumo_first_turn: tsumo && closed && r.chance(20),
                ..cond
            };
            let ura = if riichi { ura.clone() } else { Vec::new() };
            let c = calc_entry.calc(win, dora.clone(), ura, Some(cond));
            h = fnv(h, format!("{c:?}").as_bytes());
        }
    }
    (h, wins, yakuman)
}

#[test]
fn yaku_results_are_unchanged() {
    let (digest, wins, yakuman) = fingerprint(30_000);
    assert!(wins > 40_000 && yakuman > 300, "wins {wins}, yakuman {yakuman}");
    assert_eq!(digest, 0xf0a6_aab6_8a56_f403, "digest {digest:#018x}, wins {wins}, yakuman {yakuman}");
}
