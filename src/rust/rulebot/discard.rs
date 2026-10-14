//! Choosing which tile to let go of, at each of the three rungs of push/fold.
//!
//! All three rungs answer the same question -- score every tile the hand can
//! spare, take the best -- and differ only in what "best" means. That scan is
//! written once in `scan_discards`; each rung supplies a sort key. Keeping them
//! as one loop is not only shorter: the scan is the bot's hottest code, and a
//! change to how a probe hand is built now lands in one place.

use super::agari::{ScoringHand, agari_points};
use super::decision::Decision;
use super::hand_shanten;
use super::tile::Tile;
use super::config::{BUILD_EV_SHANTEN, EV_PER_DANGER, FOLD_SHANTEN, RIICHI_DAMA_PT, TENPAI_EV_PER_DANGER};
use super::{AKA_TILE_TYPES, aka_slot};

use std::cmp::Reverse;

/// Score every discard the hand can make and return the best-scoring tile type.
///
/// `score` receives the tile and the hand with that tile already removed, and
/// returns the key to minimise on -- or `None` to rule the discard out. Ties go
/// to the lower tile index, which is what scanning in order already gives.
fn scan_discards<K, F>(
    hand: &[u8; 34],
    forbidden: &[bool; 34],
    mut score: F,
) -> Option<usize>
where
    K: Ord,
    F: FnMut(usize, &[u8; 34]) -> Option<K>,
{
    let mut probe = *hand;
    let mut best: Option<(usize, K)> = None;
    for tile_type in 0..34 {
        if hand[tile_type] == 0 || forbidden[tile_type] {
            continue;
        }
        probe[tile_type] -= 1;
        let key = score(tile_type, &probe);
        probe[tile_type] += 1;
        if let Some(key) = key
            && best.as_ref().is_none_or(|(_, b)| key < *b)
        {
            best = Some((tile_type, key));
        }
    }
    best.map(|(tile_type, _)| tile_type)
}

/// Tiles that would lower the shanten of `hand`, counted by how many of each
/// tile type could still be drawn.
///
/// `seen` counts every tile already visible anywhere -- own hand, ponds, melds,
/// dora indicators -- so `4 - seen` is what is genuinely left in the wall and
/// in opponents' hands. Counting `4 - own hand` instead, as the first cut of
/// this did, values a shape whose tiles are all in the pond exactly as highly
/// as a live one.
fn ukeire(hand: &[u8; 34], seen: &[u8; 34], n_sets: u8, shanten_now: i8) -> u32 {
    let mut total = 0;
    let mut probe = *hand;
    for t in 0..34 {
        if seen[t] >= 4 {
            continue;
        }
        probe[t] += 1;
        if hand_shanten::shanten_of(&probe, n_sets) < shanten_now {
            total += u32::from(4 - seen[t]);
        }
        probe[t] -= 1;
    }
    total
}

/// How many ways the hand can take `tile_type`, relative to having to draw it
/// (`kobalab_ev`'s far-from-tenpai ukeire).
///
/// A tile the hand can pon comes from any of the three opponents as well as
/// from the wall; one it can chi comes from the player to the left only. Plain
/// ukeire prices all three the same, which is wrong for a bot that calls: a
/// pair waiting on its third copy is far likelier to get it than a lone tile.
/// The weights are kobalab's.
pub(super) fn call_ways(hand: &[u8; 34], tile_type: usize) -> u32 {
    if hand[tile_type] >= 2 {
        return 4; // pon, from any seat
    }
    if tile_type >= 27 {
        return 1; // an honor that is not already a pair can only be drawn
    }
    let n = (tile_type % 9) as i32;
    let base = tile_type - n as usize;
    let has = |offset: i32| -> bool {
        let i = n + offset;
        (0..9).contains(&i) && hand[base + i as usize] > 0
    };
    let chi = has(-2) && has(-1) || has(-1) && has(1) || has(1) && has(2);
    if chi { 2 } else { 1 }
}

/// Whether a tile contributes to no block the hand could still build.
///
/// A second copy is a pair, which is a block already, so only a lone tile can
/// be isolated. For numbers the reach is two either way: a neighbour makes a
/// run, and one tile further makes a kanchan, both of which are blocks worth
/// keeping. Honors have no such reach, which is why a lone one is isolated
/// from the moment it is drawn.
fn is_isolated(hand: &[u8; 34], tile_type: usize) -> bool {
    if hand[tile_type] > 1 {
        return false;
    }
    if tile_type >= 27 {
        return true;
    }
    let n = (tile_type % 9) as i32;
    let base = tile_type - n as usize;
    [-2, -1, 1, 2].iter().all(|&offset| {
        let i = n + offset;
        !(0..9).contains(&i) || hand[base + i as usize] == 0
    })
}

/// How readily the hand can spare a tile, lowest going first.
///
/// Only isolated tiles are ranked; anything holding a block together sorts
/// last and is never reached by this tie break. Among the isolated ones the
/// order is by how many blocks the tile could still join: an honor only a
/// pair or a triplet, a terminal one run, and a 4, 5 or 6 four of them.
fn isolation_rank(hand: &[u8; 34], tile_type: usize) -> u8 {
    if !is_isolated(hand, tile_type) {
        return u8::MAX;
    }
    if tile_type >= 27 {
        return 0;
    }
    let n = tile_type % 9;
    match n.min(8 - n) {
        0 => 1, // 1, 9
        1 => 2, // 2, 8
        2 => 3, // 3, 7
        _ => 4, // 4, 5, 6
    }
}
impl Decision<'_> {
    /// Which of the 34 tile types to discard, every rung included.
    ///
    /// Falls back to the attacking discard whenever a rung declines to name a
    /// tile: every rung but that one can rule out its whole candidate set, and
    /// a hand must always discard something.
    pub(super) fn choose_discard(&self) -> Option<usize> {
        match self.threat_danger() {
            Some(danger) => self.threatened_discard(&danger),
            None => self.best_discard(),
        }
        .or_else(|| self.best_discard())
    }

    /// The discard while someone is threatening: fold, or push or fold.
    ///
    /// Two rungs. From two shanten out the hand folds on the shanten number
    /// alone; at one shanten and tenpai, where the decision actually lives,
    /// value and danger are weighed on one axis (`push_or_fold`). Without a
    /// DP value to weigh, a one-shanten hand holds its shape safely and a
    /// tenpai hand plays on.
    ///
    /// The hand holds 3n+2 tiles here, so `shanten` reads it at its best
    /// discard; a 3n+1 reading would be one too high whenever a
    /// shanten-advancing discard exists, and fold a tenpai hand.
    fn threatened_discard(&self, danger: &[u32; 34]) -> Option<usize> {
        let shanten = self.state.shanten();
        if shanten >= FOLD_SHANTEN {
            // Far enough out to fold on the shanten number alone, which is both
            // the accepted way to play it and the one place expected value is
            // most expensive to get: the single-player DP runs deepest at high
            // shanten early in the hand, exactly where the answer was never in
            // doubt. So never ask.
            return self.safest_discard(danger);
        }
        // Tenpai and one shanten are the same decision on different scales,
        // so they share the rule and differ only in the coefficient.
        let per_danger = if shanten == 0 { TENPAI_EV_PER_DANGER } else { EV_PER_DANGER };
        self.push_or_fold(danger, per_danger).or_else(|| {
            if shanten > 0 {
                self.cautious_discard(danger)
            } else {
                self.best_discard()
            }
        })
    }

    /// The one-shanten decision: give the hand up, or push with the tile that
    /// pays the least value for the danger it avoids.
    ///
    /// There is no threshold. Giving the hand up is a candidate like any other:
    /// the safest tile scored against what the hand is still worth once it is
    /// gone. A hand worth nothing loses that comparison on its own, without
    /// anyone having to say where nothing begins.
    ///
    /// The candidates are the DP table's discards as they stand, so the
    /// candidate set costs nothing to build. `per_danger` at 0 leaves pure
    /// value; large enough, pure danger.
    fn push_or_fold(&self, danger: &[u32; 34], per_danger: f32) -> Option<usize> {
        let (_, table) = self.hand_ev()?;
        let score = |tile_type: usize, value: f32| per_danger.mul_add(-(danger[tile_type] as f32), value);

        // Giving the hand up is a candidate like any other: the safest tile the
        // hand holds, at whatever a hand nobody is playing on is worth. The
        // table cannot supply it -- it lists only discards that keep the hand
        // advancing -- so it is scored here at zero.
        let folding = self.safest_discard(danger).map(|tile_type| (tile_type, score(tile_type, 0.)));
        let forbidden = self.state.forbidden_discards();
        table
            .iter()
            .filter_map(|c| {
                let tile_type = c.tile.plain().tile_type_index();
                if forbidden[tile_type] {
                    return None;
                }
                let ev = *c.ev.first()?;
                Some((tile_type, score(tile_type, ev)))
            })
            .chain(folding)
            // The first strict maximum, not the last: the table arrives sorted
            // by value, and folding is chained behind it, so a tie goes to the
            // better-placed push. That is what makes `per_danger` at 0 exactly
            // the plain highest-value discard -- the identity a sweep can check
            // against, and the only way turning the knob up from there is a
            // change of one thing at a time.
            .fold(None, |best: Option<(usize, f32)>, cur| match best {
                Some(b) if b.1 >= cur.1 => Some(b),
                _ => Some(cur),
            })
            .map(|(tile_type, _)| tile_type)
    }

    /// Attacking discard: get closer first, then keep the wider wait, and only
    /// then keep the value.
    ///
    /// Dora never overrules ukeire, so the hand cannot be made slower to hold a
    /// red five -- it just stops giving one away for nothing.
    pub(super) fn best_discard(&self) -> Option<usize> {
        if self.state.shanten() <= BUILD_EV_SHANTEN
            && let Some(tile_type) = self.ev_best_discard()
        {
            return Some(tile_type);
        }
        let n_sets = self.state.n_sets();
        let seen = self.state.seen_counts();
        let dora = self.dora_value();
        let hand = self.state.hand();
        let chi_runs = self.state.chi_runs();
        let pon_sets = self.state.pon_sets();

        scan_discards(&hand, self.state.forbidden_discards(), |tile_type, probe| {
            let shanten = hand_shanten::shanten_of(probe, n_sets);
            // Width is still counted against the plain shanten -- what advances
            // the hand at all -- while the ranking asks what advances it towards
            // something it can win with.
            let width = ukeire(probe, &seen, n_sets, shanten);
            let rank = self.yaku_shanten(probe, chi_runs, pon_sets, n_sets);
            // The spare-tile order is judged on the hand as it stands, not on
            // the probe: the probe has already given the tile up, so a pair
            // would read there as a lone tile and every pair would look isolated.
            Some((rank, Reverse(width), dora[tile_type], isolation_rank(&hand, tile_type)))
        })
    }

    /// Whether to declare riichi, given the hand can.
    ///
    /// A hand that already pays well enough on a silent ron stays dama: riichi
    /// would add little and cost the freedom to fold. The wait is the one the
    /// discard the hand would make anyway leaves, since that is the shape
    /// riichi would lock in.
    pub(super) fn should_riichi(&self) -> bool {
        let Some(discard) = self.best_discard() else {
            return true;
        };
        let n_sets = self.state.n_sets();
        let seen = self.state.seen_counts();
        let mut probe = self.state.hand();
        probe[discard] -= 1;

        let mut waits = Vec::with_capacity(4);
        for tile_type in 0..34 {
            if seen[tile_type] >= 4 {
                continue;
            }
            probe[tile_type] += 1;
            if hand_shanten::shanten_of(&probe, n_sets) == -1 {
                waits.push(tile_type);
            }
            probe[tile_type] -= 1;
        }
        self.dama_value(&probe, &waits) < RIICHI_DAMA_PT
    }

    /// Dora in the own ankans, which the concealed counts no longer hold: four
    /// copies each, and a five's kan holds its red five.
    fn closed_kan_dora(&self) -> u8 {
        let dora = self.state.dora_per_tile_type();
        self.state
            .closed_kans()
            .iter()
            .map(|&k| 4 * dora[k as usize] + u8::from(super::aka_slot(k as usize).is_some()))
            .sum()
    }

    /// What the hand pays on a ron with nothing announced, at its worst wait.
    ///
    /// Zero when any wait has no yaku, since a hand that cannot win silently on
    /// one of its tiles is not a hand that can afford to stay quiet.
    fn dama_value(&self, probe: &[u8; 34], waits: &[usize]) -> u32 {
        let dora = self.state.dora_per_tile_type();
        let dora_count: u8 = (0..34)
            .map(|t| probe[t] * dora[t])
            .sum::<u8>()
            + self.state.akas_in_hand().iter().filter(|&&held| held).count() as u8
            + self.closed_kan_dora();
        let mut hand = *probe;
        let mut worst = u32::MAX;
        for &tile_type in waits {
            hand[tile_type] += 1;
            let inp = ScoringHand {
                hand: &hand,
                is_menzen: true,
                chi_runs: &[],
                pon_sets: &[],
                open_kans: &[],
                closed_kans: self.state.closed_kans(),
                round_wind: self.state.round_wind(),
                player_wind: self.state.player_wind(),
                win_tile_type: tile_type as u8,
                is_ron: true,
            };
            let value = agari_points(&inp, false, dora_count).map_or(0, |v| v as u32);
            hand[tile_type] -= 1;
            worst = worst.min(value);
        }
        if worst == u32::MAX { 0 } else { worst }
    }

    /// The discard the single-player table values most.
    ///
    /// `max_ev_table` arrives sorted by expected value, so the head is the
    /// answer -- the table has already weighed speed, width, the wait it leaves
    /// and what the hand scores, which is every term `best_discard` was
    /// approximating separately.
    fn ev_best_discard(&self) -> Option<usize> {
        let (_, table) = self.hand_ev()?;
        let forbidden = self.state.forbidden_discards();
        table
            .iter()
            .map(|c| c.tile.plain().tile_type_index())
            .find(|&t| !forbidden[t])
    }

    /// Safest discard among those that keep the hand's current shanten.
    ///
    /// The middle rung, which kobalab's majiang-ai describes as "push as far as
    /// suji": at one-shanten the hand is close enough to be worth keeping but
    /// not worth feeding a no-suji middle tile for. So hold the shape where it
    /// is and, among the discards that do, take the safest, breaking ties on
    /// ukeire so the kept shape is the wider one.
    ///
    /// `None` when every discard costs shanten, and the caller then falls back
    /// to the outright fold.
    fn cautious_discard(&self, danger: &[u32; 34]) -> Option<usize> {
        let n_sets = self.state.n_sets();
        let seen = self.state.seen_counts();
        let keep = self.state.shanten();

        scan_discards(&self.state.hand(), self.state.forbidden_discards(), |tile_type, probe| {
            let shanten = hand_shanten::shanten_of(probe, n_sets);
            // Anything that sets the hand back is not holding the shape.
            (shanten <= keep).then(|| {
                let width = ukeire(probe, &seen, n_sets, shanten);
                (danger[tile_type], Reverse(width))
            })
        })
    }

    /// Safest discard, breaking ties on how little it costs the hand.
    ///
    /// Folding is not the same as throwing the hand away: among equally safe
    /// tiles there is usually one the hand misses least, and keeping that
    /// distinction is what lets the bot fold for a few turns and still come
    /// back if the danger passes.
    fn safest_discard(&self, danger: &[u32; 34]) -> Option<usize> {
        let n_sets = self.state.n_sets();

        scan_discards(&self.state.hand(), self.state.forbidden_discards(), |tile_type, probe| {
            Some((danger[tile_type], hand_shanten::shanten_of(probe, n_sets)))
        })
    }

    /// How much value each tile type carries, counting an owned red five.
    fn dora_value(&self) -> [u8; 34] {
        let mut dora = self.state.dora_per_tile_type();
        let akas = self.state.akas_in_hand();
        for (slot, &tile_type) in AKA_TILE_TYPES.iter().enumerate() {
            if akas[slot] {
                dora[tile_type] += 1;
            }
        }
        dora
    }

    /// The tile id to actually put on the table for a chosen tile type.
    ///
    /// A red five has its own id. Only emit it when it is the sole five of that
    /// suit in hand, so the plain copy goes first.
    pub(super) fn emit_tile(&self, tile_type: usize) -> Tile {
        match aka_slot(tile_type) {
            Some(slot) if self.state.akas_in_hand()[slot] && self.state.hand()[tile_type] == 1 => {
                Tile::new(tile_type as u8, true)
            }
            _ => Tile::new(tile_type as u8, false),
        }
    }
}
