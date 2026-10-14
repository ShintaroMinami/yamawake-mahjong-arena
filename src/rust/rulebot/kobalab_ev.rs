//! kobalab's `eval_shoupai` / `eval_fulou`, ported so the two evaluators can be
//! compared on the same hands.
//!
//! Port of majiang-ai (https://github.com/kobalab/majiang-ai), MIT License,
//! Copyright (c) 2021 Satoshi Kobayashi; see docs/THIRD_PARTY_NOTICES.md.
//!
//! The single-player DP (`dp::Solver`) is a full dynamic program over the
//! remaining draws. This is a depth-limited expectation instead: from a hand
//! within three of tenpai, take
//! every tile that advances it, weigh it by how many are left and how many draws
//! remain, recurse, and divide by a fixed width per shanten step. Memoisation is
//! what makes it affordable. Whether the cheaper model is also worse is the
//! question this exists to answer, so it is a faithful port rather than an
//! improvement -- the deliberate deviations are listed below.
//!
//! Scoring matches the single-player DP's conventions exactly, so the numbers are
//! comparable: the win is a tsumo, and a concealed hand is credited with riichi
//! plus menzen tsumo while an open one gets nothing. kobalab makes the same two
//! assumptions (`get_defen` passes no ron tile, and sets `hupai.lizhi` from
//! `shoupai.menqian`), so this is not a place the two designs differ.
//!
//! Deviations from majiang-ai, all in branches our configuration never
//! reaches or where the repo already has an equivalent:
//!
//! * The `back` argument -- kobalab's furiten guard for shanten-down discards --
//!   is dropped. `select_fulou` passes `''`, which never matches a tile.
//! * The three-shanten-and-worse branch weighs ukeire with `call_ways`
//!   rather than kobalab's yaku-aware `tingpai`. Shanten only falls inside the
//!   recursion, so the branch is reachable only if the caller starts there.
//! * Kuikae is not enforced on the discard after a simulated call.
//! * The cache is keyed by the hand alone, as kobalab's is. The remaining-tile
//!   counts differ between paths that reach the same hand, so this trades a
//!   little accuracy for the memoisation to work at all.

use super::agari::{ScoringHand, agari_points};
use super::decision::Decision;
use super::hand_shanten;
use std::collections::HashMap as AHashMap;

/// Normalisation per shanten step, `[8, 8*4, 8*4*2]` in majiang-ai.
///
/// The recursion sums over every ukeire tile without dividing by how many
/// there were, so the raw total grows with each step away from tenpai. These
/// bring the steps back onto one scale.
const WIDTH: [f32; 3] = [8., 32., 64.];

/// The shanten at which the expectation stops and ukeire counting starts.
const EV_SHANTEN: i8 = 3;

pub(super) const NO_TILE: u8 = u8::MAX;
const MAX_MELDS: usize = 4;

/// The melds a hand holds, including any this evaluation is pretending to make.
///
/// Fixed arrays rather than vectors: the recursion clones this at every call,
/// and a hand can hold four melds at most.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Melds {
    chi_runs: [u8; MAX_MELDS],
    pon_sets: [u8; MAX_MELDS],
    n_chi: u8,
    n_pon: u8,
    /// Dora sitting in the melds, which the hand array no longer sees.
    dora_count: u8,
}

impl Melds {
    fn chi_runs(&self) -> &[u8] {
        &self.chi_runs[..self.n_chi as usize]
    }

    fn pon_sets(&self) -> &[u8] {
        &self.pon_sets[..self.n_pon as usize]
    }
}

#[derive(PartialEq, Eq, Hash)]
struct Key {
    hand: [u8; 34],
    melds: Melds,
    drawn_tile_type: u8,
}

/// One hand's evaluation, holding the wall model the recursion walks.
pub(super) struct KobalabEv {
    /// Winds as 34-format tile types (27..30 = E..N).
    round_wind: u8,
    player_wind: u8,
    open_kans: Vec<u8>,
    closed_kans: Vec<u8>,
    /// Dora carried by each tile kind, red fives aside.
    dora_per_tile_type: [u8; 34],
    /// Red fives held concealed. They move into a meld when one is called, which
    /// this ignores -- they stay dora either way.
    n_akas: u8,
    /// Tiles nobody has seen, so the ones that could still be drawn.
    unseen: [u8; 34],
    n_unseen: f32,
    /// Tiles still in the wall, `_n_zimo` in majiang-ai -- the wall, not this
    /// seat's share of it. A step of the recursion is one draw for this seat,
    /// which takes four tiles off the wall, so that is what a step spends.
    drawable_count: f32,
    cache: AHashMap<Key, f32>,
}

impl Decision<'_> {
    /// Set up kobalab's evaluator for the hand as it stands.
    pub(super) fn kobalab_ev(&self) -> KobalabEv {
        let state = self.state;
        let mut unseen = [4; 34];
        for (tile_type, &seen) in state.seen_counts().iter().enumerate() {
            unseen[tile_type] = 4_u8.saturating_sub(seen);
        }
        let n_unseen = unseen.iter().map(|&n| f32::from(n)).sum();

        KobalabEv {
            round_wind: state.round_wind(),
            player_wind: state.player_wind(),
            open_kans: state.open_kans().to_vec(),
            closed_kans: state.closed_kans().to_vec(),
            dora_per_tile_type: state.dora_per_tile_type(),
            n_akas: state.akas_in_hand().iter().filter(|&&b| b).count() as u8,
            unseen,
            n_unseen,
            drawable_count: f32::from(state.drawable_count()),
            cache: AHashMap::new(),
        }
    }
}

impl KobalabEv {
    /// The melds the hand actually holds, before any are simulated.
    pub(super) fn melds_of(&self, chi_runs: &[u8], pon_sets: &[u8]) -> Melds {
        let mut melds = Melds {
            chi_runs: [0; MAX_MELDS],
            pon_sets: [0; MAX_MELDS],
            n_chi: 0,
            n_pon: 0,
            dora_count: 0,
        };
        for &c in chi_runs.iter().take(MAX_MELDS) {
            melds.chi_runs[melds.n_chi as usize] = c;
            melds.n_chi += 1;
            melds.dora_count += (0..3).map(|i| self.dora_of(c + i)).sum::<u8>();
        }
        for &p in pon_sets.iter().take(MAX_MELDS) {
            melds.pon_sets[melds.n_pon as usize] = p;
            melds.n_pon += 1;
            melds.dora_count += self.dora_of(p) * 3;
        }
        for &k in self.open_kans.iter().chain(self.closed_kans.iter()) {
            melds.dora_count += self.dora_of(k) * 4;
        }
        melds
    }

    /// Count a kan of `tile_type` as made: closed for an ankan, open for a
    /// kakan (whose pon the caller drops from `pon_sets`). The cache is keyed
    /// without the kans, so it is cleared.
    pub(super) fn add_kan(&mut self, tile_type: u8, closed: bool) {
        if closed {
            self.closed_kans.push(tile_type);
        } else {
            self.open_kans.push(tile_type);
        }
        self.cache.clear();
    }

    /// How many dora one copy of a tile is worth, red fives aside.
    fn dora_of(&self, tile_type: u8) -> u8 {
        self.dora_per_tile_type[tile_type as usize]
    }

    /// What a hand is worth, in points.
    ///
    /// The entry point. A 3n+1 hand is valued as it stands; a 3n+2 one is
    /// valued at its best discard, which is what makes this usable on a hand
    /// the seat does not hold yet -- give it the tiles a call would leave and
    /// it prices the call.
    pub(super) fn value(
        &mut self,
        hand: &[u8; 34],
        chi_runs: &[u8],
        pon_sets: &[u8],
        n_sets: u8,
        drawn_tile_type: u8,
    ) -> f32 {
        let melds = self.melds_of(chi_runs, pon_sets);
        let mut work = *hand;
        self.eval(&mut work, melds, n_sets, drawn_tile_type)
    }

    /// Expected score of a hand, in points.
    ///
    /// `drawn_tile_type` is the tile that arrived last, which is the winning
    /// tile if the hand is complete; pass `NO_TILE` when nothing was just drawn.
    fn eval(&mut self, hand: &mut [u8; 34], melds: Melds, n_sets: u8, drawn_tile_type: u8) -> f32 {
        let key = Key {
            hand: *hand,
            melds,
            drawn_tile_type,
        };
        if let Some(&hit) = self.cache.get(&key) {
            return hit;
        }

        let shanten = hand_shanten::shanten_of(hand, n_sets);
        let concealed: u8 = hand.iter().sum();
        let value = if shanten == -1 {
            self.score(hand, &melds, drawn_tile_type)
        } else if concealed % 3 == 2 {
            self.best_of_discards(hand, melds, n_sets, shanten)
        } else if shanten < EV_SHANTEN {
            self.expectation(hand, melds, n_sets, shanten)
        } else {
            self.ukeire(hand, n_sets, shanten)
        };

        self.cache.insert(key, value);
        value
    }

    /// A 3n+2 hand is worth the best its discards are worth.
    fn best_of_discards(
        &mut self,
        hand: &mut [u8; 34],
        melds: Melds,
        n_sets: u8,
        shanten: i8,
    ) -> f32 {
        let mut best = 0_f32;
        for tile_type in 0..34 {
            if hand[tile_type] == 0 {
                continue;
            }
            hand[tile_type] -= 1;
            // A discard that gives ground back is never the one to make here.
            if hand_shanten::shanten_of(hand, n_sets) <= shanten {
                best = best.max(self.eval(hand, melds, n_sets, NO_TILE));
            }
            hand[tile_type] += 1;
        }
        best
    }

    /// Within three of tenpai: average over the tiles that advance the hand.
    ///
    /// The call bonus is added for every ukeire tile except at tenpai, where
    /// there is nothing left to call for.
    fn expectation(
        &mut self,
        hand: &mut [u8; 34],
        melds: Melds,
        n_sets: u8,
        shanten: i8,
    ) -> f32 {
        let mut total = 0.;
        for tile_type in 0..34 {
            if self.unseen[tile_type] == 0 {
                continue;
            }
            hand[tile_type] += 1;
            let advances = hand_shanten::shanten_of(hand, n_sets) < shanten;
            if advances {
                // Spend the tile before recursing: it cannot also be drawn later,
                // and a step of the recursion is one turn for each of four seats.
                self.unseen[tile_type] -= 1;
                self.n_unseen -= 1.;
                self.drawable_count -= 4.;

                let mut value = self.eval(hand, melds, n_sets, tile_type as u8);
                hand[tile_type] -= 1;
                if shanten > 0 {
                    value += self.eval_fulou(hand, melds, n_sets, tile_type);
                }

                self.unseen[tile_type] += 1;
                self.n_unseen += 1.;
                self.drawable_count += 4.;
                total += value * self.draw_chance(tile_type);
            } else {
                hand[tile_type] -= 1;
            }
        }
        total / WIDTH[shanten as usize]
    }

    /// Three away or worse: ukeire, weighed by how a tile can be had.
    ///
    /// Points mean nothing this far out, so the value changes unit here -- which
    /// is why the thresholds a caller compares against only hold nearer tenpai.
    fn ukeire(&self, hand: &[u8; 34], n_sets: u8, shanten: i8) -> f32 {
        let mut probe = *hand;
        let mut total = 0.;
        for tile_type in 0..34 {
            if self.unseen[tile_type] == 0 {
                continue;
            }
            probe[tile_type] += 1;
            if hand_shanten::shanten_of(&probe, n_sets) < shanten {
                total += f32::from(self.unseen[tile_type]) * super::discard::call_ways(hand, tile_type) as f32;
            }
            probe[tile_type] -= 1;
        }
        total
    }

    /// What a tile being drawn is worth as a weight: how many are left, thinned
    /// by how much of the wall is still to come.
    fn draw_chance(&self, tile_type: usize) -> f32 {
        if self.drawable_count <= 0. {
            return 0.;
        }
        f32::from(self.unseen[tile_type]) * self.drawable_count / self.n_unseen
    }

    /// What being able to call this tile adds, kobalab's `eval_fulou`.
    ///
    /// A tile that can be ponned comes from any of the three other seats; one
    /// that can only be chi'd comes from the one seat above. When the pon is
    /// worth more it is what every seat would be called on, hence three times;
    /// otherwise two seats offer a pon and the upper seat a chi.
    fn eval_fulou(
        &mut self,
        hand: &mut [u8; 34],
        melds: Melds,
        n_sets: u8,
        tile_type: usize,
    ) -> f32 {
        let now = hand_shanten::shanten_of(hand, n_sets);
        let after = n_sets - 1;
        let mut pon = 0_f32;
        let mut chi = 0_f32;

        if hand[tile_type] >= 2 && (melds.n_pon as usize + melds.n_chi as usize) < MAX_MELDS {
            hand[tile_type] -= 2;
            if hand_shanten::shanten_of(hand, after) < now {
                let mut with = melds;
                with.pon_sets[with.n_pon as usize] = tile_type as u8;
                with.n_pon += 1;
                with.dora_count += self.dora_of(tile_type as u8) * 3;
                pon = self.eval(hand, with, after, tile_type as u8);
            }
            hand[tile_type] += 2;
        }

        if tile_type < 27 && (melds.n_pon as usize + melds.n_chi as usize) < MAX_MELDS {
            let rank = (tile_type % 9) as i32;
            let base = tile_type - rank as usize;
            for offsets in [[-2, -1], [-1, 1], [1, 2]] {
                let taken: Vec<usize> = offsets
                    .iter()
                    .filter_map(|&o| {
                        let i = rank + o;
                        (0..9).contains(&i).then_some(base + i as usize)
                    })
                    .collect();
                if taken.len() != 2 || taken.iter().any(|&t| hand[t] == 0) {
                    continue;
                }
                for &t in &taken {
                    hand[t] -= 1;
                }
                if hand_shanten::shanten_of(hand, after) < now {
                    let mut with = melds;
                    let low = taken.iter().copied().min().unwrap().min(tile_type);
                    with.chi_runs[with.n_chi as usize] = low as u8;
                    with.n_chi += 1;
                    with.dora_count += (0..3).map(|i| self.dora_of(low as u8 + i)).sum::<u8>();
                    chi = chi.max(self.eval(hand, with, after, tile_type as u8));
                }
                for &t in &taken {
                    hand[t] += 1;
                }
            }
        }

        if pon > chi { pon * 3. } else { pon * 2. + chi }
    }

    /// The hand's score as a tsumo, or zero when it has no yaku to win with.
    pub(super) fn score(&self, hand: &[u8; 34], melds: &Melds, win_tile_type: u8) -> f32 {
        if win_tile_type == NO_TILE {
            return 0.;
        }
        let is_menzen = melds.n_chi == 0 && melds.n_pon == 0 && self.open_kans.is_empty();
        // Riichi plus menzen tsumo while concealed, nothing once open. The same
        // split the single-player DP makes, and the one kobalab makes.
        let dora_count = (0..34)
            .map(|t| hand[t] * self.dora_per_tile_type[t])
            .sum::<u8>()
            + self.n_akas
            + melds.dora_count;
        let inp = ScoringHand {
            hand,
            is_menzen,
            chi_runs: melds.chi_runs(),
            pon_sets: melds.pon_sets(),
            open_kans: &self.open_kans,
            closed_kans: &self.closed_kans,
            round_wind: self.round_wind,
            player_wind: self.player_wind,
            win_tile_type,
            is_ron: false,
        };
        agari_points(&inp, is_menzen, dora_count).map_or(0., |v| v as f32)
    }
}
