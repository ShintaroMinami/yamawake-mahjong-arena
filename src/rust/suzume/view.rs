//! What one suzume decision may read: the seat's own hand and the public
//! table, from riichienv's `Observation`, the live wall count, and the
//! engine's legal actions (the only source of what may be done).

use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::types::{MeldType, standard_next_dora_tile};

use crate::eval::feat::tiles::{aka_slot_of_tid, is_aka_tid, is_yaochu};

/// One seat's view at one decision.
#[derive(Clone, Debug)]
pub struct MjView {
    /// This seat (absolute).
    pub seat: u8,
    /// Concealed hand, 34-format counts (14 tiles on the own turn).
    pub hand: [u8; 34],
    pub akas_in_hand: [bool; 3],
    /// Copies visible from this seat: hand, every river, every meld, dora
    /// indicators (`feat::seen`).
    pub seen_counts: [u8; 34],
    /// Red fives visible from this seat (hand, rivers, melds).
    pub akas_seen: [bool; 3],
    pub dora_indicator_types: Vec<u8>,
    /// Dora han each tile type carries (indicators only; red fives apart).
    pub dora_per_tile_type: [u8; 34],
    /// Winds as 34-format tile types (27..30 = E..N).
    pub round_wind: u8,
    pub player_wind: u8,
    /// This seat's points.
    pub score: i32,
    /// The kyoku within the round wind (0-based; honba not counted).
    pub kyoku_index: u8,
    /// Every seat's points, by absolute seat (riichi sticks on the table not
    /// counted), the honba and the riichi sticks on the table: the situation
    /// the push / fold state model reads (`pf::situation`).
    pub scores: [i32; 4],
    pub honba: u8,
    pub sticks: u32,
    /// This seat's points minus the best of the other seats' (riichi
    /// sticks on the table not counted): > 0 is a unique top.
    pub lead: i32,
    pub drawable_count: u8,
    /// The tile just drawn (136-format id), on the own turn.
    pub drawn_tile: Option<u8>,
    /// Riichi declared by this seat (from the declaration on).
    pub in_riichi: bool,
    /// This seat's melds as (meld type, 136-format tiles, opened): the
    /// meld type in `score::Scorer`'s code (0 chi, 1 pon, 2 daiminkan,
    /// 3 ankan, 4 kakan); an ankan is not opened.
    pub own_melds: Vec<(u8, Vec<u8>, bool)>,
    /// Seats that draw before this seat's next draw (0 on the own turn; while
    /// reacting to a discard, the seats between the discarder and us).
    pub draws_before_us: u8,
    /// Nobody has called and this seat has not discarded yet.
    pub double_riichi_possible: bool,
    /// Tile types in this seat's river (called ones included): the
    /// furiten the DP's ron model checks.
    pub own_discards: [bool; 34],
    /// This seat's discards so far (called ones included).
    pub n_own_discards: u8,
    /// How many discards ago (anyone's, 1 = the latest) each tile type was
    /// last discarded this kyoku; `NEVER` when it was not, or before a river
    /// record is applied (`apply_river`).
    pub last_ago: [u8; 34],
    /// What each other seat's river says about its wait, by absolute seat
    /// (this seat's own entry stays empty).
    pub threats: [Threat; 4],
    pub legal: Vec<Action>,
}

/// One other seat as a threat: whether it is in riichi and which tile
/// types it cannot ron.
#[derive(Clone, Copy, Debug)]
pub struct Threat {
    /// Riichi declared (`Observation::riichi_declared`).
    pub riichi: bool,
    /// Tile types it cannot ron (genbutsu): its own river (furiten), and,
    /// once a river record is applied (`MjView::apply_river`), every tile
    /// anyone discarded after its riichi declaration (passing one in riichi
    /// is furiten too).
    pub safe: [bool; 34],
    /// Tile types in its own river (called ones included).
    pub river: [bool; 34],
    /// Its seat wind as a tile type (27..30 = E..N).
    pub wind: u8,
    /// Its discards so far (called ones included).
    pub n_discards: u8,
    /// Discards by anyone after its riichi declaration (0 until a river
    /// record is applied).
    pub passed: u8,
    /// Its melds (chi, pon, daiminkan, kakan, ankan: every one is visible).
    pub n_melds: u8,
    /// Dora han in its melds: the indicated dora and the red fives.
    pub meld_dora: u8,
    /// Its pon / kan melds of a yakuhai (a dragon, the round wind, its own
    /// seat wind).
    pub yakuhai_melds: u8,
    /// It has melds and their number tiles are of one suit (honors allowed).
    pub meld_flush: bool,
    /// It has melds and every meld tile is a 2..8.
    pub meld_tanyao: bool,
    /// It has melds and every meld holds a 1, a 9 or an honor.
    pub meld_chanta: bool,
    /// Its discards (called ones included) by kind: m, p, s, honors.
    pub suit_discards: [u8; 4],
    /// Its own discard index of its first discard of each kind (`NEVER`
    /// when none).
    pub suit_first: [u8; 4],
    // Read from a river record (`MjView::apply_river`; empty before):
    /// Its own discard index (0 = its first discard) of its first discard
    /// of each tile type (`NEVER` when it has not discarded one).
    pub first_discard: [u8; 34],
    /// Its riichi-declaring discard: (own discard index, tile type).
    pub decl: Option<(u8, u8)>,
    /// It discarded the red five of the suit (m, p, s) at or before its
    /// riichi-declaring discard (any time so far without a declaration).
    pub red_pre: [bool; 3],
}

/// "Never" in `Threat::first_discard` and `MjView::last_ago`.
pub const NEVER: u8 = u8::MAX;

impl Default for Threat {
    fn default() -> Self {
        Self {
            riichi: false,
            safe: [false; 34],
            river: [false; 34],
            wind: 27,
            n_discards: 0,
            passed: 0,
            n_melds: 0,
            meld_dora: 0,
            yakuhai_melds: 0,
            meld_flush: false,
            meld_tanyao: false,
            meld_chanta: false,
            suit_discards: [0; 4],
            suit_first: [NEVER; 4],
            first_discard: [NEVER; 34],
            decl: None,
            red_pre: [false; 3],
        }
    }
}

/// One discard of the kyoku; a river record lists them in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiverRow {
    /// The discarding seat (absolute).
    pub seat: u8,
    pub tile_type: u8,
    /// The riichi-declaring discard.
    pub riichi: bool,
    /// A red five.
    pub red: bool,
}

/// The kyoku's discards in order, from its mjai events (from `start_kyoku`
/// on; other event types are skipped).
pub fn river_from_mjai(events: &[serde_json::Value]) -> Result<Vec<RiverRow>, String> {
    let mut pending = [false; 4];
    let mut rows = Vec::new();
    for e in events {
        let actor = || e.get("actor").and_then(serde_json::Value::as_u64).map(|a| a as usize % 4);
        match e.get("type").and_then(serde_json::Value::as_str) {
            Some("reach") => pending[actor().ok_or("reach actor")?] = true,
            Some("dahai") => {
                let a = actor().ok_or("dahai actor")?;
                let tid = crate::eval::feat::tiles::parse_tile(e["pai"].as_str().ok_or("dahai pai")?)?;
                rows.push(RiverRow {
                    seat: a as u8,
                    tile_type: tid / 4,
                    riichi: std::mem::take(&mut pending[a]),
                    red: is_aka_tid(tid),
                });
            }
            _ => {}
        }
    }
    Ok(rows)
}

impl MjView {
    /// The view of the seat `obs` belongs to.
    pub fn from_observation(obs: &Observation, legal: Vec<Action>) -> Self {
        let drawable_count = obs.drawable_count;
        let seat = obs.player_id as usize;
        let mut hand = [0u8; 34];
        let mut akas_in_hand = [false; 3];
        for &tid in &obs.hands[seat] {
            hand[(tid / 4) as usize] += 1;
            if let Some(slot) = aka_slot_of_tid(tid) {
                akas_in_hand[slot] = true;
            }
        }
        let mut akas_seen = akas_in_hand;
        let mut mark_aka = |tid: u32| {
            if let Some(slot) = aka_slot_of_tid(tid) {
                akas_seen[slot] = true;
            }
        };
        for seat_discards in &obs.discards {
            seat_discards.iter().for_each(|&t| mark_aka(t));
        }
        for seat_melds in &obs.melds {
            for m in seat_melds {
                m.tiles.iter().for_each(|&t| mark_aka(u32::from(t)));
            }
        }
        let dora_indicator_types: Vec<u8> =
            obs.dora_indicators.iter().map(|&t| (t / 4) as u8).collect();
        let mut dora_per_tile_type = [0u8; 34];
        for &t in &dora_indicator_types {
            dora_per_tile_type[standard_next_dora_tile(t) as usize] += 1;
        }
        let indicator_types: Vec<usize> = dora_indicator_types.iter().map(|&t| t as usize).collect();
        let hand_i: [i32; 34] = std::array::from_fn(|t| i32::from(hand[t]));
        let seen = crate::eval::feat::seen::seen_counts(&hand_i, obs, &indicator_types);
        let seen_counts: [u8; 34] = std::array::from_fn(|t| seen[t] as u8);
        let nobody_called = obs.melds.iter().all(Vec::is_empty);
        let own_melds = obs.melds[seat]
            .iter()
            .map(|m| (m.meld_type as u8, m.tiles.clone(), m.opened))
            .collect();
        // While reacting, the discarder is the seat whose river ends with
        // the offered tile (searched from kamicha).
        let offered = legal.iter().find_map(|a| match a.action_type {
            ActionType::Chi | ActionType::Pon | ActionType::Daiminkan | ActionType::Ron => a.tile,
            _ => None,
        });
        let draws_before_us = offered
            .and_then(|t| (1..4).find(|&k| obs.discards[(seat + 4 - k) % 4].last() == Some(&u32::from(t))))
            .map_or(0, |k| (k - 1) as u8);
        let mut own_discards = [false; 34];
        for &tid in &obs.discards[seat] {
            own_discards[(tid / 4) as usize] = true;
        }
        let mut threats = [Threat::default(); 4];
        for (s, th) in threats.iter_mut().enumerate().filter(|&(s, _)| s != seat) {
            th.riichi = obs.riichi_declared[s];
            th.wind = 27 + ((s + 4 - obs.oya as usize) % 4) as u8;
            th.n_discards = obs.discards[s].len() as u8;
            for &tid in &obs.discards[s] {
                th.river[(tid / 4) as usize] = true;
            }
            th.safe = th.river;
            th.n_melds = obs.melds[s].len() as u8;
            th.meld_dora = obs.melds[s]
                .iter()
                .flat_map(|m| m.tiles.iter())
                .map(|&t| dora_per_tile_type[usize::from(t / 4)] + u8::from(is_aka_tid(t)))
                .sum();
            let melds = &obs.melds[s];
            let round_wind = 27 + obs.round_wind.min(3);
            th.yakuhai_melds = melds
                .iter()
                .filter(|m| m.meld_type != MeldType::Chi)
                .filter(|m| m.tiles.first().is_some_and(|&t| t / 4 >= 31 || t / 4 == round_wind || t / 4 == th.wind))
                .count() as u8;
            let types = || melds.iter().flat_map(|m| m.tiles.iter().map(|&t| t / 4));
            let mut suits = types().filter(|&t| t < 27).map(|t| t / 9);
            let first = suits.next();
            th.meld_flush = !melds.is_empty() && suits.all(|x| Some(x) == first);
            th.meld_tanyao = !melds.is_empty() && types().all(|t| t < 27 && (1..8).contains(&(t % 9)));
            th.meld_chanta = !melds.is_empty()
                && melds.iter().all(|m| m.tiles.iter().any(|&t| is_yaochu(usize::from(t / 4))));
            for (j, &tid) in obs.discards[s].iter().enumerate() {
                let k = ((tid / 4) as usize / 9).min(3);
                th.suit_discards[k] += 1;
                th.suit_first[k] = th.suit_first[k].min(j.min(usize::from(NEVER - 1)) as u8);
            }
        }
        Self {
            seat: seat as u8,
            hand,
            akas_in_hand,
            seen_counts,
            akas_seen,
            dora_indicator_types,
            dora_per_tile_type,
            round_wind: 27 + obs.round_wind.min(3),
            player_wind: 27 + ((seat + 4 - obs.oya as usize) % 4) as u8,
            score: obs.scores[seat],
            kyoku_index: obs.kyoku_index,
            scores: obs.scores,
            honba: obs.honba,
            sticks: obs.riichi_sticks,
            lead: obs.scores[seat]
                - (0..obs.scores.len()).filter(|&s| s != seat).map(|s| obs.scores[s]).max().unwrap_or(obs.scores[seat]),
            drawable_count,
            drawn_tile: obs.drawn_tile,
            in_riichi: obs.riichi_declared[seat],
            own_melds,
            draws_before_us,
            double_riichi_possible: nobody_called && obs.discards[seat].is_empty(),
            own_discards,
            n_own_discards: obs.discards[seat].len() as u8,
            last_ago: [NEVER; 34],
            threats,
            legal,
        }
    }

    /// Read the kyoku's discards so far, in order (`rows`): for each seat
    /// in riichi, the tiles discarded after its declaration become safe; for
    /// every seat, its first discard of each tile type, its declaring
    /// discard and the red fives it cut before it; and how long ago each
    /// tile type was last discarded. A riichi seat whose declaration is not
    /// in `rows` keeps its own river as its only safe tiles.
    pub fn apply_river(&mut self, rows: &[RiverRow]) {
        let n = rows.len();
        self.last_ago = [NEVER; 34];
        let mut own_index = [0u8; 4];
        for (i, r) in rows.iter().enumerate() {
            self.last_ago[usize::from(r.tile_type)] = (n - i).min(usize::from(NEVER - 1)) as u8;
            let s = usize::from(r.seat % 4);
            let th = &mut self.threats[s];
            let j = own_index[s];
            own_index[s] = j.saturating_add(1);
            let first = &mut th.first_discard[usize::from(r.tile_type)];
            *first = (*first).min(j);
            if r.red && th.decl.is_none() {
                if let Some(slot) = crate::eval::feat::tiles::aka_slot(usize::from(r.tile_type)) {
                    th.red_pre[slot] = true;
                }
            }
            if r.riichi && th.decl.is_none() {
                th.decl = Some((j, r.tile_type));
            }
        }
        for (s, th) in self.threats.iter_mut().enumerate().filter(|(_, th)| th.riichi) {
            if let Some(i) = rows.iter().position(|r| r.riichi && usize::from(r.seat) == s) {
                for r in &rows[i + 1..] {
                    th.safe[usize::from(r.tile_type)] = true;
                }
                th.passed = (rows.len() - i - 1) as u8;
            }
        }
    }

    /// Other seats in riichi.
    pub fn riichi_threats(&self) -> impl Iterator<Item = (usize, &Threat)> {
        self.threats.iter().enumerate().filter(|(_, th)| th.riichi)
    }

    /// The first legal action of a type.
    pub fn find(&self, action_type: ActionType) -> Option<&Action> {
        self.legal.iter().find(|a| a.action_type == action_type)
    }

    /// Copies of each tile type not visible from this seat.
    pub fn unseen(&self) -> [u8; 34] {
        std::array::from_fn(|t| 4u8.saturating_sub(self.seen_counts[t]))
    }

    /// Dragons, the round wind and the player's wind.
    pub fn is_yakuhai(&self, tile_type: u8) -> bool {
        tile_type >= 31 || tile_type == self.round_wind || tile_type == self.player_wind
    }

    /// `h` (a concealed hand, e.g. after a call and its discard) holds the
    /// yakuhai `t` exactly twice and the pair can still be ponned: at least
    /// one of the other two copies is not visible outside the own hand
    /// (rivers, melds, dora indicators). `seen_counts` includes the own
    /// hand, so with two copies held the pair is dead when it reaches 4.
    /// E.g. a chun pair with the other two chun in the rivers is dead.
    pub fn live_yakuhai_pair(&self, h: &[u8; 34], t: usize) -> bool {
        h[t] == 2 && self.is_yakuhai(t as u8) && self.seen_counts[t] < 4
    }

    /// This seat holds a chi, pon or open kan (no riichi, no menzen yaku).
    pub fn is_open(&self) -> bool {
        self.own_melds.iter().any(|m| m.2)
    }

    /// The drawn tile as (tile type, aka).
    pub fn drawn_tile_kind(&self) -> Option<(u8, bool)> {
        self.drawn_tile.map(|t| (t / 4, is_aka_tid(t)))
    }
}
