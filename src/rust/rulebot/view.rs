//! The bot's one-turn view of the table, built from riichienv state.
//!
//! Everything is SELF-RELATIVE: seat 0 is this hand, 1 is shimocha,
//! 2 toimen, 3 kamicha -- the convention every rule file is written in.
//!
//! Construction takes the riichienv `Observation` plus the kyoku's mjai
//! events (the stream `decide` and the Python `SeatTracker` keep), because the
//! observation alone does not carry the aligned discards, riichi tiles or
//! the reaction context.

use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::types::{standard_next_dora_tile, MeldType};
use serde_json::Value;

use super::hand_shanten::shanten_of;
use super::tile::Tile;
use crate::eval::feat::tiles::{aka_slot_of_tid, is_aka_tid, parse_tile};

/// One aligned-discards cell: the discard a seat made on that go-around.
#[derive(Clone, Copy)]
pub struct DiscardCell {
    pub tile: Tile,
    pub discard_is_riichi: bool,
}

/// What the seat may do right now: the subset of the engine's legal actions
/// the rules read.
#[derive(Clone, Copy, Default)]
pub struct AllowedActions {
    pub discard: bool,
    pub riichi: bool,
    pub win: bool,
    pub pon: bool,
    pub chi_low: bool,
    pub chi_mid: bool,
    pub chi_high: bool,
    /// Relative seat of the discarder being reacted to (0 when not reacting).
    pub discarder: u8,
    /// Tile types the engine offers an ankan / a kakan of, as bit masks.
    pub ankan: u64,
    pub kakan: u64,
}

impl AllowedActions {
    /// Flags from the engine's legal actions; chi sides follow riichienv's
    /// own chi encoding (38 low, 39 mid, 40 high: the called tile's place
    /// in the run).
    pub fn from_legal(legal: &[Action], discarder: u8) -> Self {
        let has = |t: ActionType| legal.iter().any(|a| a.action_type == t);
        let chi = |id: i32| {
            legal
                .iter()
                .any(|a| a.action_type == ActionType::Chi && a.encode().ok() == Some(id))
        };
        Self {
            discard: has(ActionType::Discard),
            riichi: has(ActionType::Riichi),
            win: has(ActionType::Ron) || has(ActionType::Tsumo),
            pon: has(ActionType::Pon),
            chi_low: chi(38),
            chi_mid: chi(39),
            chi_high: chi(40),
            discarder,
            ankan: kan_mask(legal, ActionType::Ankan),
            kakan: kan_mask(legal, ActionType::Kakan),
        }
    }

    pub const fn any_chi(&self) -> bool {
        self.chi_low || self.chi_mid || self.chi_high
    }
}

/// The tile types of the legal kans of one kind, as a bit mask.
fn kan_mask(legal: &[Action], kind: ActionType) -> u64 {
    legal
        .iter()
        .filter(|a| a.action_type == kind)
        .filter_map(|a| a.tile)
        .fold(0, |m, t| m | 1 << (t / 4))
}

/// What this seat can see, self-relative.
pub struct BotView {
    pub(super) hand: [u8; 34],
    pub(super) akas_in_hand: [bool; 3],
    pub(super) n_sets: u8,
    /// Winds as 34-format tile types (27..30 = E..N); `obs.round_wind` is 0..3.
    pub(super) round_wind: u8,
    pub(super) player_wind: u8,
    pub(super) oya: u8,
    pub(super) scores: [i32; 4],
    pub(super) seen_counts: [u8; 34],
    pub(super) dora_per_tile_type: [u8; 34],
    pub(super) visible_dora: [u8; 4],
    pub(super) drawable_count: u8,
    pub(super) in_riichi: [bool; 4],
    pub(super) chi_runs: Vec<u8>,
    pub(super) pon_sets: Vec<u8>,
    pub(super) open_kans: Vec<u8>,
    pub(super) closed_kans: Vec<u8>,
    pub(super) meld_tiles: [Vec<Vec<Tile>>; 4],
    pub(super) closed_kan_tile_types: [Vec<u8>; 4],
    pub(super) aligned_discards: [Vec<Option<DiscardCell>>; 4],
    /// Each seat's discards in order, self-relative seats (unlike the
    /// absolute, 136-id `obs.discards`).
    pub(super) discards: [Vec<Tile>; 4],
    pub(super) last_discard: Option<Tile>,
    pub(super) drawn_tile: Option<Tile>,
    pub(super) allowed: AllowedActions,
    pub(super) dora_indicator_types: Vec<u8>,
    /// Own melds with physical red info: (dp meld type, tiles).
    pub(super) own_meld_tiles: Vec<(u8, Vec<Tile>)>,
    /// Tile types the engine refuses to discard right now (kuikae).
    /// The rules themselves never filter the swap-call discard; riichienv
    /// forbids it, so `arena::rulebot_decision` (the loop and `decide`)
    /// feeds the ban in from the legal actions.
    pub(super) forbidden_discards: [bool; 34],
}

impl BotView {
    /// Build the view for the seat `obs` belongs to.
    ///
    /// `kyoku_events`: the kyoku's mjai events so far (from start_kyoku).
    /// `allowed`: what the engine offers now. The live wall count is the
    /// observation's.
    pub fn build(obs: &Observation, kyoku_events: &[Value], allowed: AllowedActions) -> Result<Self, String> {
        let drawable_count = obs.drawable_count;
        let player_id = obs.player_id as usize;
        let rel = |abs: usize| (abs + 4 - player_id) % 4;

        let mut hand = [0u8; 34];
        let mut akas_in_hand = [false; 3];
        for &tid in &obs.hands[player_id] {
            hand[(tid / 4) as usize] += 1;
            if let Some(slot) = aka_slot_of_tid(tid) {
                akas_in_hand[slot] = true;
            }
        }
        let n_tiles: u8 = hand.iter().sum();
        let n_sets = n_tiles / 3;

        let oya_abs = obs.oya as usize;
        let round_wind = 27 + obs.round_wind.min(3);

        // Rivers laid out by go-around, so column k holds each seat's k-th
        // turn of the kyoku: a seat that comes before the dealer in our
        // self-relative order has had no turn in the dealer's first
        // go-around, so its first cell is empty.
        let oya_rel = rel(oya_abs);
        let mut aligned_discards: [Vec<Option<DiscardCell>>; 4] =
            std::array::from_fn(|r| if r < oya_rel { vec![None] } else { Vec::new() });
        let mut discards: [Vec<Tile>; 4] = Default::default();
        let mut last_discard = None;
        let mut riichi_stage = [false; 4];
        let mut in_riichi = [false; 4];

        for event in kyoku_events {
            let t = event
                .get("type")
                .and_then(Value::as_str)
                .ok_or("event without type")?;
            match t {
                "dahai" => {
                    let actor = event["actor"].as_u64().ok_or("dahai actor")? as usize;
                    let tid = parse_tile(event["pai"].as_str().ok_or("dahai pai")?)?;
                    let tile = Tile::from_tid(tid);
                    let r = rel(actor);
                    let discard_is_riichi = riichi_stage[actor];
                    riichi_stage[actor] = false;
                    aligned_discards[r].push(Some(DiscardCell { tile, discard_is_riichi }));
                    discards[r].push(tile);
                    last_discard = Some(tile);
                }
                "reach" => {
                    riichi_stage[event["actor"].as_u64().ok_or("reach actor")? as usize] =
                        true;
                }
                "reach_accepted" => {
                    in_riichi
                        [rel(event["actor"].as_u64().ok_or("accepted actor")? as usize)] =
                        true;
                }
                "chi" | "pon" | "daiminkan" => {
                    let actor = event["actor"].as_u64().ok_or("call actor")? as usize;
                    let target = event["target"].as_u64().ok_or("call target")? as usize;
                    // A call moves the turn straight to the caller: every
                    // seat strictly between the discarder and the caller
                    // loses a turn, i.e. gets an empty cell. (A chi is made
                    // by the next seat, so it skips nobody; the claimed
                    // discard keeps its own cell either way.)
                    let skipped = (actor + 4 - target) % 4;
                    for step in 1..skipped {
                        aligned_discards[rel((target + step) % 4)].push(None);
                    }
                }
                _ => {}
            }
        }

        // Melds of every seat, plain tile types; own lists split by type.
        let mut meld_tiles: [Vec<Vec<Tile>>; 4] = Default::default();
        let mut closed_kan_tile_types: [Vec<u8>; 4] = Default::default();
        let (mut chi_runs, mut pon_sets, mut open_kans, mut closed_kans) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut own_meld_tiles: Vec<(u8, Vec<Tile>)> = Vec::new();
        for abs in 0..4 {
            let r = rel(abs);
            for m in &obs.melds[abs] {
                let tiles: Vec<Tile> = m
                    .tiles
                    .iter()
                    .map(|&tid| {
                        Tile::from_tid(tid)
                    })
                    .collect();
                let min_tile_type = tiles.iter().map(|t| t.tile_type).min().unwrap_or(0);
                if m.meld_type == MeldType::Ankan {
                    closed_kan_tile_types[r].push(min_tile_type);
                } else {
                    meld_tiles[r].push(tiles.clone());
                }
                if abs == player_id {
                    let meld_type = match m.meld_type {
                        MeldType::Chi => {
                            chi_runs.push(min_tile_type);
                            0
                        }
                        MeldType::Pon => {
                            pon_sets.push(min_tile_type);
                            1
                        }
                        MeldType::Daiminkan | MeldType::Kakan => {
                            open_kans.push(min_tile_type);
                            2
                        }
                        MeldType::Ankan => {
                            closed_kans.push(min_tile_type);
                            3
                        }
                    };
                    own_meld_tiles.push((meld_type, tiles.clone()));
                }
            }
        }

        // Dora machinery, event-timed via obs.dora_indicators (live view).
        let mut dora_per_tile_type = [0u8; 34];
        for &indicator in &obs.dora_indicators {
            dora_per_tile_type[standard_next_dora_tile((indicator / 4) as u8) as usize] += 1;
        }
        let indicator_types: Vec<usize> =
            obs.dora_indicators.iter().map(|&t| (t / 4) as usize).collect();
        let dora_indicator_types: Vec<u8> =
            indicator_types.iter().map(|&t| t as u8).collect();
        let seen_counts = {
            let mut counts_i = [0i32; 34];
            for t in 0..34 {
                counts_i[t] = i32::from(hand[t]);
            }
            let seen = crate::eval::feat::seen::seen_counts(&counts_i, obs, &indicator_types);
            let mut out = [0u8; 34];
            for t in 0..34 {
                out[t] = seen[t] as u8;
            }
            out
        };

        // Dora each seat is known to hold: this hand in full (concealed +
        // reds + melds), the others only what their melds show.
        let mut visible_dora = [0u8; 4];
        for t in 0..34 {
            visible_dora[0] += hand[t] * dora_per_tile_type[t];
        }
        visible_dora[0] += akas_in_hand.iter().filter(|&&b| b).count() as u8;
        for abs in 0..4 {
            let r = rel(abs);
            let mut meld_dora = 0u8;
            for m in &obs.melds[abs] {
                for &tid in &m.tiles {
                    meld_dora += dora_per_tile_type[(tid / 4) as usize];
                    if is_aka_tid(tid) {
                        meld_dora += 1;
                    }
                }
            }
            visible_dora[r] += meld_dora;
        }

        let mut scores = [0i32; 4];
        for r in 0..4 {
            scores[r] = obs.scores[(player_id + r) % 4];
        }
        // NOTE: obs.riichi_declared is deliberately NOT folded in here. It
        // flips at the declaration, but the threat logic runs on ACCEPTED
        // riichi only -- a declarer whose reach tile is still up for calls
        // is not yet a threat (e.g. a pon on the reach tile itself). The reach_accepted rows in the event stream are the
        // source of truth.

        let drawn_tile = obs.drawn_tile.map(|tid| {
            Tile::from_tid(tid)
        });

        Ok(Self {
            hand,
            akas_in_hand,
            n_sets,
            round_wind,
            player_wind: 27 + ((player_id + 4 - oya_abs) % 4) as u8,
            oya: rel(oya_abs) as u8,
            scores,
            seen_counts,
            dora_per_tile_type,
            visible_dora,
            drawable_count,
            in_riichi,
            chi_runs,
            pon_sets,
            open_kans,
            closed_kans,
            meld_tiles,
            closed_kan_tile_types,
            aligned_discards,
            discards,
            last_discard,
            drawn_tile,
            allowed: allowed,
            dora_indicator_types,
            own_meld_tiles,
            forbidden_discards: [false; 34],
        })
    }

    // ---- accessors ----

    pub fn hand(&self) -> [u8; 34] {
        self.hand
    }
    pub fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }
    pub fn n_sets(&self) -> u8 {
        self.n_sets
    }
    pub fn round_wind(&self) -> u8 {
        self.round_wind
    }
    pub fn player_wind(&self) -> u8 {
        self.player_wind
    }
    pub fn oya(&self) -> u8 {
        self.oya
    }
    pub fn scores(&self) -> [i32; 4] {
        self.scores
    }
    pub fn seen_counts(&self) -> [u8; 34] {
        self.seen_counts
    }
    pub fn dora_per_tile_type(&self) -> [u8; 34] {
        self.dora_per_tile_type
    }
    pub fn visible_dora(&self) -> [u8; 4] {
        self.visible_dora
    }
    pub fn drawable_count(&self) -> u8 {
        self.drawable_count
    }
    pub fn in_riichi(&self) -> [bool; 4] {
        self.in_riichi
    }
    pub fn self_in_riichi(&self) -> bool {
        self.in_riichi[0]
    }
    pub fn chi_runs(&self) -> &[u8] {
        &self.chi_runs
    }
    pub fn pon_sets(&self) -> &[u8] {
        &self.pon_sets
    }
    pub fn open_kans(&self) -> &[u8] {
        &self.open_kans
    }
    pub fn closed_kans(&self) -> &[u8] {
        &self.closed_kans
    }
    pub fn meld_tiles(&self, player: usize) -> &[Vec<Tile>] {
        &self.meld_tiles[player]
    }
    pub fn meld_count(&self, player: usize) -> usize {
        self.meld_tiles[player].len() + self.closed_kan_tile_types[player].len()
    }
    pub fn discard_count(&self, player: usize) -> usize {
        self.discards[player].len()
    }
    pub fn discard_mask(&self, player: usize) -> [bool; 34] {
        let mut out = [false; 34];
        for tile in &self.discards[player] {
            out[tile.plain().tile_type_index()] = true;
        }
        out
    }
    pub fn passed_since_riichi(&self, player: usize) -> [bool; 34] {
        if !self.in_riichi[player] {
            return [false; 34];
        }
        let mut out = self.discard_mask(player);
        let Some(riichi_declaration_index) = self.aligned_discards[player]
            .iter()
            .position(|item| item.is_some_and(|k| k.discard_is_riichi))
        else {
            return out;
        };
        for row in &self.aligned_discards {
            for item in row.iter().skip(riichi_declaration_index + 1).flatten() {
                out[item.tile.plain().tile_type_index()] = true;
            }
        }
        out
    }
    pub fn last_discard(&self) -> Option<Tile> {
        self.last_discard
    }
    pub fn drawn_tile(&self) -> Option<Tile> {
        self.drawn_tile
    }
    pub fn allowed(&self) -> AllowedActions {
        self.allowed
    }
    pub fn dora_indicator_types(&self) -> Vec<u8> {
        self.dora_indicator_types.clone()
    }
    /// Mark tile types the engine will not accept as a discard now.
    pub fn set_forbidden_discards(&mut self, forbidden: [bool; 34]) {
        self.forbidden_discards = forbidden;
    }
    pub fn forbidden_discards(&self) -> &[bool; 34] {
        &self.forbidden_discards
    }
    pub fn own_meld_tiles(&self) -> &[(u8, Vec<Tile>)] {
        &self.own_meld_tiles
    }
    pub fn discards_of(&self, player: usize) -> Vec<super::tile::Tile> {
        self.discards[player].clone()
    }
    /// Shanten of the hand as held: a 3n+1 hand as is, a 3n+2 hand read as
    /// its best discard (and -1 when the draw completes it) -- the property
    /// `hand_shanten` gates.
    pub fn shanten(&self) -> i8 {
        shanten_of(&self.hand, self.n_sets)
    }
}
