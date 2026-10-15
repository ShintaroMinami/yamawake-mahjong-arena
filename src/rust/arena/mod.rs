//! Pure-Rust lockstep self-play driver over riichienv-core.
//!
//! N games run in Rust; Python is entered once per lockstep tick with every
//! pending decision as a serialized riichienv `Observation` (each carries
//! the seat's new masked events and its legal actions) and answers with one
//! exact action spec per row. Rulebot and suzume bots are decided
//! in-loop and never surface.
//!
//! Correctness rules:
//!   - an illegal action does NOT panic riichienv-core: `step` records
//!     `last_error` and force-ryukyokus the round, so every step is
//!     followed by a hard `last_error` check here;
//!   - mjai events are drained through our own cursor into the kyoku's
//!     event history, never via the consumed-once `obs.new_events()`;
//!   - `reacting_to` mirrors the live rule: the log's last event is a
//!     dahai/kakan by another actor.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use serde_json::Value;

use crate::suzume::{SuzumeConfig, SeatMemo};
use crate::rulebot::config::RulebotConfig;


/// One pending decision of one seat, kept between collect() and apply().
pub struct Pending {
    pub game: usize,
    pub seat: u8,
    pub legal: Vec<Action>,
}

/// One game driven natively.
pub struct GameSlot {
    pub state: GameState,
    event_cursor: usize,
    /// The current kyoku's parsed events (from its start_kyoku), the
    /// public history a rulebot's BotView is rebuilt from.
    kyoku_events: Vec<Value>,
    /// Rulebot / suzume decisions of the CURRENT tick, held until the games with
    /// external players step in apply(). Cleared by step().
    auto_actions: HashMap<u8, Action>,
    /// Every suzume decision of the game: (mjai log length at the
    /// decision, seat, deciding rule id). The decision's own event, if it
    /// makes one, is the first one by that seat from that log index on.
    pub suzume_trace: Vec<(u32, u8, u8)>,
    /// Per seat: what a suzume bot keeps across its decisions within
    /// the kyoku (cleared at each start_kyoku; untouched for other seats).
    suzume_memo: [SeatMemo; 4],
    /// Per seat: the in-loop bot's decisions and the wall-clock time they
    /// took (nanoseconds, observation building included).
    pub bot_decisions: [u64; 4],
    pub bot_nanos: [u64; 4],
    /// Wall-clock time of all this game's work in the loop (nanoseconds):
    /// engine steps, observations, events and the in-loop bots.
    pub work_nanos: u64,
}

impl GameSlot {
    pub fn new(
        game_mode: u8,
        seed: u64,
        rule: GameRule,
        wall_table: Option<HashMap<(u8, u8), Vec<u8>>>,
        split_wall: bool,
    ) -> Result<Self, String> {
        let mut state = GameState::new(game_mode, false, Some(seed), 0, rule);
        // Mirror the Python wrapper's `RiichiEnv.reset()`: clear the logs
        // and re-initialize round 1, which advances the wall's hand_index
        // once. The Python arena plays on THESE walls, so the shim must
        // reproduce them exactly (verified against pre/post-reset mjai
        // logs of the same seed).
        state.reset();
        // Duplicate-eval wall controls must be armed before the round 1
        // init below so its wall already comes from the table.
        state.wall_table = wall_table;
        state.split_wall = split_wall;
        state._initialize_round(0, 0, 0, 0, None, None);
        let mut slot = Self {
            state,
            event_cursor: 0,
            kyoku_events: Vec::new(),
            auto_actions: HashMap::new(),
            suzume_trace: Vec::new(),
            suzume_memo: Default::default(),
            bot_decisions: [0; 4],
            bot_nanos: [0; 4],
            work_nanos: 0,
        };
        slot.drain_events()?;
        Ok(slot)
    }

    /// Fold the mjai lines appended since the last drain (deal, discards,
    /// calls ...) into the kyoku's event history.
    fn drain_events(&mut self) -> Result<(), String> {
        while self.event_cursor < self.state.mjai_log.len() {
            let line = &self.state.mjai_log[self.event_cursor];
            let event: Value = serde_json::from_str(line)
                .map_err(|e| format!("bad mjai line from core: {e}"))?;
            let t = event
                .get("type")
                .and_then(Value::as_str)
                .ok_or("mjai event without type")?;
            if t == "start_kyoku" {
                self.kyoku_events.clear();
                self.suzume_memo.iter_mut().for_each(SeatMemo::clear);
            }
            self.kyoku_events.push(event);
            self.event_cursor += 1;
        }
        Ok(())
    }

    /// The seats that must act now (empty when the game is done).
    pub fn actors(&self) -> Vec<u8> {
        if self.state.is_done {
            return Vec::new();
        }
        match self.state.phase {
            Phase::WaitAct => vec![self.state.current_player],
            Phase::WaitResponse => self.state.active_players.clone(),
        }
    }

    /// Apply the chosen actions and drain the resulting events. Any core
    /// error is fatal: a silently force-ryukyoku'd round must never pass
    /// as data.
    pub fn step(&mut self, actions: &HashMap<u8, Action>) -> Result<(), String> {
        self.auto_actions.clear();
        self.state.step(actions);
        if let Some(err) = &self.state.last_error {
            return Err(format!("riichienv-core rejected a step: {err}"));
        }
        self.drain_events()
    }

    /// Decide one rulebot in-loop (`rulebot::decide::decide` on the seat's
    /// observation and the kyoku's events).
    pub fn rulebot_action(&mut self, seat: u8, cfg: &RulebotConfig) -> Result<Action, String> {
        let obs = self.state.get_observation(seat);
        crate::rulebot::decide::decide(&obs, &self.kyoku_events, cfg, false).map(|(a, _)| a)
    }

    /// Decide one tsumogiri bot in-loop (`tsumogiri`).
    pub fn tsumogiri_action(&mut self, seat: u8) -> Result<Action, String> {
        tsumogiri(&self.state.get_observation(seat))
    }

    /// Decide one suzume bot in-loop (`suzume::decide_observation` with the seat's
    /// kyoku memo); the rule that decided is traced.
    pub fn suzume_action(&mut self, seat: u8, cfg: &SuzumeConfig) -> Result<Action, String> {
        let obs = self.state.get_observation(seat);
        let memo = &mut self.suzume_memo[usize::from(seat)];
        let (action, rule) = crate::suzume::decide_observation(&obs, &self.kyoku_events, cfg, memo)?;
        self.suzume_trace.push((self.state.mjai_log.len() as u32, seat, rule));
        Ok(action)
    }
}

/// The tsumogiri bot's move: discard the drawn tile, else pass, else the
/// first legal action (as the Python `PythonTsumogiriAgent`).
pub fn tsumogiri(obs: &riichienv_core::observation::Observation) -> Result<Action, String> {
    let legal = obs.legal_actions_method();
    legal
        .iter()
        .find(|a| a.action_type == ActionType::Discard && a.tile == obs.drawn_tile)
        .or_else(|| legal.iter().find(|a| a.action_type == ActionType::Pass))
        .or_else(|| legal.first())
        .cloned()
        .ok_or_else(|| format!("seat {} has no legal actions but must act", obs.player_id))
}

/// N-game lockstep driver: collect() -> (pending observations) -> apply(specs).
pub struct LockstepLoop {
    pub slots: Vec<GameSlot>,
    pub pending: Vec<Pending>,
    /// Per game: the seats decided in-loop and their bots. A game whose
    /// every actor is in-loop advances inside collect() without surfacing a
    /// row at all -- such a game runs to completion in the first collect()
    /// call. Every other seat holds an observation-lane player.
    pub bot_seats: Vec<Vec<(u8, Bot)>>,
}

/// A bot decided inside the loop.
#[derive(Clone, Copy, Debug)]
pub enum Bot {
    Rulebot(RulebotConfig),
    Suzume(SuzumeConfig),
    Tsumogiri,
}

/// One collect() tick's rows, in pending/apply order.
pub struct Batch {
    pub games: Vec<usize>,
    pub seats: Vec<u8>,
    /// Base64 observations (each carries the seat's new masked events and
    /// its legal actions).
    pub obs: Vec<String>,
}

/// One exact action: (riichienv action_type, tile, consume_tiles).
pub type ActionSpec = (u8, Option<u8>, Vec<u8>);

impl LockstepLoop {
    pub fn new(
        seeds: &[u64],
        game_mode: u8,
        bot_seats: Vec<Vec<(u8, Bot)>>,
        wall_tables: Vec<Option<HashMap<(u8, u8), Vec<u8>>>>,
        split_wall: bool,
    ) -> Result<Self, String> {
        if !bot_seats.is_empty() && bot_seats.len() != seeds.len() {
            return Err("bot_seats must be empty or one entry per game".into());
        }
        if !wall_tables.is_empty() && wall_tables.len() != seeds.len() {
            return Err("wall_tables must be empty or one entry per game".into());
        }
        let mut wall_tables = wall_tables;
        wall_tables.resize(seeds.len(), None);
        let slots = seeds
            .iter()
            .zip(wall_tables)
            .map(|(&s, wt)| {
                GameSlot::new(game_mode, s, GameRule::default_tenhou(), wt, split_wall)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bot_seats = if bot_seats.is_empty() { vec![Vec::new(); seeds.len()] } else { bot_seats };
        Ok(Self { slots, pending: Vec::new(), bot_seats })
    }

    /// Every pending decision across all live games. None = every game is
    /// finished.
    pub fn collect(&mut self) -> Result<Option<Batch>, String> {
        if !self.pending.is_empty() {
            return Err("collect() called with unapplied decisions".into());
        }
        // Advance every game in parallel (rayon; one worker owns one game,
        // so no slot is touched from two threads), then flatten in slot
        // order so the row order stays deterministic.
        use rayon::prelude::*;
        let bot_seats = &self.bot_seats;
        let per_slot: Vec<Vec<(u8, String, Vec<Action>)>> = self
            .slots
            .par_iter_mut()
            .enumerate()
            .map(|(gi, slot)| {
                let t_work = std::time::Instant::now();
                let rows = (|| {
                // Advance past ticks whose every actor is a rulebot / suzume
                // seat; stop at the first tick that needs an external decision
                // (or at game end) and hold the tick's in-loop decisions for
                // apply() to merge in.
                loop {
                    let actors = slot.actors();
                    if actors.is_empty() {
                        return Ok(Vec::new());
                    }
                    let mut auto: HashMap<u8, Action> = HashMap::new();
                    let mut external: Vec<u8> = Vec::new();
                    for seat in actors {
                        let t0 = std::time::Instant::now();
                        let action = match bot_seats[gi].iter().find(|(s, _)| *s == seat) {
                            Some((_, Bot::Rulebot(cfg))) => slot.rulebot_action(seat, cfg)?,
                            Some((_, Bot::Suzume(cfg))) => slot.suzume_action(seat, cfg)?,
                            Some((_, Bot::Tsumogiri)) => slot.tsumogiri_action(seat)?,
                            None => {
                                external.push(seat);
                                continue;
                            }
                        };
                        let i = usize::from(seat);
                        slot.bot_decisions[i] += 1;
                        slot.bot_nanos[i] += t0.elapsed().as_nanos() as u64;
                        auto.insert(seat, action);
                    }
                    if external.is_empty() {
                        slot.step(&auto)?;
                        continue;
                    }
                    slot.auto_actions = auto;
                    return external
                        .into_iter()
                        .map(|seat| {
                            let obs = slot.state.get_observation(seat);
                            let legal = obs.legal_actions_method();
                            if legal.is_empty() {
                                return Err(format!(
                                    "seat {seat} has no legal actions but must act"
                                ));
                            }
                            let b64 = obs
                                .serialize_to_base64()
                                .map_err(|e| format!("obs serialize failed: {e:?}"))?;
                            Ok((seat, b64, legal))
                        })
                        .collect::<Result<Vec<_>, String>>();
                }
                })();
                slot.work_nanos += t_work.elapsed().as_nanos() as u64;
                rows
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut batch = Batch { games: Vec::new(), seats: Vec::new(), obs: Vec::new() };
        for (gi, rows) in per_slot.into_iter().enumerate() {
            for (seat, b64, legal) in rows {
                batch.games.push(gi);
                batch.seats.push(seat);
                batch.obs.push(b64);
                self.pending.push(Pending { game: gi, seat, legal });
            }
        }
        if batch.games.is_empty() {
            return Ok(None);
        }
        Ok(Some(batch))
    }

    /// Apply the tick's decisions and advance every stepped game.
    ///
    /// `actions`: one EXACT action spec (action_type, tile, consume_tiles)
    /// per collect() row, in row order. The physical tile is the agent's
    /// choice (a tsumogiri and a tedashi of the same tile type differ), so the
    /// spec is matched literally against the legal actions.
    pub fn apply(&mut self, actions: &[ActionSpec]) -> Result<(), String> {
        if actions.len() != self.pending.len() {
            return Err(format!(
                "apply() got {} decisions for {} pending",
                actions.len(),
                self.pending.len()
            ));
        }
        let mut per_game: HashMap<usize, HashMap<u8, Action>> = HashMap::new();
        for (p, (action_type, tile, consume_tiles)) in self.pending.iter().zip(actions) {
            let mut want = consume_tiles.clone();
            want.sort_unstable();
            let action = p
                .legal
                .iter()
                .find(|a| {
                    a.action_type as u8 == *action_type
                        && a.tile == *tile
                        && a.consume_tiles == want
                })
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "action ({action_type}, {tile:?}, {want:?}) is not legal \
                         for game {} seat {}",
                        p.game, p.seat
                    )
                })?;
            per_game.entry(p.game).or_default().insert(p.seat, action);
        }
        self.pending.clear();
        // Step every acted game in parallel; each worker owns its slot.
        use rayon::prelude::*;
        let mut acts: Vec<Option<HashMap<u8, Action>>> =
            (0..self.slots.len()).map(|_| None).collect();
        for (gi, actions) in per_game {
            acts[gi] = Some(actions);
        }
        self.slots
            .par_iter_mut()
            .zip(acts.into_par_iter())
            .try_for_each(|(slot, actions)| match actions {
                Some(mut actions) => {
                    // The tick's rulebot / suzume decisions act together with the
                    // external ones they were computed alongside.
                    for (seat, action) in slot.auto_actions.clone() {
                        actions.insert(seat, action);
                    }
                    let t_work = std::time::Instant::now();
                    let r = slot.step(&actions);
                    slot.work_nanos += t_work.elapsed().as_nanos() as u64;
                    r
                }
                None => Ok(()),
            })
    }

    /// Per-game final scores (only meaningful once a game is done).
    pub fn scores(&self) -> Vec<[i32; 4]> {
        self.slots
            .iter()
            .map(|s| {
                let mut sc = [0i32; 4];
                for i in 0..4 {
                    sc[i] = s.state.players[i].score;
                }
                sc
            })
            .collect()
    }
}
