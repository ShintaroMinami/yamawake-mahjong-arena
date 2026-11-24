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
//!
//! A game can be forked at a pending decision (`LockstepLoop::fork`,
//! 2026-11-25): each fork is a new game of the loop with the parent's whole
//! engine state, which plays one given move and then the rest of its kyoku
//! through the same collect() / apply(); its result is read with
//! `fork_result` and its index freed with `release`.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use serde_json::{Value, json};

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
    /// Per seat, the riichi / call (chi / pon / daiminkan) options of whatever plays it (A1 at first; every agent,
    /// in-loop or external, since 2026-10-30):
    /// [riichi legal, riichi taken, call legal, call taken] over decisions,
    /// then the same four over kyoku (a kyoku counts once if the option was
    /// legal in any decision, and as taken if it was taken in any).
    pub a1_options: [[u32; 8]; 4],
    /// Per seat, this kyoku's [riichi legal, riichi taken, call legal, call
    /// taken] seen so far (cleared at each start_kyoku).
    a1_kyoku_seen: [[bool; 4]; 4],
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
    /// Decisions fixed before the next collect() asks anyone: a fork's own
    /// move and the in-loop decisions of its tick (`branch_off`); the next
    /// tick takes them.
    held: HashMap<u8, Action>,
    /// For a fork of the loop: where it left its parent (`branch_off`).
    pub branch: Option<Branch>,
    /// Released (`LockstepLoop::release`): it takes no more decisions and
    /// its index serves the next fork.
    released: bool,
    /// A fork without in-loop bots (`branch_off`): its new events stay as
    /// mjai lines, parsed only for its result (`kyoku_events` holds the two
    /// events before its fork point).
    lean: bool,
}

/// Where a fork of the loop left its parent game (`LockstepLoop::fork`).
#[derive(Clone, Debug)]
pub struct Branch {
    /// The parent game, the seat whose decision was forked and the fork's move there.
    pub parent: usize,
    pub seat: u8,
    pub action: Action,
    /// The fork's kyoku events before its move (its result's events start here) and the scores then.
    pub k0: usize,
    pub scores0: [i32; 4],
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
        Self::from_state(state)
    }

    /// A slot on a state already set up (`playout` builds one mid-kyoku): its
    /// mjai log so far is drained into the kyoku's events (from the last
    /// `start_kyoku`).
    pub fn from_state(state: GameState) -> Result<Self, String> {
        let mut slot = Self {
            state,
            event_cursor: 0,
            kyoku_events: Vec::new(),
            auto_actions: HashMap::new(),
            suzume_trace: Vec::new(),
            a1_options: [[0; 8]; 4],
            a1_kyoku_seen: [[false; 4]; 4],
            suzume_memo: Default::default(),
            bot_decisions: [0; 4],
            bot_nanos: [0; 4],
            work_nanos: 0,
            held: HashMap::new(),
            branch: None,
            released: false,
            lean: false,
        };
        slot.drain_events()?;
        Ok(slot)
    }

    /// The current kyoku's events (from its `start_kyoku`).
    pub fn kyoku_events(&self) -> &[Value] {
        &self.kyoku_events
    }

    /// A copy of the game for a separate continuation (`playout`): the state and the kyoku's events, with empty bot
    /// memos and counters.
    pub fn fork(&self) -> Self {
        Self {
            state: self.state.clone(),
            event_cursor: self.event_cursor,
            kyoku_events: self.kyoku_events.clone(),
            auto_actions: HashMap::new(),
            suzume_trace: Vec::new(),
            a1_options: [[0; 8]; 4],
            a1_kyoku_seen: self.a1_kyoku_seen,
            suzume_memo: Default::default(),
            bot_decisions: [0; 4],
            bot_nanos: [0; 4],
            work_nanos: 0,
            held: HashMap::new(),
            branch: None,
            released: false,
            lean: false,
        }
    }

    /// A fork of this game at its current tick (`LockstepLoop::fork`): the engine state with every hidden tile
    /// (hands, live wall or per-seat queues, dead wall, dora and ura indicators), the tick's in-loop decisions and
    /// `action` for `seat` held for its first tick, and each seat's suzume decision state (`SeatMemo::decisions`).
    /// It stops at its kyoku's end (`defer_next_round`). To keep a fork small it holds no log before the fork point
    /// but each seat's unread lines (the read positions moved to match), no wall table (it deals no next kyoku), and
    /// the kyoku's parsed events only when `history` (its in-loop bots read them; else the last two, which mark a
    /// rinshan draw after a kan).
    pub fn branch_off(&mut self, parent: usize, seat: u8, action: Action, history: bool) -> Result<GameSlot, String> {
        let kyoku_events = if history {
            self.kyoku_events.clone()
        } else {
            let all = self.events_now()?;
            all[all.len().saturating_sub(2)..].to_vec()
        };
        // Clone the state without its logs and wall table, then copy the unread lines.
        let st = &mut self.state;
        let kept = (std::mem::take(&mut st.mjai_log), std::mem::take(&mut st.mjai_log_per_player), st.wall_table.take());
        let mut state = st.clone();
        (st.mjai_log, st.mjai_log_per_player, st.wall_table) = kept;
        for i in 0..4 {
            state.mjai_log_per_player[i] = st.mjai_log_per_player[i][st.player_event_counts[i]..].to_vec();
            state.player_event_counts[i] = 0;
            state.kyoku_start_event_counts[i] = 0;
        }
        state.mjai_log = st.mjai_log[self.event_cursor..].to_vec();
        state.defer_next_round = true;
        let mut held = self.auto_actions.clone();
        held.insert(seat, action.clone());
        let scores0 = std::array::from_fn(|i| self.state.players[i].score);
        Ok(GameSlot {
            state,
            event_cursor: 0,
            auto_actions: HashMap::new(),
            suzume_trace: Vec::new(),
            a1_options: [[0; 8]; 4],
            a1_kyoku_seen: self.a1_kyoku_seen,
            suzume_memo: std::array::from_fn(|i| self.suzume_memo[i].decisions()),
            bot_decisions: [0; 4],
            bot_nanos: [0; 4],
            work_nanos: 0,
            held,
            branch: Some(Branch { parent, seat, action, k0: kyoku_events.len(), scores0 }),
            kyoku_events,
            released: false,
            lean: !history,
        })
    }

    /// The kyoku's events (`kyoku_events`), a lean fork's mjai lines parsed and appended.
    pub fn events_now(&self) -> Result<std::borrow::Cow<'_, [Value]>, String> {
        if !self.lean {
            return Ok(std::borrow::Cow::Borrowed(&self.kyoku_events));
        }
        let mut out = self.kyoku_events.clone();
        for line in &self.state.mjai_log {
            out.push(serde_json::from_str(line).map_err(|e| format!("bad mjai line from core: {e}"))?);
        }
        Ok(std::borrow::Cow::Owned(out))
    }

    /// Whether the game takes no more decisions: it is over, released, or a fork whose kyoku has ended.
    pub fn finished(&self) -> bool {
        self.released || self.state.is_done || (self.branch.is_some() && self.state.needs_initialize_next_round)
    }

    /// Release a fork: drop its logs, events and memos (`LockstepLoop::release`).
    fn retire(&mut self) {
        self.released = true;
        self.state.mjai_log = Vec::new();
        self.state.mjai_log_per_player = Default::default();
        self.state.player_event_counts = [0; 4];
        self.state.kyoku_start_event_counts = [0; 4];
        self.event_cursor = 0;
        self.kyoku_events = Vec::new();
        self.suzume_memo = Default::default();
        self.suzume_trace = Vec::new();
        self.auto_actions.clear();
        self.held.clear();
        self.lean = false;
    }

    /// Swap the seats' suzume memos with `memos` (`playout` carries the caches from one play-out to the next).
    pub fn swap_memos(&mut self, memos: &mut [SeatMemo; 4]) {
        std::mem::swap(&mut self.suzume_memo, memos);
    }

    /// One in-loop bot's decision for `seat`.
    pub fn bot_action(&mut self, seat: u8, bot: &Bot) -> Result<Action, String> {
        match bot {
            Bot::Rulebot(cfg) => self.rulebot_action(seat, cfg),
            Bot::Suzume(cfg) => self.suzume_action(seat, cfg),
            Bot::Tsumogiri => self.tsumogiri_action(seat),
            Bot::A1(a1) => self.a1_action(seat, a1),
        }
    }

    /// Fold the mjai lines appended since the last drain (deal, discards,
    /// calls ...) into the kyoku's event history.
    fn drain_events(&mut self) -> Result<(), String> {
        if self.lean {
            // Kept as lines (`events_now`).
            self.event_cursor = self.state.mjai_log.len();
            return Ok(());
        }
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
                self.a1_kyoku_seen = [[false; 4]; 4];
            }
            self.kyoku_events.push(event);
            self.event_cursor += 1;
        }
        Ok(())
    }

    /// The seats that must act now (empty when the game is done; `finished`).
    pub fn actors(&self) -> Vec<u8> {
        if self.finished() {
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
        self.drain_events()?;
        if self.branch.is_some() {
            // A fork keeps each seat's unread lines only, and (unless lean) no drained line.
            let st = &mut self.state;
            for i in 0..4 {
                st.mjai_log_per_player[i].drain(..st.player_event_counts[i]);
                st.player_event_counts[i] = 0;
            }
            if !self.lean {
                st.mjai_log.clear();
                self.event_cursor = 0;
            }
        }
        Ok(())
    }

    /// Decide one rulebot in-loop (`rulebot::decide::decide` on the seat's
    /// observation and the kyoku's events).
    pub fn rulebot_action(&mut self, seat: u8, cfg: &RulebotConfig) -> Result<Action, String> {
        let obs = self.state.get_observation(seat);
        let action = crate::rulebot::decide::decide(&obs, &self.kyoku_events, cfg, false)
            .map(|(a, _)| a)
            .map_err(|e| format!("rulebot seat {seat} at mjai line {}: {e}", self.state.mjai_log.len()))?;
        self.count_options(seat, &obs.legal_actions_method(), &action);
        Ok(action)
    }

    /// Decide one tsumogiri bot in-loop (`tsumogiri`).
    pub fn tsumogiri_action(&mut self, seat: u8) -> Result<Action, String> {
        let obs = self.state.get_observation(seat);
        let action = tsumogiri(&obs)?;
        self.count_options(seat, &obs.legal_actions_method(), &action);
        Ok(action)
    }

    /// Decide one suzume bot in-loop (`suzume::decide_observation` with the seat's
    /// kyoku memo); the rule that decided is traced.
    pub fn suzume_action(&mut self, seat: u8, cfg: &SuzumeConfig) -> Result<Action, String> {
        let obs = self.state.get_observation(seat);
        let memo = &mut self.suzume_memo[usize::from(seat)];
        let (action, rule) = crate::suzume::decide_observation(&obs, &self.kyoku_events, cfg, memo)?;
        self.suzume_trace.push((self.state.mjai_log.len() as u32, seat, rule));
        self.count_options(seat, &obs.legal_actions_method(), &action);
        Ok(action)
    }
}

impl GameSlot {
    /// Decide one A1 bot in-loop (`suzume::a1::decide` with the seat's kyoku memo).
    pub fn a1_action(&mut self, seat: u8, a1: &crate::suzume::a1::A1Config) -> Result<Action, String> {
        self.a1_explained(seat, a1).map(|e| e.action)
    }

    /// `a1_action` with what chose the action and the options' scores (`suzume::a1::decide_explained`).
    pub fn a1_explained(&mut self, seat: u8, a1: &crate::suzume::a1::A1Config) -> Result<crate::suzume::a1::Explained, String> {
        let obs = self.state.get_observation(seat);
        let e = crate::suzume::a1::decide_explained(&obs, &self.kyoku_events, a1, &mut self.suzume_memo[usize::from(seat)])?;
        self.count_options(seat, &obs.legal_actions_method(), &e.action);
        Ok(e)
    }

    /// Count one decision's riichi / call options into `a1_options` (decisions and kyoku).
    pub fn count_options(&mut self, seat: u8, legal: &[Action], action: &Action) {
        let is_call = |t: ActionType| matches!(t, ActionType::Chi | ActionType::Pon | ActionType::Daiminkan);
        let riichi_legal = legal.iter().any(|a| a.action_type == ActionType::Riichi);
        let call_legal = legal.iter().any(|a| is_call(a.action_type));
        let hit = [riichi_legal, riichi_legal && action.action_type == ActionType::Riichi,
                   call_legal, call_legal && is_call(action.action_type)];
        let (n, seen) = (&mut self.a1_options[usize::from(seat)], &mut self.a1_kyoku_seen[usize::from(seat)]);
        for i in 0..4 {
            n[i] += u32::from(hit[i]);
            n[4 + i] += u32::from(hit[i] && !seen[i]);
            seen[i] |= hit[i];
        }
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
    /// Indices of released forks, for the next forks (`release`).
    free: Vec<usize>,
}

/// The legal action of `legal` that the exact spec names (its consume tiles in any order).
fn legal_action(legal: &[Action], (action_type, tile, consume_tiles): &ActionSpec) -> Option<Action> {
    let mut want = consume_tiles.clone();
    want.sort_unstable();
    legal.iter().find(|a| a.action_type as u8 == *action_type && a.tile == *tile && a.consume_tiles == want).cloned()
}

/// The table rule by name: `"tenhou"` (Tenhou's yaku and game flow) or
/// `"mjsoul"` (Mahjong Soul's yaku; the engine's stock game flow).
pub fn game_rule(name: &str) -> Result<GameRule, String> {
    match name {
        "tenhou" => Ok(GameRule::default_tenhou()),
        "mjsoul" => Ok(GameRule::default_mjsoul()),
        _ => Err(format!("unknown game rule {name:?} (tenhou or mjsoul)")),
    }
}

/// The play limits an in-loop bot (rulebot, A1) can take, both off by default (2026-11-19): with `win` false it
/// never declares tsumo or ron; with `call` false it never calls chi, pon or daiminkan (ankan stays). The bot
/// decides as if the engine had not offered those actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub win: bool,
    pub call: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self { win: true, call: true }
    }
}

impl Limits {
    /// Whether an action of type `at` stays legal under these limits.
    pub fn keeps(&self, at: ActionType) -> bool {
        (self.win || !matches!(at, ActionType::Tsumo | ActionType::Ron))
            && (self.call || !matches!(at, ActionType::Chi | ActionType::Pon | ActionType::Daiminkan))
    }
}

/// A bot decided inside the loop.
#[derive(Clone, Copy, Debug)]
pub enum Bot {
    Rulebot(RulebotConfig),
    Suzume(SuzumeConfig),
    Tsumogiri,
    /// A1 (`suzume::a1`), a research bot timed against suzume.
    A1(crate::suzume::a1::A1Config),
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
        rule: GameRule,
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
                GameSlot::new(game_mode, s, rule, wt, split_wall)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bot_seats = if bot_seats.is_empty() { vec![Vec::new(); seeds.len()] } else { bot_seats };
        Ok(Self { slots, pending: Vec::new(), bot_seats, free: Vec::new() })
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
                // apply() to merge in. A new fork's first tick starts from its
                // held decisions (`GameSlot::branch_off`).
                let mut held = std::mem::take(&mut slot.held);
                loop {
                    let actors = slot.actors();
                    if actors.is_empty() {
                        return Ok(Vec::new());
                    }
                    let mut auto: HashMap<u8, Action> = std::mem::take(&mut held);
                    let mut external: Vec<u8> = Vec::new();
                    for seat in actors {
                        if auto.contains_key(&seat) {
                            continue;
                        }
                        let t0 = std::time::Instant::now();
                        let action = match bot_seats[gi].iter().find(|(s, _)| *s == seat) {
                            Some((_, bot)) => slot.bot_action(seat, bot)?,
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
        let slots = &mut self.slots;
        for (p, spec) in self.pending.iter().zip(actions) {
            let action = legal_action(&p.legal, spec).ok_or_else(|| {
                format!("action {spec:?} is not legal for game {} seat {}", p.game, p.seat)
            })?;
            slots[p.game].count_options(p.seat, &p.legal, &action);
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

    /// Fork game `game` at the decision of `seat` that the last collect() returned, before apply(): one new game per
    /// action of `actions` (exact specs, as apply() takes them), their indices returned in order (a released fork's
    /// index is reused). Each fork holds the parent's whole engine state (`GameSlot::branch_off`), plays its action
    /// for `seat` and then the rest of its kyoku as a game of the loop: every seat keeps the parent's lane and bot,
    /// and its decisions come through collect() / apply() with the other games'. Another seat deciding in the same
    /// tick (a call or a ron on the same discard) is asked again in the fork's first collect(); the tick's in-loop
    /// decisions carry over. The parent is not changed. A fork depends only on the parent's state, its action and the
    /// agents' moves; with `split_wall` every seat draws from its own queue, so a call in one fork does not shift
    /// another seat's draws (sibling forks share every seat's next draws, unless a queue runs dry).
    pub fn fork(&mut self, game: usize, seat: u8, actions: &[ActionSpec]) -> Result<Vec<usize>, String> {
        let p = self
            .pending
            .iter()
            .find(|p| p.game == game && p.seat == seat)
            .ok_or_else(|| format!("game {game} seat {seat} has no decision pending (fork after collect(), before apply())"))?;
        let moves = actions
            .iter()
            .map(|a| legal_action(&p.legal, a).ok_or_else(|| format!("action {a:?} is not legal for game {game} seat {seat}")))
            .collect::<Result<Vec<_>, _>>()?;
        let history = !self.bot_seats[game].is_empty();
        let forks: Vec<GameSlot> = moves
            .into_iter()
            .map(|a| self.slots[game].branch_off(game, seat, a, history))
            .collect::<Result<_, _>>()?;
        let bots = self.bot_seats[game].clone();
        Ok(forks
            .into_iter()
            .map(|f| match self.free.pop() {
                Some(i) => {
                    self.slots[i] = f;
                    self.bot_seats[i] = bots.clone();
                    i
                }
                None => {
                    self.slots.push(f);
                    self.bot_seats.push(bots.clone());
                    self.slots.len() - 1
                }
            })
            .collect())
    }

    /// The result of fork `game` once its kyoku has ended, `None` while it plays: `playout::kyoku_result` from the
    /// fork's move on (with `events`, the kyoku's events from that move on), plus `parent`, `seat` and `action` (its
    /// exact spec).
    pub fn fork_result(&self, game: usize, events: bool) -> Result<Option<Value>, String> {
        let slot = self.slots.get(game).ok_or_else(|| format!("no game {game}"))?;
        let b = slot.branch.as_ref().ok_or_else(|| format!("game {game} is not a fork"))?;
        if slot.released {
            return Err(format!("fork {game} was released"));
        }
        if !slot.finished() {
            return Ok(None);
        }
        let mut out = crate::playout::kyoku_result(&slot.state, &slot.events_now()?, b.k0, &b.scores0, events);
        let mut consume = b.action.consume_tiles.clone();
        consume.sort_unstable();
        out["parent"] = json!(b.parent);
        out["seat"] = json!(b.seat);
        out["action"] = json!([b.action.action_type as u8, b.action.tile, consume]);
        Ok(Some(out))
    }

    /// Release forks: they take no more decisions, their memory is dropped and their indices serve the next forks.
    /// Only forks without a pending decision can be released (a fork that has not finished is abandoned).
    pub fn release(&mut self, games: &[usize]) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for &g in games {
            let slot = self.slots.get(g).ok_or_else(|| format!("no game {g}"))?;
            if slot.branch.is_none() {
                return Err(format!("game {g} is not a fork"));
            }
            if slot.released || !seen.insert(g) {
                return Err(format!("fork {g} is already released"));
            }
            if self.pending.iter().any(|p| p.game == g) {
                return Err(format!("fork {g} has a decision pending (release it after apply())"));
            }
        }
        for &g in games {
            self.slots[g].retire();
            self.free.push(g);
        }
        Ok(())
    }

    /// Whether every game takes no more decisions (`GameSlot::finished`).
    pub fn done(&self) -> bool {
        self.slots.iter().all(GameSlot::finished)
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

#[cfg(test)]
mod tests;
