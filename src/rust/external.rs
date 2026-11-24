//! suzume outside the arena, from Rust: the crate's public entry.
//!
//! A [`Suzume`] is a bot: a built-in A1 set (`suzume`, `suzume-rla225`) or an A1 model directory, with its
//! options. [`Suzume::choose`] is one decision on an empty memo: the move with suzume's score of every legal
//! action, a riichi as one move with the discard declared with it ([`Choice`]). [`SuzumeSeat`] plays one seat
//! through a kyoku, keeping the seat's memo between decisions as the arena keeps it. The Python
//! `yamawake.external.SuzumePlayer` goes through these same functions, so its moves and scores are these.
//!
//! The crate re-exports the riichienv-core fork it is built on ([`crate::riichienv_core`]) and `serde_json`, so a
//! caller's `Observation` and mjai events are the types these functions take.
//!
//! ```no_run
//! use yamawake::external::{Suzume, SuzumeSeat, observation_from_base64};
//! use yamawake::serde_json::Value;
//!
//! # fn main() -> Result<(), String> {
//! # let (obs_b64, kyoku_events): (String, Vec<Value>) = unimplemented!();
//! let bot = Suzume::from_spec("suzume")?;                       // Copy + Send + Sync
//! let obs = observation_from_base64(&obs_b64, &kyoku_events)?;  // Python's obs.serialize_to_base64()
//! let c = bot.choose(&obs, &kyoku_events)?;                     // as SuzumePlayer("suzume").choose(obs, events)
//! let (action, riichi_tile) = (c.action.clone(), c.riichi_tile);
//! let mut seat = SuzumeSeat::new(bot);                          // one seat through a kyoku
//! let a = seat.decide(&obs, &kyoku_events)?;
//! # Ok(()) }
//! ```

use riichienv_core::action::Action;
use riichienv_core::observation::Observation;
use serde_json::{Value, json};

use crate::arena::Limits;
use crate::suzume::SeatMemo;
use crate::suzume::a1::{A1Config, A1Models, BUILTIN, Precision};

/// One legal move of a decision ([`Choice::legal`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    /// Its mjai form: a riichi is `{"type": "reach", "pai": ..., "tsumogiri": ...}`, one per declaring discard.
    pub mjai: Value,
    /// Its engine action; for a riichi the discard declared with it.
    pub action: Action,
    /// Whether it declares riichi.
    pub riichi: bool,
}

/// One decision with its scores (`docs/play_out_spec.md` section 5).
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// The legal moves, a riichi spelled out per declaring discard.
    pub legal: Vec<Move>,
    /// The move played, in mjai form (a riichi with its discard).
    pub choice: Value,
    /// Its index in `legal`.
    pub choice_index: Option<usize>,
    /// The engine action to play now, the one [`Suzume::decide`] gives: for a riichi the declaration
    /// (`ActionType::Riichi`).
    pub action: Action,
    /// For a riichi, the tile (136 id) the seat discards after the declaration: the discard its next decision
    /// plays.
    pub riichi_tile: Option<u8>,
    /// What decided the move: `"model"` (the best score), or a rule (`"win"`, `"kyushu_kyuhai"`, `"riichi_tile"`,
    /// `"riichi_group"`, `"one_discard"`, `"pass"`, `"tsumogiri"`, `"first"`; play_out_spec.md section 5).
    pub basis: String,
    /// Per legal move, the log-probability of its option in the decision (`None` for a move no model scored);
    /// `None` when no model ran (a rule's decision).
    pub scores: Option<Vec<Option<f64>>>,
    /// The P / M / F weights of the experts' mixture, for a set with P / M / F experts.
    pub pmf: Option<[f64; 3]>,
}

impl Choice {
    /// The JSON the Python `SuzumePlayer.choose` returns: `legal` (mjai), `choice`, `choice_index`, `basis`,
    /// `scores` and, with experts, `pmf`.
    pub fn to_json(&self) -> Value {
        let legal: Vec<&Value> = self.legal.iter().map(|m| &m.mjai).collect();
        let mut out = json!({"legal": legal, "choice": self.choice, "choice_index": self.choice_index,
                             "basis": self.basis, "scores": self.scores});
        if let Some(p) = self.pmf {
            out["pmf"] = json!(p);
        }
        out
    }
}

/// A suzume bot: an A1 set with its options. It holds no per-game state, so one value serves any number of seats
/// and threads.
#[derive(Clone, Copy, Debug)]
pub struct Suzume {
    a1: A1Config,
}

impl Suzume {
    /// The built-in sets' names, the default (`suzume`) first.
    pub fn builtin_sets() -> impl Iterator<Item = &'static str> {
        BUILTIN.iter().map(|b| b.name)
    }

    /// The bot of `set` (a built-in set's name, or an A1 model directory) with `options`, as the Python player's
    /// config: `aggression` (push / fold, in [-1, 1]), `call_shift` / `riichi_shift` (numbers), `win` / `call`
    /// (0 / 1, the play limits), `mlp` (`f64`, the default, or `f32`: the networks' arithmetic); a directory set
    /// also takes `pmf_fixed` and the suzume options of its features set. Each value is read as its text.
    ///
    /// # Errors
    /// An unknown set or option, or a value out of range.
    pub fn new<K: AsRef<str>, V: AsRef<str>>(set: &str, options: impl IntoIterator<Item = (K, V)>)
            -> Result<Self, String> {
        let options = options.into_iter().map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string())).collect();
        a1_config(set, options).map(|a1| Self { a1 })
    }

    /// The bot of a player spec, `<set>[:KEY=VALUE...]` (e.g. `suzume`, `suzume:aggression=0.3:call=0`, `suzume:mlp=f32`), with
    /// [`Suzume::new`]'s options. The style targets of the Python spec (calibrated in Python) are not taken: give
    /// the options they calibrate to.
    ///
    /// # Errors
    /// As [`Suzume::new`], or a part that is not `KEY=VALUE`.
    pub fn from_spec(spec: &str) -> Result<Self, String> {
        let mut parts = spec.split(':');
        let set = parts.next().unwrap_or_default();
        let options = parts
            .map(|p| p.split_once('=').ok_or_else(|| format!("{spec}: {p:?} is not KEY=VALUE")))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(set, options)
    }

    /// The bot of an A1 config (the Python bindings, which read the config from a dict).
    pub(crate) fn of(a1: A1Config) -> Self {
        Self { a1 }
    }

    /// One decision on an empty memo, with its scores: what a new [`SuzumeSeat`] chooses.
    ///
    /// `obs` is the seat's observation; `kyoku_events` the kyoku's mjai events as the seat saw them, from its
    /// `start_kyoku` (others' draws may be masked).
    ///
    /// # Errors
    /// No legal action, or events the decision cannot read.
    pub fn choose(&self, obs: &Observation, kyoku_events: &[Value]) -> Result<Choice, String> {
        crate::playout::a1_choice(obs, kyoku_events, &self.a1, &mut SeatMemo::default())
    }

    /// One decision on an empty memo, the move only ([`Choice::action`] without the scores).
    ///
    /// # Errors
    /// As [`Suzume::choose`].
    pub fn decide(&self, obs: &Observation, kyoku_events: &[Value]) -> Result<Action, String> {
        crate::suzume::a1::decide(obs, kyoku_events, &self.a1, &mut SeatMemo::default())
    }
}

/// One seat of suzume through a game: the bot and the seat's kyoku memo (the tile chosen with a riichi, for the
/// discard after the declaration, and exact caches), forgotten when the kyoku's `start_kyoku` changes.
pub struct SuzumeSeat {
    bot: Suzume,
    memo: SeatMemo,
    kyoku: Option<Value>,
}

impl SuzumeSeat {
    /// A seat of `bot` with an empty memo.
    pub fn new(bot: Suzume) -> Self {
        Self { bot, memo: SeatMemo::default(), kyoku: None }
    }

    /// [`Suzume::choose`] with the seat's memo.
    ///
    /// # Errors
    /// As [`Suzume::choose`].
    pub fn choose(&mut self, obs: &Observation, kyoku_events: &[Value]) -> Result<Choice, String> {
        self.enter(kyoku_events);
        crate::playout::a1_choice(obs, kyoku_events, &self.bot.a1, &mut self.memo)
    }

    /// [`Suzume::decide`] with the seat's memo: after a riichi it chose, the discard declared with it.
    ///
    /// # Errors
    /// As [`Suzume::choose`].
    pub fn decide(&mut self, obs: &Observation, kyoku_events: &[Value]) -> Result<Action, String> {
        self.enter(kyoku_events);
        crate::suzume::a1::decide(obs, kyoku_events, &self.bot.a1, &mut self.memo)
    }

    /// Forget the memo (a new game on the same seat; a new kyoku forgets it by itself).
    pub fn reset(&mut self) {
        self.memo.clear();
        self.kyoku = None;
    }

    /// The P / M / F state probabilities on this decision (`a1::state_probs`), with the seat's memo.
    pub(crate) fn state_probs(&mut self, obs: &Observation, kyoku_events: &[Value]) -> Result<Option<[f64; 3]>, String> {
        self.enter(kyoku_events);
        crate::suzume::a1::state_probs(obs, kyoku_events, &self.bot.a1, &mut self.memo)
    }

    /// Start over on a new kyoku (its `start_kyoku`, the first event, differs from the last one seen).
    fn enter(&mut self, kyoku_events: &[Value]) {
        let start = kyoku_events.first().filter(|e| e["type"] == "start_kyoku");
        if start.is_some() && start != self.kyoku.as_ref() {
            self.memo.clear();
            self.kyoku = start.cloned();
        }
    }
}

/// The live wall at a kyoku's start (riichienv's `drawable_count` then).
const LIVE_WALL: usize = 70;

/// An observation from its base64 JSON (riichienv's `Observation.serialize_to_base64()`), with the fork's two
/// fields set from the kyoku's events: a stock riichienv `Observation` (one that went through Python's riichienv,
/// or an outside server's) has no `drawable_count` and no `last_discarder`. The live wall is 70 less the kyoku's
/// draws (rinshan included, as the engine counts them) and the last discarder is the actor of the kyoku's last
/// `dahai`; both equal the engine's own values.
///
/// # Errors
/// Text that is not an observation.
pub fn observation_from_base64(obs_b64: &str, kyoku_events: &[Value]) -> Result<Observation, String> {
    let mut obs = Observation::deserialize_from_base64(obs_b64).map_err(|e| format!("obs deserialize failed: {e:?}"))?;
    let kind = |e: &Value, t: &str| e.get("type").and_then(Value::as_str) == Some(t);
    let draws = kyoku_events.iter().filter(|e| kind(e, "tsumo")).count();
    obs.drawable_count = LIVE_WALL.saturating_sub(draws) as u8;
    obs.last_discarder =
        kyoku_events.iter().rev().find(|e| kind(e, "dahai")).and_then(|e| e.get("actor")?.as_u64()).map(|a| a as u8);
    Ok(obs)
}

/// Take `key`'s value out of `options`.
fn take(options: &mut Vec<(String, String)>, key: &str) -> Option<String> {
    options.iter().position(|(k, _)| k == key).map(|i| options.remove(i).1)
}

/// An A1 bot from its set and options ([`Suzume::new`]; the Python player's config less its `"a1"`).
pub(crate) fn a1_config(set: &str, mut options: Vec<(String, String)>) -> Result<A1Config, String> {
    let mut number = |key: &str| -> Result<f64, String> {
        match take(&mut options, key) {
            Some(v) => v.trim().parse().map_err(|_| format!("{key} {v}: not a number")),
            None => Ok(0.0),
        }
    };
    let (aggression, call_shift, riichi_shift) = (number("aggression")?, number("call_shift")?, number("riichi_shift")?);
    let mut limits = Limits::default();
    for (key, flag) in [("win", &mut limits.win), ("call", &mut limits.call)] {
        if let Some(v) = take(&mut options, key) {
            *flag = match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" => true,
                "0" | "false" => false,
                s => return Err(format!("{key} {s}: not 0 or 1")),
            };
        }
    }
    let mlp = take(&mut options, "mlp").map(|v| v.parse::<Precision>()).transpose()?.unwrap_or_default();
    let a1 = match A1Config::builtin(set) {
        Some(c) => {
            if !options.is_empty() {
                return Err(format!("{set} takes only aggression / call_shift / riichi_shift / win / call / mlp"));
            }
            A1Config::new(c.cfg, c.models, None, aggression, call_shift, riichi_shift)?
        }
        None => {
            let pmf = take(&mut options, "pmf_fixed");
            let models = A1Models::load(set)?;
            let cfg = crate::suzume::SuzumeConfig::from_pairs(options.iter().map(|(k, v)| (k.as_str(), v.as_str())))?;
            A1Config::new(cfg, models, pmf.as_deref(), aggression, call_shift, riichi_shift)?
        }
    };
    Ok(A1Config { limits, mlp, ..a1 })
}
