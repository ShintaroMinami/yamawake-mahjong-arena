//! PyO3 bindings (`yamawake._core`): the lockstep loop, the rulebot,
//! suzume, and the single-player DP.
//!
//! `run_sp(input_dict)` returns a list of candidate dicts; the DP (scoring
//! included) runs in pure Rust with the GIL released.

// pyo3 0.22's macros expand to unsafe calls inside unsafe fns, which the
// 2024 edition warns on once per bound argument; nothing here is unsafe.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::eval::dp::{Akas, Counts, Input, Solver};

mod features;

fn counts_from(obj: &Bound<'_, PyAny>) -> PyResult<Counts> {
    let v: Vec<u8> = obj.extract()?;
    let mut c = [0u8; 34];
    for (i, x) in v.iter().enumerate().take(34) {
        c[i] = *x;
    }
    Ok(c)
}

fn akas_from(obj: &Bound<'_, PyAny>) -> PyResult<Akas> {
    let v: Vec<bool> = obj.extract()?;
    Ok([v[0], v[1], v[2]])
}

#[pyfunction]
fn run_sp<'py>(
    py: Python<'py>,
    inp: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let get = |k: &str| -> PyResult<Bound<'py, PyAny>> {
        inp.get_item(k)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()))
    };
    let input = Input {
        hand: counts_from(&get("hand")?)?,
        akas_in_hand: akas_from(&get("reds_in_hand")?)?,
        seen_counts: counts_from(&get("seen_counts")?)?,
        akas_seen: akas_from(&get("reds_seen")?)?,
        draws_left: get("draws_left")?.extract()?,
        own_turn: get("own_turn")?.extract()?,
        last_draw_is_haitei: get("last_draw_is_haitei")?.extract()?,
        double_riichi_possible: get("double_riichi_possible")?.extract()?,
        riichi_when_tenpai: get("riichi_when_tenpai")?.extract()?,
        is_menzen: get("is_menzen")?.extract()?,
        player_wind: get("jikaze")?.extract()?,
        round_wind: get("bakaze")?.extract()?,
        dora_indicators: get("dora_indicators")?.extract()?,
        meld_dora_han: get("meld_dora_han")?.extract()?,
        melds: get("melds")?.extract()?,
        // Optional, so a caller that predates it still works.
        collect_tiles: match inp.get_item("collect_tiles")? {
            Some(v) => v.extract()?,
            None => Vec::new(),
        },
        // `run_sp` values self-drawn wins only, as it always has.
        ron: None,
    };

    let cands = py.allow_threads(|| {
        let mut solver = Solver::new(&input);
        solver.run()
    });

    let out = PyList::empty_bound(py);
    for c in cands {
        let d = PyDict::new_bound(py);
        d.set_item("tile", c.tile_type)?;
        d.set_item("discard_is_red", c.discard_is_aka)?;
        d.set_item("required", c.effective_tiles)?;
        d.set_item("tenpai", c.tenpai)?;
        d.set_item("win", c.win)?;
        d.set_item("exp", c.ev)?;
        // Present on the no-discard root, and on any discard candidate named
        // by `collect_tiles`; the caller reshapes them to (34, 3, t) and
        // (34, 34).
        if let Some(v) = c.per_tile {
            d.set_item("per_tile", v)?;
        }
        if let Some(v) = c.per_edge {
            d.set_item("per_edge", v)?;
        }
        out.append(d)?;
    }
    Ok(out)
}

/// Decode one game's packed wall table: consecutive 138-byte records of
/// (kyoku ordinal, honba, 136 wall tiles), each wall a permutation of
/// 0..136. An empty blob means "no table" (seeded shuffle, as today).
fn parse_wall_table(
    blob: &[u8],
) -> Result<Option<std::collections::HashMap<(u8, u8), Vec<u8>>>, String> {
    if blob.is_empty() {
        return Ok(None);
    }
    if blob.len() % 138 != 0 {
        return Err(format!(
            "wall table blob length {} is not a multiple of 138",
            blob.len()
        ));
    }
    let mut table = std::collections::HashMap::with_capacity(blob.len() / 138);
    for rec in blob.chunks_exact(138) {
        let wall = rec[2..].to_vec();
        let mut seen = [false; 136];
        for &t in &wall {
            if t as usize >= 136 || seen[t as usize] {
                return Err(format!(
                    "wall for kyoku {} honba {} is not a 136-tile permutation",
                    rec[0], rec[1]
                ));
            }
            seen[t as usize] = true;
        }
        if table.insert((rec[0], rec[1]), wall).is_some() {
            return Err(format!("duplicate wall table key ({}, {})", rec[0], rec[1]));
        }
    }
    Ok(Some(table))
}

/// N-game lockstep self-play driver. Python enters once per tick:
/// `collect()` hands back every pending decision as a base64 riichienv
/// Observation, `apply(actions)` answers with one exact action spec per row.
/// Rulebot and suzume bots are decided inside the loop and never surface.
#[pyclass(module = "yamawake._core")]
struct SelfPlayLoop {
    inner: crate::arena::LockstepLoop,
}

#[pymethods]
impl SelfPlayLoop {
    #[new]
    #[pyo3(signature = (seeds, game_mode=2, bot_seats=None, wall_tables=None, split_wall=false))]
    fn new(
        seeds: Vec<u64>,
        game_mode: u8,
        bot_seats: Option<Vec<Vec<(u8, String, Bound<'_, PyDict>)>>>,
        wall_tables: Option<Vec<Vec<u8>>>,
        split_wall: bool,
    ) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        // Per game: [(seat, "rulebot" / "suzume", config dict), ...]: the
        // seats of the in-loop bots.
        let bot_seats = bot_seats
            .unwrap_or_default()
            .into_iter()
            .map(|game| {
                game.into_iter()
                    .map(|(seat, kind, cfg)| Ok((seat, bot_from(&kind, &cfg)?)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        // Per game: one packed wall-table blob of 138-byte records
        // (kyoku ordinal, honba, 136 wall tiles); empty blob = no table.
        // A game with a table plays each (kyoku, honba) on the table's
        // wall regardless of renchan; missing keys fall back to the
        // seeded shuffle.
        let wall_tables = wall_tables
            .unwrap_or_default()
            .into_iter()
            .map(|blob| parse_wall_table(&blob).map_err(PyValueError::new_err))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = crate::arena::LockstepLoop::new(&seeds, game_mode, bot_seats, wall_tables, split_wall)
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Every pending decision, or None when every game is finished.
    /// Keys: n, games, seats, obs (base64 Observations), row-aligned.
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        use pyo3::exceptions::PyValueError;
        let inner = &mut self.inner;
        let batch = py
            .allow_threads(|| inner.collect())
            .map_err(PyValueError::new_err)?;
        let Some(b) = batch else { return Ok(None) };
        let out = PyDict::new_bound(py);
        out.set_item("n", b.games.len())?;
        out.set_item("games", b.games)?;
        out.set_item("seats", b.seats)?;
        out.set_item("obs", b.obs)?;
        Ok(Some(out))
    }

    /// Apply one exact (action_type, tile, consume_tiles) per collect() row,
    /// in order.
    fn apply(&mut self, py: Python<'_>, actions: Vec<(u8, Option<u8>, Vec<u8>)>) -> PyResult<()> {
        use pyo3::exceptions::PyValueError;
        let inner = &mut self.inner;
        py.allow_threads(|| inner.apply(&actions))
            .map_err(PyValueError::new_err)
    }

    /// Per-game final scores (meaningful once a game is done).
    fn scores(&self) -> Vec<Vec<i32>> {
        self.inner.scores().iter().map(|s| s.to_vec()).collect()
    }

    /// Full-information mjai log of one game.
    fn mjai_log(&self, game: usize) -> PyResult<Vec<String>> {
        self.inner
            .slots
            .get(game)
            .map(|s| s.state.mjai_log.clone())
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(game))
    }

    /// The full-information mjai lines of one game from index `start` on
    /// (to follow a game incrementally without copying the whole log).
    fn mjai_log_from(&self, game: usize, start: usize) -> PyResult<Vec<String>> {
        self.inner
            .slots
            .get(game)
            .map(|s| s.state.mjai_log.get(start..).map_or_else(Vec::new, <[String]>::to_vec))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(game))
    }

    /// Every suzume decision of one game, in order: (mjai log index at the
    /// decision, seat, deciding rule name). The decision's own event, if
    /// any, is the first event by that seat from that index on.
    fn suzume_trace(&self, game: usize) -> PyResult<Vec<(u32, u8, &'static str)>> {
        let slot = self
            .inner
            .slots
            .get(game)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(game))?;
        Ok(slot.suzume_trace.iter().map(|&(i, seat, rule)| (i, seat, crate::suzume::decider_name(rule))).collect())
    }

    /// Per game and seat: the in-loop bot's (decisions, seconds); zeros for
    /// the seats decided outside the loop.
    fn bot_times(&self) -> Vec<Vec<(u64, f64)>> {
        self.inner
            .slots
            .iter()
            .map(|s| (0..4).map(|i| (s.bot_decisions[i], s.bot_nanos[i] as f64 * 1e-9)).collect())
            .collect()
    }

    /// Per game: the loop's wall-clock work on it in seconds (engine,
    /// observations, events and the in-loop bots).
    fn work_times(&self) -> Vec<f64> {
        self.inner.slots.iter().map(|s| s.work_nanos as f64 * 1e-9).collect()
    }

    fn done(&self) -> bool {
        self.inner.slots.iter().all(|s| s.state.is_done)
    }
}

/// An in-loop bot from its kind ("rulebot" / "suzume" / "tsumogiri") and config dict.
fn bot_from(kind: &str, config: &Bound<'_, PyDict>) -> PyResult<crate::arena::Bot> {
    match kind {
        "rulebot" => Ok(crate::arena::Bot::Rulebot(rulebot_config_from(config)?)),
        "suzume" => Ok(crate::arena::Bot::Suzume(suzume_config_from(config)?)),
        "tsumogiri" => Ok(crate::arena::Bot::Tsumogiri),
        "a1" => Ok(crate::arena::Bot::A1(a1_config_from(config)?)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "no in-loop bot {kind:?} (rulebot, suzume, tsumogiri, a1)"
        ))),
    }
}

fn rulebot_config_from(overrides: &Bound<'_, PyDict>) -> PyResult<crate::rulebot::RulebotConfig> {
    use pyo3::exceptions::PyKeyError;
    let mut cfg = crate::rulebot::RulebotConfig::default();
    for (k, v) in overrides.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "call_ev_sp" => cfg.call_ev_sp = v.extract::<bool>()?,
            "call_sp_flat_pass" => cfg.call_sp_flat_pass = v.extract::<bool>()?,
            "kan" => cfg.kan = v.extract::<bool>()?,
            _ => return Err(PyKeyError::new_err(format!("unknown knob {key}"))),
        }
    }
    Ok(cfg)
}

/// An A1 bot from its config: {"a1": <a built-in set's player name>} (`suzume::a1::BUILTIN`), or {"a1": <A1 model
/// directory>, optional "pmf_fixed": the P / M / F mixture weights ("P", "M", "F" or "p,m,f"; `a1::parse_pmf`),
/// the rest: the suzume config of its features set}; both take optional "aggression": push / fold (a number in [-1, 1];
/// `A1Config::aggression`) and "call_tendency" / "riichi_tendency" (numbers; `A1Config`).
fn a1_config_from(config: &Bound<'_, PyDict>) -> PyResult<crate::suzume::a1::A1Config> {
    use pyo3::exceptions::{PyKeyError, PyValueError};
    let rest = config.copy()?;
    let a1: String = rest
        .get_item("a1")?
        .ok_or_else(|| PyKeyError::new_err("an a1 bot needs \"a1\": <built-in set or model directory>"))?
        .extract()?;
    rest.del_item("a1")?;
    let delta: f64 = match rest.get_item("aggression")? {
        Some(v) => v.str()?.to_string().trim().parse().map_err(|_| PyValueError::new_err(format!("aggression {v}: not a number")))?,
        None => 0.0,
    };
    rest.del_item("aggression").ok();
    let call_tendency: f64 = match rest.get_item("call_tendency")? {
        Some(v) => v.str()?.to_string().trim().parse().map_err(|_| PyValueError::new_err(format!("call_tendency {v}: not a number")))?,
        None => 0.0,
    };
    rest.del_item("call_tendency").ok();
    let riichi_tendency: f64 = match rest.get_item("riichi_tendency")? {
        Some(v) => v.str()?.to_string().trim().parse().map_err(|_| PyValueError::new_err(format!("riichi_tendency {v}: not a number")))?,
        None => 0.0,
    };
    rest.del_item("riichi_tendency").ok();
    if let Some(c) = crate::suzume::a1::A1Config::builtin(&a1) {
        if !rest.is_empty() {
            return Err(PyValueError::new_err(format!("{a1} takes only aggression / call_tendency / riichi_tendency")));
        }
        return crate::suzume::a1::A1Config::new(c.cfg, c.models, None, delta, call_tendency, riichi_tendency)
            .map_err(PyValueError::new_err);
    }
    let pmf: Option<String> = rest.get_item("pmf_fixed")?.map(|v| v.str().map(|s| s.to_string())).transpose()?;
    rest.del_item("pmf_fixed").ok();
    let models = crate::suzume::a1::A1Models::load(&a1).map_err(PyValueError::new_err)?;
    crate::suzume::a1::A1Config::new(suzume_config_from(&rest)?, models, pmf.as_deref(), delta, call_tendency, riichi_tendency)
        .map_err(PyValueError::new_err)
}

/// An suzume config from an option dict (each value read as its string
/// form, as `SuzumeConfig::set` takes it).
fn suzume_config_from(knobs: &Bound<'_, PyDict>) -> PyResult<crate::suzume::SuzumeConfig> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (k, v) in knobs.iter() {
        pairs.push((k.extract()?, v.str()?.to_string()));
    }
    crate::suzume::SuzumeConfig::from_pairs(pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// The legacy suzume model sets' player names (`suzume::params::ModelSet`).
#[pyfunction]
fn suzume_model_sets() -> Vec<&'static str> {
    crate::suzume::params::ModelSet::ALL.iter().map(|s| s.name()).collect()
}

/// The built-in A1 sets' player names (`suzume::a1::BUILTIN`), the default `suzume` first.
#[pyfunction]
fn a1_model_sets() -> Vec<&'static str> {
    crate::suzume::a1::BUILTIN.iter().map(|b| b.name).collect()
}

/// Every suzume rule in registry order, as dicts: name, mode, decision,
/// description.
#[pyfunction]
fn suzume_rules<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty_bound(py);
    for rule in crate::suzume::registry::RULES {
        let info = rule.info();
        let d = PyDict::new_bound(py);
        d.set_item("name", info.name)?;
        d.set_item("mode", info.mode.as_str())?;
        d.set_item("decision", info.decision.as_str())?;
        d.set_item("description", info.description)?;
        out.append(d)?;
    }
    Ok(out)
}

/// The effective suzume options of an option dict (raises on an unknown
/// option or a bad value): kan, call, riichi, fold, fold_shanten, pf.
#[pyfunction]
fn suzume_config<'py>(py: Python<'py>, knobs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let cfg = suzume_config_from(knobs)?;
    let d = PyDict::new_bound(py);
    d.set_item("kan", cfg.kan)?;
    d.set_item("call", cfg.call)?;
    d.set_item("riichi", cfg.riichi)?;
    d.set_item("fold", cfg.fold)?;
    d.set_item("fold_shanten", cfg.fold_shanten)?;
    d.set_item("pf", cfg.pf)?;
    Ok(d)
}

/// One decision of an in-loop bot on a position, as the arena's loop makes
/// it (`rulebot::decide::decide` / `suzume::decide_observation`): `kind`
/// "rulebot" or "suzume" with its config dict (a rulebot's knobs; suzume'
/// options and `models`), the seat's serialized observation, the mjai events
/// of the kyoku so far (from its `start_kyoku`, one JSON line each; they
/// also give the live wall, `observation_with_wall`). A suzume bot starts
/// from an empty kyoku memo (`SuzumeSeat` keeps one).
///
/// Returns {"action": (action_type, tile, consume_tiles)}, plus "rule" (the
/// deciding suzume rule; `a1` for an A1 bot) or, with `debug`, "debug" (the
/// rulebot's branch inputs).
#[pyfunction]
#[pyo3(signature = (kind, config, obs_b64, kyoku_events, debug=false))]
fn decide_one<'py>(
    py: Python<'py>,
    kind: &str,
    config: &Bound<'py, PyDict>,
    obs_b64: String,
    kyoku_events: Vec<String>,
    debug: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::arena::Bot;
    use pyo3::exceptions::PyValueError;
    let bot = bot_from(kind, config)?;
    let (action, rule, dbg) = py
        .allow_threads(|| -> Result<_, String> {
            let events: Vec<serde_json::Value> = kyoku_events
                .iter()
                .map(|l| serde_json::from_str(l).map_err(|e| format!("bad event: {e}")))
                .collect::<Result<_, _>>()?;
            let obs = observation_with_wall(&obs_b64, &events)?;
            match &bot {
                Bot::Rulebot(cfg) => {
                    let (a, d) = crate::rulebot::decide::decide(&obs, &events, cfg, debug)?;
                    Ok((a, None, d))
                }
                Bot::Suzume(cfg) => {
                    let mut memo = crate::suzume::SeatMemo::default();
                    let (a, r) = crate::suzume::decide_observation(&obs, &events, cfg, &mut memo)?;
                    Ok((a, Some(crate::suzume::decider_name(r)), None))
                }
                Bot::Tsumogiri => Ok((crate::arena::tsumogiri(&obs)?, None, None)),
                Bot::A1(a1) => {
                    let mut memo = crate::suzume::SeatMemo::default();
                    Ok((crate::suzume::a1::decide(&obs, &events, a1, &mut memo)?, Some("a1"), None))
                }
            }
        })
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new_bound(py);
    out.set_item("action", (action.action_type as u8, action.tile, action.consume_tiles.clone()))?;
    if let Some(r) = rule {
        out.set_item("rule", r)?;
    }
    if let Some(d) = dbg {
        out.set_item("debug", d)?;
    }
    Ok(out)
}

/// The live wall at a kyoku's start (riichienv's `drawable_count` then).
const LIVE_WALL: usize = 70;

/// An observation from its base64 JSON, with the fork's two fields set from
/// the kyoku's events: a stock riichienv `Observation` (one that went through
/// Python's riichienv, or an outside server's) has no `drawable_count` and no
/// `last_discarder`. The live wall is 70 less the kyoku's draws (rinshan
/// included, as the engine counts them) and the last discarder is the actor
/// of the kyoku's last `dahai`; both equal the engine's own values.
fn observation_with_wall(
    obs_b64: &str,
    events: &[serde_json::Value],
) -> Result<riichienv_core::observation::Observation, String> {
    let mut obs = riichienv_core::observation::Observation::deserialize_from_base64(obs_b64)
        .map_err(|e| format!("obs deserialize failed: {e:?}"))?;
    let kind = |e: &serde_json::Value, t: &str| e.get("type").and_then(serde_json::Value::as_str) == Some(t);
    let draws = events.iter().filter(|e| kind(e, "tsumo")).count();
    obs.drawable_count = LIVE_WALL.saturating_sub(draws) as u8;
    obs.last_discarder = events
        .iter()
        .rev()
        .find(|e| kind(e, "dahai"))
        .and_then(|e| e.get("actor")?.as_u64())
        .map(|a| a as u8);
    Ok(obs)
}

/// One suzume seat played outside the arena: the bot and its kyoku memo.
///
/// `decide` is `decide_one("suzume", ...)` with the memo kept between the
/// seat's decisions (the riichi tile, the push / fold history), as the arena
/// keeps it per seat; the memo starts over when the kyoku's `start_kyoku`
/// line changes.
#[pyclass(module = "yamawake._core")]
struct SuzumeSeat {
    cfg: crate::suzume::SuzumeConfig,
    /// An A1 bot (a config with `"a1"`, `a1_config_from`): it decides instead of `cfg`.
    a1: Option<crate::suzume::a1::A1Config>,
    memo: crate::suzume::SeatMemo,
    kyoku: Option<String>,
}

#[pymethods]
impl SuzumeSeat {
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        if config.contains("a1")? {
            let a1 = a1_config_from(config)?;
            return Ok(Self { cfg: a1.cfg, a1: Some(a1), memo: Default::default(), kyoku: None });
        }
        Ok(Self { cfg: suzume_config_from(config)?, a1: None, memo: Default::default(), kyoku: None })
    }

    /// `(action_type, tile, consume_tiles, rule)` for the seat's observation
    /// (`serialize_to_base64()`) and the kyoku's mjai lines so far.
    fn decide(
        &mut self,
        py: Python<'_>,
        obs_b64: String,
        kyoku_events: Vec<String>,
    ) -> PyResult<(u8, Option<u8>, Vec<u8>, &'static str)> {
        let start = kyoku_events.first().filter(|l| l.contains("\"start_kyoku\"")).cloned();
        if start.is_some() && start != self.kyoku {
            self.memo.clear();
            self.kyoku = start;
        }
        let (cfg, a1, memo) = (&self.cfg, &self.a1, &mut self.memo);
        let (action, rule) = py
            .allow_threads(|| -> Result<_, String> {
                let events: Vec<serde_json::Value> = kyoku_events
                    .iter()
                    .map(|l| serde_json::from_str(l).map_err(|e| format!("bad event: {e}")))
                    .collect::<Result<_, _>>()?;
                let obs = observation_with_wall(&obs_b64, &events)?;
                match a1 {
                    Some(a1) => Ok((crate::suzume::a1::decide(&obs, &events, a1, memo)?, "a1")),
                    None => crate::suzume::decide_observation(&obs, &events, cfg, memo)
                        .map(|(a, r)| (a, crate::suzume::decider_name(r))),
                }
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((action.action_type as u8, action.tile, action.consume_tiles.clone(), rule))
    }

    /// An A1 seat's P / M / F state probabilities on this decision (`a1::state_probs`; the memo kept as
    /// `decide` keeps it); None without `state.json` or off an own-turn discard out of riichi.
    fn a1_state_probs(&mut self, py: Python<'_>, obs_b64: String, kyoku_events: Vec<String>) -> PyResult<Option<[f64; 3]>> {
        let Some(a1) = self.a1 else {
            return Err(pyo3::exceptions::PyValueError::new_err("not an A1 seat"));
        };
        let start = kyoku_events.first().filter(|l| l.contains("\"start_kyoku\"")).cloned();
        if start.is_some() && start != self.kyoku {
            self.memo.clear();
            self.kyoku = start;
        }
        let memo = &mut self.memo;
        py.allow_threads(|| -> Result<_, String> {
            let events: Vec<serde_json::Value> = kyoku_events
                .iter()
                .map(|l| serde_json::from_str(l).map_err(|e| format!("bad event: {e}")))
                .collect::<Result<_, _>>()?;
            let obs = observation_with_wall(&obs_b64, &events)?;
            crate::suzume::a1::state_probs(&obs, &events, &a1, memo)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Forget the kyoku memo (the next decision starts a fresh kyoku).
    fn reset(&mut self) {
        self.memo.clear();
        self.kyoku = None;
    }
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sp, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_rules, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_model_sets, m)?)?;
    m.add_function(wrap_pyfunction!(a1_model_sets, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_config, m)?)?;
    m.add_function(wrap_pyfunction!(decide_one, m)?)?;
    m.add_class::<SelfPlayLoop>()?;
    m.add_class::<SuzumeSeat>()?;
    m.add_class::<features::SuzumeFeatures>()?;
    // The state model's expert input names (`suzume::pf`; 2026-10-23), for suzume-train's rows.
    m.add("PMF_TILE", crate::suzume::pf::PMF_TILE.to_vec())?;
    m.add("PMF_SUMMARY", crate::suzume::pf::PMF_SUMMARY.to_vec())?;
    Ok(())
}
