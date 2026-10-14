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

    fn done(&self) -> bool {
        self.inner.slots.iter().all(|s| s.state.is_done)
    }
}

/// An in-loop bot from its kind ("rulebot" / "suzume") and config dict.
fn bot_from(kind: &str, config: &Bound<'_, PyDict>) -> PyResult<crate::arena::Bot> {
    match kind {
        "rulebot" => Ok(crate::arena::Bot::Rulebot(rulebot_config_from(config)?)),
        "suzume" => Ok(crate::arena::Bot::Suzume(suzume_config_from(config)?)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!("no in-loop bot {kind:?} (rulebot, suzume)"))),
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

/// The suzume model sets' player names (`suzume::params::ModelSet`), the
/// default first; `suzume` also names the default.
#[pyfunction]
fn suzume_model_sets() -> Vec<&'static str> {
    crate::suzume::params::ModelSet::ALL.iter().map(|s| s.name()).collect()
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
/// of the kyoku so far (from its `start_kyoku`, one JSON line each). A
/// suzume bot starts from an empty kyoku memo.
///
/// Returns {"action": (action_type, tile, consume_tiles)}, plus "rule" (the
/// deciding suzume rule) or, with `debug`, "debug" (the rulebot's branch
/// inputs).
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
            let obs = riichienv_core::observation::Observation::deserialize_from_base64(&obs_b64)
                .map_err(|e| format!("obs deserialize failed: {e:?}"))?;
            let events: Vec<serde_json::Value> = kyoku_events
                .iter()
                .map(|l| serde_json::from_str(l).map_err(|e| format!("bad event: {e}")))
                .collect::<Result<_, _>>()?;
            match &bot {
                Bot::Rulebot(cfg) => {
                    let (a, d) = crate::rulebot::decide::decide(&obs, &events, cfg, debug)?;
                    Ok((a, None, d))
                }
                Bot::Suzume(cfg) => {
                    let mut memo = crate::suzume::SeatMemo::default();
                    let (a, r) = crate::suzume::decide_observation(&obs, &events, cfg, &mut memo)?;
                    Ok((a, Some(r), None))
                }
            }
        })
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new_bound(py);
    out.set_item("action", (action.action_type as u8, action.tile, action.consume_tiles.clone()))?;
    if let Some(r) = rule {
        out.set_item("rule", crate::suzume::decider_name(r))?;
    }
    if let Some(d) = dbg {
        out.set_item("debug", d)?;
    }
    Ok(out)
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sp, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_rules, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_model_sets, m)?)?;
    m.add_function(wrap_pyfunction!(suzume_config, m)?)?;
    m.add_function(wrap_pyfunction!(decide_one, m)?)?;
    m.add_class::<SelfPlayLoop>()?;
    Ok(())
}
