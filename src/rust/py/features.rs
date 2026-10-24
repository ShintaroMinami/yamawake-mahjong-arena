//! Feature dumps of suzume's decisions, for fitting models: the universal
//! rows (`suzume::universal`), the push models' scores, the kan decision's
//! rows and the fold features against every other seat
//! (`suzume::fold_model`), each from one seat's observation and the kyoku's
//! mjai events, on the view a suzume decision plays on
//! (`suzume::view_of`). Nothing here is used in play.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::suzume::ctx::Ctx;
use crate::suzume::fold_model::{FEATURES, FoldInputs, features, seat_features};
use crate::suzume::rules::universal_mlp::mlp_choice;
use crate::suzume::universal::{UContext, Universal, kan_universal, kind_of, riichi_keep_row, u_names};
use crate::suzume::view::river_from_mjai;
use crate::suzume::{MjView, SeatMemo, SuzumeConfig, pf, view_of};

use super::{observation_with_wall, suzume_config_from};

/// The feature dumps of one model set and option dict (`config`, as a
/// suzume player's; `models` may name a directory set, `dir:<path>`).
///
/// Each method takes a seat's observation (`serialize_to_base64()`) and the
/// kyoku's mjai lines so far (from its `start_kyoku`). The DP's caches are
/// kept per seat within a kyoku, as a suzume seat keeps them; they do not
/// change any value. Call in game order for speed; any order gives the same
/// values.
#[pyclass(module = "yamawake._core")]
pub struct SuzumeFeatures {
    cfg: SuzumeConfig,
    /// The A1 set of `state_inputs` (config key `"a1"`: an A1 model directory with P / M / F experts).
    a1: Option<&'static crate::suzume::a1::A1Models>,
    /// Each seat's memo and the `start_kyoku` line it belongs to.
    memos: [(SeatMemo, Option<String>); 4],
}

/// The decision of one call: its view and the seat's memo, lent to a context.
struct Prepared {
    view: MjView,
    seat: usize,
}

impl SuzumeFeatures {
    fn prepare(&mut self, obs_b64: &str, kyoku_events: &[String], river: bool) -> Result<Prepared, String> {
        let events: Vec<serde_json::Value> = kyoku_events
            .iter()
            .map(|l| serde_json::from_str(l).map_err(|e| format!("bad event: {e}")))
            .collect::<Result<_, _>>()?;
        let obs = observation_with_wall(obs_b64, &events)?;
        let seat = usize::from(obs.player_id);
        let start = kyoku_events.first().filter(|l| l.contains("\"start_kyoku\"")).cloned();
        let (memo, kyoku) = &mut self.memos[seat];
        if start.is_none() || start != *kyoku {
            memo.clear();
            *kyoku = start;
        }
        Ok(Prepared { view: view_of(&obs, &events, &self.cfg, river)?, seat })
    }

    /// Runs `f` on a context of the prepared decision with the seat's memo.
    fn with_ctx<T>(&mut self, p: &Prepared, f: impl FnOnce(&Ctx) -> T) -> T {
        let memo = std::mem::take(&mut self.memos[p.seat].0);
        let ctx = Ctx::with_memo(&p.view, &self.cfg, memo);
        let out = f(&ctx);
        self.memos[p.seat].0 = ctx.into_memo();
        out
    }
}

fn context_dict<'py>(py: Python<'py>, c: &UContext, view: &MjView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("turn", c.turn)?;
    d.set_item("progress", c.progress)?;
    d.set_item("round_bin", c.round_bin)?;
    d.set_item("top", c.top)?;
    d.set_item("oya", c.oya)?;
    d.set_item("shanten_ref", c.shanten_ref)?;
    d.set_item("opp_riichi", view.riichi_threats().count())?;
    Ok(d)
}

fn universal_dict<'py>(py: Python<'py>, u: &Universal, view: &MjView) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("names", u_names())?;
    out.set_item("options", u.options.clone())?;
    out.set_item("rows", u.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
    out.set_item("state", u.state.clone())?;
    out.set_item("context", context_dict(py, &u.context, view)?)?;
    out.set_item("sit", pf::situation(view).to_vec())?;
    // own minus each other seat's score / 25000, from the shimocha (the A1 `seat:score_diff` columns)
    out.set_item("score_diff", crate::suzume::a1::score_diffs(view).to_vec())?;
    Ok(out)
}

#[pymethods]
impl SuzumeFeatures {
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let rest = config.copy()?;
        let a1: Option<String> = rest.get_item("a1")?.map(|v| v.extract()).transpose()?;
        rest.del_item("a1").ok();
        let a1 = a1.map(|d| crate::suzume::a1::A1Models::load(&d).map_err(PyValueError::new_err)).transpose()?;
        Ok(Self { cfg: suzume_config_from(&rest)?, a1, memos: Default::default() })
    }

    /// The P / M / F state model's inputs on an own-turn discard decision (`a1::state_inputs`, the code path of
    /// play; the config needs `"a1"`, the A1 set whose experts are read): as `fold_features` ("names", "seats",
    /// "features" [seat][tile type][feature], "legal", "sit"), plus "pmf_tile_names" (`pf::PMF_TILE`), "pmf_tile"
    /// [tile type][4], "pmf_summary_names" (`pf::PMF_SUMMARY`), "pmf_summary" and "experts" (whether the
    /// decision's kind has experts; else the pmf values are zeros). None without a legal discard.
    fn state_inputs<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(md) = self.a1 else { return Err(PyValueError::new_err("state_inputs needs the config key \"a1\"")) };
        let events: Vec<serde_json::Value> = kyoku_events
            .iter()
            .map(|l| serde_json::from_str(l).map_err(|e| PyValueError::new_err(format!("bad event: {e}"))))
            .collect::<Result<_, _>>()?;
        let p = self.prepare(&obs_b64, &kyoku_events, false).map_err(PyValueError::new_err)?;
        let r = self.with_ctx(&p, |ctx| crate::suzume::a1::state_inputs(ctx, md, &events)).map_err(PyValueError::new_err)?;
        let Some(x) = r else { return Ok(None) };
        let out = PyDict::new_bound(py);
        out.set_item("names", FEATURES.to_vec())?;
        out.set_item("seats", vec![1, 2, 3])?;
        out.set_item("features", x.feats.map(|(_, f)| f.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).to_vec())?;
        out.set_item("legal", x.legal)?;
        out.set_item("sit", x.sit.to_vec())?;
        out.set_item("pmf_tile_names", crate::suzume::pf::PMF_TILE.to_vec())?;
        out.set_item("pmf_tile", x.tile.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
        out.set_item("pmf_summary_names", crate::suzume::pf::PMF_SUMMARY.to_vec())?;
        out.set_item("pmf_summary", x.summary.to_vec())?;
        out.set_item("experts", x.experts)?;
        Ok(Some(out))
    }

    /// The decision's universal rows (`universal::universal`; `change`
    /// computes `A_change_yaku` / `A_change_wait`): {"names", "decision_kind"
    /// (0 discard, 1 riichi, 2 tenpai, 3 call; None out of the models'
    /// scope), "options": [(kind, action id)], "rows", "state": (name, id),
    /// "context", "sit" (`pf::situation`, `pf::SIT_NAMES`)}.
    #[pyo3(signature = (obs_b64, kyoku_events, change=false))]
    fn universal<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>, change: bool) -> PyResult<Bound<'py, PyDict>> {
        let p = self.prepare(&obs_b64, &kyoku_events, false).map_err(PyValueError::new_err)?;
        let (u, kind) = self.with_ctx(&p, |ctx| (ctx.universal(change), kind_of(ctx)));
        let out = universal_dict(py, &u, &p.view)?;
        out.set_item("decision_kind", kind)?;
        Ok(out)
    }

    /// The push models' scoring of the decision (`universal_mlp::mlp_choice`):
    /// {"model": the model's kind, "options", "scores", "choice" (an index
    /// into options)}, or None when no model applies.
    fn universal_mlp<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let p = self.prepare(&obs_b64, &kyoku_events, false).map_err(PyValueError::new_err)?;
        let r = self.with_ctx(&p, |ctx| mlp_choice(ctx).map(|(md, u, scores, best)| (md.kind.clone(), u.options.clone(), scores, best)));
        let Some((kind, options, scores, best)) = r else { return Ok(None) };
        let out = PyDict::new_bound(py);
        out.set_item("model", kind)?;
        out.set_item("options", options)?;
        out.set_item("scores", scores)?;
        out.set_item("choice", best)?;
        Ok(Some(out))
    }

    /// The kan decision's rows (`universal::kan_universal`) when an ankan or
    /// kakan is legal: as `universal`, with `keep` first (the option the
    /// decision's push model picks; in riichi the tsumogiri) and one option
    /// per kan (id 42 + tile type); None without a legal kan or a keep option.
    fn universal_kan<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let p = self.prepare(&obs_b64, &kyoku_events, false).map_err(PyValueError::new_err)?;
        let u = self.with_ctx(&p, |ctx| {
            let base = if ctx.view.in_riichi {
                riichi_keep_row(ctx)
            } else {
                mlp_choice(ctx).map(|(_, u, _, best)| (u.options[best], u.rows[best]))
            };
            base.and_then(|(b, row)| kan_universal(ctx, b, row))
        });
        u.map(|u| universal_dict(py, &u, &p.view)).transpose()
    }

    /// The fold features of an own-turn discard decision against every other
    /// seat (`fold_model::seat_features`, the river applied): {"names":
    /// `FEATURES`, "seats": [relative seat 1..3, from the shimocha],
    /// "features": [seat][tile type][feature] (raw), "legal": the legal
    /// discard types, "shanten_after": [tile type] (None off the legal
    /// ones), "sit"}; None without a legal discard, unless `any_decision` (then
    /// any decision: the features of every tile type, with "legal" empty and the
    /// shanten columns 0 when no discard is legal, e.g. for the seat columns of a
    /// call decision). `hand_mlp` is the set's push model's; the tables are the
    /// set's (`params::Models::tables`).
    #[pyo3(signature = (obs_b64, kyoku_events, any_decision=false))]
    fn fold_features<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>, any_decision: bool) -> PyResult<Option<Bound<'py, PyDict>>> {
        if crate::suzume::params::set_models(self.cfg.models).tables.is_none() {
            return Err(PyValueError::new_err(format!(
                "model set {} has no fold tables (discard_F.json or fold_tables.json)",
                self.cfg.models.name()
            )));
        }
        let p = self.prepare(&obs_b64, &kyoku_events, true).map_err(PyValueError::new_err)?;
        let r = self.with_ctx(&p, |ctx| {
            if ctx.discard_cands().is_empty() && !any_decision {
                return None;
            }
            let shanten = FoldInputs::of(ctx).shanten_after;
            let feats = seat_features(ctx);
            Some((feats.map(|(_, f)| f.iter().map(|r| r.to_vec()).collect::<Vec<_>>()), pf::legal_types(ctx), shanten))
        });
        let Some((feats, legal, shanten)) = r else { return Ok(None) };
        let out = PyDict::new_bound(py);
        out.set_item("names", FEATURES.to_vec())?;
        out.set_item("seats", vec![1, 2, 3])?;
        out.set_item("features", feats.to_vec())?;
        out.set_item("legal", legal)?;
        out.set_item("shanten_after", shanten.to_vec())?;
        out.set_item("sit", pf::situation(&p.view).to_vec())?;
        Ok(Some(out))
    }

    /// The fold features of an own-turn discard decision with no other seat
    /// in riichi, against each other seat as if it had declared riichi on its
    /// latest discard and the riichi were accepted now (the hypothetical riichi
    /// of the first push / fold labels): the observation's `riichi_declared`
    /// set for that seat (nothing else changed) and that discard flagged as
    /// the declaring one. The features are those of the river cut right after
    /// that discard, but `last_ago` and `other_wait_est` those of the whole
    /// river. {"seats": [relative seat of each scored seat, 1..3; the seats
    /// with a discard], "features": [seat][tile type][feature], "legal",
    /// "shanten_after"}; None without a legal discard, an error when some
    /// other seat is in riichi.
    fn fold_features_hypo<'py>(&mut self, py: Python<'py>, obs_b64: String, kyoku_events: Vec<String>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if crate::suzume::params::set_models(self.cfg.models).tables.is_none() {
            return Err(PyValueError::new_err(format!("model set {} has no fold tables", self.cfg.models.name())));
        }
        let events: Vec<serde_json::Value> = kyoku_events
            .iter()
            .map(|l| serde_json::from_str(l).map_err(|e| PyValueError::new_err(format!("bad event: {e}"))))
            .collect::<Result<_, _>>()?;
        let obs = observation_with_wall(&obs_b64, &events).map_err(PyValueError::new_err)?;
        let seat = usize::from(obs.player_id);
        if (0..4).any(|s| s != seat && obs.riichi_declared[s]) {
            return Err(PyValueError::new_err("another seat is in riichi: use fold_features"));
        }
        let rows = river_from_mjai(&events).map_err(PyValueError::new_err)?;
        let legal = obs.legal_actions_method();
        const FROM_FULL: [usize; 2] = [6, 15];          // other_wait_est, last_ago
        let mut seats = Vec::new();
        let mut feats = Vec::new();
        let mut legal_types = None;
        for k in 1..=3usize {
            let s = (seat + k) % 4;
            let Some(last) = rows.iter().rposition(|r| usize::from(r.seat) == s) else { continue };
            let mut o = obs.clone();
            o.riichi_declared[s] = true;
            let mut flagged = rows.clone();
            flagged[last].riichi = true;
            let mut one = |river: &[crate::suzume::view::RiverRow]| {
                let mut view = MjView::from_observation(&o, legal.clone());
                view.apply_river(river);
                let p = Prepared { view, seat };
                self.with_ctx(&p, |ctx| {
                    if ctx.discard_cands().is_empty() {
                        return None;
                    }
                    let inp = FoldInputs::of(ctx);
                    Some((features(ctx.models.tables(), ctx.view, s, &inp), pf::legal_types(ctx), inp.shanten_after))
                })
            };
            let Some((trunc, lt, sh)) = one(&flagged[..=last]) else { return Ok(None) };
            let Some((full, _, _)) = one(&flagged) else { return Ok(None) };
            let merged: Vec<Vec<f64>> = (0..34)
                .map(|t| {
                    let mut r = trunc[t].to_vec();
                    for c in FROM_FULL {
                        r[c] = full[t][c];
                    }
                    r
                })
                .collect();
            seats.push(k);
            feats.push(merged);
            legal_types.get_or_insert((lt, sh));
        }
        let out = PyDict::new_bound(py);
        let (lt, sh) = match legal_types {
            Some(x) => x,
            None => {
                // No other seat has discarded yet: the legal types alone.
                let view = view_of(&obs, &events, &self.cfg, false).map_err(PyValueError::new_err)?;
                let p = Prepared { view, seat };
                match self.with_ctx(&p, |ctx| (!ctx.discard_cands().is_empty()).then(|| (pf::legal_types(ctx), FoldInputs::of(ctx).shanten_after))) {
                    Some(x) => x,
                    None => return Ok(None),
                }
            }
        };
        out.set_item("names", FEATURES.to_vec())?;
        out.set_item("seats", seats)?;
        out.set_item("features", feats)?;
        out.set_item("legal", lt)?;
        out.set_item("shanten_after", sh.to_vec())?;
        Ok(Some(out))
    }
}
