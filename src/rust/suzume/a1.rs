//! A1: suzume without the push / mawashi / fold split (one MLP per decision
//! kind, fitted on every position of the teacher's games), decided in-loop.
//! Since 2026-10-21 the player `suzume` is an A1 set (`BUILTIN`,
//! `params/suzume/`); the earlier suzume plays on as the legacy sets
//! (`legacy-suzume-mlp3` ...).
//!
//! A Rust port of suzume-train's Python `A1Agent` (`suzume_train/a1.py`); it
//! plays the same moves. It is its own bot (`arena::Bot::A1`): nothing here
//! changes the legacy sets.
//!
//! The rows are the universal rows (`universal::universal`, `change` on) and
//! 114 danger columns (`a1_names`): for a discard-like option (discard,
//! dama, break, riichi) the fold features of its tile type against the three
//! other seats from the shimocha (`fold_model::features`, the river applied,
//! `hand_mlp` left out) and `seat:score_diff` (own minus that seat's score,
//! / 25000; 2026-10-19); for another option (pass, chi, pon, daiminkan) only
//! the `seat:*` columns; for a kan decision none. The models read their
//! inputs by name, so a model fitted before `seat:score_diff` plays as
//! before. The decision order:
//!
//! 1. a win (tsumo / ron) or kyushu kyuhai whenever legal;
//! 2. right after its own riichi, the discard the riichi option named;
//! 3. the kan decision (ankan / kakan legal): the kan model compares `keep`
//!    (this decision's own choice) with each kan;
//! 4. the per-decision model's best option (ties keep the earlier);
//! 5. the only legal discard tile, else pass, else the drawn tile, else the
//!    first legal discard, else the first legal action.
//!
//! The context (`cfg`) is a suzume config on the features set (the fold
//! tables, and the push models whose choice the kan rows are first built
//! with, as the Python player's `SuzumeFeatures`).

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use riichienv_core::action::{Action, ActionType};
use serde_json::Value;

use super::config::SuzumeConfig;
use super::ctx::Ctx;
use super::fold_model::{FEATURES, FoldInputs, features};
use super::memo::SeatMemo;
use super::mlp::{Hidden, Sit, add_bias, parse_hidden, parse_sit};
use super::params::{num, nums, positions, strs};
use super::rules::universal_mlp::mlp_choice;
use super::universal::{N_U, kan_universal, kind_of, riichi_keep_row, u_names};
use super::view::{MjView, river_from_mjai};
use super::view_of;

/// The fold features of the danger columns (`foldall.INPUTS` without `hand_mlp`).
const DANGER: [&str; 37] = [
    "patterns_log", "turn", "dshanten", "honor", "riichi_yaku", "safe_sum", "other_wait_est", "ryanmen_log",
    "tanki_unseen", "held", "outside_28", "outside_37", "outside_46", "decl_outside_suji", "last_ago",
    "patterns:kanchan", "patterns:penchan", "patterns:shanpon", "order:others_safe", "supply:safe_left",
    "supply:enough", "tile_kind:dora", "tile_kind:terminal", "tile_kind:two_eight", "genbutsu:genbutsu",
    "shanten:best_shanten", "seat:riichi", "seat:n_melds", "seat:meld_dora", "seat:dealer", "seat:n_discards",
    "seat:yakuhai_melds", "seat:meld_flush", "seat:meld_tanyao", "seat:meld_chanta", "suit:discards", "suit:first",
];
/// The danger columns not among the fold features: own minus that seat's score / 25000 (2026-10-19).
const EXTRA: [&str; 1] = ["seat:score_diff"];
const N_PER: usize = DANGER.len() + EXTRA.len();
const N_D: usize = 3 * N_PER;
const N_ROW: usize = N_U + N_D;
const RED_TIDS: [u8; 3] = [16, 52, 88];

/// Every column an A1 model may read: the universal names, then `o<k>:<feature>` (k = 1..3 from the shimocha).
fn a1_names() -> &'static [String] {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| {
        let mut v = u_names().to_vec();
        for k in 1..=3 {
            v.extend(DANGER.iter().chain(EXTRA.iter()).map(|n| format!("o{k}:{n}")));
        }
        v
    })
}

/// The `FEATURES` index of each danger feature.
fn danger_idx() -> &'static [usize; 37] {
    static I: OnceLock<[usize; 37]> = OnceLock::new();
    I.get_or_init(|| std::array::from_fn(|j| FEATURES.iter().position(|f| *f == DANGER[j]).expect("danger feature")))
}

/// One A1 model file: `w2 . relu(... relu(W1 [x; state one-hot; sit] + b1) ...) + b2`, x (and the
/// situation inputs `sit`, when the file has `sit_inputs`: the hanchan models, 2026-10-19) standardized.
struct Model {
    cols: Vec<usize>,
    states: Vec<String>,
    sit: Option<Sit>,
    mean: Vec<f64>,
    std: Vec<f64>,
    net: Hidden,
    w2: Vec<f64>,
    b2: f64,
}

impl Model {
    /// The model in JSON `text`; `label` names it in errors.
    fn parse(text: &str, label: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("{label}: {e}"))?;
        let at = |e: String| format!("{label}: {e}");
        let (inputs, states) = (strs(&v, &["inputs"]).map_err(at)?, strs(&v, &["state_onehot_order"]).map_err(at)?);
        let n_own = inputs.len() + states.len();
        let sit = parse_sit(&v, n_own).map_err(at)?;
        let n_sit = sit.as_ref().map_or(0, |s| s.mean.len());
        let md = Self {
            cols: positions(&inputs, a1_names()).map_err(at)?,
            net: parse_hidden(&v, n_own + n_sit).map_err(at)?,
            sit,
            states,
            mean: nums(&v, &["mean"]).map_err(at)?,
            std: nums(&v, &["std"]).map_err(at)?,
            w2: nums(&v, &["w2"]).map_err(at)?,
            b2: num(&v, &["b2"]).map_err(at)?,
        };
        if md.w2.len() != md.net.out_width() || md.mean.len() != inputs.len() || md.std.len() != inputs.len() {
            return Err(format!("{label}: w2 / mean / std do not match the network"));
        }
        Ok(md)
    }

    /// The one-hot index of state `name` (`<name>/opp` or `<name>/noopp` first, then `name`; as the Python player).
    fn state_index(&self, name: &str, opp: bool) -> Option<usize> {
        let full = format!("{name}{}", if opp { "/opp" } else { "/noopp" });
        self.states.iter().position(|s| *s == full).or_else(|| self.states.iter().position(|s| s == name))
    }

    fn score(&self, row: &[f64; N_ROW], state: Option<usize>, sit: &[f64]) -> f64 {
        let n_in = self.cols.len();
        let x: Vec<f64> = self.cols.iter().enumerate().map(|(j, &c)| (row[c] - self.mean[j]) / self.std[j]).collect();
        let mut z = self.net.pre_dense(&x);
        add_bias(&mut z, sit);
        if let Some(si) = state {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk += self.net.w1_row(k)[n_in + si];
            }
        }
        self.b2 + self.w2.iter().zip(self.net.activate(z)).map(|(w, h)| w * h).sum::<f64>()
    }

    /// The index of the best row (the first on a tie), the situation of view `v` added when the model reads it.
    fn best(&self, rows: &[[f64; N_ROW]], state: &str, opp: bool, v: &MjView) -> usize {
        let si = self.state_index(state, opp);
        let sit = self.sit.as_ref().map_or_else(Vec::new, |s| s.bias(&self.net, v));
        let mut best = (0, f64::NEG_INFINITY);
        for (i, r) in rows.iter().enumerate() {
            let s = self.score(r, si, &sit);
            if s > best.1 {
                best = (i, s);
            }
        }
        best.0
    }
}

/// The models of an A1 directory (`discard0`, `discard12`, `discard3`, `call`, `kan`; or the first A1's names:
/// `discard_P_t` + `riichi_P` for discard0, `discard_P_12`, `discard_P_3`, `call_P`, `kan`). Optional `discard1` /
/// `discard2` (2026-10-19, the split of discard12) take the 1- / 2-shanten discards when present.
pub struct A1Models {
    dir: String,
    discard12: Model,
    discard1: Option<Model>,
    discard2: Option<Model>,
    discard3: Model,
    tenpai: Model,
    riichi: Model,
    call: Model,
    kan: Model,
}

impl std::fmt::Debug for A1Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A1Models({})", self.dir)
    }
}

impl A1Models {
    /// The models of directory `dir`, read on first use in this process (the same path again gives the same set).
    pub fn load(dir: &str) -> Result<&'static Self, String> {
        static SETS: OnceLock<Mutex<Vec<&'static A1Models>>> = OnceLock::new();
        let mut sets = SETS.get_or_init(|| Mutex::new(Vec::new())).lock().expect("A1 sets");
        if let Some(m) = sets.iter().find(|m| m.dir == dir) {
            return Ok(m);
        }
        let p = Path::new(dir);
        let read = |name: &str| -> Result<Option<String>, String> {
            let f = p.join(format!("{name}.json"));
            match std::fs::read_to_string(&f) {
                Ok(t) => Ok(Some(t)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(format!("{}: {e}", f.display())),
            }
        };
        let md: &'static Self = Box::leak(Box::new(Self::from_reader(dir, read)?));
        sets.push(md);
        Ok(md)
    }

    /// The models of set `label`, each file's JSON text read by `read` (`None` when absent).
    fn from_reader(label: &str, read: impl Fn(&str) -> Result<Option<String>, String>) -> Result<Self, String> {
        let file = |name: &str| format!("{label}/{name}.json");
        let opt = |name: &str| read(name)?.map(|t| Model::parse(&t, &file(name))).transpose();
        // The A1 names (2026-10-19: discard0 for every tenpai decision, riichi legal or not; discard12, discard3,
        // call, kan), else the first A1's (discard_P_t + riichi_P, discard_P_12, discard_P_3, call_P, kan).
        let m = |names: &[&str]| -> Result<Model, String> {
            for n in names {
                if let Some(md) = opt(n)? {
                    return Ok(md);
                }
            }
            Err(format!("{}: missing", file(names[0])))
        };
        Ok(Self {
            dir: label.to_string(),
            discard12: m(&["discard12", "discard_P_12"])?,
            discard1: opt("discard1")?,
            discard2: opt("discard2")?,
            discard3: m(&["discard3", "discard_P_3"])?,
            tenpai: m(&["discard0", "discard_P_t"])?,
            riichi: m(&["discard0", "riichi_P"])?,
            call: m(&["call", "call_P"])?,
            kan: m(&["kan"])?,
        })
    }

    /// The built-in set of player name `name` (`BUILTIN`), parsed on first use.
    pub fn builtin(name: &str) -> Option<&'static Self> {
        static M: [OnceLock<A1Models>; BUILTIN.len()] = [const { OnceLock::new() }; BUILTIN.len()];
        let i = BUILTIN.iter().position(|b| b.name == name)?;
        let b = &BUILTIN[i];
        Some(M[i].get_or_init(|| {
            let read = |n: &str| Ok(b.files.iter().find(|(f, _)| *f == n).map(|(_, t)| t.to_string()));
            Self::from_reader(&format!("params/suzume/{}", b.dir), read).unwrap_or_else(|e| panic!("{e}"))
        }))
    }
}

/// One built-in A1 set: its player name, its directory under `params/suzume/` and its files (name, JSON text).
pub struct Builtin {
    pub name: &'static str,
    dir: &'static str,
    files: &'static [(&'static str, &'static str)],
}

macro_rules! a1_files {
    ($dir:literal, [$($f:literal),*]) => {
        &[$(($f, include_str!(concat!("../../../params/suzume/", $dir, "/", $f, ".json")))),*]
    };
}

/// The built-in A1 sets, the default (`suzume`) first: the discard1 / discard2 split sets fitted on the
/// Mortal-teacher games (`suzume`) and the rlactor-225-teacher games (`suzume-rla225`), and their discard12
/// counterparts (`-d12`, for reference; 2026-10-21).
pub static BUILTIN: [Builtin; 4] = [
    Builtin {
        name: "suzume",
        dir: "suzume",
        files: a1_files!("suzume", ["discard0", "discard12", "discard1", "discard2", "discard3", "call", "kan"]),
    },
    Builtin {
        name: "suzume-rla225",
        dir: "rla225",
        files: a1_files!("rla225", ["discard0", "discard12", "discard1", "discard2", "discard3", "call", "kan"]),
    },
    Builtin { name: "suzume-d12", dir: "d12", files: a1_files!("d12", ["discard0", "discard12", "discard3", "call", "kan"]) },
    Builtin {
        name: "suzume-rla225-d12",
        dir: "rla225_d12",
        files: a1_files!("rla225_d12", ["discard0", "discard12", "discard3", "call", "kan"]),
    },
];

/// An in-loop A1: its models and the suzume config of its features set.
#[derive(Clone, Copy, Debug)]
pub struct A1Config {
    pub cfg: SuzumeConfig,
    pub models: &'static A1Models,
}

impl A1Config {
    /// The bot of built-in set `name` (`BUILTIN`): its models with the built-in features set
    /// (`params::features_set`, `pf=0`, `fold=0`, as suzume-train's `A1RustAgent` plays a directory set).
    pub fn builtin(name: &str) -> Option<Self> {
        let models = A1Models::builtin(name)?;
        let cfg = SuzumeConfig { pf: false, fold: false, models: super::params::features_set(), ..SuzumeConfig::default() };
        Some(Self { cfg, models })
    }
}

/// The tile type and red flag a discard-like option id names.
fn tile_of(id: u16) -> (u8, bool) {
    if id >= 34 { ([4, 13, 22][usize::from(id - 34)], true) } else { (id as u8, false) }
}

/// The legal action of option `(kind, id)`, the first in legal order (as the Python player's `option_action`).
fn option_action(legal: &[Action], kind: &str, id: u16) -> Option<Action> {
    let red = |t: u8| RED_TIDS.contains(&t);
    let of = |at: ActionType| legal.iter().filter(move |a| a.action_type == at);
    match kind {
        "discard" | "dama" | "break" => {
            let (t, r) = tile_of(id);
            of(ActionType::Discard).find(|a| a.tile.is_some_and(|x| x / 4 == t && red(x) == r)).cloned()
        }
        "riichi" => of(ActionType::Riichi).next().cloned(),
        "pass" => of(ActionType::Pass).next().cloned(),
        "chi" | "pon" => {
            let r = id >= 76;
            let at = if kind == "chi" { ActionType::Chi } else { ActionType::Pon };
            of(at).find(|a| {
                if a.consume_tiles.iter().any(|&c| red(c)) != r {
                    return false;
                }
                if kind == "pon" {
                    return true;
                }
                let called = a.tile.map_or(0, |t| t / 4);
                let pos = a.consume_tiles.iter().filter(|&&c| c / 4 < called).count() as u16;
                id - if r { 76 } else { 38 } == pos
            })
            .cloned()
        }
        "daiminkan" | "ankan" | "kakan" => {
            let at = match kind {
                "daiminkan" => ActionType::Daiminkan,
                "ankan" => ActionType::Ankan,
                _ => ActionType::Kakan,
            };
            of(at).find(|a| a.tile.is_some_and(|t| u16::from(t / 4) + 42 == id)).cloned()
        }
        _ => None,
    }
}

/// Own score minus each other seat's (from the shimocha), / 25000.
pub fn score_diffs(v: &MjView) -> [f64; 3] {
    let s = usize::from(v.seat);
    std::array::from_fn(|k| (f64::from(v.scores[s]) - f64::from(v.scores[(s + k + 1) % 4])) / 25000.0)
}

/// The danger columns of one option from the features `x` [seat][tile type][feature] and the score
/// differences `diff` (`score_diffs`).
fn danger_row(kind: &str, id: u16, x: &[[[f64; super::fold_model::N_FEATURES]; 34]; 3], diff: &[f64; 3]) -> [f64; N_D] {
    let mut out = [0.0; N_D];
    let di = danger_idx();
    let n = DANGER.len();
    match kind {
        "discard" | "dama" | "break" | "riichi" => {
            let t = usize::from(tile_of(id).0);
            for s in 0..3 {
                for j in 0..n {
                    out[s * N_PER + j] = x[s][t][di[j]];
                }
                out[s * N_PER + n] = diff[s];
            }
        }
        "keep" | "ankan" | "kakan" => {}
        _ => {
            for s in 0..3 {
                for j in 0..n {
                    if DANGER[j].starts_with("seat:") {
                        out[s * N_PER + j] = x[s][0][di[j]];
                    }
                }
                out[s * N_PER + n] = diff[s];
            }
        }
    }
    out
}

/// One A1 decision on the seat's observation, the kyoku's events and the seat's kyoku memo.
pub fn decide(
    obs: &riichienv_core::observation::Observation,
    kyoku_events: &[Value],
    a1: &A1Config,
    memo: &mut SeatMemo,
) -> Result<Action, String> {
    let legal = obs.legal_actions_method();
    if legal.is_empty() {
        return Err(format!("A1 at seat {} has no legal actions", obs.player_id));
    }
    for at in [ActionType::Tsumo, ActionType::Ron, ActionType::KyushuKyuhai] {
        if let Some(a) = legal.iter().find(|a| a.action_type == at) {
            return Ok(a.clone());
        }
    }
    if let Some((t, r)) = memo.riichi_tile.take() {
        let id = if r { 34 + u16::from(t / 9) } else { u16::from(t) };
        if let Some(a) = option_action(&legal, "discard", id) {
            return Ok(a);
        }
    }
    let cfg = &a1.cfg;
    let md = a1.models;
    let view = view_of(obs, kyoku_events, cfg, false)?;
    let ctx = Ctx::with_memo(&view, cfg, std::mem::take(memo));
    let opp = view.riichi_threats().next().is_some();
    // This decision's universal options, rows (with the danger columns) and its model's choice.
    let uni = (|| {
        let u = ctx.universal(true);
        if u.options.is_empty() {
            return Ok(None);
        }
        let Some(kind) = kind_of(&ctx) else { return Ok(None) };
        let mut rv = view.clone();
        rv.apply_river(&river_from_mjai(kyoku_events)?);
        // The shanten columns read the hand only, so the decision's own candidates serve.
        let mut shanten_after = [None; 34];
        for c in ctx.discard_cands() {
            shanten_after[usize::from(c.tile_type)] = Some(c.shanten);
        }
        let inp = FoldInputs { shanten_after, hand_mlp: [0.0; 34] };
        let x: [_; 3] = std::array::from_fn(|k| {
            features(ctx.models.tables(), &rv, (usize::from(rv.seat) + k + 1) % 4, &inp)
        });
        let diff = score_diffs(&view);
        let rows: Vec<[f64; N_ROW]> = u
            .options
            .iter()
            .zip(&u.rows)
            .map(|(&(k, id), r)| {
                let mut row = [0.0; N_ROW];
                row[..N_U].copy_from_slice(r);
                row[N_U..].copy_from_slice(&danger_row(k, id, &x, &diff));
                row
            })
            .collect();
        let model = match kind {
            0 if u.context.shanten_ref >= 3 => &md.discard3,
            0 if u.context.shanten_ref == 1 && md.discard1.is_some() => md.discard1.as_ref().unwrap(),
            0 if u.context.shanten_ref == 2 && md.discard2.is_some() => md.discard2.as_ref().unwrap(),
            0 => &md.discard12,
            1 => &md.riichi,
            2 => &md.tenpai,
            _ => &md.call,
        };
        let best = model.best(&rows, &u.state.0, opp, &view);
        Ok::<_, String>(Some((u, best)))
    })();
    let out = (|| {
        let uni = uni?;
        if legal.iter().any(|a| matches!(a.action_type, ActionType::Ankan | ActionType::Kakan)) {
            // The kan rows are built with the features set's keep, then keep becomes A1's own choice (as Python).
            let base = if view.in_riichi {
                riichi_keep_row(&ctx)
            } else {
                mlp_choice(&ctx).map(|(_, u, _, best)| (u.options[best], u.rows[best]))
            };
            if let Some(mut k) = base.and_then(|(b, row)| kan_universal(&ctx, b, row)) {
                if let Some((u, j)) = &uni {
                    k.rows[0] = u.rows[*j];
                    k.options[0] = u.options[*j];
                }
                let rows: Vec<[f64; N_ROW]> = k
                    .rows
                    .iter()
                    .map(|r| {
                        let mut row = [0.0; N_ROW];
                        row[..N_U].copy_from_slice(r);
                        row
                    })
                    .collect();
                let best = md.kan.best(&rows, &k.state.0, opp, &view);
                if best > 0 {
                    let (kind, id) = k.options[best];
                    if let Some(a) = option_action(&legal, kind, id) {
                        return Ok(Some(a));
                    }
                }
            }
        }
        if let Some((u, j)) = &uni {
            let (kind, id) = u.options[*j];
            if let Some(a) = option_action(&legal, kind, id) {
                ctx.memo().riichi_tile = (kind == "riichi").then(|| tile_of(id));
                return Ok(Some(a));
            }
        }
        Ok::<_, String>(None)
    })();
    *memo = ctx.into_memo();
    if let Some(a) = out? {
        return Ok(a);
    }
    let discards: Vec<&Action> = legal.iter().filter(|a| a.action_type == ActionType::Discard).collect();
    let first = discards.first().map(|a| a.tile);
    if !discards.is_empty() && discards.iter().all(|a| a.tile == first.flatten()) {
        return Ok(discards[0].clone());
    }
    if let Some(a) = legal.iter().find(|a| a.action_type == ActionType::Pass) {
        return Ok(a.clone());
    }
    if let Some(a) = discards.iter().find(|a| a.tile == obs.drawn_tile) {
        return Ok((*a).clone());
    }
    Ok(discards.first().map_or_else(|| legal[0].clone(), |a| (*a).clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_set_parses_with_its_features_set() {
        for b in &BUILTIN {
            let c = A1Config::builtin(b.name).expect("a built-in set");
            assert_eq!(c.models.discard1.is_some(), !b.name.ends_with("-d12"), "{}", b.name);
            assert_eq!(c.models.discard2.is_some(), c.models.discard1.is_some(), "{}", b.name);
            assert!(!c.cfg.pf && !c.cfg.fold);
            assert!(c.cfg.check_models().is_ok());
            assert!(super::super::params::set_models(c.cfg.models).tables.is_some());
        }
        assert_eq!(BUILTIN[0].name, "suzume");
        assert!(A1Config::builtin("legacy-suzume-mlp3").is_none());
    }
}
