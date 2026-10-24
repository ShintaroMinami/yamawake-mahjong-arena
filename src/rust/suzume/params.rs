//! The fixed parameters of suzume, read from `params/` at the repository
//! root: the DP's constants and the weights the universal features read
//! (`params/suzume.json`), and the models of each legacy model set
//! (`ModelSet`): a hand model set (`params/legacy/hand/<hand>/`, one file per
//! decision and push / fold state, `MODEL_FILES`) and a push / fold state
//! model (`params/legacy/state/<state>.json`). The current sets
//! (`params/suzume/`) are A1's (`suzume::a1::BUILTIN`); their features set is
//! here (`features_set`).
//!
//! They are not options: the files are embedded at build time and parsed
//! once on first use, so a caller cannot vary them, and the models always
//! run on the values they were fitted with. Changing one means editing the
//! file and rebuilding. A missing or malformed key stops the program on
//! first use. A suzume bot picks its model set by its name (`ModelSet`, in its
//! config); a decision reads it from its context (`Ctx::models`).
//!
//! For fitting new models, a set can also be read from a directory at run
//! time (`ModelSet::Dir`, the name `dir:<path>`; `load_dir` lists its files).

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

use super::mlp::{Hidden, Sit, matrix, parse_hidden, parse_sit, vector};

use super::rules::tile_features::{N_TF, TF_NAMES};
use super::rules::value_features::{N_VF, VF_NAMES};

/// `params/suzume.json`.
#[derive(Debug)]
pub struct Params {
    /// The DP values hands from this shanten (after the best discard) in.
    pub dp_max_shanten: i8,
    /// The DP's scale on the chance an opponent discard is a winning tile.
    pub ron_rate: f32,
    /// Points the riichi stick costs when the hand does not win.
    pub riichi_deposit: f32,
    /// The linear value score's weights, in `VF_NAMES` order.
    pub value_weights: [f32; N_VF],
    /// The value-band tie pick's band (score units).
    pub tie_band: f32,
    /// The value-band tie pick's weights, in `TF_NAMES` order.
    pub tie_weights: [f32; N_TF],
    /// A unique top leads by more than this (the 1-layer set's outside bias).
    pub context_top_margin: i32,
}

/// One per-decision MLP (`params/legacy/hand/<hand>/`, `MODEL_FILES`).
#[derive(Debug)]
pub struct Umlp {
    pub kind: String,
    /// Universal column names, in input order.
    pub inputs: Vec<String>,
    /// The inputs as universal row indices (`universal::u_names`).
    pub cols: Vec<usize>,
    /// The state one-hot order (universal state names).
    pub states: Vec<String>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// The hidden layers; the first reads `inputs.len() + states.len()`
    /// columns.
    pub net: Hidden,
    /// The head: one score from the last layer.
    pub w2: Vec<f64>,
    pub b2: f64,
    /// Outside bias by round bin (`waits::rc_round_bin`), unique top, dealer
    /// (`outside_bias`; zero in a file without it, as in every 3-layer set).
    pub out_round: [f64; 4],
    pub out_top: f64,
    pub out_dealer: f64,
    /// The situation inputs (`sit_inputs`; after the state one-hot), if any.
    pub sit: Option<Sit>,
}

/// The fold model (`params/legacy/hand/<hand>/discard_F.json`; `suzume::fold_model`):
/// one MLP scoring each discard against one riichi seat, and the tables its
/// features read.
#[derive(Debug)]
pub struct FoldMlp {
    /// Feature names (`fold_model::FEATURES`), in input order.
    pub inputs: Vec<String>,
    /// The inputs as `fold_model::FEATURES` indices.
    pub cols: Vec<usize>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// The hidden layers; the first reads `inputs.len()` columns.
    pub net: Hidden,
    /// The head: one score from the last layer.
    pub w2: Vec<f64>,
    pub b2: f64,
    /// The tables and caps its features read.
    pub tables: FoldTables,
    /// The soft minimum's temperature (`"tau"`): a tile type's score is
    /// `-tau ln sum exp(-s / tau)` over its scores against every other seat
    /// (the file must say `"seats": "all"`, `"combine": "softmin"`).
    pub tau: f64,
    /// The situation inputs (`sit_inputs`; after `inputs`), if any.
    pub sit: Option<Sit>,
}

/// The tables and caps the fold features read (`fold_model::features`): a
/// fold-model file's, or a `fold_tables.json` of a directory set.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldTables {
    /// The rate a non-genbutsu tile type is the riichi seat's wait, by
    /// pattern bin (rows) and own-turn bin (columns), row-major.
    pub wait_table: Vec<f64>,
    /// Pattern-count bin edges: the bin is the number of edges below the
    /// count.
    pub pattern_edges: Vec<f64>,
    /// Own-turn bin edges: the bin is the number of edges at or below the
    /// own discards so far.
    pub turn_edges: Vec<u8>,
    /// The `turn` feature is min(own discards, cap) / cap.
    pub turn_cap: f64,
    /// The `last_ago` feature is min(ago, cap) / cap.
    pub last_ago_cap: f64,
    /// A riichi seat's early discards (the outside features): its first
    /// this many.
    pub early_discards: u8,
}

/// The push / fold state model (`params/legacy/state/<state>.json`; `suzume::pf`):
/// each (other seat, legal discard type) row of the fold model's features,
/// next to the situation (`sit`) -> `relu(W1 [(x - mean) / std, sit] + b1)`
/// and any further layers
/// (`mlp::Hidden`), pooled over the rows by the mean and the max ->
/// `W2 [mean, max] + b2`, the P / M / F logits.
///
/// `sit` (`"sit_inputs"`, `pf::SIT_NAMES`; none in a file without them): the
/// decision's round, honba, sticks, rank and score gaps (`pf::situation`),
/// standardized by `sit_mean` / `sit_std`. Since 2026-10-23 (the A1 P / M / F
/// mixture's state model) `inputs` may add `pf::PMF_TILE` names after the fold
/// features and `sit_inputs` `pf::PMF_SUMMARY` names after `SIT_NAMES`: the
/// experts' inputs (`a1::pmf_inputs`), e.g.
/// `"inputs": ["patterns_log", ..., "pmf:p_diff", "pmf:f_diff"]`,
/// `"sit_inputs": ["round_E1", ..., "has_down", "pmf:top_p1:f", "pmf:has2"]`.
#[derive(Debug)]
pub struct StateMlp {
    /// Feature names (`fold_model::FEATURES`, then any `pf::PMF_TILE`), in
    /// input order.
    pub inputs: Vec<String>,
    /// The inputs as `pf::state_input_names` indices (a `PMF_TILE` one at
    /// `N_FEATURES + j`).
    pub cols: Vec<usize>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// The decision-level inputs: 0, or `pf::SIT_NAMES.len()` plus the
    /// summary columns.
    pub n_sit: usize,
    /// The `pf::PMF_SUMMARY` index of each summary column after `SIT_NAMES`.
    pub summary_cols: Vec<usize>,
    pub sit_mean: Vec<f64>,
    pub sit_std: Vec<f64>,
    /// The hidden layers; the first reads `inputs.len() + n_sit`
    /// columns.
    pub net: Hidden,
    /// 3 rows of twice the last layer's width (the mean pool, then the max
    /// pool), row-major.
    pub w2: Vec<f64>,
    pub b2: [f64; 3],
}

impl FoldMlp {
    /// A tile type's score from its scores against the other seats: the
    /// soft minimum `-tau ln sum exp(-s / tau)`.
    pub fn combine(&self, s: impl Iterator<Item = f64>) -> f64 {
        -self.tau * s.map(|x| (-x / self.tau).exp()).sum::<f64>().ln()
    }
}

const PARAMS_JSON: &str = include_str!("../../../params/suzume.json");

/// The push / fold MLPs over the universal features (`pf=1`), in
/// `Models` field order: riichi in the mawashi state (also the fold state's),
/// call in the mawashi state, call in the fold state. Each has the inputs and
/// states of its push model (`riichi`, `call`).
pub const PF_MODEL_KINDS: [&str; 3] = ["riichi_M", "call_M", "call_F"];

/// The file names (`params/legacy/hand/<hand>/<name>.json`) of the per-decision
/// models: the P-state models of each decision (the M / F ones are named by
/// their `PF_MODEL_KINDS`, the fold and mawashi models `discard_F` /
/// `discard_M`).
pub const MODEL_FILES: [&str; 6] = ["discard_P_12", "discard_P_3", "discard_P_t", "riichi_P", "call_P", "kan"];

/// A legacy set of the models suzume played with until 2026-10-21 (the
/// player `suzume` is now the A1 set, `suzume::a1`), picked by the player
/// name: a hand model set and a state model. The names say what they were
/// fitted on (`flat` East 1 games, `normal` hanchan games) and how deep they
/// are (`mlp1` one hidden layer, `mlp3` three).
///
/// | player | hand (`params/legacy/hand/`) | state (`params/legacy/state/`) |
/// |---|---|---|
/// | `legacy-suzume-mlp3` (the default of a `SuzumeConfig`) | `flat_mlp3` | `normal_mlp3` |
/// | `legacy-suzume-mlp3-flat` (East 1) | `flat_mlp3` | `flat_mlp3` |
/// | `legacy-suzume-mlp3-avoid4` (last-place avoidance) | `normal_mlp3` (with the situation inputs) | `normal_mlp3` |
/// | `legacy-suzume-mlp1` (light) | `flat_mlp1` | `normal_mlp1` |
/// | `legacy-suzume-mlp1-flat` (light, East 1) | `flat_mlp1` | `flat_mlp1` |
///
/// `Dir` is a set read from a directory at run time (`dir:<path>`,
/// `load_dir`), for fitting new models, or the A1 sets' features set
/// (`features_set`); it is not one of `ALL`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSet {
    #[default]
    Mlp3,
    Mlp3Flat,
    Mlp3Avoid4,
    Mlp1,
    Mlp1Flat,
    /// The `n`-th directory set loaded in this process (`DIR_SETS`).
    Dir(u16),
}

impl ModelSet {
    pub const ALL: [ModelSet; 5] = [ModelSet::Mlp3, ModelSet::Mlp3Flat, ModelSet::Mlp3Avoid4, ModelSet::Mlp1, ModelSet::Mlp1Flat];

    /// The player name of the set (`dir:<path>` for a directory set, its
    /// path as loaded).
    pub fn name(self) -> &'static str {
        match self {
            ModelSet::Mlp3 => "legacy-suzume-mlp3",
            ModelSet::Mlp3Flat => "legacy-suzume-mlp3-flat",
            ModelSet::Mlp3Avoid4 => "legacy-suzume-mlp3-avoid4",
            ModelSet::Mlp1 => "legacy-suzume-mlp1",
            ModelSet::Mlp1Flat => "legacy-suzume-mlp1-flat",
            ModelSet::Dir(n) => dir_sets().lock().expect("dir sets")[usize::from(n)].0,
        }
    }

    /// The built-in set of a player name (the player `suzume` is not one of
    /// these sets since 2026-10-21: it is the A1 set `suzume::a1::BUILTIN`).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The set of a name: a built-in one (`from_name`) or `dir:<path>`, read
    /// on first use in this process (`load_dir`; the same path again gives
    /// the same set).
    pub fn parse(name: &str) -> Result<Self, String> {
        if let Some(path) = name.strip_prefix("dir:") {
            return Self::from_dir(path);
        }
        Self::from_name(name).ok_or_else(|| format!("not a model set: {name:?}"))
    }

    fn from_dir(path: &str) -> Result<Self, String> {
        let name = format!("dir:{path}");
        let mut sets = dir_sets().lock().expect("dir sets");
        if let Some(n) = sets.iter().position(|(m, _)| *m == name) {
            return Ok(ModelSet::Dir(n as u16));
        }
        let models = load_dir(Path::new(path))?;
        let n = u16::try_from(sets.len()).map_err(|_| "too many directory model sets".to_string())?;
        sets.push((Box::leak(name.into_boxed_str()), Box::leak(Box::new(models))));
        Ok(ModelSet::Dir(n))
    }
}

/// The directory sets loaded in this process: (name, models).
fn dir_sets() -> &'static Mutex<Vec<(&'static str, &'static Models)>> {
    static D: OnceLock<Mutex<Vec<(&'static str, &'static Models)>>> = OnceLock::new();
    D.get_or_init(|| Mutex::new(Vec::new()))
}

/// The JSON texts of one model set.
struct SetFiles {
    hand: &'static str,
    umlp: [&'static str; 6],
    pf_umlp: [&'static str; 3],
    fold: &'static str,
    mawashi: &'static str,
    state: (&'static str, &'static str),
}

macro_rules! set_files {
    ($hand:literal, $state:literal) => {
        SetFiles {
            hand: $hand,
            umlp: [
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/discard_P_12.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/discard_P_3.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/discard_P_t.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/riichi_P.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/call_P.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/kan.json")),
            ],
            pf_umlp: [
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/riichi_M.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/call_M.json")),
                include_str!(concat!("../../../params/legacy/hand/", $hand, "/call_F.json")),
            ],
            fold: include_str!(concat!("../../../params/legacy/hand/", $hand, "/discard_F.json")),
            mawashi: include_str!(concat!("../../../params/legacy/hand/", $hand, "/discard_M.json")),
            state: ($state, include_str!(concat!("../../../params/legacy/state/", $state, ".json"))),
        }
    };
}

/// The files of each set, in `ModelSet::ALL` order.
static SET_FILES: [SetFiles; ModelSet::ALL.len()] = [
    set_files!("flat_mlp3", "normal_mlp3"),
    set_files!("flat_mlp3", "flat_mlp3"),
    set_files!("normal_mlp3", "normal_mlp3"),
    set_files!("flat_mlp1", "normal_mlp1"),
    set_files!("flat_mlp1", "flat_mlp1"),
];

/// The parsed models of one set.
pub struct Models {
    /// The push (P) per-decision MLPs (`MODEL_FILES`).
    pub discard12: Umlp,
    pub discard3: Umlp,
    pub tenpai: Umlp,
    pub riichi: Umlp,
    pub call: Umlp,
    pub kan: Umlp,
    /// An optional split of `discard12` by the lowest after-discard shanten
    /// (`discard_P_1` for 1, `discard_P_2` for 2; a directory set with both,
    /// 2026-10-18): when present they decide those discards and `discard12`
    /// is not read. No built-in set has them.
    pub discard_split: Option<(Umlp, Umlp)>,
    /// The fold model (`fold=1` with `pf=0`, and the F state of `pf=1`);
    /// every built-in set has one.
    pub fold: Option<FoldMlp>,
    /// The tables the fold features read: the fold model's, or a directory
    /// set's `fold_tables.json`.
    pub tables: Option<FoldTables>,
    /// The push / fold models (`pf=1`); every built-in set has them.
    pub pf: Option<PfModels>,
}

/// The models only push / fold (`pf=1`) reads.
pub struct PfModels {
    /// The push / fold MLPs (`PF_MODEL_KINDS`): riichi in M (also F's), call
    /// in M and in F, each with its push model's inputs and states.
    pub riichi_m: Umlp,
    pub call_m: Umlp,
    pub call_f: Umlp,
    /// A fold-model file fitted to the mawashi state's discards, reading the
    /// fold model's features and tables.
    pub mawashi: FoldMlp,
    pub state: StateMlp,
}

impl Models {
    /// The push per-decision MLPs, in `MODEL_FILES` order (for the tests).
    #[cfg(test)]
    pub fn push_models(&self) -> [&Umlp; 6] {
        [&self.discard12, &self.discard3, &self.tenpai, &self.riichi, &self.call, &self.kan]
    }

    /// The fold model (a set without one cannot take `fold=1` or `pf=1`,
    /// `SuzumeConfig::from_pairs`).
    pub fn fold(&self) -> &FoldMlp {
        self.fold.as_ref().expect("a model set without discard_F (fold=1 / pf=1 need it)")
    }

    /// The fold features' tables.
    pub fn tables(&self) -> &FoldTables {
        self.tables.as_ref().expect("a model set without fold tables (discard_F.json or fold_tables.json)")
    }

    /// The push / fold models.
    pub fn pf(&self) -> &PfModels {
        self.pf.as_ref().expect("a model set without the push / fold models (pf=1 needs them)")
    }
}

/// Checks the models a set's fold and push / fold parts share: the mawashi
/// model reads the fold model's inputs and tables, the state model its inputs.
fn check_pf(fold: &FoldMlp, pf: &PfModels, push: [&Umlp; 2]) -> Result<(), String> {
    if pf.mawashi.inputs != fold.inputs || pf.mawashi.tables != fold.tables {
        return Err("discard_M.json: inputs / tables differ from discard_F.json's".into());
    }
    if pf.state.inputs != fold.inputs {
        return Err("state model: inputs differ from the fold model's".into());
    }
    let [riichi, call] = push;
    for (md, base) in [(&pf.riichi_m, riichi), (&pf.call_m, call), (&pf.call_f, call)] {
        if (&md.kind, &md.inputs, &md.states) != (&base.kind, &base.inputs, &base.states) {
            return Err(format!("{}: kind / inputs / states differ from its push model's", md.kind));
        }
    }
    Ok(())
}

fn parse_set(f: &SetFiles) -> Models {
    let d = f.hand;
    fn umlps<const N: usize>(d: &str, files: &[&str; N], texts: &[&str; N]) -> [Umlp; N] {
        std::array::from_fn(|i| parse_umlp(texts[i]).unwrap_or_else(|e| panic!("params/legacy/hand/{d}/{}.json: {e}", files[i])))
    }
    let fold = parse_fold(f.fold).unwrap_or_else(|e| panic!("params/legacy/hand/{d}/discard_F.json: {e}"));
    let mawashi = parse_fold(f.mawashi).unwrap_or_else(|e| panic!("params/legacy/hand/{d}/discard_M.json: {e}"));
    let (sname, stext) = f.state;
    let state = parse_state(stext).unwrap_or_else(|e| panic!("params/legacy/state/{sname}.json: {e}"));
    let [discard12, discard3, tenpai, riichi, call, kan] = umlps(d, &MODEL_FILES, &f.umlp);
    let [riichi_m, call_m, call_f] = umlps(d, &PF_MODEL_KINDS, &f.pf_umlp);
    let pf = PfModels { riichi_m, call_m, call_f, mawashi, state };
    check_pf(&fold, &pf, [&riichi, &call]).unwrap_or_else(|e| panic!("params/legacy/hand/{d} (state {sname}): {e}"));
    let tables = Some(fold.tables.clone());
    Models { discard12, discard3, tenpai, riichi, call, kan, discard_split: None, fold: Some(fold), tables, pf: Some(pf) }
}

/// A model set read from directory `dir`, laid out as a hand set of
/// `params/legacy/hand/` plus its state model:
///
/// | file | |
/// |---|---|
/// | `discard_P_12`, `discard_P_3`, `discard_P_t`, `riichi_P`, `call_P`, `kan` (`.json`) | required |
/// | `discard_F.json` | the fold model (`fold=1`, `pf=1`) |
/// | `riichi_M`, `call_M`, `call_F`, `discard_M`, `state` (`.json`) | the push / fold models (`pf=1`): all or none, and only with `discard_F.json` |
/// | `fold_tables.json` | the fold features' tables (`FoldTables` keys) when there is no `discard_F.json` |
///
/// Without a fold model or the push / fold models, the set plays only with
/// the options that do not need them (`pf=0`, and `fold=0` without a fold
/// model).
pub fn load_dir(dir: &Path) -> Result<Models, String> {
    let read = |name: &str| -> Result<Option<String>, String> {
        let p = dir.join(format!("{name}.json"));
        match std::fs::read_to_string(&p) {
            Ok(t) => Ok(Some(t)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", p.display())),
        }
    };
    load_with(read, &dir.display().to_string())
}

/// A model set laid out as `load_dir` lists, its files read by `read` (the
/// JSON text of `<name>.json`, `None` when absent); `label` names the set in
/// errors.
fn load_with(read: impl Fn(&str) -> Result<Option<String>, String>, label: &str) -> Result<Models, String> {
    let at = |name: &str, e: String| format!("{label}/{name}.json: {e}");
    let umlp = |name: &str| -> Result<Option<Umlp>, String> {
        read(name)?.map(|t| parse_umlp(&t).map_err(|e| at(name, e))).transpose()
    };
    let need = |name: &str| -> Result<Umlp, String> { umlp(name)?.ok_or_else(|| at(name, "missing".into())) };
    let [discard12, discard3, tenpai, riichi, call, kan] = [0, 1, 2, 3, 4, 5].map(|i| need(MODEL_FILES[i]));
    let (discard12, discard3, tenpai, riichi, call, kan) = (discard12?, discard3?, tenpai?, riichi?, call?, kan?);
    let discard_split = match (umlp("discard_P_1")?, umlp("discard_P_2")?) {
        (Some(a), Some(b)) => Some((a, b)),
        (None, None) => None,
        _ => return Err(format!("{label}: discard_P_1 and discard_P_2 come both or none")),
    };
    let fold = read("discard_F")?.map(|t| parse_fold(&t).map_err(|e| at("discard_F", e))).transpose()?;
    let pf_names = ["riichi_M", "call_M", "call_F", "discard_M", "state"];
    let present: Vec<bool> = pf_names.iter().map(|n| read(n).map(|t| t.is_some())).collect::<Result<_, _>>()?;
    let pf = if present.iter().all(|&p| p) {
        let fold = fold.as_ref().ok_or_else(|| at("discard_F", "missing (the push / fold models need it)".into()))?;
        let mawashi = parse_fold(&read("discard_M")?.unwrap_or_default()).map_err(|e| at("discard_M", e))?;
        let state = parse_state(&read("state")?.unwrap_or_default()).map_err(|e| at("state", e))?;
        let [riichi_m, call_m, call_f] = [0, 1, 2].map(|i| need(pf_names[i]));
        let pf = PfModels { riichi_m: riichi_m?, call_m: call_m?, call_f: call_f?, mawashi, state };
        check_pf(fold, &pf, [&riichi, &call]).map_err(|e| format!("{label}: {e}"))?;
        Some(pf)
    } else if present.iter().any(|&p| p) {
        return Err(format!("{label}: the push / fold models {pf_names:?} come all or none"));
    } else {
        None
    };
    let tables = match (&fold, read("fold_tables")?) {
        (Some(f), None) => Some(f.tables.clone()),
        (None, Some(t)) => {
            let v: Value = serde_json::from_str(&t).map_err(|e| at("fold_tables", e.to_string()))?;
            Some(parse_tables(&v).map_err(|e| at("fold_tables", e))?)
        }
        (None, None) => None,
        (Some(_), Some(_)) => return Err(format!("{label}: fold_tables.json next to discard_F.json (the fold model's tables are used)")),
    };
    Ok(Models { discard12, discard3, tenpai, riichi, call, kan, discard_split, fold, tables, pf })
}

/// The features set of the built-in A1 sets (`suzume::a1::BUILTIN`), `params/suzume/features/`: the six push
/// models of `legacy-suzume-mlp3` (`params/legacy/hand/flat_mlp3/`, copied) and the A1 sets' fold tables
/// (`fold_tables.json`), laid out as a `load_dir` set without a fold model. A1 builds its rows (and the kan rows'
/// first choice) with it. It is a directory-like set (`ModelSet::Dir`) named `suzume-features`, made on first use;
/// no player name gives it.
pub fn features_set() -> ModelSet {
    const FILES: [(&str, &str); 7] = [
        ("discard_P_12", include_str!("../../../params/suzume/features/discard_P_12.json")),
        ("discard_P_3", include_str!("../../../params/suzume/features/discard_P_3.json")),
        ("discard_P_t", include_str!("../../../params/suzume/features/discard_P_t.json")),
        ("riichi_P", include_str!("../../../params/suzume/features/riichi_P.json")),
        ("call_P", include_str!("../../../params/suzume/features/call_P.json")),
        ("kan", include_str!("../../../params/suzume/features/kan.json")),
        ("fold_tables", include_str!("../../../params/suzume/features/fold_tables.json")),
    ];
    static S: OnceLock<ModelSet> = OnceLock::new();
    *S.get_or_init(|| {
        let read = |name: &str| Ok(FILES.iter().find(|(n, _)| *n == name).map(|(_, t)| t.to_string()));
        let models = load_with(read, "params/suzume/features").unwrap_or_else(|e| panic!("{e}"));
        let mut sets = dir_sets().lock().expect("dir sets");
        let n = u16::try_from(sets.len()).expect("too many directory model sets");
        sets.push(("suzume-features", Box::leak(Box::new(models))));
        ModelSet::Dir(n)
    })
}

/// The parsed models of set `s`.
pub fn set_models(s: ModelSet) -> &'static Models {
    static M: [OnceLock<Models>; ModelSet::ALL.len()] = [const { OnceLock::new() }; ModelSet::ALL.len()];
    match s {
        ModelSet::Dir(n) => dir_sets().lock().expect("dir sets")[usize::from(n)].1,
        _ => {
            let i = ModelSet::ALL.iter().position(|&x| x == s).expect("a built-in set");
            M[i].get_or_init(|| parse_set(&SET_FILES[i]))
        }
    }
}

/// The parsed `params/suzume.json`.
pub fn params() -> &'static Params {
    static P: OnceLock<Params> = OnceLock::new();
    P.get_or_init(|| parse_params(PARAMS_JSON).unwrap_or_else(|e| panic!("params/suzume.json: {e}")))
}

/// Each of `inputs` as its index in `names`.
pub(crate) fn positions<S: AsRef<str>>(inputs: &[String], names: &[S]) -> Result<Vec<usize>, String> {
    inputs
        .iter()
        .map(|n| names.iter().position(|x| x.as_ref() == n).ok_or_else(|| format!("no input column {n}")))
        .collect()
}

pub(crate) fn get<'a>(v: &'a Value, path: &[&str]) -> Result<&'a Value, String> {
    path.iter().try_fold(v, |v, k| v.get(*k).ok_or_else(|| format!("missing key {}", path.join("."))))
}

pub(crate) fn num(v: &Value, path: &[&str]) -> Result<f64, String> {
    get(v, path)?.as_f64().ok_or_else(|| format!("{} is not a number", path.join(".")))
}

pub(crate) fn nums(v: &Value, path: &[&str]) -> Result<Vec<f64>, String> {
    vector(get(v, path)?, &path.join("."))
}

pub(crate) fn strs(v: &Value, path: &[&str]) -> Result<Vec<String>, String> {
    get(v, path)?
        .as_array()
        .ok_or_else(|| format!("{} is not a list", path.join(".")))?
        .iter()
        .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("{} holds a non-string", path.join("."))))
        .collect()
}

/// A table of named weights in `names` order (every name required, no other).
fn named<const N: usize>(v: &Value, path: &[&str], names: &[&str; N]) -> Result<[f32; N], String> {
    let table = get(v, path)?.as_object().ok_or_else(|| format!("{} is not a table", path.join(".")))?;
    if let Some(k) = table.keys().find(|k| !names.contains(&k.as_str())) {
        return Err(format!("{}: unknown name {k}", path.join(".")));
    }
    let mut out = [0f32; N];
    for (o, n) in out.iter_mut().zip(names) {
        *o = f32_of(table.get(*n).ok_or_else(|| format!("{}.{n} missing", path.join(".")))?)
            .ok_or_else(|| format!("{}.{n} is not a number", path.join(".")))?;
    }
    Ok(out)
}

/// A number read straight into f32 from its decimal form (not through f64,
/// whose rounding to f32 can differ by one unit from the literal's).
fn f32_of(v: &Value) -> Option<f32> {
    v.as_number()?.to_string().parse().ok()
}

fn parse_params(text: &str) -> Result<Params, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    Ok(Params {
        dp_max_shanten: num(&v, &["dp", "max_shanten"])? as i8,
        ron_rate: f32_of(get(&v, &["dp", "ron_rate"])?).ok_or("dp.ron_rate is not a number")?,
        riichi_deposit: f32_of(get(&v, &["dp", "riichi_deposit"])?).ok_or("dp.riichi_deposit is not a number")?,
        value_weights: named(&v, &["value_features", "weights"], &VF_NAMES)?,
        tie_band: f32_of(get(&v, &["tile_features", "tie_band"])?).ok_or("tile_features.tie_band is not a number")?,
        tie_weights: named(&v, &["tile_features", "tie_weights"], &TF_NAMES)?,
        context_top_margin: num(&v, &["context", "top_margin"])? as i32,
    })
}

fn parse_umlp(text: &str) -> Result<Umlp, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let (inputs, states) = (strs(&v, &["inputs"])?, strs(&v, &["state_onehot_order"])?);
    let sit = parse_sit(&v, inputs.len() + states.len())?;
    let net = parse_hidden(&v, inputs.len() + states.len() + sit.as_ref().map_or(0, |s| s.mean.len()))?;
    let co = v.get("outside_bias").and_then(|o| o.get("coefficients"));
    let c = |k: &str| co.and_then(|c| c.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
    let md = Umlp {
        kind: get(&v, &["kind"])?.as_str().ok_or("kind is not a string")?.to_string(),
        cols: positions(&inputs, crate::suzume::universal::u_names())?,
        inputs,
        states,
        mean: nums(&v, &["mean"])?,
        std: nums(&v, &["std"])?,
        net,
        w2: nums(&v, &["w2"])?,
        b2: num(&v, &["b2"])?,
        out_round: [c("b_E1"), c("b_E2-4"), c("b_S1-3"), c("b_all-last")],
        out_top: c("unique_top"),
        out_dealer: c("dealer"),
        sit,
    };
    let n_in = md.inputs.len();
    if md.w2.len() != md.net.out_width() {
        return Err("w2 does not match the last layer".into());
    }
    if md.mean.len() != n_in || md.std.len() != n_in || md.std.iter().any(|&s| s <= 0.0) {
        return Err("mean / std do not match the inputs".into());
    }
    Ok(md)
}

fn rows(v: &Value, key: &str) -> Result<(usize, Vec<f64>), String> {
    let (n, _, out) = matrix(get(v, &[key])?, key)?;
    Ok((n, out))
}

fn parse_tables(v: &Value) -> Result<FoldTables, String> {
    let (n_pat, wait_table) = rows(v, "wait_table")?;
    let t = FoldTables {
        wait_table,
        pattern_edges: nums(v, &["pattern_edges"])?,
        turn_edges: nums(v, &["turn_edges"])?.iter().map(|&x| x as u8).collect(),
        turn_cap: num(v, &["turn_cap"])?,
        last_ago_cap: num(v, &["last_ago_cap"])?,
        early_discards: num(v, &["early_discards"])? as u8,
    };
    if n_pat != t.pattern_edges.len() + 1 || t.wait_table.len() != n_pat * (t.turn_edges.len() + 1) {
        return Err("wait_table is not (pattern bins) x (turn bins)".into());
    }
    if t.turn_cap <= 0.0 || t.last_ago_cap <= 0.0 {
        return Err("turn_cap / last_ago_cap must be positive".into());
    }
    Ok(t)
}

fn parse_fold(text: &str) -> Result<FoldMlp, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let inputs = strs(&v, &["inputs"])?;
    let sit = parse_sit(&v, inputs.len())?;
    let net = parse_hidden(&v, inputs.len() + sit.as_ref().map_or(0, |s| s.mean.len()))?;
    let md = FoldMlp {
        cols: positions(&inputs, &crate::suzume::fold_model::FEATURES)?,
        inputs,
        mean: nums(&v, &["mean"])?,
        std: nums(&v, &["std"])?,
        net,
        w2: nums(&v, &["w2"])?,
        b2: num(&v, &["b2"])?,
        tables: parse_tables(&v)?,
        tau: num(&v, &["tau"])?,
        sit,
    };
    if v.get("seats").and_then(Value::as_str) != Some("all") || v.get("combine").and_then(Value::as_str) != Some("softmin") {
        return Err("a fold-model file scores every other seat by the soft minimum (\"seats\": \"all\", \"combine\": \"softmin\")".into());
    }
    let n_in = md.inputs.len();
    if md.w2.len() != md.net.out_width() {
        return Err("w2 does not match the last layer".into());
    }
    if md.mean.len() != n_in || md.std.len() != n_in || md.std.iter().any(|&s| s <= 0.0) {
        return Err("mean / std do not match the inputs".into());
    }
    Ok(md)
}

pub(crate) fn parse_state(text: &str) -> Result<StateMlp, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let (n_out, w2) = rows(&v, "W2")?;
    let b2 = nums(&v, &["b2"])?;
    if v.get("prev_inputs").is_some() {
        return Err("prev_inputs (previous-decision inputs) are no longer read".into());
    }
    let sit: Vec<String> = if v.get("sit_inputs").is_some() { strs(&v, &["sit_inputs"])? } else { Vec::new() };
    let n_base = crate::suzume::pf::SIT_NAMES.len();
    if !sit.is_empty() && (sit.len() < n_base || sit[..n_base] != crate::suzume::pf::SIT_NAMES) {
        return Err(format!("sit_inputs must start with {:?}", crate::suzume::pf::SIT_NAMES));
    }
    let summary_cols = if sit.is_empty() { Vec::new() } else { positions(&sit[n_base..], &crate::suzume::pf::PMF_SUMMARY)? };
    let sit_stat = |k: &str| -> Result<Vec<f64>, String> { if sit.is_empty() { Ok(Vec::new()) } else { nums(&v, &[k]) } };
    let inputs = strs(&v, &["inputs"])?;
    let net = parse_hidden(&v, inputs.len() + sit.len())?;
    let md = StateMlp {
        cols: positions(&inputs, crate::suzume::pf::state_input_names())?,
        inputs,
        mean: nums(&v, &["mean"])?,
        std: nums(&v, &["std"])?,
        n_sit: sit.len(),
        summary_cols,
        sit_mean: sit_stat("sit_mean")?,
        sit_std: sit_stat("sit_std")?,
        net,
        w2,
        b2: b2.try_into().map_err(|_| "b2 must hold 3 numbers".to_string())?,
    };
    let n_in = md.inputs.len();
    if md.sit_mean.len() != md.n_sit || md.sit_std.len() != md.n_sit || md.sit_std.iter().any(|&s| s <= 0.0) {
        return Err("sit_mean / sit_std do not match sit_inputs".into());
    }
    if n_out != 3 || md.w2.len() != 3 * 2 * md.net.out_width() {
        return Err("W2 must be 3 x twice the last layer".into());
    }
    if md.mean.len() != n_in || md.std.len() != n_in || md.std.iter().any(|&s| s <= 0.0) {
        return Err("mean / std do not match the inputs".into());
    }
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_file_parses() {
        let p = params();
        assert_eq!(p.dp_max_shanten, 2);
        assert!(p.ron_rate > 0.0 && p.riichi_deposit == 1000.0);
        let ms = set_models(ModelSet::default());
        let kinds = ["discard12", "discard3", "tenpai", "riichi", "call", "kan"];
        assert_eq!(ms.push_models().map(|x| x.kind.as_str()), kinds);
        let (f, pf) = (ms.fold(), ms.pf());
        assert_eq!(f.net.w1.len(), f.net.widths[0] * f.inputs.len());
        for (md, base) in [(&pf.riichi_m, &ms.riichi), (&pf.call_m, &ms.call), (&pf.call_f, &ms.call)] {
            assert_eq!((&md.kind, &md.inputs, &md.states), (&base.kind, &base.inputs, &base.states));
        }
        assert_eq!(pf.mawashi.inputs, f.inputs);
        assert_eq!(pf.state.inputs, f.inputs);
        assert_eq!(ms.tables(), &f.tables);
    }

    /// Every model set parses (`parse_set` checks the shared inputs); the
    /// sets' depths and state models are those `ModelSet` states.
    #[test]
    fn every_model_set_parses() {
        let depth = |s: ModelSet| set_models(s).discard12.net.widths.len();
        assert_eq!(ModelSet::ALL.map(depth), [3, 3, 3, 1, 1]);
        let sit = |s: ModelSet| set_models(s).pf().state.n_sit;
        assert_eq!(ModelSet::ALL.map(sit), [21, 0, 21, 21, 0]);
        let umlp_sit = |s: ModelSet| set_models(s).push_models().iter().all(|m| m.sit.is_some()) && set_models(s).fold().sit.is_some();
        assert_eq!(ModelSet::ALL.map(umlp_sit), [false, false, true, false, false]);
        for s in ModelSet::ALL {
            assert_eq!(ModelSet::from_name(s.name()), Some(s));
        }
        assert_eq!(ModelSet::from_name("suzume"), None);
        assert_eq!(ModelSet::from_name("suzume-flat"), None);
    }

    #[test]
    fn a_missing_or_unknown_weight_is_an_error() {
        let mut v: Value = serde_json::from_str(PARAMS_JSON).unwrap();
        v["value_features"]["weights"].as_object_mut().unwrap().remove("pairs");
        assert!(parse_params(&v.to_string()).unwrap_err().contains("pairs"));
        let mut v: Value = serde_json::from_str(PARAMS_JSON).unwrap();
        v["tile_features"]["tie_weights"]["bogus"] = 1.0.into();
        assert!(parse_params(&v.to_string()).unwrap_err().contains("bogus"));
    }
}
