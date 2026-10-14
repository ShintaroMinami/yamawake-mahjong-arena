//! The fixed parameters of suzume, read from `params/` at the repository
//! root: the DP's constants and the weights the universal features read
//! (`params/suzume.json`), and the models of each model set (`ModelSet`):
//! a hand model set (`params/hand/<hand>/`, one file per decision and
//! push / fold state, `MODEL_FILES`) and a push / fold state model
//! (`params/state/<state>.json`).
//!
//! They are not options: the files are embedded at build time and parsed
//! once on first use, so a caller cannot vary them, and the models always
//! run on the values they were fitted with. Changing one means editing the
//! file and rebuilding. A missing or malformed key stops the program on
//! first use. A suzume bot picks its model set by its name (`ModelSet`, in its
//! config); a decision reads it from its context (`Ctx::models`).

use std::sync::OnceLock;

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

/// One per-decision MLP (`params/hand/<hand>/`, `MODEL_FILES`).
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

/// The fold model (`params/hand/<hand>/discard_F.json`; `suzume::fold_model`):
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
    /// The soft minimum's temperature (`"tau"`): a tile type's score is
    /// `-tau ln sum exp(-s / tau)` over its scores against every other seat
    /// (the file must say `"seats": "all"`, `"combine": "softmin"`).
    pub tau: f64,
    /// The situation inputs (`sit_inputs`; after `inputs`), if any.
    pub sit: Option<Sit>,
}

/// The push / fold state model (`params/state/<state>.json`; `suzume::pf`):
/// each (other seat, legal discard type) row of the fold model's features,
/// next to the situation (`sit`) -> `relu(W1 [(x - mean) / std, sit] + b1)`
/// and any further layers
/// (`mlp::Hidden`), pooled over the rows by the mean and the max ->
/// `W2 [mean, max] + b2`, the P / M / F logits.
///
/// `sit` (`"sit_inputs"`, `pf::SIT_NAMES`; none in a file without them): the
/// decision's round, honba, sticks, rank and score gaps (`pf::situation`),
/// standardized by `sit_mean` / `sit_std`.
#[derive(Debug)]
pub struct StateMlp {
    /// Feature names (`fold_model::FEATURES`), in input order (the fold
    /// model's).
    pub inputs: Vec<String>,
    /// The inputs as `fold_model::FEATURES` indices.
    pub cols: Vec<usize>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// The situation inputs: 0, or `pf::SIT_NAMES.len()`.
    pub n_sit: usize,
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

/// The file names (`params/hand/<hand>/<name>.json`) of the per-decision
/// models: the P-state models of each decision (the M / F ones are named by
/// their `PF_MODEL_KINDS`, the fold and mawashi models `discard_F` /
/// `discard_M`).
pub const MODEL_FILES: [&str; 6] = ["discard_P_12", "discard_P_3", "discard_P_t", "riichi_P", "call_P", "kan"];

/// A set of the models suzume plays with, picked by the player name: a hand
/// model set and a state model. The names say what they were fitted on
/// (`T1` East 1 games, `TN` hanchan games) and how deep they are (`dep1`
/// one hidden layer, `dep3` three).
///
/// | player | hand (`params/hand/`) | state (`params/state/`) |
/// |---|---|---|
/// | `suzume-mlp3` (also `suzume`, the default) | `T1_dep3` | `TN_dep3` |
/// | `suzume-mlp3-flat` (East 1) | `T1_dep3` | `T1_dep3` |
/// | `suzume-mlp3-avoid4` (last-place avoidance) | `TN_dep3` (with the situation inputs) | `TN_dep3` |
/// | `suzume-mlp1` (light) | `T1_dep1` | `TN_dep1` |
/// | `suzume-mlp1-flat` (light, East 1) | `T1_dep1` | `T1_dep1` |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSet {
    #[default]
    Mlp3,
    Mlp3Flat,
    Mlp3Avoid4,
    Mlp1,
    Mlp1Flat,
}

impl ModelSet {
    pub const ALL: [ModelSet; 5] = [ModelSet::Mlp3, ModelSet::Mlp3Flat, ModelSet::Mlp3Avoid4, ModelSet::Mlp1, ModelSet::Mlp1Flat];

    /// The player name of the set.
    pub const fn name(self) -> &'static str {
        match self {
            ModelSet::Mlp3 => "suzume-mlp3",
            ModelSet::Mlp3Flat => "suzume-mlp3-flat",
            ModelSet::Mlp3Avoid4 => "suzume-mlp3-avoid4",
            ModelSet::Mlp1 => "suzume-mlp1",
            ModelSet::Mlp1Flat => "suzume-mlp1-flat",
        }
    }

    /// The set of a player name (`suzume` is the default set's other name).
    pub fn from_name(name: &str) -> Option<Self> {
        if name == "suzume" {
            return Some(Self::default());
        }
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The set's position in `ALL` (for per-set caches).
    pub const fn index(self) -> usize {
        self as usize
    }
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
                include_str!(concat!("../../../params/hand/", $hand, "/discard_P_12.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/discard_P_3.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/discard_P_t.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/riichi_P.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/call_P.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/kan.json")),
            ],
            pf_umlp: [
                include_str!(concat!("../../../params/hand/", $hand, "/riichi_M.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/call_M.json")),
                include_str!(concat!("../../../params/hand/", $hand, "/call_F.json")),
            ],
            fold: include_str!(concat!("../../../params/hand/", $hand, "/discard_F.json")),
            mawashi: include_str!(concat!("../../../params/hand/", $hand, "/discard_M.json")),
            state: ($state, include_str!(concat!("../../../params/state/", $state, ".json"))),
        }
    };
}

/// The files of each set, in `ModelSet::ALL` order.
static SET_FILES: [SetFiles; ModelSet::ALL.len()] = [
    set_files!("T1_dep3", "TN_dep3"),
    set_files!("T1_dep3", "T1_dep3"),
    set_files!("TN_dep3", "TN_dep3"),
    set_files!("T1_dep1", "TN_dep1"),
    set_files!("T1_dep1", "T1_dep1"),
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
    /// The push / fold MLPs (`PF_MODEL_KINDS`): riichi in M (also F's), call
    /// in M and in F, each with its push model's inputs and states.
    pub riichi_m: Umlp,
    pub call_m: Umlp,
    pub call_f: Umlp,
    pub fold: FoldMlp,
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
}

fn parse_set(f: &SetFiles) -> Models {
    let d = f.hand;
    fn umlps<const N: usize>(d: &str, files: &[&str; N], texts: &[&str; N]) -> [Umlp; N] {
        std::array::from_fn(|i| parse_umlp(texts[i]).unwrap_or_else(|e| panic!("params/hand/{d}/{}.json: {e}", files[i])))
    }
    let fold = parse_fold(f.fold).unwrap_or_else(|e| panic!("params/hand/{d}/discard_F.json: {e}"));
    let mawashi = parse_fold(f.mawashi).unwrap_or_else(|e| panic!("params/hand/{d}/discard_M.json: {e}"));
    let same = mawashi.inputs == fold.inputs
        && mawashi.wait_table == fold.wait_table
        && mawashi.pattern_edges == fold.pattern_edges
        && mawashi.turn_edges == fold.turn_edges
        && mawashi.turn_cap == fold.turn_cap
        && mawashi.last_ago_cap == fold.last_ago_cap
        && mawashi.early_discards == fold.early_discards;
    assert!(same, "params/hand/{d}/discard_M.json: inputs / tables differ from discard_F.json's");
    let (sname, stext) = f.state;
    let state = parse_state(stext).unwrap_or_else(|e| panic!("params/state/{sname}.json: {e}"));
    assert!(state.inputs == fold.inputs, "params/state/{sname}.json: inputs differ from hand {d}'s fold model");
    let [discard12, discard3, tenpai, riichi, call, kan] = umlps(d, &MODEL_FILES, &f.umlp);
    let [riichi_m, call_m, call_f] = umlps(d, &PF_MODEL_KINDS, &f.pf_umlp);
    Models { discard12, discard3, tenpai, riichi, call, kan, riichi_m, call_m, call_f, fold, mawashi, state }
}

/// The parsed models of set `s`.
pub fn set_models(s: ModelSet) -> &'static Models {
    static M: [OnceLock<Models>; ModelSet::ALL.len()] = [const { OnceLock::new() }; ModelSet::ALL.len()];
    M[s.index()].get_or_init(|| parse_set(&SET_FILES[s.index()]))
}

/// The parsed `params/suzume.json`.
pub fn params() -> &'static Params {
    static P: OnceLock<Params> = OnceLock::new();
    P.get_or_init(|| parse_params(PARAMS_JSON).unwrap_or_else(|e| panic!("params/suzume.json: {e}")))
}

/// Each of `inputs` as its index in `names`.
fn positions<S: AsRef<str>>(inputs: &[String], names: &[S]) -> Result<Vec<usize>, String> {
    inputs
        .iter()
        .map(|n| names.iter().position(|x| x.as_ref() == n).ok_or_else(|| format!("no input column {n}")))
        .collect()
}

fn get<'a>(v: &'a Value, path: &[&str]) -> Result<&'a Value, String> {
    path.iter().try_fold(v, |v, k| v.get(*k).ok_or_else(|| format!("missing key {}", path.join("."))))
}

fn num(v: &Value, path: &[&str]) -> Result<f64, String> {
    get(v, path)?.as_f64().ok_or_else(|| format!("{} is not a number", path.join(".")))
}

fn nums(v: &Value, path: &[&str]) -> Result<Vec<f64>, String> {
    vector(get(v, path)?, &path.join("."))
}

fn strs(v: &Value, path: &[&str]) -> Result<Vec<String>, String> {
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

fn parse_fold(text: &str) -> Result<FoldMlp, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let inputs = strs(&v, &["inputs"])?;
    let sit = parse_sit(&v, inputs.len())?;
    let net = parse_hidden(&v, inputs.len() + sit.as_ref().map_or(0, |s| s.mean.len()))?;
    let (n_pat, wait_table) = rows(&v, "wait_table")?;
    let turn_edges = nums(&v, &["turn_edges"])?;
    let md = FoldMlp {
        cols: positions(&inputs, &crate::suzume::fold_model::FEATURES)?,
        inputs,
        mean: nums(&v, &["mean"])?,
        std: nums(&v, &["std"])?,
        net,
        w2: nums(&v, &["w2"])?,
        b2: num(&v, &["b2"])?,
        wait_table,
        pattern_edges: nums(&v, &["pattern_edges"])?,
        turn_edges: turn_edges.iter().map(|&x| x as u8).collect(),
        turn_cap: num(&v, &["turn_cap"])?,
        last_ago_cap: num(&v, &["last_ago_cap"])?,
        early_discards: num(&v, &["early_discards"])? as u8,
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
    if n_pat != md.pattern_edges.len() + 1 || md.wait_table.len() != n_pat * (md.turn_edges.len() + 1) {
        return Err("wait_table is not (pattern bins) x (turn bins)".into());
    }
    if md.turn_cap <= 0.0 || md.last_ago_cap <= 0.0 {
        return Err("turn_cap / last_ago_cap must be positive".into());
    }
    Ok(md)
}

fn parse_state(text: &str) -> Result<StateMlp, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let (n_out, w2) = rows(&v, "W2")?;
    let b2 = nums(&v, &["b2"])?;
    if v.get("prev_inputs").is_some() {
        return Err("prev_inputs (previous-decision inputs) are no longer read".into());
    }
    let sit: Vec<String> = if v.get("sit_inputs").is_some() { strs(&v, &["sit_inputs"])? } else { Vec::new() };
    if !sit.is_empty() && sit != crate::suzume::pf::SIT_NAMES {
        return Err(format!("sit_inputs must be {:?}", crate::suzume::pf::SIT_NAMES));
    }
    let sit_stat = |k: &str| -> Result<Vec<f64>, String> { if sit.is_empty() { Ok(Vec::new()) } else { nums(&v, &[k]) } };
    let inputs = strs(&v, &["inputs"])?;
    let net = parse_hidden(&v, inputs.len() + sit.len())?;
    let md = StateMlp {
        cols: positions(&inputs, &crate::suzume::fold_model::FEATURES)?,
        inputs,
        mean: nums(&v, &["mean"])?,
        std: nums(&v, &["std"])?,
        n_sit: sit.len(),
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
        let f = &ms.fold;
        assert_eq!(f.net.w1.len(), f.net.widths[0] * f.inputs.len());
        for (md, base) in [(&ms.riichi_m, &ms.riichi), (&ms.call_m, &ms.call), (&ms.call_f, &ms.call)] {
            assert_eq!((&md.kind, &md.inputs, &md.states), (&base.kind, &base.inputs, &base.states));
        }
        assert_eq!(ms.mawashi.inputs, f.inputs);
        assert_eq!(ms.state.inputs, f.inputs);
    }

    /// Every model set parses (`parse_set` checks the shared inputs); the
    /// sets' depths and state models are those `ModelSet` states.
    #[test]
    fn every_model_set_parses() {
        let depth = |s: ModelSet| set_models(s).discard12.net.widths.len();
        assert_eq!(ModelSet::ALL.map(depth), [3, 3, 3, 1, 1]);
        let sit = |s: ModelSet| set_models(s).state.n_sit;
        assert_eq!(ModelSet::ALL.map(sit), [21, 0, 21, 21, 0]);
        let umlp_sit = |s: ModelSet| set_models(s).push_models().iter().all(|m| m.sit.is_some()) && set_models(s).fold.sit.is_some();
        assert_eq!(ModelSet::ALL.map(umlp_sit), [false, false, true, false, false]);
        for s in ModelSet::ALL {
            assert_eq!(ModelSet::from_name(s.name()), Some(s));
        }
        assert_eq!(ModelSet::from_name("suzume"), Some(ModelSet::Mlp3));
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
