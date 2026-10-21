//! The switches of suzume.
//!
//! Only what a caller may legitimately vary is a switch; everything the
//! fitted models depend on (the DP's ron rate, the riichi deposit, the DP's
//! shanten range, the weights the universal features read, the MLPs) is a
//! fixed parameter read from `params/` (`params.rs`), so no option can put a
//! model off the data it was fitted on. Switches are set from string pairs (`set`), the one format the
//! CLI (`--opt KEY=VALUE`), the player spec and Python dicts all reduce to:
//!
//! | key | value | default |
//! |---|---|---|
//! | `kan` | 0/1: make kans: ankan / kakan when the kan MLP prefers them, daiminkan when the call MLP does; 0 never kans | 1 |
//! | `call` | 0/1: chi and pon when the call MLP prefers them; 0 never chi / pon (daiminkan follows `kan`) | 1 |
//! | `riichi` | 0/1: declare riichi when the riichi MLP prefers it (and value hands with riichi in reach); 0 never riichi | 1 |
//! | `fold` | 0/1: betaori against a riichi from `fold_shanten` (`fold_discard`; used only with `pf=0`); 0 off | 1 |
//! | `fold_shanten` | 0..=8: the best after-discard shanten from which `fold` folds | 1 |
//! | `pf` | 0/1: push / fold by the predicted P / M / F state (`suzume::pf`; `fold` is then off); 0 off | 1 |

pub use super::params::ModelSet;

/// Live-wall tiles at the start of a kyoku (136 - 14 dead wall - 4 x 13
/// dealt): the reference of the progress features.
pub const DRAWABLE_AT_START: f32 = 70.0;

/// The switches of suzume. `Default` is the reference behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuzumeConfig {
    /// Make kans when the kan MLP prefers them (`kan_mlp`).
    pub kan: bool,
    /// Chi and pon when the call MLP prefers them.
    pub call: bool,
    /// Declare riichi when the riichi MLP prefers it; also whether the DP
    /// values a closed hand with riichi in reach.
    pub riichi: bool,
    /// Betaori against a riichi (`fold_discard`; used only without `pf`).
    pub fold: bool,
    /// The best after-discard shanten from which `fold` folds.
    pub fold_shanten: i8,
    /// Push / fold by the predicted state (`suzume::pf`).
    /// Replaces `fold`.
    pub pf: bool,
    /// The legacy model set, picked by the player name (`legacy-suzume-mlp3`,
    /// `legacy-suzume-mlp3-flat`, `legacy-suzume-mlp3-avoid4`, `legacy-suzume-mlp1`,
    /// `legacy-suzume-mlp1-flat`; `params::ModelSet`). Not an
    /// option: each set is its own player.
    pub models: ModelSet,
}

impl Default for SuzumeConfig {
    fn default() -> Self {
        Self { kan: true, call: true, riichi: true, fold: true, fold_shanten: 1, pf: true, models: ModelSet::Mlp3 }
    }
}

impl SuzumeConfig {
    /// Set one switch from its string form.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let bad = |e: &dyn std::fmt::Display| format!("suzume option {key}={value:?}: {e}");
        let flag = || match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" => Ok(true),
            "0" | "false" => Ok(false),
            _ => Err(bad(&"must be 0 or 1")),
        };
        match key {
            "kan" => self.kan = flag()?,
            "call" => self.call = flag()?,
            "riichi" => self.riichi = flag()?,
            "fold" => self.fold = flag()?,
            "pf" => self.pf = flag()?,
            "models" => self.models = ModelSet::parse(value.trim()).map_err(|e| bad(&e))?,
            "fold_shanten" => {
                let v: i8 = value.trim().parse().map_err(|e| bad(&e))?;
                if !(0..=8).contains(&v) {
                    return Err(bad(&"must be 0..=8"));
                }
                self.fold_shanten = v;
            }
            _ => return Err(format!("unknown suzume option {key:?} (options: kan, call, riichi, fold, fold_shanten, pf)")),
        }
        Ok(())
    }

    /// The default config with `pairs` applied in order.
    pub fn from_pairs<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<Self, String> {
        let mut cfg = Self::default();
        for (k, v) in pairs {
            cfg.set(k, v)?;
        }
        cfg.check_models()?;
        Ok(cfg)
    }

    /// Whether the model set has the models the switches play with: `pf=1`
    /// needs the push / fold models, `fold=1` (with `pf=0`) the fold model
    /// (a directory set may lack them, `params::load_dir`).
    pub fn check_models(&self) -> Result<(), String> {
        let m = super::params::set_models(self.models);
        let name = self.models.name();
        if self.pf && m.pf.is_none() {
            return Err(format!("suzume model set {name} has no push / fold models: use pf=0"));
        }
        if self.fold && !self.pf && m.fold.is_none() {
            return Err(format!("suzume model set {name} has no fold model (discard_F): use fold=0"));
        }
        Ok(())
    }
}
