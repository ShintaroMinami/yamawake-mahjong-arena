#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, set_all, from_py_object)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    pub is_kokushi_musou_13machi_double: bool,
    pub is_suuankou_tanki_double: bool,
    pub is_junsei_chuurenpoutou_double: bool,
    pub is_daisuushii_double: bool,
    pub yakuman_pao_is_liability_only: bool,
    pub sanchaho_is_draw: bool,

    pub kuikae_forbidden: bool,

    /// yamawake patch: the game flow (all off is riichienv's stock flow).
    /// An abortive draw in the all-last keeps the deal but never ends the
    /// game by the dealer's lead (Tenhou, MJSoul); off, it ends the game
    /// with the dealer on top with 30,000 or more, as a dealer win does.
    #[serde(default)]
    pub abortive_draw_plays_on: bool,
    /// yamawake patch: a nagashi mangan keeps the deal when the dealer is
    /// tenpai (Tenhou, MJSoul); off, only when the dealer made it.
    #[serde(default)]
    pub nagashi_renchan_on_tenpai: bool,
    /// yamawake patch: in the extension round (West in a hanchan), a kept
    /// deal ends the game only as in the all-last, with the dealer on top
    /// with 30,000 or more (Tenhou); off, anyone's 30,000 ends it (MJSoul).
    #[serde(default)]
    pub extension_renchan_plays_on: bool,
    /// yamawake patch: in the extension round, a deal kept by a multiple
    /// ron that includes the dealer plays on as `extension_renchan_plays_on`
    /// has every kept deal play on (MJSoul, and Tenhou through that flag).
    #[serde(default)]
    pub extension_multi_ron_renchan_plays_on: bool,
    /// yamawake patch: the riichi sticks left on the table at the end of a
    /// game of rounds go to first place (Tenhou, MJSoul); off, to nobody.
    #[serde(default)]
    pub leftover_sticks_to_top: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_tenhou()
    }
}

impl GameRule {
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,

            abortive_draw_plays_on: true,
            nagashi_renchan_on_tenpai: true,
            extension_renchan_plays_on: true,
            extension_multi_ron_renchan_plays_on: true,
            leftover_sticks_to_top: true,
        }
    }

    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,

            abortive_draw_plays_on: true,
            nagashi_renchan_on_tenpai: true,
            extension_renchan_plays_on: false,
            extension_multi_ron_renchan_plays_on: true,
            leftover_sticks_to_top: true,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, is_suuankou_tanki_double=false, is_junsei_chuurenpoutou_double=false, is_daisuushii_double=false, yakuman_pao_is_liability_only=false, sanchaho_is_draw=false, kuikae_forbidden=true, abortive_draw_plays_on=false, nagashi_renchan_on_tenpai=false, extension_renchan_plays_on=false, extension_multi_ron_renchan_plays_on=false, leftover_sticks_to_top=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        is_suuankou_tanki_double: bool,
        is_junsei_chuurenpoutou_double: bool,
        is_daisuushii_double: bool,
        yakuman_pao_is_liability_only: bool,
        sanchaho_is_draw: bool,
        kuikae_forbidden: bool,
        abortive_draw_plays_on: bool,
        nagashi_renchan_on_tenpai: bool,
        extension_renchan_plays_on: bool,
        extension_multi_ron_renchan_plays_on: bool,
        leftover_sticks_to_top: bool,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            is_suuankou_tanki_double,
            is_junsei_chuurenpoutou_double,
            is_daisuushii_double,
            yakuman_pao_is_liability_only,
            sanchaho_is_draw,
            kuikae_forbidden,
            abortive_draw_plays_on,
            nagashi_renchan_on_tenpai,
            extension_renchan_plays_on,
            extension_multi_ron_renchan_plays_on,
            leftover_sticks_to_top,
        }
    }

    #[staticmethod]
    #[pyo3(name = "default_tenhou")]
    pub fn py_default_tenhou() -> Self {
        Self::default_tenhou()
    }

    #[staticmethod]
    #[pyo3(name = "default_mjsoul")]
    pub fn py_default_mjsoul() -> Self {
        Self::default_mjsoul()
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, is_suuankou_tanki_double={}, is_junsei_chuurenpoutou_double={}, is_daisuushii_double={}, yakuman_pao_is_liability_only={}, sanchaho_is_draw={}, kuikae_forbidden={}, abortive_draw_plays_on={}, nagashi_renchan_on_tenpai={}, extension_renchan_plays_on={}, extension_multi_ron_renchan_plays_on={}, leftover_sticks_to_top={})",
            self.allows_ron_on_ankan_for_kokushi_musou,
            self.is_kokushi_musou_13machi_double,
            self.is_suuankou_tanki_double,
            self.is_junsei_chuurenpoutou_double,
            self.is_daisuushii_double,
            self.yakuman_pao_is_liability_only,
            self.sanchaho_is_draw,
            self.kuikae_forbidden,
            self.abortive_draw_plays_on,
            self.nagashi_renchan_on_tenpai,
            self.extension_renchan_plays_on,
            self.extension_multi_ron_renchan_plays_on,
            self.leftover_sticks_to_top
        )
    }
}
