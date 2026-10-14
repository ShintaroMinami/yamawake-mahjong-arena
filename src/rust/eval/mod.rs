//! Hand evaluation shared by the bots: the single-player DP (`dp`; method
//! credited in docs/THIRD_PARTY_NOTICES.md, no third-party code), its scoring
//! (`score`), shanten with one tile added or removed (`shanten`) and the tile
//! helpers (`feat`).

pub mod dp;
pub mod feat;
pub mod hash;
pub mod score;
pub mod shanten;
