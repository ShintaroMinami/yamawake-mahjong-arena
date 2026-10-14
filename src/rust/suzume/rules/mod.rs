//! The rules themselves, one small unit each; `registry.rs` orders them.

pub mod calls;
pub(crate) mod discard;
pub mod fold;
mod kyushu;
pub mod waits;
pub mod tile_features;
pub mod universal_mlp;
pub(crate) mod value_features;
mod win;

pub use fold::{FoldDiscard, PfDiscard};
pub use discard::DiscardForced;
pub use kyushu::KyushuAlways;
pub use universal_mlp::{KanMlp, UniversalMlp};
pub use win::WinAlways;
