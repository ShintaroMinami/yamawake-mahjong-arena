//! The bot's chosen action, named after riichienv's action vocabulary
//! (`py.rs` writes it out as the matching subset of mjai events).
//!
//! Actor fields are dropped (the caller knows whose decision this is);
//! no diagnostic metadata is carried.

use super::tile::Tile;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BotAction {
    Discard { tile: Tile, tsumogiri: bool },
    Riichi,
    Pon { target: u8, tile: Tile, consume_tiles: [Tile; 2] },
    Chi { target: u8, tile: Tile, consume_tiles: [Tile; 2] },
    Win,
    Pass,
    /// A closed kan of this tile type.
    Ankan { tile_type: u8 },
    /// An added kan on the pon of this tile type.
    Kakan { tile_type: u8 },
}
