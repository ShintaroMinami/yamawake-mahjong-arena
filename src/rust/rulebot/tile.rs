//! The tile value the rules work with: a tile type (0..34) with an aka flag.
//!
//! `next_tile_type`/`prev_tile_type` stay within the 0..34 tile-type space
//! the callers guarantee (chi construction on numbers).

use super::aka_slot;
use crate::eval::feat::tiles::is_aka_tid;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tile {
    pub tile_type: u8,
    pub is_aka: bool,
}

impl Tile {
    pub const fn new(tile_type: u8, is_aka: bool) -> Self {
        Self { tile_type, is_aka }
    }

    /// The tile a 136-format id stands for.
    pub fn from_tid(tid: u8) -> Self {
        Self::new(tid / 4, is_aka_tid(tid))
    }

    /// The same tile type without the aka flag.
    pub const fn plain(self) -> Self {
        Self { tile_type: self.tile_type, is_aka: false }
    }

    pub const fn tile_type_index(self) -> usize {
        self.tile_type as usize
    }

    /// The aka copy of this tile type when one exists, else the plain tile.
    pub const fn as_aka(self) -> Self {
        Self {
            tile_type: self.tile_type,
            is_aka: aka_slot(self.tile_type as usize).is_some(),
        }
    }

    /// The next tile type up, plain.
    pub const fn next_tile_type(self) -> Self {
        Self { tile_type: self.tile_type + 1, is_aka: false }
    }

    /// The next tile type down, plain.
    pub const fn prev_tile_type(self) -> Self {
        Self { tile_type: self.tile_type - 1, is_aka: false }
    }
}
