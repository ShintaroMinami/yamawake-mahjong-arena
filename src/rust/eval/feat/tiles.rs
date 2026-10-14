//! Tile notation.

use riichienv_core::parser::mjai_to_tid;

/// 136-format red five ids, in aka-slot order (5m, 5p, 5s).
pub const AKA_TIDS: [u8; 3] = [16, 52, 88];

/// The tile types that carry a red five, in aka-slot order.
pub const AKA_TILE_TYPES: [usize; 3] = [4, 13, 22];

/// Whether a 136-format id is a red five.
pub fn is_aka_tid(tid: u8) -> bool {
    AKA_TIDS.contains(&tid)
}

/// The red-five slot a tile type occupies, if it carries one at all.
pub const fn aka_slot(tile_type: usize) -> Option<usize> {
    match tile_type {
        4 => Some(0),
        13 => Some(1),
        22 => Some(2),
        _ => None,
    }
}

/// The red-five slot of a 136-format id, if it is a red five.
pub fn aka_slot_of_tid(tid: u32) -> Option<usize> {
    AKA_TIDS.iter().position(|&a| u32::from(a) == tid)
}

/// A terminal or an honor (a tile type).
pub const fn is_yaochu(tile_type: usize) -> bool {
    tile_type >= 27 || tile_type % 9 == 0 || tile_type % 9 == 8
}

/// mjai tile string -> 136-format id, via riichienv's own parser.
///
/// Errors on anything outside the 136 ids (riichienv maps '8z'/'9z' past
/// the end) -- an unknown notation means the event source changed and must
/// not be guessed around.
pub fn parse_tile(s: &str) -> Result<u8, String> {
    mjai_to_tid(s)
        .filter(|&tid| tid < 136)
        .ok_or_else(|| format!("unknown tile notation {s:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dialects() {
        assert_eq!(parse_tile("5mr").unwrap(), 16);
        assert_eq!(parse_tile("0m").unwrap(), 16);
        assert_eq!(parse_tile("5m").unwrap() / 4, 4);
        assert_eq!(parse_tile("1z").unwrap() / 4, 27);
        assert_eq!(parse_tile("7z").unwrap() / 4, 33);
        assert_eq!(parse_tile("E").unwrap() / 4, 27);
        assert_eq!(parse_tile("9s").unwrap() / 4, 26);
        assert!(parse_tile("x").is_err());
        assert!(parse_tile("8z").is_err());
        assert!(parse_tile("9z").is_err());
    }
}
