//! kobalab's `Player#xiangting`: how far the hand is from a tenpai it can win.
//!
//! Plain shanten counts steps to a complete hand, which for an open hand is the
//! wrong question -- a melded hand with no yaku cannot win at all, so the shape
//! it is closest to may be worth nothing. This takes the minimum over the
//! handful of yaku a hand can commit to without a search, and reports how far
//! the nearest one is.
//!
//! Seven routes: menzen, yakuhai (with the pair-and-wait case), tanyao, toitoi,
//! and a flush in each suit. Six of them are the standard shanten function run
//! over a masked hand -- zero out the tiles the yaku forbids and the ordinary
//! calculator answers for it -- and toitoi is closed form. That masking trick is
//! why this is cheap enough to run on every call candidate.
//!
//! Left out, as in majiang-ai: sanshoku, ittsuu, chanta, and chiitoi as a goal
//! of its own (`shanten_of` still finds it inside the menzen route).

use super::decision::Decision;
use super::hand_shanten;

/// A hand's kans by tile type: open (daiminkan, kakan) and closed (ankan).
#[derive(Clone, Copy)]
pub(super) struct Kans<'a> {
    pub open: &'a [u8],
    pub closed: &'a [u8],
}

impl Kans<'_> {
    fn iter(&self) -> impl Iterator<Item = &u8> {
        self.open.iter().chain(self.closed.iter())
    }
}

/// Unreachable, for a route this hand has already given up.
const INF: i8 = i8::MAX;

/// Tiles a flush in `suit` may keep: that suit and the honors.
fn mask_to_suit(hand: &[u8; 34], suit: usize) -> [u8; 34] {
    let mut out = *hand;
    for other in 0..3 {
        if other != suit {
            out[other * 9..other * 9 + 9].fill(0);
        }
    }
    out
}

/// Tiles tanyao may keep: 2 through 8 of each suit.
fn mask_to_simples(hand: &[u8; 34]) -> [u8; 34] {
    let mut out = *hand;
    for suit in 0..3 {
        out[suit * 9] = 0;
        out[suit * 9 + 8] = 0;
    }
    out[27..34].fill(0);
    out
}

/// The three tiles a chi holds, given the lowest of them.
const fn chi_tiles(low: u8) -> [u8; 3] {
    [low, low + 1, low + 2]
}

/// Whether a tile is a terminal or an honor.
const fn is_terminal_or_honor(tile_type: u8) -> bool {
    crate::eval::feat::tiles::is_yaochu(tile_type as usize)
}

impl Decision<'_> {
    /// Shanten to the nearest tenpai this hand could actually win from.
    ///
    /// `hand`, `chi_runs` and `pon_sets` are passed rather than read off the state so
    /// a call candidate can be measured before it is made. Kans come from the
    /// state (`yaku_shanten_with_kans` takes them too).
    pub(super) fn yaku_shanten(
        &self,
        hand: &[u8; 34],
        chi_runs: &[u8],
        pon_sets: &[u8],
        n_sets: u8,
    ) -> i8 {
        let kans = Kans { open: self.state.open_kans(), closed: self.state.closed_kans() };
        self.yaku_shanten_with_kans(hand, chi_runs, pon_sets, kans, n_sets)
    }

    /// `yaku_shanten` with the kans given, so a kan can be measured before it
    /// is declared.
    pub(super) fn yaku_shanten_with_kans(
        &self,
        hand: &[u8; 34],
        chi_runs: &[u8],
        pon_sets: &[u8],
        kans: Kans<'_>,
        n_sets: u8,
    ) -> i8 {
        let melds = chi_runs.len() + pon_sets.len() + kans.open.len();
        let open = melds > 0;

        let menzen_shanten = if open {
            INF
        } else {
            hand_shanten::shanten_of(hand, n_sets)
        };

        [
            menzen_shanten,
            self.yakuhai_shanten(hand, pon_sets, kans, n_sets),
            self.tanyao_shanten(hand, chi_runs, pon_sets, kans, n_sets),
            self.toitoi_shanten(hand, chi_runs, pon_sets, kans),
            self.flush_shanten(hand, chi_runs, pon_sets, kans, 0, n_sets),
            self.flush_shanten(hand, chi_runs, pon_sets, kans, 1, n_sets),
            self.flush_shanten(hand, chi_runs, pon_sets, kans, 2, n_sets),
        ]
        .into_iter()
        .min()
        .unwrap_or(INF)
    }

    /// Round wind, seat wind and the three dragons.
    fn is_yakuhai_tile(&self, tile_type: u8) -> bool {
        matches!(tile_type, 31..=33) || tile_type == self.state.round_wind() || tile_type == self.state.player_wind()
    }

    /// Shanten to a hand carrying a yakuhai triplet.
    ///
    /// A triplet already held, melded or kanned settles it, and the hand is then worth
    /// its plain shanten. Otherwise a pair still counts, but only while a third
    /// copy is unseen -- kobalab prices that as the pon it would take, so the
    /// hand is measured one meld lighter and one step further out.
    fn yakuhai_shanten(&self, hand: &[u8; 34], pon_sets: &[u8], kans: Kans<'_>, n_sets: u8) -> i8 {
        let held = (27..34).any(|tile_type| self.is_yakuhai_tile(tile_type as u8) && hand[tile_type] >= 3)
            || pon_sets.iter().chain(kans.iter()).any(|&tile_type| self.is_yakuhai_tile(tile_type));
        if held {
            return hand_shanten::shanten_of(hand, n_sets);
        }
        if n_sets == 0 {
            return INF;
        }
        let seen = self.state.seen_counts();
        let backing = (27..34).find(|&tile_type| {
            self.is_yakuhai_tile(tile_type as u8) && hand[tile_type] == 2 && seen[tile_type] < 4
        });
        match backing {
            Some(tile_type) => {
                let mut probe = *hand;
                probe[tile_type] -= 2;
                hand_shanten::shanten_of(&probe, n_sets - 1).saturating_add(1)
            }
            None => INF,
        }
    }

    /// Shanten to a tanyao hand, which no meld may have spoiled.
    fn tanyao_shanten(&self, hand: &[u8; 34], chi_runs: &[u8], pon_sets: &[u8], kans: Kans<'_>, n_sets: u8) -> i8 {
        let spoiled = chi_runs
            .iter()
            .any(|&low| chi_tiles(low).into_iter().any(is_terminal_or_honor))
            || pon_sets.iter().chain(kans.iter()).copied().any(is_terminal_or_honor);
        if spoiled {
            return INF;
        }
        hand_shanten::shanten_of(&mask_to_simples(hand), n_sets)
    }

    /// Shanten to toitoi, counted rather than searched.
    ///
    /// Every block is a triplet or the pair, so the shanten is fixed by how
    /// many of each the hand already has -- no calculator needed. A chi rules
    /// the route out outright.
    fn toitoi_shanten(&self, hand: &[u8; 34], chi_runs: &[u8], pon_sets: &[u8], kans: Kans<'_>) -> i8 {
        if !chi_runs.is_empty() {
            return INF;
        }
        let mut triplets = pon_sets.len() + kans.open.len() + kans.closed.len();
        let mut pairs = 0;
        for &count in hand.iter() {
            if count >= 3 {
                triplets += 1;
            } else if count == 2 {
                pairs += 1;
            }
        }
        // Four triplets and a pair is the whole hand; anything past that is
        // spare and cannot count towards the shanten.
        if triplets + pairs > 5 {
            pairs = 5 - triplets.min(5);
        }
        8 - (triplets as i8) * 2 - pairs as i8
    }

    /// Shanten to a flush in `suit`, honors included -- honitsu, not chinitsu.
    fn flush_shanten(
        &self,
        hand: &[u8; 34],
        chi_runs: &[u8],
        pon_sets: &[u8],
        kans: Kans<'_>,
        suit: usize,
        n_sets: u8,
    ) -> i8 {
        let in_suit = |tile_type: u8| tile_type >= 27 || (tile_type as usize) / 9 == suit;
        let spoiled = chi_runs.iter().any(|&low| !in_suit(low))
            || pon_sets.iter().chain(kans.iter()).copied().any(|tile_type| !in_suit(tile_type));
        if spoiled {
            return INF;
        }
        hand_shanten::shanten_of(&mask_to_suit(hand, suit), n_sets)
    }
}

#[cfg(test)]
mod test {
    use super::{is_terminal_or_honor, mask_to_simples, mask_to_suit};

    #[test]
    fn masks_keep_only_what_the_yaku_allows() {
        let mut hand = [0_u8; 34];
        hand[0] = 1; // 1m
        hand[4] = 1; // 5m
        hand[8] = 1; // 9m
        hand[9] = 1; // 1p
        hand[30] = 1; // north

        let simples = mask_to_simples(&hand);
        assert_eq!(simples[0], 0, "1m is not a simple");
        assert_eq!(simples[8], 0, "9m is not a simple");
        assert_eq!(simples[9], 0, "1p is not a simple");
        assert_eq!(simples[30], 0, "honors are not simples");
        assert_eq!(simples[4], 1, "5m survives");

        let manzu = mask_to_suit(&hand, 0);
        assert_eq!(manzu[9], 0, "another suit is dropped");
        assert_eq!(manzu[30], 1, "honors stay in a honitsu");
        assert_eq!(manzu[0], 1);
    }

    #[test]
    fn terminal_or_honor_covers_terminals_and_honors() {
        for tile_type in [0, 8, 9, 17, 18, 26, 27, 33] {
            assert!(is_terminal_or_honor(tile_type), "{tile_type} is a terminal or honor");
        }
        for tile_type in [1, 7, 10, 16, 19, 25] {
            assert!(!is_terminal_or_honor(tile_type), "{tile_type} is a simple");
        }
    }
}
