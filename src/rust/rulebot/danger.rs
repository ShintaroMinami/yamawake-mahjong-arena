//! How much each tile risks, and who is worth risking it against.

use super::decision::Decision;
use super::config::{DANGER_PER_THREAT, MELD_MIN_VALUE};

impl Decision<'_> {
    /// Combined danger against every threat, or `None` if nobody threatens.
    ///
    /// A threat is a riichi, or a melded hand that reads as tenpai
    /// (`open_threat`), and both weigh the same. Against two threats the worst case is what matters, so take the max
    /// rather than the sum: a tile that is genbutsu to one but wide open to
    /// the other is not half safe.
    pub(super) fn threat_danger(&self) -> Option<[u32; 34]> {
        let in_riichi = self.state.in_riichi();

        let mut worst = [0_u32; 34];
        let mut threats = 0_u32;
        // Seat 0 is this hand; only the other three can be threatening it.
        for (player, &player_in_riichi) in in_riichi.iter().enumerate().skip(1) {
            if !(player_in_riichi || self.open_threat(player)) {
                continue;
            }
            threats += 1;
            for (w, v) in worst.iter_mut().zip(self.danger(player)) {
                *w = (*w).max(v);
            }
        }
        if threats == 0 {
            return None;
        }
        // Two riichi is not one riichi. The worst-case tile is still the worst
        // case, but the price of walking into it has gone up.
        let factor = DANGER_PER_THREAT.powi(threats as i32 - 1);
        if factor != 1. {
            for value in &mut worst {
                *value = (*value as f32 * factor).round() as u32;
            }
        }
        Some(worst)
    }

    /// Whether a player with no riichi should be read as tenpai.
    ///
    /// A riichi announces itself; a melded hand does not, so the count of melds
    /// against how far the hand has gone stands in for the announcement. The
    /// common reading is three melds inside the first six discards, two by the
    /// twelfth, and one after that -- a player who has called that much, that
    /// early, has been building towards something and is close to done.
    ///
    /// Reading the melds for a yaku instead would answer a different question:
    /// what the hand is worth, not how near it is. Nearness is what the discard
    /// turns on. Damaten is left out entirely -- it shows nothing to count, and
    /// guessing at it would fold good hands against nothing.
    fn open_threat(&self, player: usize) -> bool {
        let melds = self.state.meld_count(player);
        // Melds early in the hand say more than melds late: three by the sixth
        // discard is a hand that has been racing, one by the thirteenth is just
        // a hand that got there.
        let tenpai_likely = melds > 0
            && melds
                >= match self.state.discard_count(player) {
                    0..=6 => 3,
                    7..=12 => 2,
                    _ => 1,
                };
        tenpai_likely && self.meld_value(player) >= MELD_MIN_VALUE
    }

    /// What a player's melds visibly promise: `dora + max(1, yakuhai pons)`.
    ///
    /// Everything here is face-up, so it is an estimate of the floor rather
    /// than of the hand. The `max(1, ..)` is because an open hand has to have a
    /// yaku from somewhere -- if none of the melds is a yakuhai triplet it is
    /// tanyao or a flush or toitoi, worth counting as one all the same.
    fn meld_value(&self, player: usize) -> u32 {
        let dragons_and_winds = self.yakuhai_of(player);
        let pon_sets = self
            .state
            .meld_tiles(player)
            .iter()
            .filter(|meld| {
                let first = meld[0].plain();
                meld.iter().all(|t| t.plain() == first) && dragons_and_winds.contains(&first.tile_type)
            })
            .count() as u32;
        u32::from(self.state.visible_dora()[player]) + pon_sets.max(1)
    }

    /// The tiles that are a yaku on their own for this player: the dragons,
    /// the round wind, and the wind of their own seat.
    fn yakuhai_of(&self, player: usize) -> [u8; 5] {
        // Seats are relative to this hand, so player N sits N turns downwind.
        let player_wind = 27 + ((self.state.player_wind() - 27) + player as u8) % 4;
        [31, 32, 33, self.state.round_wind(), player_wind]
    }

    /// How dangerous each tile is against one threat, lower being safer.
    ///
    /// Enumerates the wait shapes that could still be waiting on the tile and
    /// sums their relative frequency, following the scheme kobalab documents
    /// for majiang-ai: tanki 1, shanpon 2, kanchan 3, penchan 3, ryanmen 10
    /// each side. Danger then falls out of which shapes remain possible rather
    /// than from a hand-written ranking, so suji, kabe and visible counts all
    /// feed in through one mechanism:
    ///
    /// * **Genbutsu** scores 0. Furiten forbids the ron outright.
    /// * **Suji** removes a ryanmen: a ryanmen on X holds X+1,X+2 and waits on
    ///   X and X+3, so X+3 sitting in their discards makes that shape furiten.
    /// * **Kabe** removes any shape needing a tile with no copies left.
    /// * **Honors** admit only tanki and shanpon, which is why they are safe.
    ///
    /// The base is weighted as kobalab's, by the copies still unseen
    /// (`base_danger`), and a dealer threat counts half again.
    fn danger(&self, player: usize) -> [u32; 34] {
        let state = self.state;
        // `discarded` drives two separate things: which tiles score zero, and which
        // ryanmen a suji rules out. Only the first may use the wider safe set --
        // a tile another player discarded says nothing about whether this hand
        // holds the suji shape, so folding it into the suji test would silently
        // erase live ryanmen.
        let discarded = state.discard_mask(player);
        // `passed_since_riichi` is empty for a player without an accepted
        // riichi, and an empty safe set reads as "nothing is genbutsu" -- the
        // one error that costs a whole hand. Melded threats fall back to the
        // discards, which are genbutsu by furiten regardless of riichi.
        let safe = if state.in_riichi()[player] {
            state.passed_since_riichi(player)
        } else {
            discarded
        };
        let seen = state.seen_counts();
        let dora = state.dora_per_tile_type();
        // How many copies they could still be holding.
        let unseen = |t: usize| 4_u8.saturating_sub(seen[t]);
        let threat_is_oya = state.oya() as usize == player;

        let mut out = [0_u32; 34];
        for tile_type in 0..34 {
            if safe[tile_type] {
                continue; // cannot be ronned
            }
            let mut d = base_danger(tile_type, unseen(tile_type));
            if tile_type < 27 {
                d += shape_danger(tile_type, &discarded, &unseen);
            }
            // Danger is a probability; folding cares about the loss. A dora is
            // worth holding back beyond what its deal-in rate alone says, and
            // for the same reason dealing into the dealer costs more than the
            // points alone: it also keeps them in the seat.
            d += d * u32::from(dora[tile_type].min(2)) / 2;
            out[tile_type] = if threat_is_oya { d * 3 / 2 } else { d };
        }
        out
    }
}

/// Tanki and shanpon, the only two shapes an honor can be waiting as, by
/// kobalab's weights.
///
/// A flat tanki 1 + shanpon 2 would price a wholly unseen honor the same as a
/// middle number -- and a lone honor nobody has shown is the classic way to
/// feed a shanpon, which is why kobalab scores three live copies of an honor
/// at 8 against a number tile's 3.
fn base_danger(tile_type: usize, unseen: u8) -> u32 {
    match (tile_type >= 27, unseen) {
        (_, 0) => 0,
        (_, 1) => 1,
        (true, 3..) => 8,
        _ => 3,
    }
}

/// The run-based shapes a number tile can complete: kanchan, penchan and
/// the two ryanmen it sits at the end of.
fn shape_danger(tile_type: usize, discarded: &[bool; 34], unseen: &impl Fn(usize) -> u8) -> u32 {
    let n = tile_type % 9; // 0-based: 0 is a 1, 8 is a 9
    let base = tile_type - n;
    let has = |offset: i32| -> bool {
        let i = n as i32 + offset;
        (0..9).contains(&i) && unseen(base + i as usize) >= 1
    };

    let mut d = 0;
    if (1..=7).contains(&n) && has(-1) && has(1) {
        d += 3; // kanchan
    }
    if (n == 2 && has(-2) && has(-1)) || (n == 6 && has(1) && has(2)) {
        d += 3; // penchan
    }
    // Upper ryanmen: they hold n+1,n+2 and wait on n and n+3.
    if n <= 6 && has(1) && has(2) && !(n <= 5 && discarded[base + n + 3]) {
        d += 10;
    }
    // Lower ryanmen: they hold n-2,n-1 and wait on n-3 and n.
    if n >= 2 && has(-1) && has(-2) && !(n >= 3 && discarded[base + n - 3]) {
        d += 10;
    }
    d
}

#[cfg(test)]
mod test {
    use super::*;

    /// Nothing visible, so every shape is still live.
    fn all_live(_: usize) -> u8 {
        4
    }

    #[test]
    fn shapes_add_up_to_kobalab_weights() {
        let clean = [false; 34];
        // A middle number sits at the end of two ryanmen and inside a kanchan:
        // 10 + 10 + 3, plus tanki and shanpon from the base.
        assert_eq!(shape_danger(4, &clean, &all_live), 23);
        assert_eq!(base_danger(4, 4), 3);
        // A terminal has only the one ryanmen above it.
        assert_eq!(shape_danger(0, &clean, &all_live), 10);
        // A 3 is the peak of the model, not the 5: it carries the penchan 1-2
        // as well as both ryanmen and the kanchan.
        assert_eq!(shape_danger(2, &clean, &all_live), 26);
        // Honors never reach `shape_danger` at all; the base is the whole of
        // it, and weighs a live honor above a number.
        assert_eq!(base_danger(27, 4), 8);
        assert_eq!(base_danger(27, 2), 3);
    }

    #[test]
    fn suji_removes_one_ryanmen_not_both() {
        // 8m in the discards makes the 5m ryanmen that waits 5-8 furiten, but says
        // nothing about the one waiting 2-5.
        let mut discarded = [false; 34];
        discarded[7] = true;
        assert_eq!(shape_danger(4, &discarded, &all_live), 13);
        // Both ends passed: only the kanchan is left.
        discarded[1] = true;
        assert_eq!(shape_danger(4, &discarded, &all_live), 3);
    }

    #[test]
    fn kabe_removes_every_shape_through_a_dead_tile() {
        let clean = [false; 34];
        // With all four 6m gone, the 5m kanchan (4-6) and the upper ryanmen
        // (6,7) both need a tile nobody can hold.
        let dead_6m = |t: usize| if t == 5 { 0 } else { 4 };
        assert_eq!(shape_danger(4, &clean, &dead_6m), 10);
        // A tile with no copies left cannot be ronned at all, whatever shape
        // it would complete.
        assert_eq!(base_danger(4, 0), 0);
    }
}
