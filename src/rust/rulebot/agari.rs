//! Hand scoring for the rules, on riichienv-core.
//!
//! The rules ask two questions: kobalab_ev's "what does this hand pay as a
//! tsumo, dora counted by hand, +2 flat han while concealed", and
//! dama_value's "what does it pay as a silent ron". Both call HandEvaluator
//! with NO dora indicators (dora added to the han afterwards) and
//! riichi/tsumo supplied as Conditions -- for a concealed hand riichi +
//! menzen tsumo is the flat +2, computed by the same rules (fu included,
//! since the evaluator knows it is a tsumo).

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::score::calculate_score;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};

use crate::eval::feat::tiles::AKA_TIDS;

/// A complete hand to score: concealed counts, melds and winds.
pub struct ScoringHand<'a> {
    /// Concealed tiles INCLUDING the winning tile, 34-format counts.
    pub hand: &'a [u8; 34],
    pub is_menzen: bool,
    pub chi_runs: &'a [u8],
    pub pon_sets: &'a [u8],
    pub open_kans: &'a [u8],
    pub closed_kans: &'a [u8],
    /// Winds as 34-format tile types (27..30 = E..N), not riichienv's 0..3 `Wind`.
    pub round_wind: u8,
    pub player_wind: u8,
    pub win_tile_type: u8,
    pub is_ron: bool,
}

fn meld(meld_type: MeldType, tile_types: &[u8], opened: bool) -> Meld {
    // Physical ids are arbitrary for yaku purposes; reds never matter here
    // because the callers count dora themselves.
    let tiles: Vec<u8> = tile_types.iter().map(|&k| k * 4).collect();
    Meld {
        meld_type,
        tiles,
        opened,
        from_who: 0,
        called_tile: None,
    }
}

/// The winning hand's total points, or `None` when it has no yaku.
///
/// `riichi_bonus`: whether the concealed hand is credited riichi (with a
/// tsumo, riichi + menzen tsumo make the flat +2; the tsumo half comes from
/// `is_ron == false`). `dora_count`: dora han counted by the caller (indicators,
/// reds, melds). Ron returns the ron payment; tsumo the all-seats total.
pub fn agari_points(inp: &ScoringHand, riichi_bonus: bool, dora_count: u8) -> Option<f64> {
    // Callers only guarantee that hand+melds form a complete shape; the
    // winning tile may live inside a just-made meld (kobalab's call
    // valuation does this). The evaluator wants the
    // winning tile taken from the concealed part, so borrow the pair
    // tile in that case -- same yaku, fu off by a hair at most.
    let effective_win_type = if inp.hand[inp.win_tile_type as usize] > 0 {
        inp.win_tile_type
    } else {
        (0..34u8)
            .find(|&t| inp.hand[t as usize] >= 2)
            .or_else(|| (0..34u8).find(|&t| inp.hand[t as usize] >= 1))?
    };
    let mut ids: Vec<u8> = Vec::with_capacity(14);
    for t in 0..34u8 {
        let mut n = inp.hand[t as usize];
        if t == effective_win_type && n > 0 {
            n -= 1; // the winning tile is passed separately
        }
        for k in 0..n {
            ids.push(t * 4 + 3 - k);
        }
    }
    let win_tid = effective_win_type * 4;

    let mut melds: Vec<Meld> = Vec::new();
    for &c in inp.chi_runs {
        melds.push(Meld {
            meld_type: MeldType::Chi,
            tiles: vec![c * 4, (c + 1) * 4, (c + 2) * 4],
            opened: true,
            from_who: 0,
            called_tile: None,
        });
    }
    for &p in inp.pon_sets {
        melds.push(meld(MeldType::Pon, &[p, p, p], true));
    }
    for &k in inp.open_kans {
        melds.push(meld(MeldType::Daiminkan, &[k, k, k, k], true));
    }
    for &k in inp.closed_kans {
        melds.push(meld(MeldType::Ankan, &[k, k, k, k], false));
    }

    let cond = Conditions {
        tsumo: !inp.is_ron,
        riichi: riichi_bonus && inp.is_menzen,
        player_wind: Wind::from(inp.player_wind - 27),
        round_wind: Wind::from(inp.round_wind - 27),
        ..Default::default()
    };
    // Our synthetic ids can collide with the red-five ids (16/52/88 are
    // just tile_type*4), and riichienv counts akadora straight off the raw ids
    // even with no indicators -- but all dora here come from the caller
    // (`dora_count`). Count the reds we fed and
    // subtract that phantom from the returned han.
    let phantom: u32 = ids
        .iter()
        .chain(melds.iter().flat_map(|m| m.tiles.iter()))
        .chain(std::iter::once(&win_tid))
        .filter(|&&t| AKA_TIDS.contains(&t))
        .count() as u32;
    let he = HandEvaluator::new(ids, melds);
    let res = he.calc(win_tid, vec![], vec![], Some(cond));
    let yaku_han = res.han.saturating_sub(phantom);
    if !res.is_win || yaku_han == 0 {
        return None;
    }
    // Scored the way riichienv's evaluator caps it: a yakuman pays its own
    // multiple whatever dora ride on it, anything else stops at kazoe
    // yakuman (13 han).
    let han = if res.yakuman {
        (yaku_han / 13 * 13) as u8
    } else {
        (yaku_han + u32::from(dora_count)).min(13) as u8
    };
    let is_oya = inp.player_wind == 27;
    let s = calculate_score(han, res.fu as u8, is_oya, !inp.is_ron, 0, 4);
    Some(s.total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ron(hand: &[u8; 34], win_tile_type: u8, dora_count: u8) -> Option<f64> {
        let inp = ScoringHand {
            hand,
            is_menzen: true,
            chi_runs: &[],
            pon_sets: &[],
            open_kans: &[],
            closed_kans: &[],
            round_wind: 27,
            player_wind: 28,
            win_tile_type,
            is_ron: true,
        };
        agari_points(&inp, false, dora_count)
    }

    /// Chinitsu + ittsu with 20 dora is kazoe yakuman, not a double.
    #[test]
    fn kazoe_stops_at_one_yakuman() {
        let mut hand = [0u8; 34];
        for (k, n) in [(0, 3), (1, 1), (2, 2), (3, 2), (4, 2), (5, 1), (6, 1), (7, 1), (8, 1)] {
            hand[k] = n;
        }
        assert_eq!(ron(&hand, 8, 20), Some(32000.0));
    }

    /// Dora riding on a yakuman add nothing.
    #[test]
    fn dora_do_not_stack_on_yakuman() {
        let mut hand = [0u8; 34];
        for (k, n) in [(31, 3), (32, 3), (33, 3), (0, 1), (1, 1), (2, 1), (8, 2)] {
            hand[k] = n;
        }
        assert_eq!(ron(&hand, 8, 13), Some(32000.0));
    }
}
