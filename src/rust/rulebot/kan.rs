//! kobalab's `select_gang`: whether to declare an ankan or a kakan.
//!
//! Port of majiang-ai (https://github.com/kobalab/majiang-ai), MIT License,
//! Copyright (c) 2021 Satoshi Kobayashi; see docs/THIRD_PARTY_NOTICES.md.
//!
//! * No kan while some seat is in riichi and the hand is not tenpai.
//! * Within two of tenpai: the kan whose hand is worth the most by
//!   `eval_shoupai` (`kobalab_ev`), if no less than the hand without it; a kan
//!   that leaves the hand three or more away is not considered. Ties go to the
//!   later tile type, as in majiang-ai.
//! * Three away or worse: the first kan that keeps the yaku-aware shanten.
//!
//! Candidates are the engine's legal ankan / kakan, in tile-type order (the
//! order of majiang-core's `get_gang_mianzi`). Daiminkan is never declared.

use super::action::BotAction;
use super::decision::Decision;
use super::hand_shanten;
use super::kobalab_ev::NO_TILE;
use super::view::AllowedActions;
use super::yaku_shanten::Kans;

/// majiang-ai's tolerance for "no less than" on evaluations.
const EV_EPS: f32 = 1e-7;

/// One kan candidate's hand and melds after the kan (before the rinshan draw).
struct AfterKan {
    hand: [u8; 34],
    pon_sets: Vec<u8>,
    n_sets: u8,
    closed: bool,
}

impl Decision<'_> {
    /// The kan to declare on the own turn, if any.
    pub(super) fn select_kan(&self, allowed: &AllowedActions) -> Option<BotAction> {
        let mask = allowed.ankan | allowed.kakan;
        if mask == 0 {
            return None;
        }
        let state = self.state;
        let hand = state.hand();
        let n_sets = state.n_sets();
        let chi_runs = state.chi_runs();
        let pon_sets = state.pon_sets();
        let now = hand_shanten::shanten_of(&hand, n_sets);
        if state.in_riichi().iter().any(|&r| r) && now > 0 {
            return None;
        }
        let mut candidates = (0..34u8).filter(|&t| mask >> t & 1 == 1).map(|t| {
            let closed = allowed.ankan >> t & 1 == 1;
            let mut after = AfterKan { hand, pon_sets: pon_sets.to_vec(), n_sets, closed };
            if closed {
                after.hand[t as usize] -= 4;
                after.n_sets -= 1;
            } else {
                after.hand[t as usize] -= 1;
                after.pon_sets.retain(|&p| p != t);
            }
            (t, after)
        });
        let pick = if now < 3 {
            let drawn = state.drawn_tile().map_or(NO_TILE, |t| t.tile_type);
            let mut best = self.kobalab_ev().value(&hand, chi_runs, pon_sets, n_sets, drawn);
            let mut pick = None;
            for (t, a) in candidates {
                if hand_shanten::shanten_of(&a.hand, a.n_sets) >= 3 {
                    continue;
                }
                let mut ev = self.kobalab_ev();
                ev.add_kan(t, a.closed);
                let v = ev.value(&a.hand, chi_runs, &a.pon_sets, a.n_sets, NO_TILE);
                if v - best > -EV_EPS {
                    best = v;
                    pick = Some((t, a.closed));
                }
            }
            pick
        } else {
            let now = self.yaku_shanten(&hand, chi_runs, pon_sets, n_sets);
            candidates
                .find(|(t, a)| {
                    let (mut open, mut closed) = (state.open_kans().to_vec(), state.closed_kans().to_vec());
                    if a.closed { closed.push(*t) } else { open.push(*t) }
                    let kans = Kans { open: &open, closed: &closed };
                    self.yaku_shanten_with_kans(&a.hand, chi_runs, &a.pon_sets, kans, a.n_sets) == now
                })
                .map(|(t, a)| (t, a.closed))
        }?;
        Some(if pick.1 { BotAction::Ankan { tile_type: pick.0 } } else { BotAction::Kakan { tile_type: pick.0 } })
    }
}
