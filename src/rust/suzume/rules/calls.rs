//! Calls as the universal features read them: the options of a call
//! decision, the view right after a chi or pon, the discard that would follow
//! it, and the yaku paths of a hand.

use riichienv_core::action::{Action, ActionType};

use crate::eval::feat::tiles::{AKA_TIDS, aka_slot, aka_slot_of_tid, is_aka_tid, is_yaochu};
use crate::suzume::params::params;
use crate::suzume::ctx::Ctx;
use crate::suzume::sp::DpRow;
use crate::suzume::view::MjView;

use super::value_features::value_ranked;

/// The flush path's share threshold: (most held suit + honors) / all tiles.
pub const FLUSH_SHARE_MIN: f64 = 0.75;

/// What an option is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallKind {
    Pass,
    Chi,
    Pon,
}

/// One option of a call decision.
#[derive(Clone, Debug)]
pub struct CallOption {
    pub kind: CallKind,
    /// For a chi, where the called tile sits in the run: 0 low, 1 mid,
    /// 2 high; 0 otherwise.
    pub chi_pos: u8,
    /// The call spends a red five from the hand.
    pub red: bool,
    /// The legal action realising it (the first legal one of its kind).
    pub action: Action,
}

/// Pass, then every distinct legal chi (by position, plain before red),
/// then every distinct legal pon (plain before red). Empty when pass is not
/// legal.
pub fn call_options(view: &MjView) -> Vec<CallOption> {
    let Some(pass) = view.find(ActionType::Pass) else { return Vec::new() };
    let mut out = vec![CallOption { kind: CallKind::Pass, chi_pos: 0, red: false, action: pass.clone() }];
    let mut calls: Vec<CallOption> = Vec::new();
    for a in &view.legal {
        let kind = match a.action_type {
            ActionType::Chi => CallKind::Chi,
            ActionType::Pon => CallKind::Pon,
            _ => continue,
        };
        let Some(called) = a.tile else { continue };
        let consumed = a.consume_tiles.clone();
        let chi_pos = if kind == CallKind::Chi {
            consumed.iter().filter(|&&t| t / 4 < called / 4).count() as u8
        } else {
            0
        };
        let red = consumed.iter().any(|&t| is_aka_tid(t));
        if !calls.iter().any(|o| (o.kind, o.chi_pos, o.red) == (kind, chi_pos, red)) {
            calls.push(CallOption { kind, chi_pos, red, action: a.clone() });
        }
    }
    calls.sort_by_key(|o| (o.kind == CallKind::Pon, o.chi_pos, o.red));
    out.extend(calls);
    out
}

/// What the hand-plan state and the yaku-path features read of a hand: a
/// 3n+1 concealed hand `h` with `melds`.
pub(crate) struct Profile {
    pub(crate) yakuhai_set: bool,
    /// Live yakuhai pairs, a double wind counting 2.
    pub(crate) yakuhai_pairs: f64,
    pub(crate) yaochu: u32,
    pub(crate) meld_yaochu: bool,
    pub(crate) flush_share: f64,
    pub(crate) flush_dead: bool,
    pub(crate) toitoi_sets: u32,
    pub(crate) chi: bool,
    /// Tile counts of the whole hand, melds included (a kan counts 4).
    pub(crate) all: [u8; 34],
}

impl Profile {
    pub(crate) fn of(v: &MjView, h: &[u8; 34], melds: &[(u8, Vec<u8>, bool)]) -> Self {
        let mut all = *h;
        let (mut set_tiles, mut chi, mut suits, mut meld_yaochu) = (0u32, false, [false; 3], false);
        let mut yakuhai_set = false;
        for (mt, ids, _) in melds {
            let t0 = (ids[0] / 4) as usize;
            for &id in ids {
                let t = (id / 4) as usize;
                all[t] += 1;
                meld_yaochu |= is_yaochu(t);
                if t < 27 {
                    suits[t / 9] = true;
                }
            }
            if *mt == 0 {
                chi = true;
            } else {
                set_tiles += 1;
                yakuhai_set |= v.is_yakuhai(t0 as u8);
            }
        }
        yakuhai_set |= (27..34).any(|t| h[t] >= 3 && v.is_yakuhai(t as u8));
        let yakuhai_pairs = (27..34).filter(|&t| v.live_yakuhai_pair(h, t)).map(|t| wind_weight(v, t)).sum();
        let yaochu: u32 = (0..34).filter(|&t| is_yaochu(t)).map(|t| u32::from(h[t])).sum();
        let total: u32 = all.iter().map(|&n| u32::from(n)).sum();
        let honors: u32 = all[27..].iter().map(|&n| u32::from(n)).sum();
        let suit_max = (0..3).map(|s| all[s * 9..s * 9 + 9].iter().map(|&n| u32::from(n)).sum::<u32>()).max();
        Profile {
            yakuhai_set,
            yakuhai_pairs,
            yaochu,
            meld_yaochu,
            flush_share: f64::from(suit_max.unwrap_or(0) + honors) / f64::from(total.max(1)),
            flush_dead: suits.iter().filter(|&&s| s).count() >= 2,
            toitoi_sets: set_tiles + h.iter().filter(|&&n| n >= 2).count() as u32,
            chi,
            all,
        }
    }

    pub(crate) fn tanyao_path(&self) -> bool {
        !self.meld_yaochu && self.yaochu <= 2
    }

    pub(crate) fn flush_path(&self) -> bool {
        !self.flush_dead && self.flush_share >= FLUSH_SHARE_MIN
    }

    pub(crate) fn toitoi_path(&self) -> bool {
        !self.chi && self.toitoi_sets >= 4
    }
}

/// A double wind (round and seat) counts 2.
fn wind_weight(v: &MjView, t: usize) -> f64 {
    if t as u8 == v.round_wind && t as u8 == v.player_wind { 2.0 } else { 1.0 }
}

/// How the discard after a call was chosen: its DP row, or by the
/// `value_features` score when the hand is out of DP range.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AfterCallValue {
    Dp(DpRow),
    Vf,
}

/// The discard the discard rules would make right after a call, in the
/// context `actx` of the view `after` it: the index in
/// `actx.discard_cands()` and how it was valued: the legal discard of
/// highest DP EV (`AfterCallValue::Dp`) when the 14 tiles' shanten equals
/// the lowest after-discard shanten, is within `params().dp_max_shanten` and
/// the DP solves (a draw left), else the linear value score's winner
/// (`value_features::value_ranked`, `AfterCallValue::Vf`). `None` without a
/// legal discard.
pub(crate) fn after_call_discard(actx: &Ctx, after: &MjView) -> Option<(usize, AfterCallValue)> {
    let cands = actx.discard_cands();
    let s_min = actx.min_shanten()?;
    // The DP values only the discards that keep the 14-tile shanten: within
    // range when a legal one does.
    let s14 = actx.tables().shanten(&after.hand);
    if s_min == s14 && s_min <= params().dp_max_shanten {
        if let Some(rows) = actx.dp(false) {
            let best = rows
                .iter()
                .filter(|r| cands.iter().any(|c| (c.tile_type, c.is_aka) == (r.tile_type, r.is_aka)))
                .reduce(|a, b| if b.ev > a.ev { b } else { a });
            if let Some(r) = best {
                let i = cands.iter().position(|c| (c.tile_type, c.is_aka) == (r.tile_type, r.is_aka))?;
                return Some((i, AfterCallValue::Dp(*r)));
            }
        }
    }
    let (i, _) = value_ranked(actx)?;
    Some((i, AfterCallValue::Vf))
}

/// The own-turn view right after `action` (a chi or pon): the meld added,
/// its tiles out of the hand, and the legal discards minus the kuikae ones
/// (riichienv's rule). Nothing drawn; the same live wall.
pub fn view_after_call(view: &MjView, action: &Action) -> MjView {
    let called = action.tile.expect("a call names its tile");
    let consumed = action.consume_tiles.clone();
    let mut hand = view.hand;
    let mut akas_in_hand = view.akas_in_hand;
    for &t in &consumed {
        hand[(t / 4) as usize] -= 1;
        if let Some(slot) = aka_slot_of_tid(u32::from(t)) {
            akas_in_hand[slot] = false;
        }
    }
    let mut tiles = consumed.clone();
    tiles.push(called);
    tiles.sort_unstable();
    let meld_type = if action.action_type == ActionType::Chi { 0 } else { 1 };
    let forbidden = kuikae_forbidden(called / 4, &consumed, meld_type == 0);
    // One discard per tile type and red five (ids as the test notation makes them).
    let mut legal = Vec::new();
    for t in 0..34u8 {
        let n = hand[t as usize];
        if n == 0 || forbidden >> t & 1 == 1 {
            continue;
        }
        let red = aka_slot(t as usize).filter(|&s| akas_in_hand[s]);
        if let Some(s) = red {
            legal.push(Action::new(ActionType::Discard, Some(AKA_TIDS[s]), vec![], action.actor));
        }
        if n > u8::from(red.is_some()) {
            legal.push(Action::new(ActionType::Discard, Some(t * 4 + 3), vec![], action.actor));
        }
    }
    let mut own_melds = view.own_melds.clone();
    own_melds.push((meld_type, tiles, true));
    MjView {
        seat: view.seat,
        hand,
        akas_in_hand,
        seen_counts: view.seen_counts,
        akas_seen: view.akas_seen,
        dora_indicator_types: view.dora_indicator_types.clone(),
        dora_per_tile_type: view.dora_per_tile_type,
        round_wind: view.round_wind,
        player_wind: view.player_wind,
        score: view.score,
        kyoku_index: view.kyoku_index,
        scores: view.scores,
        honba: view.honba,
        sticks: view.sticks,
        lead: view.lead,
        drawable_count: view.drawable_count,
        drawn_tile: None,
        in_riichi: false,
        own_melds,
        draws_before_us: 0,
        double_riichi_possible: false,
        own_discards: view.own_discards,
        n_own_discards: view.n_own_discards,
        last_ago: view.last_ago,
        threats: view.threats,
        legal,
    }
}

/// Tile types (bit mask) that may not be discarded right after calling
/// `called` with `consumed` (riichienv's kuikae rule): the called type, and
/// for a chi taking the two tiles on one side of it, the type three away
/// on the other side (suji kuikae).
pub fn kuikae_forbidden(called: u8, consumed: &[u8], is_chi: bool) -> u64 {
    let mut mask = 1u64 << called;
    if is_chi {
        let mut c: Vec<u8> = consumed.iter().map(|&t| t / 4).collect();
        c.sort_unstable();
        if c == [called + 1, called + 2] && called % 9 <= 5 {
            mask |= 1 << (called + 3);
        } else if called >= 2 && c == [called - 2, called - 1] && called % 9 >= 3 {
            mask |= 1 << (called - 3);
        }
    }
    mask
}
