//! The universal features U: every option of one decision as one row, the
//! input of the per-decision MLPs (`rules::universal_mlp`), the same rows the
//! models were fitted on.
//! Design: docs/suzume.md.
//!
//! Decision kinds and their options (`(kind, action id)`, the ids the models were fitted on:
//! tiles 0-33, red fives 34-36, chi 38-40 / red 76-78 by the called tile's
//! place, pon 41 / red 79, pass 82; a riichi option carries its tile):
//!
//! | decision | options | A (the hand after) | values |
//! |---|---|---|---|
//! | `discard` (lowest after-discard shanten >= 1) | `discard` per legal discard | the 13 tiles left | DP (`Ctx::dp(assume_riichi)`) within `params().dp_max_shanten`, shanten-keeping discards only |
//! | `riichi` (riichi legal) | `dama` per tenpai-keeping discard, `break` per other discard, then `riichi` per tenpai-keeping discard | the 13 tiles left | dama DP row; `sp::pass_value` of the 13 tiles for a break; riichi DP row minus `deposit * (1 - win)` |
//! | `tenpai` (lowest after-discard shanten 0, riichi not legal) | `dama` / `break` per legal discard | the 13 tiles left | dama DP row; `sp::pass_value` for a break |
//! | `kan` (`kan_universal`: an ankan / kakan legal; in riichi too) | `keep` (the row of the option the decision's own MLP picks, its kind and id; in riichi the tsumogiri, `riichi_keep_row`) then each kan (`ankan` / `kakan`, id 42 + tile type) | keep: as that option; a kan: the concealed tiles after it with the kan among the melds, the replacement draw next | keep: as that option; a kan: `sp::pass_value` of the hand after it (riichi assumed when the hand stays closed and affordable) |
//! | `call` (chi or pon legal) | `pass`, then each `call_options` chi / pon, then each `daiminkan` (id 42 + tile type) | pass: the 13 tiles held; a chi / pon: after the meld and the discard that follows (`calls::after_call_discard`); a daiminkan: as a kan option | pass: the DP's no-discard root (riichi assumed when affordable); a chi / pon: the DP row of that discard, or none out of DP range; a daiminkan: as a kan option |
//!
//! Anything else (in riichi, no discard, a declaring discard) is `none` with
//! no option. Riichi's declaring discard right after the riichi action is
//! not told apart from a tenpai decision here: callers skip it.
//!
//! Features (`U_NAMES`; units raw, points in thousands):
//!
//! - A, the hand after the option: `A_ev` / `A_win` / `A_tenpai` (DP; 0
//!   outside, `A_in_dp`), `A_shanten`, `A_ukeire` (the DP row's when in DP,
//!   else `UkeireTables::ukeire`), `A_ukeire2` (two-step ukeire when the
//!   shanten is the lowest of the decision's options and above 0, else 0),
//!   waits by kind, `A_tanki`, `A_ron_points` / `A_dama_tsumo_points`
//!   (dama, per wait weighted by unseen copies; tenpai only), `A_menzen`,
//!   `A_no_yaku` (in DP with win exactly 0), `A_break_tenpai` /
//!   `A_break_yaku_tenpai` (`break` options), yaku paths
//!   (`calls::Profile`), `A_change_yaku` / `A_change_wait` (tenpai only, with
//!   `change`), `A_wait_suji` (the own river plus the option's discard),
//!   `A_dora` (the whole hand, melds included);
//! - B, the moved tile (the discard; for a call the called tile on the hand
//!   before the call) and, for a call, `B_d_*` of the discard after it (on
//!   the hand after the meld); 0 for pass: `tile_features` plus
//!   `suji_held`, `B_pair_break_4plus`, `B_suit_share`;
//!   `B_chi_from_ryanmen` (calls);
//! - O: `O_riichi`, `O_call`, `O_break`, `O_ankan` / `O_kakan` (kan
//!   options; a daiminkan, a call option, has no flag of its own);
//! - R over all options of the decision (pass included): `R_ev_pct` (in-DP
//!   options against the best in-DP one, 0 for the others), `R_ev_near5`,
//!   `R_uk2_short`, `R_sh_loss`;
//! - products: `X_adv_progress` ((reference shanten - `A_shanten`) x
//!   progress; the reference is the lowest after-discard shanten on the own
//!   turn, the 13 tiles held while reacting), `X_progress2_call`.
//!
//! The state (one split for every decision): decision kind x hand type
//! (`chiitoi`: closed with 4+ pairs; `flush`: (most held suit + honors) / all
//! tiles >= 0.7 with the melds in one suit at most; `yakuhai`: a yakuhai
//! triplet held or melded, or a yakuhai held twice; else `general`) x turn
//! (`turn` 1-6 `early`, else `later`), on the hand before the
//! decision.

use riichienv_core::action::ActionType;

use crate::eval::feat::tiles::{aka_slot, is_aka_tid};
use crate::eval::hash::pack;
use crate::suzume::config::DRAWABLE_AT_START;
use crate::suzume::params::params;
use crate::suzume::ctx::{Ctx, DiscardCand};
use crate::suzume::rules::calls::{
    AfterCallValue, CallKind, Profile, after_call_discard, call_options, view_after_call,
};
use crate::suzume::rules::waits::{is_tanki, rc_round_bin, wait_value, waits_of};
use crate::suzume::rules::tile_features::{
    AKA, DORA, DORA_ADJ1, DORA_ADJ2, DRAGON, GUEST_WIND, ISOLATED, IS_28, NB_UNSEEN1, NB_UNSEEN2, PAIR_BREAK,
    SET_BREAK, TANYAO_STEP, TERMINAL, UNSEEN, YAKUHAI_WIND, tie_features,
};
use crate::suzume::sp::{deposit_value, pass_value};
use crate::suzume::view::MjView;
use crate::eval::score::Scorer;

/// The tile features of one moved tile, in `B_*` / `B_d_*` order.
pub const N_TILE: usize = 19;
const TILE_NAMES: [&str; N_TILE] = [
    "kind_honor_yaku",
    "kind_honor_guest",
    "kind_19",
    "kind_28",
    "double_wind",
    "dora",
    "unseen",
    "isolated",
    "set_break",
    "pair_break",
    "term_suji_held",
    "edge_suji_held",
    "tanyao_step",
    "pair_break_4plus",
    "suit_share",
    "nb_unseen1",
    "nb_unseen2",
    "dora_adj1",
    "dora_adj2",
];

/// Number of features.
pub const N_U: usize = 38 + 2 * N_TILE;

/// Feature names, in row order.
pub fn u_names() -> &'static [String] {
    static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let mut v: Vec<String> = HEAD.iter().map(|s| s.to_string()).collect();
        v.extend(TILE_NAMES.iter().map(|s| format!("B_{s}")));
        v.extend(TILE_NAMES.iter().map(|s| format!("B_d_{s}")));
        v
    })
}

const HEAD: [&str; 38] = [
    "A_ev",
    "A_win",
    "A_tenpai",
    "A_in_dp",
    "A_shanten",
    "A_ukeire",
    "A_ukeire2",
    "A_wait_honor",
    "A_wait_edge",
    "A_wait_mid",
    "A_tanki",
    "A_ron_points",
    "A_menzen",
    "A_no_yaku",
    "O_riichi",
    "O_call",
    "O_break",
    "R_ev_pct",
    "R_ev_near5",
    "R_uk2_short",
    "R_sh_loss",
    "X_adv_progress",
    "X_progress2_call",
    "A_break_tenpai",
    "A_break_yaku_tenpai",
    "A_path_tanyao",
    "A_path_flush",
    "A_path_toitoi",
    "A_yakuhai_pairs",
    "A_yakuhai_set",
    "A_change_yaku",
    "A_change_wait",
    "A_dama_tsumo_points",
    "A_wait_suji",
    "A_dora",
    "B_chi_from_ryanmen",
    "O_ankan",
    "O_kakan",
];

const A_EV: usize = 0;
const A_WIN: usize = 1;
const A_TENPAI: usize = 2;
const A_IN_DP: usize = 3;
const A_SHANTEN: usize = 4;
const A_UKEIRE: usize = 5;
const A_UKEIRE2: usize = 6;
const A_WAIT_HONOR: usize = 7;
const A_WAIT_EDGE: usize = 8;
const A_WAIT_MID: usize = 9;
const A_TANKI: usize = 10;
const A_RON_POINTS: usize = 11;
const A_MENZEN: usize = 12;
const A_NO_YAKU: usize = 13;
const O_RIICHI: usize = 14;
const O_CALL: usize = 15;
const O_BREAK: usize = 16;
const R_EV_PCT: usize = 17;
const R_EV_NEAR5: usize = 18;
const R_UK2_SHORT: usize = 19;
const R_SH_LOSS: usize = 20;
const X_ADV_PROGRESS: usize = 21;
const X_PROGRESS2_CALL: usize = 22;
const A_BREAK_TENPAI: usize = 23;
const A_BREAK_YAKU_TENPAI: usize = 24;
const A_PATH_TANYAO: usize = 25;
const A_PATH_FLUSH: usize = 26;
const A_PATH_TOITOI: usize = 27;
const A_YAKUHAI_PAIRS: usize = 28;
const A_YAKUHAI_SET: usize = 29;
const A_CHANGE_YAKU: usize = 30;
const A_CHANGE_WAIT: usize = 31;
const A_DAMA_TSUMO_POINTS: usize = 32;
const A_WAIT_SUJI: usize = 33;
const A_DORA: usize = 34;
const B_CHI_FROM_RYANMEN: usize = 35;
const O_ANKAN: usize = 36;
const O_KAKAN: usize = 37;
/// First `B_*` tile feature; `B_d_*` follow at `B_D`.
pub const B: usize = 38;
pub const B_D: usize = B + N_TILE;

/// The decision-level context (the same for every option).
#[derive(Clone, Copy, Debug, Default)]
pub struct UContext {
    pub turn: i32,
    /// `1 - drawable_count / 70`, clamped to 0..1.
    pub progress: f64,
    /// `waits::rc_round_bin` (0 E1, 1 E2-4, 2 S1-3, 3 all-last).
    pub round_bin: usize,
    /// Leads every other seat (`lead > params().context_top_margin`).
    pub top: bool,
    pub oya: bool,
    /// `X_adv_progress`'s reference shanten.
    pub shanten_ref: i8,
}

/// One decision's options and rows.
#[derive(Clone, Debug)]
pub struct Universal {
    pub options: Vec<(&'static str, u16)>,
    pub rows: Vec<[f64; N_U]>,
    /// The state name (`kind/hand type/turn`) and id.
    pub state: (String, usize),
    pub context: UContext,
}

const KINDS: [&str; 5] = ["discard", "riichi", "tenpai", "call", "kan"];
const HAND_TYPES: [&str; 4] = ["chiitoi", "flush", "yakuhai", "general"];

/// The action id of a discard (the module docs' ids).
fn tile_id(t: u8, aka: bool) -> u16 {
    if aka { 34 + u16::from(t / 9) } else { u16::from(t) }
}

/// The scorer the dama checks use: this seat's winds and dora, `melds` as
/// the DP gets them.
fn scorer_of(v: &MjView, melds: &[(u8, Vec<u8>, bool)]) -> Scorer {
    let spec = melds.iter().map(|(t, ids, opened)| (*t, ids.clone(), *opened, 0)).collect();
    Scorer::new(false, v.player_wind, v.round_wind, v.dora_indicator_types.clone(), 0, spec)
}

/// The hand type (`HAND_TYPES`) of a view's hand before the decision.
fn hand_type(v: &MjView) -> usize {
    let h = &v.hand;
    let open = v.is_open();
    if !open && h.iter().filter(|&&n| n >= 2).count() >= 4 {
        return 0;
    }
    let mut all = *h;
    let mut suits = [false; 3];
    let mut set = false;
    for (mt, ids, _) in &v.own_melds {
        for &id in ids {
            let t = usize::from(id / 4);
            all[t] += 1;
            if t < 27 {
                suits[t / 9] = true;
            }
        }
        set |= *mt != 0 && v.is_yakuhai(ids[0] / 4);
    }
    let n: u32 = all.iter().map(|&x| u32::from(x)).sum();
    let honors: u32 = all[27..].iter().map(|&x| u32::from(x)).sum();
    let suit = (0..3).map(|s| all[s * 9..s * 9 + 9].iter().map(|&x| u32::from(x)).sum::<u32>()).max().unwrap_or(0);
    if suits.iter().filter(|&&s| s).count() <= 1 && n > 0 && f64::from(suit + honors) / f64::from(n) >= 0.7 {
        return 1;
    }
    if set || (27..34).any(|t| v.is_yakuhai(t as u8) && h[t] >= 2) {
        return 2;
    }
    3
}

/// `term_suji_held` and `edge_suji_held` of discarding tile type `t` from
/// the concealed hand `h` (before the discard): a 1 with the 4 of its suit
/// held or a 9 with the 6; a 2 or 8 with the 5. Honors: 0.
pub fn suji_held(h: &[u8; 34], t: usize) -> (f64, f64) {
    if t >= 27 {
        return (0.0, 0.0);
    }
    let (base, r) = (t - t % 9, t % 9);
    let has = |rank0: usize| h[base + rank0] > 0;
    let term = (r == 0 && has(3)) || (r == 8 && has(5));
    let edge = (r == 1 || r == 7) && has(4);
    (f64::from(u8::from(term)), f64::from(u8::from(edge)))
}

/// The turn `floor((69 - drawable_count) / 4) + 1`.
pub fn turn(drawable_count: u8) -> i32 {
    (69 - i32::from(drawable_count)).div_euclid(4) + 1
}

/// The tile features of `c` moved out of `ctx.view.hand` (for a called tile,
/// the hand before the call).
fn tile_block(ctx: &Ctx, c: DiscardCand) -> [f64; N_TILE] {
    let t = &tie_features(ctx, &[c], &[0.0])[0];
    let h = &ctx.view.hand;
    let tt = usize::from(c.tile_type);
    let (term, edge) = suji_held(h, tt);
    let pairs = h.iter().filter(|&&n| n >= 2).count();
    let n: u32 = h.iter().map(|&x| u32::from(x)).sum();
    let hon: u32 = h[27..].iter().map(|&x| u32::from(x)).sum();
    let suit = |s: usize| h[s * 9..s * 9 + 9].iter().map(|&x| u32::from(x)).sum::<u32>();
    let own = if tt < 27 { suit(tt / 9) } else { (0..3).map(suit).max().unwrap_or(0) };
    let b = |x: bool| f64::from(u8::from(x));
    [
        b(t[YAKUHAI_WIND] > 0.0 || t[DRAGON] > 0.0),
        t[GUEST_WIND],
        t[TERMINAL],
        t[IS_28],
        b(t[YAKUHAI_WIND] >= 2.0),
        t[DORA] + t[AKA],
        t[UNSEEN],
        t[ISOLATED],
        t[SET_BREAK],
        t[PAIR_BREAK],
        term,
        edge,
        t[TANYAO_STEP],
        b(t[PAIR_BREAK] > 0.0 && pairs >= 4),
        f64::from(own + hon) / f64::from(n.max(1)),
        t[NB_UNSEEN1],
        t[NB_UNSEEN2],
        t[DORA_ADJ1],
        t[DORA_ADJ2],
    ]
}

/// The DP values of an option's hand: (EV in points, win, tenpai, ukeire).
type DpVals = (f64, f64, f64, Option<u32>);

/// One option's hand after it, as `a_block` reads it.
struct After<'a, 'c> {
    /// The decision's context (its unseen counts, winds, dora and river
    /// hold for every option's hand).
    ctx: &'a Ctx<'c>,
    h13: [u8; 34],
    akas: [bool; 3],
    melds: &'a [(u8, Vec<u8>, bool)],
    /// The own river with the option's discard.
    river: [bool; 34],
    shanten: i8,
    dp: Option<DpVals>,
    is_break: bool,
}

type Melds = Vec<(u8, Vec<u8>, bool)>;

/// What `a_block` reads of a tenpai hand's waits.
#[derive(Clone, Copy)]
struct Waits {
    mask: u64,
    ron: f64,
    tsumo: f64,
    tanki: bool,
    change: Option<(u32, u32)>,
}

/// Per-decision caches of the hand features (exact: each keyed by what it
/// reads; the decision fixes the unseen counts, winds, dora and river), and
/// one scorer per meld set (its ron verdicts shared across the options).
#[derive(Default)]
struct Shared {
    scorers: std::cell::RefCell<Vec<(Melds, Scorer)>>,
    two_step: std::cell::RefCell<Vec<([u8; 34], u64)>>,
    breaks: std::cell::RefCell<Vec<(([u8; 34], [bool; 3], Melds), (f64, f64))>>,
    waits: std::cell::RefCell<Vec<(([u8; 34], [bool; 3], Melds), Waits)>>,
    /// Dama ron possible by (hand, red fives, scorer).
    ron13: std::cell::RefCell<crate::eval::hash::FastMap<(u128, [bool; 3], usize), bool>>,
}

impl Shared {
    /// The index of the scorer of `melds` (made on first use).
    fn scorer_id(&self, v: &MjView, melds: &[(u8, Vec<u8>, bool)]) -> usize {
        let mut s = self.scorers.borrow_mut();
        match s.iter().position(|(m, _)| m.as_slice() == melds) {
            Some(i) => i,
            None => {
                s.push((melds.to_vec(), scorer_of(v, melds)));
                s.len() - 1
            }
        }
    }

    /// `f` with the scorer of `melds`.
    fn with_scorer<R>(&self, v: &MjView, melds: &[(u8, Vec<u8>, bool)], f: impl FnOnce(&mut Scorer) -> R) -> R {
        let i = self.scorer_id(v, melds);
        f(&mut self.scorers.borrow_mut()[i].1)
    }

    /// A 13-tile hand's waits (`waits::waits_of`) when it is tenpai,
    /// `None` when it is not.
    fn tenpai_waits(&self, ctx: &Ctx, h: &[u8; 34]) -> Option<u64> {
        let tenpai = ctx.tables().shanten(h) == 0;
        tenpai.then(|| waits_of(ctx, h))
    }

    /// Some wait of the tenpai hand `h` (red fives `ak`) rons with a yaku
    /// without riichi (`wait_value`'s `ron_possible`), with the scorer `sid`.
    fn ron_possible(&self, sid: usize, h: &[u8; 34], ak: &[bool; 3], waits: u64, unseen: &[u8; 34]) -> bool {
        let key = (pack(h), *ak, sid);
        if let Some(&r) = self.ron13.borrow().get(&key) {
            return r;
        }
        let r = wait_value(&mut self.scorers.borrow_mut()[sid].1, h, ak, waits, unseen, false).ron_possible;
        self.ron13.borrow_mut().insert(key, r);
        r
    }

    /// The break counts of a tenpai-breaking hand `h13`: a
    /// draw allows a tenpai discard (other than itself, which gives `h13`
    /// back) iff it lowers the shanten of a 1-shanten `h13`, so only those
    /// draws (`UkeireTables::ukeire`) are tried; the dama ron checks cached.
    fn break_counts(&self, ctx: &Ctx, sid: usize, h13: &[u8; 34], akas: &[bool; 3], own: &[bool; 34]) -> (f64, f64) {
        let unseen = ctx.view.unseen();
        let s13 = ctx.tables().shanten(h13);
        if s13 != 1 {
            return (0.0, 0.0);
        }
        let (_, draws) = ctx.tables().ukeire(h13);
        let (mut any, mut yaku) = (0u32, 0u32);
        for t in (0..34).filter(|&t| draws >> t & 1 == 1) {
            let mut h14 = *h13;
            h14[t] += 1;
            let mut unseen2 = unseen;
            unseen2[t] -= 1;
            let (mut a, mut y) = (false, false);
            for x in (0..34).filter(|&x| x != t && h14[x] > 0) {
                if a && y {
                    break;
                }
                let mut h = h14;
                h[x] -= 1;
                let Some(waits) = self.tenpai_waits(ctx, &h) else { continue };
                a = true;
                if (0..34).any(|w| waits >> w & 1 == 1 && own[w]) {
                    continue;
                }
                let mut ak = *akas;
                if let Some(s) = aka_slot(x).filter(|&s| ak[s] && h14[x] == 1) {
                    ak[s] = false;
                }
                y |= self.ron_possible(sid, &h, &ak, waits, &unseen2);
            }
            any += u32::from(a) * u32::from(unseen[t]);
            yaku += u32::from(y) * u32::from(unseen[t]);
        }
        (f64::from(any), f64::from(yaku))
    }

    /// `waits::change_counts`, its 13-tile hands cached (`base`: the
    /// hand's waits and their unseen copies).
    fn change_counts(&self, ctx: &Ctx, sid: usize, h13: &[u8; 34], akas: &[bool; 3], base: (u64, u32)) -> (u32, u32) {
        let unseen = ctx.view.unseen();
        let own = &ctx.view.own_discards;
        let (mut yaku, mut wider) = (0, 0);
        for t in 0..34 {
            if unseen[t] == 0 || base.0 >> t & 1 == 1 {
                continue;
            }
            let mut h14 = *h13;
            h14[t] += 1;
            let mut unseen2 = unseen;
            unseen2[t] -= 1;
            let (mut y, mut wd) = (false, false);
            for x in (0..34).filter(|&x| x != t && h14[x] > 0) {
                if y && wd {
                    break;
                }
                let mut h = h14;
                h[x] -= 1;
                let Some(waits) = self.tenpai_waits(ctx, &h) else { continue };
                if (0..34).any(|w| waits >> w & 1 == 1 && own[w]) {
                    continue;
                }
                let mut ak = *akas;
                if let Some(s) = aka_slot(x).filter(|&s| ak[s] && h14[x] == 1) {
                    ak[s] = false;
                }
                y |= self.ron_possible(sid, &h, &ak, waits, &unseen2);
                let n: u32 = (0..34).filter(|&w| waits >> w & 1 == 1).map(|w| u32::from(unseen2[w])).sum();
                wd |= n > base.1;
            }
            yaku += u32::from(y) * u32::from(unseen[t]);
            wider += u32::from(wd) * u32::from(unseen[t]);
        }
        (yaku, wider)
    }
}

/// The A features of one option's hand (R, O, C and the products later).
/// `two_step`: compute `A_ukeire2` (else 0).
fn a_block(a: &After, sh: &Shared, two_step: bool, change: bool, row: &mut [f64; N_U]) {
    let v = a.ctx.view;
    let unseen = v.unseen();
    if let Some((ev, win, tenpai, _)) = a.dp {
        row[A_EV] = ev / 1000.0;
        row[A_WIN] = win;
        row[A_TENPAI] = tenpai;
        row[A_IN_DP] = 1.0;
        row[A_NO_YAKU] = f64::from(u8::from(win == 0.0));
    }
    row[A_SHANTEN] = f64::from(a.shanten);
    {
        row[A_UKEIRE] = match a.dp {
            Some((_, _, _, Some(u))) => f64::from(u),
            _ => f64::from(a.ctx.tables().ukeire(&a.h13).0),
        };
    }
    if two_step {
        let hit = sh.two_step.borrow().iter().find(|(h, _)| *h == a.h13).map(|x| x.1);
        let n = hit.unwrap_or_else(|| {
            let n = a.ctx.tables().two_step(&a.h13, a.shanten).0;
            sh.two_step.borrow_mut().push((a.h13, n));
            n
        });
        row[A_UKEIRE2] = n as f64;
    }
    row[A_MENZEN] = f64::from(u8::from(!a.melds.iter().any(|m| m.2)));
    {
        let p = Profile::of(v, &a.h13, a.melds);
        row[A_PATH_TANYAO] = f64::from(u8::from(p.tanyao_path()));
        row[A_PATH_FLUSH] = f64::from(u8::from(p.flush_path()));
        row[A_PATH_TOITOI] = f64::from(u8::from(p.toitoi_path()));
        row[A_YAKUHAI_PAIRS] = p.yakuhai_pairs;
        row[A_YAKUHAI_SET] = f64::from(u8::from(p.yakuhai_set));
        let meld_akas = a.melds.iter().flat_map(|m| m.1.iter()).filter(|&&id| is_aka_tid(id)).count();
        row[A_DORA] = (0..34).map(|t| f64::from(p.all[t]) * f64::from(v.dora_per_tile_type[t])).sum::<f64>()
            + (a.akas.iter().filter(|&&k| k).count() + meld_akas) as f64;
    }
    let key = || (a.h13, a.akas, a.melds.to_vec());
    if a.is_break {
        let hit = sh.breaks.borrow().iter().find(|(k, _)| k.0 == a.h13 && k.1 == a.akas && k.2 == a.melds).map(|x| x.1);
        let b = hit.unwrap_or_else(|| {
            let sid = sh.scorer_id(v, a.melds);
            let b = sh.break_counts(a.ctx, sid, &a.h13, &a.akas, &a.river);
            sh.breaks.borrow_mut().push((key(), b));
            b
        });
        (row[A_BREAK_TENPAI], row[A_BREAK_YAKU_TENPAI]) = b;
    }
    if a.shanten != 0 {
        return;
    }
    let hit = sh.waits.borrow().iter().find(|(k, _)| k.0 == a.h13 && k.1 == a.akas && k.2 == a.melds).map(|x| x.1);
    let w = match hit {
        Some(w) if w.change.is_some() || !change => w,
        _ => {
            let mask = waits_of(a.ctx, &a.h13);
            let base = sh.with_scorer(v, a.melds, |s| wait_value(s, &a.h13, &a.akas, mask, &unseen, true));
            let tanki = is_tanki(a.ctx, &a.h13);
            let change = change.then(|| {
                let sid = sh.scorer_id(v, a.melds);
                sh.change_counts(a.ctx, sid, &a.h13, &a.akas, (mask, base.unseen))
            });
            let w = Waits { mask, ron: base.ron_points, tsumo: base.tsumo_points, tanki, change };
            sh.waits.borrow_mut().push((key(), w));
            w
        }
    };
    row[A_RON_POINTS] = w.ron / 1000.0;
    row[A_DAMA_TSUMO_POINTS] = w.tsumo / 1000.0;
    row[A_TANKI] = f64::from(u8::from(w.tanki));
    let waits = w.mask;
    for w in (0..34).filter(|&w| waits >> w & 1 == 1) {
        let n = f64::from(unseen[w]);
        if w >= 27 {
            row[A_WAIT_HONOR] += n;
            continue;
        }
        let rank = w % 9;
        row[if (2..=6).contains(&rank) { A_WAIT_MID } else { A_WAIT_EDGE }] += n;
        let (lo, hi) = (rank >= 3 && a.river[w - 3], rank <= 5 && a.river[w + 3]);
        let suji = match rank {
            0..=2 => hi,
            6..=8 => lo,
            _ => lo && hi,
        };
        if suji {
            row[A_WAIT_SUJI] += n;
        }
    }
    if let (true, Some((yaku, wider))) = (change, w.change) {
        row[A_CHANGE_YAKU] = f64::from(yaku);
        row[A_CHANGE_WAIT] = f64::from(wider);
    }
}

/// The hand after discarding `c` from the view's hand: (13 tiles, red
/// fives, river).
fn after_discard(v: &MjView, c: DiscardCand) -> ([u8; 34], [bool; 3], [bool; 34]) {
    let t = usize::from(c.tile_type);
    let mut h = v.hand;
    h[t] -= 1;
    let mut akas = v.akas_in_hand;
    if c.is_aka {
        akas[aka_slot(t).expect("a red discard is a five")] = false;
    }
    let mut river = v.own_discards;
    river[t] = true;
    (h, akas, river)
}

/// The view after discarding `c`, as `own_turn_rows` values a `break` option
/// with `sp::pass_value` (the 13 tiles, the others draw first).
fn view_after_discard(v: &MjView, c: DiscardCand) -> MjView {
    let (hand, akas, river) = after_discard(v, c);
    let mut after = v.clone();
    after.hand = hand;
    after.akas_in_hand = akas;
    after.own_discards = river;
    after.drawn_tile = None;
    after.draws_before_us = 3;
    after.double_riichi_possible = false;
    after
}

/// The decision kind of a view (`KINDS` index: 0 discard, 1 riichi, 2
/// tenpai, 3 call), or `None`.
pub(crate) fn kind_of(ctx: &Ctx) -> Option<usize> {
    let v = ctx.view;
    if v.legal.iter().any(|a| matches!(a.action_type, ActionType::Chi | ActionType::Pon))
        && v.find(ActionType::Pass).is_some()
    {
        return Some(3);
    }
    if v.in_riichi || !v.legal.iter().any(|a| a.action_type == ActionType::Discard) {
        return None;
    }
    if v.find(ActionType::Riichi).is_some() {
        return Some(1);
    }
    Some(if ctx.min_shanten()? == 0 { 2 } else { 0 })
}

fn context_of(ctx: &Ctx) -> UContext {
    let v = ctx.view;
    let round_bin = rc_round_bin(v.round_wind, v.kyoku_index);
    let top = v.lead > params().context_top_margin;
    UContext {
        turn: turn(v.drawable_count),
        progress: 1.0 - (f64::from(v.drawable_count) / f64::from(DRAWABLE_AT_START)).min(1.0),
        round_bin,
        top,
        oya: v.player_wind == 27,
        shanten_ref: 0,
    }
}

/// The legal action of option `(kind, id)` of this decision (`None` if
/// none is legal).
pub(crate) fn option_action(ctx: &Ctx, kind: &str, id: u16) -> Option<riichienv_core::action::Action> {
    let v = ctx.view;
    match kind {
        "riichi" => v.find(ActionType::Riichi).cloned(),
        "pass" => v.find(ActionType::Pass).cloned(),
        "daiminkan" => v
            .legal
            .iter()
            .find(|a| a.action_type == ActionType::Daiminkan && a.tile.is_some_and(|t| 42 + u16::from(t / 4) == id))
            .cloned(),
        "chi" | "pon" => call_options(v).into_iter().find_map(|o| {
            let base = if o.kind == CallKind::Chi { 38 + u16::from(o.chi_pos) } else { 41 };
            let oid = if o.red { base - 38 + 76 } else { base };
            (o.kind != CallKind::Pass && oid == id).then_some(o.action)
        }),
        _ => {
            let (t, aka) = if id >= 34 { (4 + 9 * (id - 34) as u8, true) } else { (id as u8, false) };
            ctx.discard_action(t, aka)
        }
    }
}

/// Every option of this decision with its U row (`change`: compute
/// `A_change_*`, which score every tenpai alternative).
pub fn universal(ctx: &Ctx, change: bool) -> Universal {
    let v = ctx.view;
    let mut context = context_of(ctx);
    let Some(kind) = kind_of(ctx) else {
        return Universal {
            options: Vec::new(),
            rows: Vec::new(),
            state: (String::new(), 0),
            context,
        };
    };
    let ht = hand_type(v);
    let later = usize::from(context.turn >= 7);
    let state_name = format!("{}/{}/{}", KINDS[kind], HAND_TYPES[ht], ["early", "later"][later]);
    let state_id = (kind * 4 + ht) * 2 + later;
    let (mut options, mut rows) = match kind {
        3 => call_rows(ctx, change, &mut context),
        _ => own_turn_rows(ctx, kind, change, &mut context),
    };
    finish(&mut rows, &options, &context);
    options.shrink_to_fit();
    rows.shrink_to_fit();
    Universal { options, rows, state: (state_name, state_id), context }
}

/// The options of an own-turn decision (`discard`, `riichi`, `tenpai`).
fn own_turn_rows(
    ctx: &Ctx,
    kind: usize,
    change: bool,
    context: &mut UContext,
) -> (Vec<(&'static str, u16)>, Vec<[f64; N_U]>) {
    let v = ctx.view;
    let cands = ctx.discard_cands().to_vec();
    let s_min = ctx.min_shanten().unwrap_or(0);
    context.shanten_ref = s_min;
    let tf: Vec<[f64; N_TILE]> = cands.iter().map(|&c| tile_block(ctx, c)).collect();
    let sh = Shared::default();
    let (mut options, mut rows) = (Vec::new(), Vec::new());
    let mut push = |kind_name: &'static str, i: usize, dp: Option<DpVals>, is_break: bool, riichi: bool| {
        let c = cands[i];
        let (h13, akas, river) = after_discard(v, c);
        let mut row = [0.0f64; N_U];
        let a = After { ctx, h13, akas, melds: &v.own_melds, river, shanten: c.shanten, dp, is_break };
        // Two-step ukeire (`A_ukeire2`) only for the shanten-keeping discards.
        a_block(&a, &sh, c.shanten == s_min && c.shanten > 0, change, &mut row);
        row[B..B + N_TILE].copy_from_slice(&tf[i]);
        row[O_RIICHI] = f64::from(u8::from(riichi));
        row[O_BREAK] = f64::from(u8::from(is_break));
        options.push((kind_name, tile_id(c.tile_type, c.is_aka)));
        rows.push(row);
    };
    let pass_of = |c: DiscardCand| -> Option<DpVals> {
        let after = view_after_discard(v, c);
        pass_value(&after, false, &mut ctx.memo()).map(|r| (r.ev, r.win, r.tenpai, None))
    };
    let row_of = |rows: Option<&[crate::suzume::sp::DpRow]>, c: DiscardCand| {
        rows?.iter().find(|r| (r.tile_type, r.is_aka) == (c.tile_type, c.is_aka)).copied()
    };
    if kind == 0 {
        let dp = if s_min <= params().dp_max_shanten { ctx.dp(ctx.assume_riichi()) } else { None };
        for (i, &c) in cands.iter().enumerate() {
            let r = row_of(dp, c).map(|r| (r.ev, r.win, r.tenpai, Some(r.ukeire)));
            push("discard", i, r, false, false);
        }
        return (options, rows);
    }
    // riichi / tenpai: dama (tenpai kept) or break, then riichi.
    let dama = ctx.dp(false);
    if kind == 1 {
        ctx.dp(true);
    }
    for (i, &c) in cands.iter().enumerate() {
        if c.shanten == 0 {
            let r = row_of(dama, c).map(|r| (r.ev, r.win, r.tenpai, Some(r.ukeire)));
            push("dama", i, r, false, false);
        } else {
            push("break", i, pass_of(c), true, false);
        }
    }
    if kind == 1 {
        let deposit = deposit_value();
        let riichi = ctx.dp(true);
        for (i, &c) in cands.iter().enumerate() {
            if c.shanten != 0 {
                continue;
            }
            if let Some(r) = row_of(riichi, c) {
                push("riichi", i, Some((r.ev - deposit * (1. - r.win), r.win, r.tenpai, Some(r.ukeire))), false, true);
            }
        }
    }
    (options, rows)
}

/// The options of a call decision: pass, then each chi / pon, then each
/// daiminkan.
fn call_rows(ctx: &Ctx, change: bool, context: &mut UContext) -> (Vec<(&'static str, u16)>, Vec<[f64; N_U]>) {
    let v = ctx.view;
    let shanten_now = ctx.tables().shanten(&v.hand);
    context.shanten_ref = shanten_now;
    let sh = Shared::default();
    let (mut options, mut rows) = (Vec::new(), Vec::new());
    for o in call_options(v) {
        let mut row = [0.0f64; N_U];
        let (kind, id) = match o.kind {
            CallKind::Pass => {
                let dp = (shanten_now <= params().dp_max_shanten)
                    .then(|| pass_value(v, ctx.assume_riichi(), &mut ctx.memo()))
                    .flatten()
                    .map(|r| (r.ev, r.win, r.tenpai, None));
                let a = After {
                    ctx,
                    h13: v.hand,
                    akas: v.akas_in_hand,
                    melds: &v.own_melds,
                    river: v.own_discards,
                    shanten: shanten_now,
                    dp,
                    is_break: false,
                };
                a_block(&a, &sh, shanten_now > 0, change, &mut row);
                ("pass", 82)
            }
            k => {
                let base = if k == CallKind::Chi { 38 + u16::from(o.chi_pos) } else { 41 };
                let id = if o.red { base - 38 + 76 } else { base };
                let after = view_after_call(v, &o.action);
                let actx = Ctx::with_memo(&after, ctx.cfg, ctx.take_memo());
                let pick = after_call_discard(&actx, &after);
                // A call that leaves no legal discard is not offered.
                let Some((i, value)) = pick else {
                    ctx.put_memo(actx.into_memo());
                    continue;
                };
                let c = actx.discard_cands()[i];
                {
                    row[B_D..B_D + N_TILE].copy_from_slice(&tile_block(&actx, c));
                }
                ctx.put_memo(actx.into_memo());
                let (h13, akas, river) = after_discard(&after, c);
                let dp = match value {
                    AfterCallValue::Dp(r) => Some((r.ev, r.win, r.tenpai, None)),
                    AfterCallValue::Vf => None,
                };
                // The decision's context serves the hand after the call: the
                // unseen counts, winds, dora and river are the same.
                let a = After { ctx, h13, akas, melds: &after.own_melds, river, shanten: c.shanten, dp, is_break: false };
                a_block(&a, &sh, c.shanten > 0, change, &mut row);
                // The called tile, on the hand before the call.
                let called = o.action.tile.expect("a call names its tile");
                let cc = DiscardCand { tile_type: called / 4, is_aka: is_aka_tid(called), shanten: 0 };
                {
                    row[B..B + N_TILE].copy_from_slice(&tile_block(ctx, cc));
                }
                if k == CallKind::Chi && called / 4 < 27 && o.chi_pos != 1 {
                    let rk = i32::from(called / 4 % 9) + 1;
                    let lo = if o.chi_pos == 0 { rk + 1 } else { rk - 2 };
                    row[B_CHI_FROM_RYANMEN] = f64::from(u8::from(!(lo == 1 || lo + 1 == 9)));
                }
                row[O_CALL] = 1.0;
                (if k == CallKind::Chi { "chi" } else { "pon" }, id)
            }
        };
        options.push((kind, id));
        rows.push(row);
    }
    // A daiminkan: the call that takes the tile as a kan (`kan_row`).
    for a in v.legal.iter().filter(|a| a.action_type == ActionType::Daiminkan) {
        rows.push(kan_row(ctx, a, &sh, change));
        options.push(("daiminkan", kan_id(a)));
    }
    (options, rows)
}

/// The view after the kan `a` (ankan, kakan or daiminkan): the kan among
/// the melds, the concealed tiles left, the replacement draw next (the new
/// kan dora is not known yet and is not counted).
pub(crate) fn view_after_kan(v: &MjView, a: &riichienv_core::action::Action) -> MjView {
    let mut after = v.clone();
    let tile = a.tile.expect("a kan names its tile");
    let t = usize::from(tile / 4);
    let take = |tid: u8, after: &mut MjView| {
        after.hand[usize::from(tid / 4)] -= 1;
        if let Some(s) = aka_slot(usize::from(tid / 4)).filter(|_| is_aka_tid(tid)) {
            after.akas_in_hand[s] = false;
        }
    };
    match a.action_type {
        ActionType::Ankan => {
            for &id in &a.consume_tiles {
                take(id, &mut after);
            }
            after.own_melds.push((3, a.consume_tiles.clone(), false));
        }
        ActionType::Kakan => {
            take(tile, &mut after);
            if let Some(m) = after.own_melds.iter_mut().find(|m| m.0 == 1 && usize::from(m.1[0] / 4) == t) {
                m.0 = 4;
                m.1.push(tile);
            }
        }
        _ => {
            for &id in &a.consume_tiles {
                take(id, &mut after);
            }
            let mut ids = a.consume_tiles.clone();
            ids.push(tile);
            after.own_melds.push((2, ids, true));
        }
    }
    after.drawn_tile = None;
    after.draws_before_us = 0;
    after.double_riichi_possible = false;
    after
}

/// In riichi: the tsumogiri as the kan decision's `keep` option and row
/// (the riichi DP row of `ctx.dp(true)` for the drawn tile, built like a
/// discard row of `own_turn_rows`).
pub fn riichi_keep_row(ctx: &Ctx) -> Option<((&'static str, u16), [f64; N_U])> {
    let v = ctx.view;
    let (t, aka) = v.drawn_tile_kind()?;
    let c = *ctx.discard_cands().iter().find(|c| (c.tile_type, c.is_aka) == (t, aka))?;
    let dp = ctx.dp(true).and_then(|rows| rows.iter().find(|r| (r.tile_type, r.is_aka) == (t, aka)).copied());
    let (h13, akas, river) = after_discard(v, c);
    let mut row = [0.0f64; N_U];
    let a = After {
        ctx,
        h13,
        akas,
        melds: &v.own_melds,
        river,
        shanten: c.shanten,
        dp: dp.map(|r| (r.ev, r.win, r.tenpai, Some(r.ukeire))),
        is_break: false,
    };
    a_block(&a, &Shared::default(), false, false, &mut row);
    row[B..B + N_TILE].copy_from_slice(&tile_block(ctx, c));
    Some((("discard", tile_id(t, aka)), row))
}

/// A kan option's id: 42 + the tile type (the teacher's kan action id).
fn kan_id(a: &riichienv_core::action::Action) -> u16 {
    42 + u16::from(a.tile.expect("a kan names its tile") / 4)
}

/// The U row of the kan `a` (ankan, kakan or daiminkan): the concealed
/// tiles after it with the kan among the melds, valued by
/// `sp::pass_value` with the replacement draw next (riichi assumed when the
/// hand stays closed and affordable); the kan tile's tile block and its
/// `O_ankan` / `O_kakan` / `O_daiminkan`.
fn kan_row(ctx: &Ctx, a: &riichienv_core::action::Action, sh: &Shared, change: bool) -> [f64; N_U] {
    let v = ctx.view;
    let after = view_after_kan(v, a);
    let shanten = ctx.tables().shanten(&after.hand);
    let assume = ctx.assume_riichi() && !after.is_open();
    let dp = {
        pass_value(&after, assume, &mut ctx.memo()).map(|r| (r.ev, r.win, r.tenpai, None))
    };
    let mut row = [0.0f64; N_U];
    let aa = After {
        ctx,
        h13: after.hand,
        akas: after.akas_in_hand,
        melds: &after.own_melds,
        river: v.own_discards,
        shanten,
        dp,
        is_break: false,
    };
    a_block(&aa, sh, shanten > 0, change, &mut row);
    let tile = a.tile.expect("a kan names its tile");
    let cc = DiscardCand { tile_type: tile / 4, is_aka: false, shanten: 0 };
    row[B..B + N_TILE].copy_from_slice(&tile_block(ctx, cc));
    match a.action_type {
        ActionType::Ankan => row[O_ANKAN] = 1.0,
        ActionType::Kakan => row[O_KAKAN] = 1.0,
        _ => {}
    }
    row
}

/// The kan decision (`KINDS` index 4) of a context where some kan is legal:
/// `keep` is `base` with its row `base_row` (the option the decision's own
/// model picks; in riichi, `riichi_keep_row`'s tsumogiri), then one row per
/// legal kan. `None` without a legal kan.
pub fn kan_universal(
    ctx: &Ctx,
    base: (&'static str, u16),
    base_row: [f64; N_U],
) -> Option<Universal> {
    let v = ctx.view;
    let kans: Vec<_> = v
        .legal
        .iter()
        .filter(|a| matches!(a.action_type, ActionType::Ankan | ActionType::Kakan))
        .cloned()
        .collect();
    if kans.is_empty() {
        return None;
    }
    let mut context = context_of(ctx);
    let ht = hand_type(v);
    let later = usize::from(context.turn >= 7);
    // After the own riichi the hand type is fixed and the choice is the kan or
    // the tsumogiri: a state of its own.
    let (state_name, state_id) = if v.in_riichi {
        (format!("{}/riichi/{}", KINDS[4], ["early", "later"][later]), 40 + later)
    } else {
        (format!("{}/{}/{}", KINDS[4], HAND_TYPES[ht], ["early", "later"][later]), (4 * 4 + ht) * 2 + later)
    };
    context.shanten_ref = ctx.min_shanten().unwrap_or(0);
    let sh = Shared::default();
    let mut options = vec![base];
    let mut rows = vec![base_row];
    for a in &kans {
        rows.push(kan_row(ctx, a, &sh, false));
        options.push((if a.action_type == ActionType::Ankan { "ankan" } else { "kakan" }, kan_id(a)));
    }
    finish(&mut rows, &options, &context);
    Some(Universal {
        options,
        rows,
        state: (state_name, state_id),
        context,
    })
}

/// The decision-level parts: R over the options and the products.
fn finish(rows: &mut [[f64; N_U]], options: &[(&'static str, u16)], c: &UContext) {
    let in_dp = |r: &[f64; N_U]| r[A_IN_DP] > 0.0;
    let top = rows.iter().filter(|r| in_dp(r)).map(|r| r[A_EV]).fold(f64::NEG_INFINITY, f64::max);
    let s_min = rows.iter().map(|r| r[A_SHANTEN]).fold(f64::INFINITY, f64::min);
    let best2 = rows
        .iter()
        .filter(|r| r[A_SHANTEN] == s_min)
        .map(|r| r[A_UKEIRE2])
        .fold(f64::NEG_INFINITY, f64::max);
    for (r, (kind, _)) in rows.iter_mut().zip(options) {
        if in_dp(r) {
            r[R_EV_PCT] = 100.0 * (r[A_EV] - top) / top.abs().max(0.001);
            r[R_EV_NEAR5] = r[R_EV_PCT].max(-5.0);
        }
        r[R_UK2_SHORT] = (r[A_UKEIRE2] - best2).max(-200.0);
        r[R_SH_LOSS] = r[A_SHANTEN] - s_min;
        let adv = if *kind == "pass" { 0.0 } else { f64::from(c.shanten_ref) - r[A_SHANTEN] };
        r[X_ADV_PROGRESS] = adv * c.progress;
        r[X_PROGRESS2_CALL] = r[O_CALL] * c.progress * c.progress;
    }
}
