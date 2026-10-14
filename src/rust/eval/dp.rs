//! Single-player DP core: tenpai/win/EV curves per shanten-keeping discard.
//!
//! A memoized per-turn expectimax; wins are scored natively (`score.rs`).
//! Method credit: THIRD_PARTY_NOTICES.md.
//!
//! # Ron (opt-in, `Input::ron`)
//!
//! Without a `RonModel` the DP counts self-drawn wins only, exactly as it
//! always has (the rulebot and `run_sp` rely on that). With one, a tenpai
//! node also wins on the opponents' discards. The events of one step, from
//! a 13-tile node at turn index `k` (just after an own discard):
//!
//! 1. the three opponents discard once each (ron chance);
//! 2. own draw `k` (tsumo chance);
//! 3. own discard, giving the node at `k + 1`.
//!
//! After the last own draw (`k = horizon`) only `RonModel::tail_discards`
//! opponent discards remain (0 when the last own draw is haitei), with a
//! ron chance and no draw. Each opponent discard is a draw from the unseen
//! tiles of the node (`d` of them, `R` of which are winning tiles that can
//! ron): `q = min(1, rate * R / d)` per discard, so a step without a ron has
//! probability `(1 - q)^3` (`(1 - q)^tail` at the end), and the ron comes on
//! wait tile type `t` with probability proportional to its unseen copies.
//! Opponent discards do not deplete the unseen tiles. Tsumo, ron and no win
//! are exclusive within a step: the tsumo at draw `k` and everything after
//! are weighted by the probability that no ron came before it.
//!
//! A ron is scored like a tsumo (dora, red fives, meld dora, exact ura
//! expectation under riichi; `Scorer::score4_ron`) with ippatsu when it
//! comes in the discards right after the tenpai discard and double riichi
//! as for tsumo; houtei is ignored. A wait whose ron has no yaku (a closed
//! yakuless hand without riichi) cannot ron, which is how dama and riichi
//! differ. Furiten: a tenpai node whose waits include a tile of
//! `RonModel::furiten` (the seat's discards before this decision) cannot
//! ron at all, at every depth and with or without riichi (the root discard
//! cannot be a wait of the hand it leaves, see `Solver::is_furiten`).
//! Discards the DP itself makes later are not tracked, nor is the furiten
//! of passing a yakuless ron.


use riichienv_core::shanten::calc_shanten_from_counts;
use crate::eval::feat::tiles::aka_slot;
use crate::eval::hash::FastMap;
use crate::eval::score::{EvalCache, Scorer};
use crate::eval::shanten::{Facts, HandShanten, facts_packed, keeping};

/// Shanten via riichienv-core's 34-count entry (no 136 conversion).
fn shanten(counts: &Counts) -> i32 {
    let n: u8 = counts.iter().sum();
    calc_shanten_from_counts(counts, n / 3) as i32
}

pub const MAX_TURNS: usize = 17;
/// The highest shanten the DP explores; hands further out value at zero.
pub const SHANTEN_DP_MAX: i32 = 3;

pub type Counts = [u8; 34];
pub type Akas = [bool; 3];

/// The opt-in ron model (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct RonModel {
    /// Scale on the chance an opponent discard is a winning tile: 1.0 is
    /// exact for opponents that discard like a random unseen tile
    /// (tsumogiri); real opponents avoid dealing in, so lower values are for
    /// calibration. 0.0 values exactly as without ron.
    pub rate: f64,
    /// Tile types in the seat's own discards (furiten).
    pub furiten: [bool; 34],
    /// Opponent discards after the last own draw (0..=3; 0 when that draw
    /// is haitei).
    pub tail_discards: u8,
}

/// What one solve is asked; equal inputs give equal outputs (suzume
/// reuses a solve by comparing inputs).
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub hand: Counts,
    pub akas_in_hand: Akas,
    pub seen_counts: Counts,
    pub akas_seen: Akas,
    pub draws_left: usize,
    pub own_turn: bool,
    pub last_draw_is_haitei: bool,
    pub double_riichi_possible: bool,
    pub riichi_when_tenpai: bool,
    pub is_menzen: bool,
    /// Winds as 34-format tile types (27..30 = E..N), not riichienv's 0..3 `Wind`.
    pub player_wind: u8,
    pub round_wind: u8,
    pub dora_indicators: Vec<u8>,
    pub meld_dora_han: u32,
    /// (meld_type 0..4, tiles_136, opened, from_who) per meld.
    pub melds: Vec<(u8, Vec<u8>, bool, i8)>,
    /// Discard candidates to gather the per-drawn-tile terms for, as 34-format
    /// tile ids; `-1` asks for all of them, an empty list for none.
    ///
    /// A candidate's node IS the afterstate of that discard -- the same hand,
    /// solved against the same unseen tiles, since a discard moves a tile from the
    /// hand to the river and `seen_counts` counts both. So the terms the
    /// afterstate's feature rows need are already computed here, and asking
    /// for them costs no extra DP.
    pub collect_tiles: Vec<i32>,
    /// Count wins by ron too; `None` values self-drawn wins only.
    pub ron: Option<RonModel>,
}

pub struct Candidate {
    pub tile_type: i32, // -1 = no discard
    pub discard_is_aka: bool,
    pub effective_tiles: Vec<(usize, u32)>,
    pub tenpai: Vec<f64>,
    /// Win probability and expected points, tsumo and ron together.
    pub win: Vec<f64>,
    pub ev: Vec<f64>,
    /// With a ron model: the ron part of `win` and `ev` (the tsumo part is
    /// the rest). Empty without one. Read only by the tests, which check the
    /// ron arithmetic with it (as with the two flags below).
    #[cfg_attr(not(test), allow(dead_code))]
    pub ron_win: Vec<f64>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub ron_ev: Vec<f64>,
    /// With a ron model, for a candidate that is tenpai: some wait wins by
    /// ron without riichi (has a yaku), furiten or not.
    #[cfg_attr(not(test), allow(dead_code))]
    pub dama_ron_possible: bool,
    /// With a ron model, for a candidate that is tenpai: a wait is in the
    /// own discards (this discard included), so the candidate gets no ron.
    #[cfg_attr(not(test), allow(dead_code))]
    pub furiten: bool,
    /// Only for the no-discard root (a 3n+1 hand, i.e. every opponent at a
    /// decision point): the per-drawn-tile decomposition of the three
    /// curves, flat `34 * 3 * horizon`, and the tiles each draw would make useful,
    /// flat `34 * 34`. That root returns one curve for the whole hand, so
    /// its tile rows would otherwise be empty; the DP reaches the curve by
    /// summing over drawable tiles, and these are those terms. A
    /// decomposition, not extra work: summing over the tile axis reproduces
    /// the curves.
    pub per_tile: Option<Vec<f64>>,
    pub per_edge: Option<Vec<f32>>,
}

/// One curve over the turns; entries from `horizon` on stay 0.
type Curve = [f64; MAX_TURNS];

/// Curves of a 3n+1 node (before choosing the next discard).
struct Node {
    tenpai: Curve,
    win: Curve,
    ev: Curve,
    /// The ron part of `win`/`ev` (all 0 without a ron model).
    ron_win: Curve,
    ron_ev: Curve,
    /// Ron (probability, value) when the node is entered at `horizon`, after
    /// the last own draw: only the tail discards are left.
    tail: (f64, f64),
    shanten: i32,
    /// The tile types whose draw lowers the shanten and are unseen (bit `t`).
    effective: u64,
}

/// A discard node's per-turn best: curves, tail, and the tiles it accepts.
struct Best {
    tenpai: Curve,
    win: Curve,
    ev: Curve,
    ron_win: Curve,
    ron_ev: Curve,
    tail: (f64, f64),
    /// The tile types any of its shanten-keeping discards accepts (bit `t`).
    accept: u64,
}

/// 34 counts (0..=4) in 3 bits each, then 3 red-five flags.
fn pack(counts: &Counts, akas: &Akas) -> u128 {
    let c = crate::eval::hash::pack(counts);
    c << 3 | u128::from(akas[0]) | u128::from(akas[1]) << 1 | u128::from(akas[2]) << 2
}

/// The memo key of a node: (hand, its red fives), (unseen, their red fives).
fn node_key(hand: &Counts, akas: &Akas, unseen: &Counts, unseen_akas: &Akas) -> (u128, u128) {
    (pack(hand, akas), pack(unseen, unseen_akas))
}

/// One copy of tile type `t` in a `pack` key (the counts sit above the 3
/// red-five flags, tile type 0 highest); keys are updated by adding and
/// subtracting these instead of packing again.
fn tile_bit(t: usize) -> u128 {
    1u128 << (102 - 3 * t)
}

/// The red-five flag of tile type `t` in a `pack` key (0 when `red` is false).
fn aka_bit(t: usize, red: bool) -> u128 {
    if red { 1u128 << aka_slot(t).expect("a red tile is a five") } else { 0 }
}

pub struct Solver<'a> {
    inp: &'a Input,
    horizon: usize,
    n_unseen: u32,
    assume_riichi: bool,
    unseen0: Counts,
    scorer: Scorer,
    nt_cache: FastMap<u32, [f64; MAX_TURNS + 1]>,
    /// Solved nodes; the caches map a key to an index here, so a hit
    /// copies nothing.
    nodes: Vec<Node>,
    bests: Vec<Best>,
    draw_cache: FastMap<(u128, u128), usize>,
    disc_cache: FastMap<(u128, u128), usize>,
    /// Furiten per tenpai hand against `RonModel::furiten`.
    furiten_cache: FastMap<u128, bool>,
}

/// A solver's tables, kept per thread between solves so their memory is
/// reused (`Solver::new` takes them cleared, dropping the solver gives them
/// back).
#[derive(Default)]
struct Tables {
    nodes: Vec<Node>,
    bests: Vec<Best>,
    draw_cache: FastMap<(u128, u128), usize>,
    disc_cache: FastMap<(u128, u128), usize>,
}

thread_local! {
    static TABLES: std::cell::RefCell<Tables> = std::cell::RefCell::new(Tables::default());
}

impl Drop for Solver<'_> {
    fn drop(&mut self) {
        let mut t = Tables {
            nodes: std::mem::take(&mut self.nodes),
            bests: std::mem::take(&mut self.bests),
            draw_cache: std::mem::take(&mut self.draw_cache),
            disc_cache: std::mem::take(&mut self.disc_cache),
        };
        t.nodes.clear();
        t.bests.clear();
        t.draw_cache.clear();
        t.disc_cache.clear();
        TABLES.with_borrow_mut(|k| {
            // Keep the larger tables of the two.
            if t.nodes.capacity() >= k.nodes.capacity() {
                *k = t;
            }
        });
    }
}

fn discard_priority(tile_type: usize, is_aka: bool) -> i32 {
    if is_aka {
        return 1;
    }
    if tile_type >= 27 {
        return 7;
    }
    let rank = (tile_type % 9 + 1) as i32;
    7 - rank.min(10 - rank)
}

/// The option of highest `value`; ties go to the higher discard priority,
/// then to the earlier option. `opts` are (priority, index into `nodes`).
fn argmax_by(opts: &[(i32, usize)], nodes: &[Node], value: impl Fn(&Node) -> f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (k, o) in opts.iter().enumerate() {
        match best {
            None => best = Some(k),
            Some(b) => {
                let ob = &opts[b];
                let (v, vb) = (value(&nodes[o.1]), value(&nodes[ob.1]));
                if v > vb || (v == vb && o.0 > ob.0) {
                    best = Some(k);
                }
            }
        }
    }
    best
}

/// The shanten of a 3n+1 hand and the improving tile types still unseen
/// (bit `t`; `improving`, kept per thread). `key`: the hand's `pack` key.
fn effective(key: u128, hand: &Counts, unseen: &Counts) -> (i32, u64) {
    let (base, mask) = facts_packed(Facts::DpImproving, key >> 3, hand, improving);
    let unseen_mask = (0..34).filter(|&t| unseen[t] > 0).fold(0u64, |m, t| m | 1 << t);
    (i32::from(base), mask & unseen_mask)
}

/// The set bits of `mask`, ascending.
fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let t = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            t
        })
    })
}

/// The shanten of a 3n+1 hand and the tile types whose draw lowers it (bit
/// `t`), among the tiles probed.
fn improving(hand: &Counts) -> (i8, u64) {
    let hs = HandShanten::new(hand);
    let base = hs.shanten();
    // exact pruning: same-suit +/-2 neighbours of held tiles, plus orphans
    // when kokushi is the binding minimum.
    const ORPHANS: [usize; 13] =
        [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut probe = [false; 34];
    for t in 0..34 {
        if hand[t] == 0 {
            continue;
        }
        if t >= 27 {
            probe[t] = true;
        } else {
            let r = t % 9;
            let lo = t - r.min(2);
            let hi = t + (8 - r).min(2);
            for x in lo..=hi {
                probe[x] = true;
            }
        }
    }
    // kokushi shanten of the 3n+1 hand: 13 - n_orphan_types - pair
    let mut n_orphan_types = 0i32;
    let mut has_pair = false;
    for &t in ORPHANS.iter() {
        if hand[t] >= 1 {
            n_orphan_types += 1;
        }
        if hand[t] >= 2 {
            has_pair = true;
        }
    }
    let kokushi_sh = 13 - n_orphan_types - if has_pair { 1 } else { 0 };
    let base = i32::from(base);
    if kokushi_sh <= base {
        for &t in ORPHANS.iter() {
            probe[t] = true;
        }
    }
    let mut mask = 0u64;
    for t in 0..34 {
        if probe[t] && hand[t] < 4 && i32::from(hs.plus(t)) < base {
            mask |= 1 << t;
        }
    }
    (base as i8, mask)
}

impl<'a> Solver<'a> {
    pub fn new(inp: &'a Input) -> Self {
        let horizon = inp.draws_left.min(MAX_TURNS);
        let mut unseen0 = [0u8; 34];
        let mut n_unseen = 0u32;
        for i in 0..34 {
            unseen0[i] = 4 - inp.seen_counts[i];
            n_unseen += unseen0[i] as u32;
        }
        let t = TABLES.with_borrow_mut(std::mem::take);
        let scorer = Scorer::new(
            inp.is_menzen && inp.riichi_when_tenpai, inp.player_wind, inp.round_wind,
            inp.dora_indicators.clone(), inp.meld_dora_han,
            inp.melds.clone());
        Solver {
            inp,
            horizon,
            n_unseen,
            assume_riichi: inp.is_menzen && inp.riichi_when_tenpai,
            unseen0,
            scorer,
            nt_cache: FastMap::default(),
            nodes: t.nodes,
            bests: t.bests,
            draw_cache: t.draw_cache,
            disc_cache: t.disc_cache,
            furiten_cache: FastMap::default(),
        }
    }

    /// `new` with the scorer's evaluator verdicts kept in `cache` (shared
    /// across solves; the values are the same as without it).
    pub fn with_eval_cache(inp: &'a Input, cache: EvalCache) -> Self {
        let mut s = Self::new(inp);
        s.scorer.attach_cache(cache);
        s
    }

    /// The candidates of `inp` with `eval` lent to the scorer for the run
    /// (`with_eval_cache`; exact, the verdicts depend on their keys alone).
    pub fn run_cached(inp: &'a Input, eval: &mut EvalCache) -> Vec<Candidate> {
        let mut solver = Self::with_eval_cache(inp, std::mem::take(eval));
        let out = solver.run();
        *eval = solver.take_eval_cache().expect("attached");
        out
    }

    /// The cache given to `with_eval_cache`, back (None without one).
    pub fn take_eval_cache(&mut self) -> Option<EvalCache> {
        self.scorer.detach_cache()
    }

    /// Whether `hand13` completed by tile type `t` is a winning shape.
    fn completes(hand13: &Counts, t: usize) -> bool {
        if hand13[t] >= 4 {
            return false;
        }
        let mut h = *hand13;
        h[t] += 1;
        shanten(&h) < 0
    }

    /// Whether the tenpai `hand13` waits on a tile of the own discards.
    ///
    /// The root discard itself never is a wait: a 13-tile hand waiting on
    /// the tile it just let go came from a complete 14-tile hand, which the
    /// DP does not value (no discard keeps its shanten).
    fn is_furiten(&mut self, hand13: &Counts) -> bool {
        let Some(ron) = self.inp.ron.as_ref() else { return false };
        let key = pack(hand13, &[false; 3]);
        if let Some(&f) = self.furiten_cache.get(&key) {
            return f;
        }
        let f = (0..34).any(|t| ron.furiten[t] && Self::completes(hand13, t));
        self.furiten_cache.insert(key, f);
        f
    }

    fn not_tsumo(&mut self, r_tot: u32) -> [f64; MAX_TURNS + 1] {
        if let Some(v) = self.nt_cache.get(&r_tot) {
            return *v;
        }
        let mut nt = [0.0f64; MAX_TURNS + 1];
        nt[0] = 1.0;
        for j in 0..self.horizon {
            let d = self.n_unseen as i64 - j as i64;
            nt[j + 1] = if d > 0 {
                nt[j] * ((d - r_tot as i64).max(0) as f64) / d as f64
            } else {
                0.0
            };
        }
        self.nt_cache.insert(r_tot, nt);
        nt
    }

    fn unseen_akas(&self) -> Akas {
        let mut w = [false; 3];
        for i in 0..3 {
            w[i] = !self.inp.akas_seen[i] && !self.inp.akas_in_hand[i];
        }
        w
    }

    /// The node of a 3n+1 hand, as an index into `nodes`.
    /// `key`: the node's `node_key`.
    fn draw(&mut self, hand: Counts, akas: Akas, unseen: Counts, unseen_akas: Akas, key: (u128, u128)) -> usize {
        self.draw_inner(hand, akas, unseen, unseen_akas, false, key).0
    }

    /// Ron terms of a tenpai node (see the module docs). Adds, per start
    /// index `i`, the ron probability and value over the remaining steps
    /// and the tail to `ron_win`/`ron_ev`, sets `tail` (the node entered at
    /// `horizon`), and returns the no-ron table `surv[i * horizon + j]`
    /// (no ron in the discards before draws `i..=j`) that weights the tsumo
    /// at draw `j`. None when no wait can ron (furiten, or no yaku).
    /// `scores[b]` is branch `b`'s (tsumo, ron) `Scorer::score4_pair`.
    #[allow(clippy::too_many_arguments)]
    fn ron_terms(
        &mut self,
        hand: &Counts,
        scores: &[(Option<[f64; 4]>, Option<[f64; 4]>)],
        branches: &[(usize, u32, bool)],
        nt: &[f64],
        inv_nti: &[f64],
        ron_win: &mut [f64],
        ron_ev: &mut [f64],
        tail: &mut (f64, f64),
    ) -> Option<Vec<f64>> {
        let inp = self.inp;
        let ron = inp.ron.as_ref()?;
        if self.is_furiten(hand) {
            return None;
        }
        // Winning tiles that can ron, and the mean ron value over them.
        let mut r_ron = 0u32;
        let mut value = [0.0f64; 4];
        for (&(_, c, _), &(_, ron4)) in branches.iter().zip(scores) {
            if let Some(sc) = ron4 {
                r_ron += c;
                for x in 0..4 {
                    value[x] += f64::from(c) * sc[x];
                }
            }
        }
        if r_ron == 0 {
            return None;
        }
        for x in value.iter_mut() {
            *x /= f64::from(r_ron);
        }
        let h = self.horizon;
        // No-ron probability of the discards before draw k (k < h), and of
        // the tail discards (k == h).
        let mut pass = [0.0f64; MAX_TURNS + 1];
        for (k, p) in pass.iter_mut().enumerate().take(h + 1) {
            let d = self.n_unseen as i64 - k as i64;
            let q = if d > 0 { (ron.rate * f64::from(r_ron) / d as f64).min(1.0) } else { 0.0 };
            let n = if k < h { 3 } else { i32::from(ron.tail_discards) };
            *p = (1.0 - q).powi(n);
        }
        // Situational han as the tsumo path counts them: double riichi when
        // tenpai from the first discard, ippatsu in the discards right after.
        let riichi = self.assume_riichi;
        let dr = riichi && inp.double_riichi_possible;
        let extra = |i: usize, k: usize| (usize::from(dr && i == 0) + usize::from(riichi && k == i)).min(3);
        let mut surv = vec![0.0; h * h];
        for i in 0..h {
            // No ron before the discards of step k (and no tsumo, via nt).
            let mut alive = 1.0;
            for k in i..=h {
                let p = nt[k] * inv_nti[i] * alive * (1.0 - pass[k]);
                ron_win[i] += p;
                ron_ev[i] += p * value[extra(i, k)];
                alive *= pass[k];
                if k < h {
                    surv[i * h + k] = alive;
                }
            }
        }
        *tail = (1.0 - pass[h], (1.0 - pass[h]) * value[extra(h, h)]);
        Some(surv)
    }

    /// The node of a 3n+1 hand as an index into `nodes`, plus with
    /// `collect` the per-drawn-tile terms. The root sets `collect` once per
    /// evaluation, so it neither reads nor writes the cache (a cached node
    /// would have no terms attached) and adds no DP work. With a ron model
    /// the terms cover the draw paths only (they no longer sum to the
    /// curves).
    #[allow(clippy::type_complexity)]
    fn draw_inner(
        &mut self,
        hand: Counts,
        akas: Akas,
        unseen: Counts,
        unseen_akas: Akas,
        collect: bool,
        key: (u128, u128),
    ) -> (usize, Option<(Vec<f64>, Vec<f32>)>) {
        debug_assert_eq!(key, node_key(&hand, &akas, &unseen, &unseen_akas));
        if !collect {
            if let Some(&n) = self.draw_cache.get(&key) {
                return (n, None);
            }
        }
        let (s, effective) = effective(key.0, &hand, &unseen);
        let horizon = self.horizon;
        let mut tenpai = [0.0; MAX_TURNS];
        let mut win = [0.0; MAX_TURNS];
        let mut ev = [0.0; MAX_TURNS];
        let ron_on = self.inp.ron.is_some();
        let mut ron_win = [0.0; MAX_TURNS];
        let mut ron_ev = [0.0; MAX_TURNS];
        let mut tail = (0.0, 0.0);
        let mut per_tile = if collect { vec![0.0f64; 34 * 3 * horizon] } else { Vec::new() };
        let mut per_edge = if collect { vec![0.0f32; 34 * 34] } else { Vec::new() };
        let r_tot: u32 = bits(effective).map(|t| u32::from(unseen[t])).sum();
        // horizon == 0 means no draw is left for this seat: every curve is empty,
        // so the branch loop can only do work that lands nowhere -- and it
        // would still recurse into `discard`, which is where the ukeire
        // union comes from, leaving a row of edges on a hand that cannot draw.
        if r_tot > 0 && s <= SHANTEN_DP_MAX && horizon > 0 {
            // branches: (tile_type, count, is_aka); at most one extra per red five.
            let mut branch_buf = [(0usize, 0u32, false); 37];
            let mut n_branches = 0;
            for tile_type in bits(effective) {
                let c = u32::from(unseen[tile_type]);
                let aka_here = aka_slot(tile_type).is_some_and(|s| unseen_akas[s])
                    && unseen[tile_type] >= 1;
                if aka_here {
                    if c > 1 {
                        branch_buf[n_branches] = (tile_type, c - 1, false);
                        n_branches += 1;
                    }
                    branch_buf[n_branches] = (tile_type, 1, true);
                } else {
                    branch_buf[n_branches] = (tile_type, c, false);
                }
                n_branches += 1;
            }
            let branches = &branch_buf[..n_branches];
            let nt = self.not_tsumo(r_tot);
            // P[i][j] = c/(n_unseen-j) * nt[j]/nt[i] for j>=i
            let mut jr = [0.0f64; MAX_TURNS];
            let mut inv_nti = [0.0f64; MAX_TURNS];
            for j in 0..horizon {
                let d = self.n_unseen as i64 - j as i64;
                jr[j] = if d <= 0 { 0.0 } else { nt[j] / d as f64 };
                inv_nti[j] = if nt[j] > 0.0 { 1.0 / nt[j] } else { 0.0 };
            }
            // With ron, a tenpai node scores each wait's tsumo and ron at
            // once (one ura distribution for both).
            let scores: Vec<(Option<[f64; 4]>, Option<[f64; 4]>)> = if s == 0 && ron_on {
                branches
                    .iter()
                    .map(|&(t, _, is_aka)| self.scorer.score4_pair(&hand, &akas, t, is_aka, &unseen))
                    .collect()
            } else {
                Vec::new()
            };
            let surv = if s == 0 && ron_on {
                self.ron_terms(&hand, &scores, branches, &nt, &inv_nti, &mut ron_win, &mut ron_ev, &mut tail)
            } else {
                None
            };
            for (b, &(tile_type, c, is_aka)) in branches.iter().enumerate() {
                let cf = c as f64;
                if s == 0 {
                    let sc4 = if ron_on {
                        scores[b].0
                    } else {
                        self.scorer.score4(&hand, &akas, tile_type, is_aka, &unseen)
                    };
                    let Some(sc4) = sc4 else { continue };
                    for i in 0..horizon {
                        for j in i..horizon {
                            let mut p = cf * inv_nti[i] * jr[j];
                            if let Some(sv) = &surv {
                                // Tsumo at draw j needs no ron before it.
                                p *= sv[i * horizon + j];
                            }
                            if p <= 0.0 {
                                continue;
                            }
                            let mut extra = 0usize;
                            if self.assume_riichi && self.inp.double_riichi_possible && i == 0 {
                                extra += 1;
                            }
                            if self.assume_riichi && j == i {
                                extra += 1;
                            }
                            if self.inp.last_draw_is_haitei && j == horizon - 1 {
                                extra += 1;
                            }
                            win[i] += p;
                            ev[i] += p * sc4[extra.min(3)];
                            if collect {
                                // At shanten 0 the hand is already tenpai, so
                                // the tenpai plane carries the hand-level 1.0
                                // rather than a share of it.
                                per_tile[tile_type * 3 * horizon + i] = 1.0;
                                per_tile[tile_type * 3 * horizon + horizon + i] += p;
                                per_tile[tile_type * 3 * horizon + 2 * horizon + i] += p * sc4[extra.min(3)];
                            }
                        }
                    }
                } else {
                    let mut nh = hand;
                    nh[tile_type] += 1;
                    let mut na = akas;
                    if is_aka {
                        na[aka_slot(tile_type).unwrap()] = true;
                    }
                    let mut nw = unseen;
                    nw[tile_type] -= 1;
                    let mut nr = unseen_akas;
                    if is_aka {
                        nr[aka_slot(tile_type).unwrap()] = false;
                    }
                    let moved = tile_bit(tile_type) + aka_bit(tile_type, is_aka);
                    let ci = self.discard(nh, na, nw, nr, (key.0 + moved, key.1 - moved));
                    let child = &self.bests[ci];
                    let (ct, cw, ce) = (&child.tenpai, &child.win, &child.ev);
                    if collect {
                        for u in bits(child.accept) {
                            per_edge[tile_type * 34 + u] = 1.0;
                        }
                    }
                    if !collect {
                        // The loops below with j outside: each accumulator
                        // (index i) still takes its terms in ascending j,
                        // so the sums are bit-identical, and the inner loop
                        // over i vectorizes. A term with p == 0 or a child
                        // value of 0 adds +0.0 (the child curves are finite
                        // and >= 0, the sums start at +0.0), which changes
                        // no sum, so the zero child values are skipped.
                        let mut a = [0.0f64; MAX_TURNS];
                        for i in 0..horizon {
                            a[i] = cf * inv_nti[i];
                        }
                        for j in 0..horizon {
                            let r = jr[j];
                            if j + 1 < horizon {
                                let (t1, w1, e1) = (ct[j + 1], cw[j + 1], ce[j + 1]);
                                if w1 == 0.0 && e1 == 0.0 {
                                    if t1 != 0.0 {
                                        for i in 0..=j {
                                            tenpai[i] += a[i] * r * t1;
                                        }
                                    }
                                } else {
                                    for i in 0..=j {
                                        let p = a[i] * r;
                                        tenpai[i] += p * t1;
                                        win[i] += p * w1;
                                        ev[i] += p * e1;
                                    }
                                }
                                if ron_on {
                                    let (rw, re) = (child.ron_win[j + 1], child.ron_ev[j + 1]);
                                    if rw == 0.0 && re == 0.0 {
                                        continue;
                                    }
                                    for i in 0..=j {
                                        let p = a[i] * r;
                                        ron_win[i] += p * rw;
                                        ron_ev[i] += p * re;
                                    }
                                }
                            } else if s == 1 {
                                let (tw, te) = child.tail;
                                for i in 0..=j {
                                    let p = a[i] * r;
                                    tenpai[i] += p;
                                    if ron_on {
                                        win[i] += p * tw;
                                        ev[i] += p * te;
                                        ron_win[i] += p * tw;
                                        ron_ev[i] += p * te;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    for i in 0..horizon {
                        for j in i..horizon {
                            let p = cf * inv_nti[i] * jr[j];
                            if p <= 0.0 {
                                continue;
                            }
                            if j + 1 < horizon {
                                tenpai[i] += p * ct[j + 1];
                                win[i] += p * cw[j + 1];
                                ev[i] += p * ce[j + 1];
                                if ron_on {
                                    ron_win[i] += p * child.ron_win[j + 1];
                                    ron_ev[i] += p * child.ron_ev[j + 1];
                                }
                                if collect {
                                    per_tile[tile_type * 3 * horizon + i] += p * ct[j + 1];
                                    per_tile[tile_type * 3 * horizon + horizon + i] += p * cw[j + 1];
                                    per_tile[tile_type * 3 * horizon + 2 * horizon + i] += p * ce[j + 1];
                                }
                            } else if s == 1 {
                                tenpai[i] += p;
                                if collect {
                                    per_tile[tile_type * 3 * horizon + i] += p;
                                }
                                if ron_on {
                                    // Tenpai on the last draw: only the
                                    // tail discards can still deal in.
                                    let (tw, te) = child.tail;
                                    win[i] += p * tw;
                                    ev[i] += p * te;
                                    ron_win[i] += p * tw;
                                    ron_ev[i] += p * te;
                                }
                            }
                        }
                    }
                }
            }
            if surv.is_some() {
                // A tenpai node's ron terms come from `ron_terms`, not children.
                for i in 0..horizon {
                    win[i] += ron_win[i];
                    ev[i] += ron_ev[i];
                }
            }
        }
        for i in 0..horizon {
            if s == 0 {
                tenpai[i] = 1.0;
            }
            tenpai[i] = tenpai[i].clamp(0.0, 1.0);
            win[i] = win[i].clamp(0.0, 1.0);
            ev[i] = ev[i].max(0.0);
            ron_win[i] = ron_win[i].clamp(0.0, 1.0);
            ron_ev[i] = ron_ev[i].max(0.0);
        }
        self.nodes.push(Node { tenpai, win, ev, ron_win, ron_ev, tail, shanten: s, effective });
        let idx = self.nodes.len() - 1;
        if collect {
            return (idx, Some((per_tile, per_edge)));
        }
        self.draw_cache.insert(key, idx);
        (idx, None)
    }

    /// Per-turn best (EV argmax) over shanten-keeping discards of a 3n+2
    /// hand, plus the union of what those discards accept -- the same
    /// relation `encode` writes into `hand_edges` for one's own hand, seen
    /// from the other end: what drawing this tile would make useful. The
    /// union is over discards because which one is best changes with the
    /// turn, and one row has to mean one thing.
    /// With a ron model the tail (entered after the last draw) is the best
    /// option's by tail value, chosen like the curves. Returns an index
    /// into `bests`.
    fn discard(
        &mut self,
        hand: Counts,
        akas: Akas,
        unseen: Counts,
        unseen_akas: Akas,
        key: (u128, u128),
    ) -> usize {
        debug_assert_eq!(key, node_key(&hand, &akas, &unseen, &unseen_akas));
        if let Some(&v) = self.disc_cache.get(&key) {
            return v;
        }
        let (_, keep) = keeping(&hand);
        let horizon = self.horizon;
        // options: (priority, node index); at most 14 types + 3 red fives.
        let mut opt_buf = [(0i32, 0usize); 17];
        let mut n_opts = 0;
        let mut accept = 0u64;
        for d in 0..34 {
            if hand[d] == 0 {
                continue;
            }
            let variants: &[bool] = if aka_slot(d).is_some_and(|s| akas[s])
            {
                if hand[d] > 1 {
                    &[false, true]
                } else {
                    &[true]
                }
            } else {
                &[false]
            };
            for &discard_aka in variants {
                let mut nh = hand;
                nh[d] -= 1;
                let mut na = akas;
                if discard_aka {
                    na[aka_slot(d).unwrap()] = false;
                }
                if keep >> d & 1 == 0 {
                    continue;
                }
                let moved = tile_bit(d) + aka_bit(d, discard_aka);
                let ni = self.draw(nh, na, unseen, unseen_akas, (key.0 - moved, key.1));
                accept |= self.nodes[ni].effective;
                opt_buf[n_opts] = (discard_priority(d, discard_aka), ni);
                n_opts += 1;
            }
        }
        let opts = &opt_buf[..n_opts];
        let nodes = &self.nodes;
        let ron_on = self.inp.ron.is_some();
        let mut tenpai = [0.0; MAX_TURNS];
        let mut win = [0.0; MAX_TURNS];
        let mut ev = [0.0; MAX_TURNS];
        let mut ron_win = [0.0; MAX_TURNS];
        let mut ron_ev = [0.0; MAX_TURNS];
        for j in 0..horizon {
            if let Some(b) = argmax_by(opts, nodes, |n| n.ev[j]) {
                let n = &nodes[opts[b].1];
                tenpai[j] = n.tenpai[j];
                win[j] = n.win[j];
                ev[j] = n.ev[j];
                if ron_on {
                    ron_win[j] = n.ron_win[j];
                    ron_ev[j] = n.ron_ev[j];
                }
            }
        }
        let tail = if ron_on {
            argmax_by(opts, nodes, |n| n.tail.1).map_or((0.0, 0.0), |b| nodes[opts[b].1].tail)
        } else {
            (0.0, 0.0)
        };
        self.bests.push(Best { tenpai, win, ev, ron_win, ron_ev, tail, accept });
        let idx = self.bests.len() - 1;
        self.disc_cache.insert(key, idx);
        idx
    }

    /// The output row of a root node; `hand13` is the node's hand.
    fn candidate(
        &mut self,
        tile_type: i32,
        discard_is_aka: bool,
        (hand13, akas13): (Counts, Akas),
        node: usize,
        collected: Option<(Vec<f64>, Vec<f32>)>,
    ) -> Candidate {
        let (per_tile, per_edge) = match collected {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };
        let ron_on = self.inp.ron.is_some();
        let (mut dama_ron_possible, mut furiten) = (false, false);
        if ron_on && self.nodes[node].shanten == 0 {
            for t in bits(self.nodes[node].effective) {
                if self.scorer.ron_has_yaku_without_riichi(&hand13, &akas13, t) {
                    dama_ron_possible = true;
                    break;
                }
            }
            furiten = self.is_furiten(&hand13);
        }
        let h = self.horizon;
        let n = &self.nodes[node];
        let ron_curve = |c: &Curve| if ron_on { c[..h].to_vec() } else { Vec::new() };
        Candidate {
            tile_type,
            discard_is_aka,
            // A root node's unseen counts are the input's.
            effective_tiles: bits(n.effective).map(|t| (t, u32::from(self.unseen0[t]))).collect(),
            tenpai: n.tenpai[..h].to_vec(),
            win: n.win[..h].to_vec(),
            ev: n.ev[..h].to_vec(),
            ron_win: ron_curve(&n.ron_win),
            ron_ev: ron_curve(&n.ron_ev),
            dama_ron_possible,
            furiten,
            per_tile,
            per_edge,
        }
    }

    pub fn run(&mut self) -> Vec<Candidate> {
        let unseen_akas = self.unseen_akas();
        let mut out = Vec::new();
        if !self.inp.own_turn {
            let (hand, akas) = (self.inp.hand, self.inp.akas_in_hand);
            let key = node_key(&hand, &akas, &self.unseen0, &unseen_akas);
            let (node, collected) = self.draw_inner(hand, akas, self.unseen0, unseen_akas, true, key);
            out.push(self.candidate(-1, false, (hand, akas), node, collected));
            return out;
        }
        let (_, keep) = keeping(&self.inp.hand);
        let collect_all = self.inp.collect_tiles.contains(&-1);
        for d in 0..34 {
            if self.inp.hand[d] == 0 {
                continue;
            }
            let collect = collect_all || self.inp.collect_tiles.contains(&(d as i32));
            let variants: &[bool] = if aka_slot(d).is_some_and(|s| self.inp.akas_in_hand[s])
            {
                if self.inp.hand[d] > 1 {
                    &[false, true]
                } else {
                    &[true]
                }
            } else {
                &[false]
            };
            for &discard_aka in variants {
                let mut nh = self.inp.hand;
                nh[d] -= 1;
                let mut na = self.inp.akas_in_hand;
                if discard_aka {
                    na[aka_slot(d).unwrap()] = false;
                }
                if keep >> d & 1 == 0 {
                    continue;
                }
                let key = node_key(&nh, &na, &self.unseen0, &unseen_akas);
                let (node, collected) = self.draw_inner(nh, na, self.unseen0, unseen_akas, collect, key);
                out.push(self.candidate(d as i32, discard_aka, (nh, na), node, collected));
            }
        }
        out.sort_by(|a, b| {
            let ka = (a.ev.first().copied().unwrap_or(0.0),
                      discard_priority(a.tile_type.max(0) as usize, a.discard_is_aka));
            let kb = (b.ev.first().copied().unwrap_or(0.0),
                      discard_priority(b.tile_type.max(0) as usize, b.discard_is_aka));
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic generator (SplitMix64) for reproducible inputs.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Random DP inputs: a closed hand of shanten <= `max_shanten`, a few
    /// visible tiles, one dora indicator; own turn (14 tiles) or not (13).
    fn random_inputs(seed: u64, n: usize, max_shanten: i32) -> Vec<Input> {
        let mut rng = Rng(seed);
        let mut out = Vec::new();
        while out.len() < n {
            let mut wall: Vec<u8> = (0..136u8).collect();
            for i in (1..wall.len()).rev() {
                wall.swap(i, rng.below(i + 1));
            }
            let own_turn = rng.below(3) > 0;
            let n_hand = if own_turn { 14 } else { 13 };
            let mut hand = [0u8; 34];
            let mut akas_in_hand = [false; 3];
            for &tid in &wall[..n_hand] {
                hand[(tid / 4) as usize] += 1;
                if let Some(s) = crate::eval::feat::tiles::aka_slot_of_tid(u32::from(tid)) {
                    akas_in_hand[s] = true;
                }
            }
            if shanten(&hand) > max_shanten {
                continue;
            }
            let mut seen = hand;
            let mut akas_seen = akas_in_hand;
            let n_seen = rng.below(25);
            for &tid in &wall[n_hand..n_hand + n_seen + 1] {
                seen[(tid / 4) as usize] += 1;
                if let Some(s) = crate::eval::feat::tiles::aka_slot_of_tid(u32::from(tid)) {
                    akas_seen[s] = true;
                }
            }
            let indicator = wall[n_hand] / 4;
            out.push(Input {
                hand,
                akas_in_hand,
                seen_counts: seen,
                akas_seen,
                draws_left: 1 + rng.below(12),
                own_turn,
                last_draw_is_haitei: rng.below(2) == 0,
                double_riichi_possible: rng.below(4) == 0,
                riichi_when_tenpai: rng.below(2) == 0,
                is_menzen: true,
                player_wind: 27 + rng.below(4) as u8,
                round_wind: 27 + rng.below(2) as u8,
                dora_indicators: vec![indicator],
                meld_dora_han: 0,
                melds: Vec::new(),
                collect_tiles: if own_turn { Vec::new() } else { vec![-1] },
                ron: None,
            });
        }
        out
    }

    /// FNV-1a over every output of the solver, f64s by their bits.
    fn digest(cands: &[Candidate]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        for c in cands {
            eat(c.tile_type as u64);
            eat(u64::from(c.discard_is_aka));
            for &(t, n) in &c.effective_tiles {
                eat(t as u64);
                eat(u64::from(n));
            }
            for v in c.tenpai.iter().chain(&c.win).chain(&c.ev) {
                eat(v.to_bits());
            }
            for v in c.per_tile.iter().flatten() {
                eat(v.to_bits());
            }
            for v in c.per_edge.iter().flatten() {
                eat(u64::from(v.to_bits()));
            }
        }
        h
    }

    fn digest_all(inputs: &[Input]) -> u64 {
        let mut h = 0u64;
        for inp in inputs {
            h = h.rotate_left(5) ^ digest(&Solver::new(inp).run());
        }
        h
    }

    /// The tsumo-only DP is pinned to the outputs it gave before ron was
    /// added (digest taken from that code), so the rulebot and `run_sp`
    /// see bit-identical values.
    #[test]
    fn tsumo_only_outputs_are_pinned() {
        let mut inputs = random_inputs(7, 40, 2);
        inputs.extend(random_inputs(11, 8, 3));
        inputs.extend(random_inputs(13, 30, 0));
        assert_eq!(digest_all(&inputs), 0xb467_e2d7_e2bc_9bbf);
    }

    fn ron(rate: f64) -> Option<RonModel> {
        Some(RonModel { rate, furiten: [false; 34], tail_discards: 0 })
    }

    /// A closed hand (no red fives) as South (non-dealer) in an East round,
    /// one dora indicator 1z; nothing visible but the hand and indicator.
    fn hand_input(hand: &str, riichi: bool, draws_left: usize, ron: Option<RonModel>) -> Input {
        let hand = crate::suzume::test_util::counts(hand);
        let n: u8 = hand.iter().sum();
        let mut seen = hand;
        seen[27] += 1;
        Input {
            hand,
            akas_in_hand: [false; 3],
            seen_counts: seen,
            akas_seen: [false; 3],
            draws_left,
            own_turn: n == 14,
            last_draw_is_haitei: false,
            double_riichi_possible: false,
            riichi_when_tenpai: riichi,
            is_menzen: true,
            player_wind: 28,
            round_wind: 27,
            dora_indicators: vec![27],
            meld_dora_han: 0,
            melds: Vec::new(),
            collect_tiles: Vec::new(),
            ron,
        }
    }

    /// The candidate discarding tile type `t`.
    fn cand(cands: &[Candidate], t: i32) -> &Candidate {
        cands.iter().find(|c| c.tile_type == t).expect("candidate")
    }

    /// 7z out leaves a closed kanchan tenpai with no yaku: dama cannot ron
    /// (and values exactly as without ron), riichi can.
    #[test]
    fn yakuless_tenpai_rons_only_with_riichi() {
        let hand = "123m456p789p24s99m7z";
        let base = Solver::new(&hand_input(hand, false, 8, None)).run();
        let dama = Solver::new(&hand_input(hand, false, 8, ron(1.0))).run();
        let (b, d) = (cand(&base, 33), cand(&dama, 33));
        assert!(!d.dama_ron_possible && !d.furiten);
        assert!(d.ron_win.iter().chain(&d.ron_ev).all(|&x| x == 0.0));
        assert_eq!(b.ev, d.ev);
        assert_eq!(b.win, d.win);
        let riichi = Solver::new(&hand_input(hand, true, 8, ron(1.0))).run();
        let r = cand(&riichi, 33);
        assert!(!r.dama_ron_possible);
        assert!(r.ron_win[0] > 0.05 && r.ron_ev[0] > 0.0, "{:?}", r.ron_win);
        let r0 = Solver::new(&hand_input(hand, true, 8, None)).run();
        assert!(r.ev[0] > cand(&r0, 33).ev[0]);
    }

    /// Tanyao tenpai: dama can ron.
    #[test]
    fn tanyao_tenpai_rons_dama() {
        let dama = Solver::new(&hand_input("234m456p678s22s68s7z", false, 8, ron(1.0))).run();
        let d = cand(&dama, 33);
        assert!(d.dama_ron_possible && !d.furiten);
        assert!(d.ron_win[0] > 0.05 && d.ron_ev[0] > 0.0);
        assert!(d.win[0] <= 1.0 && d.win[0] > d.ron_win[0]);
    }

    /// One draw left, two tail discards: tsumo + ron + no win sum to 1, and
    /// the curves match the event model computed by hand.
    #[test]
    fn one_step_probabilities_match_the_event_model() {
        // 13 tiles, waiting on 7s only (3 unseen); tanyao.
        let hand = "234m456p678s22s68s";
        for riichi in [false, true] {
            let mut inp = hand_input(hand, riichi, 1, ron(1.0));
            inp.ron.as_mut().unwrap().tail_discards = 2;
            let c = Solver::new(&inp).run().remove(0);
            let unseen: Counts = std::array::from_fn(|t| 4 - inp.seen_counts[t]);
            let n = f64::from(unseen.iter().map(|&x| u32::from(x)).sum::<u32>());
            let r = f64::from(unseen[24]);
            let a0 = (1.0 - r / n).powi(3);
            let p_ron0 = 1.0 - a0;
            let p_tsumo = a0 * r / n;
            let p_tail = a0 * (1.0 - r / n) * (1.0 - (1.0 - r / (n - 1.0)).powi(2));
            let p_none = a0 * (1.0 - r / n) * (1.0 - r / (n - 1.0)).powi(2);
            assert!((p_ron0 + p_tsumo + p_tail + p_none - 1.0).abs() < 1e-12);
            assert!((c.win[0] - (p_ron0 + p_tsumo + p_tail)).abs() < 1e-12);
            assert!((c.ron_win[0] - (p_ron0 + p_tail)).abs() < 1e-12);
            // Values: ippatsu (+1) on the first discards and the draw under riichi.
            let ipp = usize::from(riichi);
            let mut sc = Scorer::new(riichi, 28, 27, vec![27], 0, Vec::new());
            let ron4 = sc.score4_ron(&inp.hand, &[false; 3], 24, false, &unseen).unwrap();
            let tsumo4 = sc.score4(&inp.hand, &[false; 3], 24, false, &unseen).unwrap();
            let ev = p_ron0 * ron4[ipp] + p_tsumo * tsumo4[ipp] + p_tail * ron4[0];
            assert!((c.ev[0] - ev).abs() < 1e-9, "{} vs {ev}", c.ev[0]);
            assert!((c.ron_ev[0] - (p_ron0 * ron4[ipp] + p_tail * ron4[0])).abs() < 1e-9);
        }
    }

    /// A wait in the own discards rules out ron.
    #[test]
    fn furiten_candidates_get_no_ron() {
        let mut model = ron(1.0).unwrap();
        model.furiten[24] = true;
        let inp = hand_input("234m456p678s22s68s7z", true, 8, Some(model));
        let c = Solver::new(&inp).run();
        let f = cand(&c, 33);
        assert!(f.furiten && f.dama_ron_possible);
        assert!(f.ron_win.iter().all(|&x| x == 0.0));
        let plain = Solver::new(&hand_input("234m456p678s22s68s7z", true, 8, None)).run();
        assert_eq!(f.ev, cand(&plain, 33).ev);
    }

    /// Every output of a solve, ron terms and flags included, as bits.
    fn full_bits(cands: &[Candidate]) -> Vec<u64> {
        let mut out = vec![digest(cands)];
        for c in cands {
            out.extend(c.ron_win.iter().chain(&c.ron_ev).map(|v| v.to_bits()));
            out.push(u64::from(c.dama_ron_possible) | u64::from(c.furiten) << 1);
        }
        out
    }

    /// A shared evaluator cache (warm from other inputs, other contexts and
    /// the same input) changes no output bit, with and without ron.
    #[test]
    fn eval_cache_is_exact() {
        let mut inputs = random_inputs(41, 60, 2);
        inputs.extend(random_inputs(43, 60, 0));
        inputs.extend(random_inputs(47, 6, 3));
        // Same hands under other contexts: ron (with furiten), a second dora
        // indicator, other winds.
        let mut more = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            let mut r = Input { ron: ron(0.3 + 0.1 * (i % 5) as f64), ..inp.clone() };
            r.ron.as_mut().unwrap().tail_discards = (i % 4) as u8;
            r.ron.as_mut().unwrap().furiten[i % 34] = i % 3 == 0;
            if i % 2 == 0 {
                r.dora_indicators.push(((i * 7) % 34) as u8);
            }
            if i % 4 == 1 {
                r.player_wind = 27 + ((r.player_wind - 27 + 1) % 4);
            }
            more.push(r);
        }
        inputs.extend(more);
        let mut cache = EvalCache::default();
        for pass in 0..2 {
            for inp in &inputs {
                let plain = full_bits(&Solver::new(inp).run());
                let mut s = Solver::with_eval_cache(inp, cache);
                let cached = full_bits(&s.run());
                cache = s.take_eval_cache().unwrap();
                assert_eq!(plain, cached, "pass {pass}");
            }
        }
        assert!(cache.hits > cache.misses, "{} hits, {} misses", cache.hits, cache.misses);
    }

    /// Without a ron model and with rate 0 the DP gives the same bits.
    #[test]
    fn rate_zero_matches_no_ron() {
        let mut inputs = random_inputs(21, 30, 2);
        inputs.extend(random_inputs(23, 30, 0));
        for mut inp in inputs {
            let a = Solver::new(&inp).run();
            inp.ron = ron(0.0);
            inp.ron.as_mut().unwrap().tail_discards = 2;
            let b = Solver::new(&inp).run();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!((x.tile_type, x.discard_is_aka), (y.tile_type, y.discard_is_aka));
                assert_eq!((&x.tenpai, &x.win, &x.ev), (&y.tenpai, &y.win, &y.ev));
                assert!(y.ron_win.iter().all(|&v| v == 0.0));
            }
        }
    }

    /// At tenpai, ron raises the win probability and keeps it a probability.
    #[test]
    fn ron_adds_to_win() {
        for mut inp in random_inputs(31, 30, 0) {
            let a = Solver::new(&inp).run();
            inp.ron = ron(1.0);
            let b = Solver::new(&inp).run();
            for (x, y) in a.iter().zip(&b) {
                for i in 0..x.win.len() {
                    assert!(y.win[i] <= 1.0 + 1e-12 && y.ron_win[i] <= y.win[i] + 1e-12);
                    assert!(y.win[i] >= x.win[i] - 1e-12, "{} < {}", y.win[i], x.win[i]);
                }
            }
        }
    }

    /// The DP without calls with a ron model, pinned to the outputs it gave
    /// before the call model was added (digest taken from that code).
    #[test]
    fn ron_outputs_are_pinned() {
        let mut inputs = random_inputs(51, 40, 2);
        for (i, inp) in inputs.iter_mut().enumerate() {
            inp.ron = ron([0.3, 1.0][i % 2]);
            inp.ron.as_mut().unwrap().tail_discards = (i % 4) as u8;
        }
        let mut h = 0u64;
        for inp in &inputs {
            let bits = full_bits(&Solver::new(inp).run());
            for b in bits {
                h = h.rotate_left(7) ^ b;
            }
        }
        assert_eq!(h, RON_PIN);
    }

    const RON_PIN: u64 = 0x7904_480b_ee32_f4a7;

    /// An open hand (South, ron rate 0.3) whose one meld is `meld`.
    fn open_hand_input(hand: &str, meld: (u8, &str), draws_left: usize) -> Input {
        let mut inp = hand_input(hand, false, draws_left, ron(0.3));
        let ids = crate::suzume::test_util::tids(meld.1);
        for &id in &ids {
            inp.seen_counts[(id / 4) as usize] += 1;
        }
        inp.is_menzen = false;
        inp.melds = vec![(meld.0, ids, true, 0)];
        inp
    }

    /// An open hand with no way to a yaku (chi of 789m, no honors, mixed
    /// suits) is worth 0.
    #[test]
    fn yakuless_open_hand_stays_zero() {
        let c = Solver::new(&open_hand_input("234p567s34m88p", (0, "789m"), 10)).run().remove(0);
        assert!(c.ev.iter().chain(&c.win).all(|&x| x == 0.0), "{:?}", c.ev);
    }
}
