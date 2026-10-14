//! Native tsumo and ron scoring for the SP DP, via riichienv-core (Apache-2.0).
//!
//! Expected tsumo totals for +0..+3
//! situational han, with exact uradora expectation over the path-updated
//! unseen tiles minus the winning tile; `score4_ron` does the same for a ron
//! (the DP's opt-in ron model). Runs entirely in Rust -- no Python callback
//! -- so the DP makes zero FFI crossings.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::score::calculate_score;
use riichienv_core::types::{standard_next_dora_tile, Conditions, Meld, MeldType, Wind};

use crate::eval::dp::{Akas, Counts};
use crate::eval::feat::tiles::{AKA_TIDS, aka_slot};
use crate::eval::hash::{FastMap, PackedHasher, pack};

/// The scoring context an evaluator verdict depends on beyond the hand:
/// (player wind, round wind, dora indicators (136 ids), melds as
/// `Scorer::new` takes them). Fixed within a kyoku except for the seat's
/// own calls and new kan dora.
type EvalContext = (u8, u8, Vec<u8>, Vec<(u8, Vec<u8>, bool, i8)>);

/// Evaluator verdicts (`Scorer::evaluate`) kept across DP solves: owned by
/// one suzume bot for one kyoku (cleared at each kyoku start), handed to
/// each `Solver` it runs and taken back afterwards.
///
/// A verdict is a pure function of the hand, its red fives, the winning
/// tile and its red-ness, tsumo/ron, riichi, and the `EvalContext`. The
/// context is interned (a few per kyoku: calls and kan dora add one each)
/// and its index goes into the key with the rest, so a hit returns exactly
/// what `evaluate` would (han and fu as riichienv counts them, dora and red
/// fives included). Key layout (u128): bits 26.. the 34 counts of the 13
/// tiles in 3 bits each, bits 14..26 the context index, bits 6..14 the
/// winning tile type, bits 0..6 flags (3 red fives held, red winning
/// tile, tsumo, riichi).
#[derive(Default)]
pub struct EvalCache {
    /// Interned contexts and their indices.
    contexts: HashMap<EvalContext, u32>,
    map: HashMap<u128, Option<Eval>, BuildHasherDefault<PackedHasher>>,
    /// Bumped whenever the context indices are reset, so a scorer that
    /// remembers its index (`Scorer::ctx_memo`) knows when it went stale.
    generation: u64,
    /// Lookups answered from the map / computed, since the last `clear`.
    pub hits: u64,
    pub misses: u64,
}

/// Context indices fit in 12 bits; past that the cache starts over.
const MAX_CONTEXTS: usize = 1 << 12;

impl EvalCache {
    /// Forget everything once more than `cap` verdicts are held (a cache
    /// kept across kyoku and games).
    pub fn bound(&mut self, cap: usize) {
        if self.map.len() > cap {
            self.clear();
        }
    }

    /// Forget everything (a new kyoku).
    pub fn clear(&mut self) {
        self.contexts.clear();
        self.map.clear();
        self.generation += 1;
        self.hits = 0;
        self.misses = 0;
    }


    /// The index of `ctx`, interned on first sight.
    fn context_id(&mut self, ctx: EvalContext) -> u32 {
        if let Some(&i) = self.contexts.get(&ctx) {
            return i;
        }
        if self.contexts.len() == MAX_CONTEXTS {
            self.contexts.clear();
            self.map.clear();
            self.generation += 1;
        }
        let id = self.contexts.len() as u32;
        self.contexts.insert(ctx, id);
        id
    }
}

/// The `EvalCache` key of one evaluation (layout on `EvalCache`).
fn eval_key(
    ctx_id: u32,
    hand13: &Counts,
    akas: &Akas,
    win_tile_type: usize,
    win_is_aka: bool,
    tsumo: bool,
    riichi: bool,
) -> u128 {
    let packed = hand13.iter().fold(0u128, |acc, &c| acc << 3 | u128::from(c));
    let flags = u128::from(akas[0])
        | u128::from(akas[1]) << 1
        | u128::from(akas[2]) << 2
        | u128::from(win_is_aka) << 3
        | u128::from(tsumo) << 4
        | u128::from(riichi) << 5;
    packed << 26 | u128::from(ctx_id) << 14 | (win_tile_type as u128) << 6 | flags
}

/// Fu values (in steps of 5) the points table holds: 0..=155.
const PTS_FU_SLOTS: usize = 32;

pub struct Scorer {
    /// Verdicts shared across solves (suzume), with this scorer's context
    /// index in it; None evaluates every time (rulebot, `run_sp`).
    cache: Option<(EvalCache, u32)>,
    /// This scorer's context index in the cache last attached, with that
    /// cache's generation (skips re-interning when a solve hands one cache
    /// back and forth between several scorers).
    ctx_memo: Option<(u64, u32)>,
    assume_riichi: bool,
    is_oya: bool,
    /// riichienv `Wind` (0..3); `new` takes them as tile types 27..30.
    player_wind: Wind,
    round_wind: Wind,
    dora_indicators: Vec<u8>,      // stored as 136-format for calc
    meld_dora_han: u32,
    /// Ura indicators turned over at a riichi win: one per dora indicator.
    n_indicators: usize,
    melds: Vec<Meld>,
    /// Dora tile type each indicator tile type points to, looked up once per DP run
    /// so the uradora loops index a table instead of re-deriving it.
    next_dora_types: [usize; 34],
    /// Points by [min(han, 13)][fu / 5][tsumo], filled on first use
    /// (negative = not yet).
    pts_table: [[[f64; 2]; PTS_FU_SLOTS]; 14],
    /// Ron verdicts per packed (hand, akas, winning tile type, winning tile
    /// aka) (`ron_key`).
    ron_cache: FastMap<u128, Option<Eval>>,
}

/// A ron verdict's key: the hand (3 bits per type), its red fives, the
/// winning tile type and whether it is red, in one `u128`.
fn ron_key(hand13: &Counts, akas: &Akas, win_tile_type: usize, win_is_aka: bool) -> u128 {
    let a = u128::from(akas[0]) | u128::from(akas[1]) << 1 | u128::from(akas[2]) << 2;
    pack(hand13) << 10 | a << 7 | (win_tile_type as u128) << 1 | u128::from(win_is_aka)
}

fn meld_type_from(v: u8) -> MeldType {
    match v {
        0 => MeldType::Chi,
        1 => MeldType::Pon,
        2 => MeldType::Daiminkan,
        3 => MeldType::Ankan,
        _ => MeldType::Kakan,
    }
}

/// The dora tile type a 34-format indicator points to.
fn next_dora_type(indicator_type: usize) -> usize {
    standard_next_dora_tile(indicator_type as u8) as usize
}

/// What the evaluator says of one winning shape: (han, fu, yakuman).
type Eval = (u32, u32, bool);

impl Scorer {
    pub fn new(
        assume_riichi: bool,
        player_wind: u8,
        round_wind: u8,
        dora_indicators: Vec<u8>,
        meld_dora_han: u32,
        melds_spec: Vec<(u8, Vec<u8>, bool, i8)>,
    ) -> Self {
        let n_indicators = dora_indicators.len();
        let melds = Self::melds_from_spec(melds_spec);
        Scorer {
            cache: None,
            ctx_memo: None,
            assume_riichi,
            is_oya: player_wind == 27,
            player_wind: Wind::from((player_wind - 27) as u8),
            round_wind: Wind::from((round_wind - 27) as u8),
            dora_indicators: dora_indicators.iter().map(|&t| t * 4).collect(),
            meld_dora_han,
            n_indicators,
            melds,
            next_dora_types: std::array::from_fn(next_dora_type),
            pts_table: [[[-1.0; 2]; PTS_FU_SLOTS]; 14],
            ron_cache: FastMap::default(),
        }
    }

    /// Use `cache` for the evaluator verdicts from now on.
    pub fn attach_cache(&mut self, mut cache: EvalCache) {
        if let Some((generation, id)) = self.ctx_memo {
            if generation == cache.generation {
                self.cache = Some((cache, id));
                return;
            }
        }
        let ctx = (
            self.player_wind as u8,
            self.round_wind as u8,
            self.dora_indicators.clone(),
            self.melds
                .iter()
                .map(|m| (m.meld_type as u8, m.tiles.clone(), m.opened, m.from_who))
                .collect(),
        );
        let id = cache.context_id(ctx);
        self.ctx_memo = Some((cache.generation, id));
        self.cache = Some((cache, id));
    }

    /// `melds_spec` (as `new` takes it) as riichienv `Meld`s.
    fn melds_from_spec(melds_spec: Vec<(u8, Vec<u8>, bool, i8)>) -> Vec<Meld> {
        melds_spec
            .into_iter()
            .map(|(meld_type, tiles, opened, from_who)| Meld {
                meld_type: meld_type_from(meld_type),
                tiles,
                opened,
                from_who,
                called_tile: None,
            })
            .collect()
    }

    /// Hand the cache back (None if none was attached).
    pub fn detach_cache(&mut self) -> Option<EvalCache> {
        self.cache.take().map(|(c, _)| c)
    }

    /// `evaluate` through the cache when one is attached.
    fn eval_cached(
        &mut self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
        win_is_aka: bool,
        tsumo: bool,
        riichi: bool,
    ) -> Option<Eval> {
        let Some((cache, id)) = self.cache.as_mut() else {
            return self.evaluate(hand13, akas, win_tile_type, win_is_aka, tsumo, riichi);
        };
        let key = eval_key(*id, hand13, akas, win_tile_type, win_is_aka, tsumo, riichi);
        if let Some(&e) = cache.map.get(&key) {
            cache.hits += 1;
            return e;
        }
        let e = self.evaluate(hand13, akas, win_tile_type, win_is_aka, tsumo, riichi);
        let (cache, _) = self.cache.as_mut().expect("attached");
        cache.misses += 1;
        cache.map.insert(key, e);
        e
    }

    fn pts(&mut self, han: u32, fu: u32, tsumo: bool) -> f64 {
        // Below yakuman, han stop at kazoe yakuman (13) -- the cap riichienv's
        // evaluator applies, needed here because meld dora, situational and
        // ura han are added after it. So the points depend on min(han, 13)
        // only, which indexes the table.
        let (h, f) = (han.min(13) as usize, (fu / 5) as usize);
        if fu % 5 != 0 || f >= PTS_FU_SLOTS {
            return calculate_score(h as u8, fu as u8, self.is_oya, tsumo, 0, 4).total as f64;
        }
        let slot = &mut self.pts_table[h][f][usize::from(tsumo)];
        if *slot < 0.0 {
            *slot = calculate_score(h as u8, fu as u8, self.is_oya, tsumo, 0, 4).total as f64;
        }
        *slot
    }

    /// The evaluator's verdict on `hand13` completed by the winning tile, or
    /// None if the shape does not win or has no yaku.
    fn evaluate(
        &self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
        win_is_aka: bool,
        tsumo: bool,
        riichi: bool,
    ) -> Option<Eval> {
        // Build the 136-id concealed hand (reds distinguished).
        let mut ids: Vec<u8> = Vec::with_capacity(14);
        for t in 0..34usize {
            let mut n = hand13[t];
            if let Some(slot) = aka_slot(t).filter(|&s| akas[s]) {
                ids.push(AKA_TIDS[slot]);
                n -= 1;
            }
            for k in 0..n {
                ids.push((t as u8) * 4 + 3 - k);
            }
        }
        let win_tid = if win_is_aka {
            AKA_TIDS[aka_slot(win_tile_type).unwrap()]
        } else {
            (win_tile_type as u8) * 4 + 2
        };
        let cond = Conditions {
            tsumo,
            riichi,
            player_wind: self.player_wind,
            round_wind: self.round_wind,
            ..Default::default()
        };
        let he = HandEvaluator::new(ids, self.melds.clone());
        let res = he.calc(win_tid, self.dora_indicators.clone(), vec![], Some(cond));
        if !res.is_win || res.han == 0 {
            return None;
        }
        Some((res.han, res.fu, res.yakuman))
    }

    /// Expected tsumo totals for +0..+3 extra situational han, or None if
    /// the winning shape has no yaku. `unseen` is the path-updated unseen counts.
    pub fn score4(
        &mut self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
        win_is_aka: bool,
        unseen: &Counts,
    ) -> Option<[f64; 4]> {
        let e = self.eval_cached(hand13, akas, win_tile_type, win_is_aka, true, self.assume_riichi)?;
        Some(self.expected4(e, hand13, win_tile_type, unseen, true))
    }

    /// `score4` for a ron on `win_tile_type`: the ron payment (riichi as the
    /// scorer assumes it), the same dora, red fives and ura expectation.
    /// None when the ron has no yaku (a closed yakuless hand without riichi).
    /// The evaluator's verdict is cached per (hand, akas, winning tile, aka);
    /// the ura expectation still follows `unseen`.
    pub fn score4_ron(
        &mut self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
        win_is_aka: bool,
        unseen: &Counts,
    ) -> Option<[f64; 4]> {
        let e = self.ron_eval(hand13, akas, win_tile_type, win_is_aka)?;
        Some(self.expected4(e, hand13, win_tile_type, unseen, false))
    }

    /// Whether a ron on `win_tile_type` has a yaku without riichi (whatever
    /// the scorer assumes), i.e. a dama hand could win on it.
    pub fn ron_has_yaku_without_riichi(
        &mut self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
    ) -> bool {
        self.eval_cached(hand13, akas, win_tile_type, false, false, false).is_some()
    }

    /// Expected tsumo and ron totals of one wait (`score4`, `score4_ron`),
    /// computing the ura distribution once for both.
    pub fn score4_pair(
        &mut self,
        hand13: &Counts,
        akas: &Akas,
        win_tile_type: usize,
        win_is_aka: bool,
        unseen: &Counts,
    ) -> (Option<[f64; 4]>, Option<[f64; 4]>) {
        let tsumo = self.eval_cached(hand13, akas, win_tile_type, win_is_aka, true, self.assume_riichi);
        let ron = self.ron_eval(hand13, akas, win_tile_type, win_is_aka);
        if tsumo.is_none() && ron.is_none() {
            return (None, None);
        }
        let ura = self.ura(hand13, win_tile_type, unseen);
        (tsumo.map(|e| self.totals(e, &ura, true)), ron.map(|e| self.totals(e, &ura, false)))
    }

    /// The cached ron verdict.
    fn ron_eval(&mut self, hand13: &Counts, akas: &Akas, win_tile_type: usize, win_is_aka: bool) -> Option<Eval> {
        let key = ron_key(hand13, akas, win_tile_type, win_is_aka);
        if let Some(&e) = self.ron_cache.get(&key) {
            return e;
        }
        let e = self.eval_cached(hand13, akas, win_tile_type, win_is_aka, false, self.assume_riichi);
        self.ron_cache.insert(key, e);
        e
    }

    /// Totals for +0..+3 extra situational han of an evaluated win.
    fn expected4(
        &mut self,
        e: Eval,
        hand13: &Counts,
        win_tile_type: usize,
        unseen: &Counts,
        tsumo: bool,
    ) -> [f64; 4] {
        if e.2 {
            return self.totals(e, &None, tsumo);
        }
        let ura = self.ura(hand13, win_tile_type, unseen);
        self.totals(e, &ura, tsumo)
    }

    /// The ura distribution (ways by ura han, and all ways) when riichi is
    /// assumed and enough unseen tiles are left to turn the indicators.
    fn ura(&self, hand13: &Counts, win_tile_type: usize, unseen: &Counts) -> Option<(UraDist, f64)> {
        if !(self.assume_riichi && self.n_indicators > 0) {
            return None;
        }
        // Exact uradora expectation: one ura indicator per dora indicator
        // (kan ura included), drawn without replacement from the unseen
        // tiles (path-updated, winning tile removed). Each
        // indicator of tile type t adds as many han as the hand holds of the
        // tile it points to -- concealed tiles plus own kans' tiles.
        let mut held = *hand13;
        held[win_tile_type] += 1;
        for m in &self.melds {
            for &id in &m.tiles {
                held[(id / 4) as usize] += 1;
            }
        }
        let mut w = *unseen;
        w[win_tile_type] = w[win_tile_type].saturating_sub(1);
        let total: u32 = w.iter().map(|&c| c as u32).sum();
        let k = self.n_indicators;
        if (total as usize) < k {
            return None;
        }
        let dist = ura_distribution_grouped(&w, &held, k, &self.next_dora_types);
        Some((dist, binom(total, k as u32)))
    }

    /// Totals for +0..+3 extra han of an evaluated win, over `ura`.
    fn totals(&mut self, (eval_han, fu, yakuman): Eval, ura: &Option<(UraDist, f64)>, tsumo: bool) -> [f64; 4] {
        if yakuman {
            // A yakuman pays its own multiple; nothing added on top counts.
            let s = calculate_score((eval_han / 13 * 13) as u8, fu as u8, self.is_oya, tsumo, 0, 4);
            return [s.total as f64; 4];
        }
        let han = eval_han + self.meld_dora_han;
        match ura {
            Some((dist, combos)) => {
                let mut out = [0.0f64; 4];
                for x in 0..4u32 {
                    let mut ev = 0.0;
                    for (h, &ways) in dist.iter().enumerate() {
                        if ways > 0.0 {
                            ev += ways / combos * self.pts(han + x + h as u32, fu, tsumo);
                        }
                    }
                    out[x as usize] = ev;
                }
                out
            }
            None => [self.pts(han, fu, tsumo), self.pts(han + 1, fu, tsumo),
                     self.pts(han + 2, fu, tsumo), self.pts(han + 3, fu, tsumo)],
        }
    }
}

/// `n` choose `r` as f64 (exact for the small counts used here).
fn binom(n: u32, r: u32) -> f64 {
    (0..r).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
}

/// At most five dora indicators (the first plus four kan dora).
const MAX_URA_INDICATORS: usize = 5;
/// Ura han 0..=4 per indicator.
const URA_LEN: usize = 4 * MAX_URA_INDICATORS + 1;
/// Ways by ura han (index), as `ura_distribution` counts them.
type UraDist = [f64; URA_LEN];

/// `ura_distribution` computed faster, with the same values: the tile
/// types whose indicator adds no han (the hand holds none of the tile they
/// point to) are pooled into one group of `c0` tiles (choosing `x` of them:
/// `binom(c0, x)` ways, Vandermonde's identity), and the counts are
/// updated in place on a stack table. Every entry is an integer below
/// 2^53 (at most `binom(136, 5)`), computed exactly in f64 whatever the
/// summation order, so the result is bit-identical.
fn ura_distribution_grouped(unseen: &Counts, held: &Counts, k: usize, next_dora_types: &[usize; 34]) -> UraDist {
    assert!(k <= MAX_URA_INDICATORS, "{k} dora indicators");
    // ways[j][h]: j indicators placed so far, h han so far.
    let mut ways = [[0.0f64; URA_LEN]; MAX_URA_INDICATORS + 1];
    let c0: u32 = (0..34)
        .filter(|&t| held[next_dora_types[t]] == 0)
        .map(|t| u32::from(unseen[t]))
        .sum();
    for (x, row) in ways.iter_mut().enumerate().take(k + 1) {
        row[0] = if x as u32 <= c0 { binom(c0, x as u32) } else { 0.0 };
    }
    for t in 0..34usize {
        let c = usize::from(unseen[t]);
        let e = usize::from(held[next_dora_types[t]]);
        if c == 0 || e == 0 {
            continue;
        }
        // Descending j: row j is read before this tile type writes into it.
        for j in (0..k).rev() {
            for h in 0..=4 * j {
                let base = ways[j][h];
                if base == 0.0 {
                    continue;
                }
                for x in 1..=c.min(k - j) {
                    ways[j + x][h + x * e] += base * binom(c as u32, x as u32);
                }
            }
        }
    }
    ways[k]
}

/// Ways (as f64 counts) to turn over `k` ura indicators from the unseen
/// tile counts `unseen`, by total ura han against `held` (index = han). Summed by
/// tile type in ascending order, so the result is deterministic. The
/// reference for `ura_distribution_grouped`, which the scorer uses.
/// `next_dora_types[t]`: the dora tile type an indicator of tile type `t` points to.
#[cfg(test)]
fn ura_distribution(unseen: &Counts, held: &Counts, k: usize, next_dora_types: &[usize; 34]) -> Vec<f64> {
    let max_han = 4 * k + 1;
    // ways[j][h]: j indicators placed so far, h han so far.
    let mut ways = vec![vec![0.0f64; max_han * 4]; k + 1];
    ways[0][0] = 1.0;
    for t in 0..34usize {
        let c = u32::from(unseen[t]);
        if c == 0 {
            continue;
        }
        let e = held[next_dora_types[t]] as usize;
        let mut next = ways.clone();
        for j in 0..k {
            for h in 0..ways[j].len() {
                let base = ways[j][h];
                if base == 0.0 {
                    continue;
                }
                for x in 1..=(c as usize).min(k - j) {
                    next[j + x][h + x * e] += base * binom(c, x as u32);
                }
            }
        }
        ways = next;
    }
    ways.swap_remove(k)
}


#[cfg(test)]
mod ura_tests {
    use super::*;

    /// Every ordered draw of `k` indicators from a small pool, by ura han.
    fn brute(unseen: &Counts, held: &Counts, k: usize) -> Vec<f64> {
        let pool: Vec<usize> = (0..34).flat_map(|t| std::iter::repeat(t).take(unseen[t] as usize)).collect();
        let mut out = vec![0.0; 4 * k + 1];
        let mut idx: Vec<usize> = (0..k).collect();
        // Unordered k-subsets of distinct physical tiles.
        loop {
            let h: usize = idx.iter().map(|&i| held[next_dora_type(pool[i])] as usize).sum();
            out[h] += 1.0;
            let mut i = k;
            while i > 0 && idx[i - 1] == pool.len() - k + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            idx[i - 1] += 1;
            for j in i..k {
                idx[j] = idx[j - 1] + 1;
            }
        }
        out
    }

    #[test]
    fn distribution_matches_enumeration() {
        let mut unseen = [0u8; 34];
        for (t, c) in [(0, 2), (1, 3), (8, 1), (27, 4), (30, 2), (33, 1)] {
            unseen[t] = c;
        }
        let mut held = [0u8; 34];
        for (t, c) in [(1, 2), (2, 3), (0, 1), (28, 4), (31, 1), (27, 1)] {
            held[t] = c;
        }
        for k in 1..=5 {
            let d = ura_distribution(&unseen, &held, k, &std::array::from_fn(next_dora_type));
            let b = brute(&unseen, &held, k);
            for h in 0..b.len() {
                assert_eq!(d.get(h).copied().unwrap_or(0.0), b[h], "k={k} h={h}");
            }
            let total: u32 = unseen.iter().map(|&c| u32::from(c)).sum();
            assert_eq!(d.iter().sum::<f64>(), binom(total, k as u32), "k={k}");
        }
    }

    /// The grouped table is bit-identical to the reference on random
    /// unseen/held counts, 1..=5 indicators.
    #[test]
    fn grouped_matches_reference_bits() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n) as u8
        };
        let next: [usize; 34] = std::array::from_fn(next_dora_type);
        for _ in 0..3000 {
            let unseen: Counts = std::array::from_fn(|_| rnd(5));
            let held: Counts = std::array::from_fn(|_| if rnd(3) == 0 { rnd(5) } else { 0 });
            for k in 1..=5 {
                let r = ura_distribution(&unseen, &held, k, &next);
                let g = ura_distribution_grouped(&unseen, &held, k, &next);
                for h in 0..r.len().max(URA_LEN) {
                    let a = r.get(h).copied().unwrap_or(0.0);
                    let b = g.get(h).copied().unwrap_or(0.0);
                    assert_eq!(a.to_bits(), b.to_bits(), "k={k} h={h}");
                }
            }
        }
    }

    #[test]
    fn single_indicator_matches_per_tile_weights() {
        let mut unseen = [0u8; 34];
        let mut held = [0u8; 34];
        for t in 0..34 {
            unseen[t] = (t % 5) as u8;
            held[t] = (t % 3) as u8;
        }
        let mut weight = [0.0f64; 6];
        for t in 0..34 {
            if unseen[t] > 0 {
                weight[held[next_dora_type(t)] as usize] += f64::from(unseen[t]);
            }
        }
        let d = ura_distribution(&unseen, &held, 1, &std::array::from_fn(next_dora_type));
        for e in 0..6 {
            assert_eq!(d.get(e).copied().unwrap_or(0.0), weight[e]);
        }
    }
}
