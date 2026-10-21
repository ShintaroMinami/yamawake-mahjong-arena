//! The fold (betaori) model: one MLP scores each discard against one other
//! seat (design: docs/suzume.md, "押し引き"; fitted to the teacher's
//! fold-state discards).
//!
//! For each tile type `t` the features are built against one other seat `r`
//! (in riichi or not; a seat not in riichi takes its early discards as the
//! ones before a declaration), from this seat's view and the kyoku's river record
//! (`MjView::apply_river`):
//!
//! * deal-in patterns against `r` (`patterns`): what `r` could hold to ron
//!   on `t`, counted from the copies not visible here (`u`), with `r`'s
//!   genbutsu (its river and every tile passed since its riichi) as
//!   furiten, and the five of a suit whose red five `r` cut before its
//!   riichi taken as not held;
//! * the rate `t` is `r`'s wait (`wait_table`, by pattern bin x own-turn
//!   bin; 0 for a genbutsu), the safe tiles left after `t`, and the same
//!   rate against the two other seats were they tenpai;
//! * reading `r`'s river: 2/8, 3/7 and 4/6 cut early (the outside tiles),
//!   the outside suji of the declaring tile;
//! * the hand: the shanten `t` gives up, the hand-building MLP's score
//!   (`universal_mlp::mlp_choice`), copies held, the tile's kind and dora,
//!   and how long ago anyone last cut it;
//! * the stakes, the same for every tile of a decision: whether `r` or this
//!   seat deals, the dora indicators shown and the dora not visible here;
//! * `r` itself (`seat:*`): riichi, melds (closed kans included), dora in its
//!   melds, discards, yakuhai pons, melds of one suit / tanyao / chanta;
//!   `r`'s discards of `t`'s suit and when it first cut that suit; this
//!   seat's best after-discard shanten.
//!
//! Scores are `w2 . relu(W1 (x - mean) / std + b1) + b2`
//! (`params/legacy/hand/<hand>/discard_F.json`); each tile type's scores against
//! every other seat are combined by their soft minimum (`FoldMlp::combine`).
//! The highest-scoring legal type is discarded (the lower type on a tie),
//! the plain copy before a red five.

use super::ctx::Ctx;
use super::mlp::add_bias;
use super::params::{FoldMlp, FoldTables};
use super::rules::universal_mlp::mlp_choice;
use super::view::{MjView, NEVER, Threat};
use crate::eval::feat::tiles::AKA_TILE_TYPES;

pub const N_FEATURES: usize = 42;

/// Every feature the model may read, by its name in the model file.
pub const FEATURES: [&str; N_FEATURES] = [
    "patterns_log",       // ln(1 + deal-in patterns against r)
    "turn",               // min(own discards, turn_cap) / turn_cap
    "dshanten",           // shanten after discarding t minus the best legal discard's
    "honor",              // t is an honor
    "riichi_yaku",        // a dragon 1; a wind: (round wind) + (r's seat wind); else 0
    "safe_sum",           // the kept tiles' safety against r (1 - est / the turn's worst rate), summed
    "other_wait_est",     // the wait rate of t against each other seat (its own river), summed
    "ryanmen_log",        // ln(1 + ryanmen patterns)
    "tanki_unseen",       // tanki patterns (copies of t unseen)
    "hand_mlp",           // the hand-building MLP's score of t minus its best
    "held",               // copies of t in the concealed hand
    "outside_28",         // r cut the 2 (8) early: t is the 1 (9)
    "outside_37",         // r cut the 3 (7) early: t is the 1 or 2 (8 or 9)
    "outside_46",         // r cut the 4 (6) early: t is the 2 or 3 (7 or 8)
    "decl_outside_suji",  // t is the outside suji of r's declaring tile
    "last_ago",           // min(discards since anyone last cut t, cap) / cap (1 if never)
    "patterns:kanchan",   // ln(1 + kanchan patterns)
    "patterns:penchan",   // ln(1 + penchan patterns)
    "patterns:shanpon",   // ln(1 + shanpon patterns)
    "order:others_safe",  // share of the two other seats for which t is genbutsu
    "supply:safe_left",   // genbutsu against r kept after discarding t
    "supply:enough",      // safe_sum >= own draws left (ceil(live wall / 4))
    "tile_kind:dora",     // dora han of t (red fives not counted)
    "tile_kind:terminal", // a 1 or 9
    "tile_kind:two_eight", // a 2 or 8
    "genbutsu:genbutsu",  // r cannot ron t
    "stake:riichi_dealer",   // r is the dealer (decision-level)
    "self_dealer:self_dealer", // this seat is the dealer (decision-level)
    "stake:dora_indicators", // dora indicators shown (decision-level)
    "stake:dora_unseen",     // dora han not visible from here: sum over types of dora han x unseen copies
    "shanten:best_shanten",  // the best legal discard's shanten (decision-level)
    "seat:riichi",           // r has declared riichi
    "seat:n_melds",          // r's melds (an ankan too)
    "seat:meld_dora",        // dora han in r's melds (red fives too)
    "seat:dealer",           // r is the dealer
    "seat:n_discards",       // r's discards (called ones included)
    "seat:yakuhai_melds",    // r's yakuhai pon / kan melds
    "seat:meld_flush",       // r's melds: number tiles of one suit (honors allowed)
    "seat:meld_tanyao",      // r's melds: every tile a 2..8
    "seat:meld_chanta",      // r's melds: each holds a 1, a 9 or an honor
    "suit:discards",         // r's discards of t's kind (m / p / s / honors)
    "suit:first",            // min(r's own index of its first discard of t's kind, turn_cap) / turn_cap; 1 if none
];

impl FoldMlp {
    /// The network's score of one feature row (`FEATURES` order), with the
    /// decision's situation share `sit` (`sit_bias`).
    pub fn score(&self, row: &[f64; N_FEATURES], sit: &[f64]) -> f64 {
        let x: Vec<f64> = self.cols.iter().enumerate().map(|(j, &c)| (row[c] - self.mean[j]) / self.std[j]).collect();
        let mut z = self.net.pre_dense(&x);
        add_bias(&mut z, sit);
        let mut s = self.b2;
        for (w, h) in self.w2.iter().zip(self.net.activate(z)) {
            s += w * h;
        }
        s
    }

    /// The situation share of a decision in view `v` (empty without
    /// `sit_inputs`).
    pub fn sit_bias(&self, v: &MjView) -> Vec<f64> {
        self.sit.as_ref().map_or_else(Vec::new, |s| s.bias(&self.net, v))
    }
}

impl FoldTables {
    /// The wait rate of a pattern count at an own-turn bin.
    fn rate(&self, patterns: f64, turn_bin: usize) -> f64 {
        let bin = self.pattern_edges.iter().filter(|&&e| e < patterns).count();
        self.wait_table[bin * (self.turn_edges.len() + 1) + turn_bin]
    }

    /// The highest wait rate of an own-turn bin.
    fn top_rate(&self, turn_bin: usize) -> f64 {
        let w = self.turn_edges.len() + 1;
        self.wait_table.chunks(w).map(|r| r[turn_bin]).fold(0.0, f64::max)
    }
}

/// Deal-in pattern counts per tile type `[ryanmen, kanchan, penchan,
/// shanpon, tanki]` against a seat with genbutsu `genbutsu`, from the unseen
/// copies `u`; `absent` types are not held by it (every pattern holding
/// one is out). A genbutsu has none.
///
/// ryanmen: held (n+1, n+2) waiting n, n+3 or (n-2, n-1) waiting n-3, n:
/// u(a) u(b), out when the other wait is genbutsu (furiten); kanchan
/// (n-1, n+1); penchan 12 for a 3, 89 for a 7; shanpon: a pair of `t` and a
/// pair of a non-genbutsu x, C(u(t), 2) * sum_x C(u(x), 2) / (33 * 6) (C(u(t),
/// 2) with nothing visible); tanki: u(t).
pub fn patterns(u: &[u8; 34], genbutsu: &[bool; 34], absent: &[bool; 34]) -> [[f64; 5]; 34] {
    let u: [f64; 34] = std::array::from_fn(|t| if absent[t] { 0.0 } else { f64::from(u[t]) });
    let pairs: [f64; 34] = std::array::from_fn(|t| u[t] * (u[t] - 1.0) / 2.0);
    let live_pairs: f64 = (0..34).filter(|&x| !genbutsu[x]).map(|x| pairs[x]).sum();
    let mut p = [[0.0; 5]; 34];
    for t in (0..34).filter(|&t| !genbutsu[t]) {
        p[t][3] = pairs[t] * (live_pairs - pairs[t]) / (33.0 * 6.0);
        p[t][4] = u[t];
        if t >= 27 {
            continue;
        }
        let (b, n) = (t - t % 9, t % 9);
        if n <= 5 && !genbutsu[b + n + 3] {
            p[t][0] += u[b + n + 1] * u[b + n + 2];
        }
        if n >= 3 && !genbutsu[b + n - 3] {
            p[t][0] += u[b + n - 2] * u[b + n - 1];
        }
        if (1..=7).contains(&n) {
            p[t][1] = u[b + n - 1] * u[b + n + 1];
        }
        if n == 2 {
            p[t][2] = u[b] * u[b + 1];
        }
        if n == 6 {
            p[t][2] = u[b + 7] * u[b + 8];
        }
    }
    p
}

/// The fives a seat cannot hold: those of the suits whose red five it cut
/// (before its riichi).
fn absent_fives(th: &Threat) -> [bool; 34] {
    let mut out = [false; 34];
    for (slot, &t) in AKA_TILE_TYPES.iter().enumerate() {
        out[t] = th.red_pre[slot];
    }
    out
}

/// What one fold decision shares across its scored seats (every other seat
/// with the default all-seat model): the shanten each
/// legal discard type leaves, and the hand-building MLP's scores.
pub struct FoldInputs {
    /// Shanten after discarding one copy of the type; `None` when not a
    /// legal discard.
    pub shanten_after: [Option<i8>; 34],
    /// The hand-building MLP's score of each discard type (the red five
    /// merged into its type by max) minus the decision's best; 0 for a
    /// type it does not score.
    pub hand_mlp: [f64; 34],
}

impl FoldInputs {
    pub fn of(ctx: &Ctx) -> Self {
        let mut shanten_after = [None; 34];
        for c in ctx.discard_cands() {
            shanten_after[usize::from(c.tile_type)] = Some(c.shanten);
        }
        let mut hand_mlp = [f64::NEG_INFINITY; 34];
        if let Some((_, u, scores, _)) = mlp_choice(ctx) {
            for ((kind, id), s) in u.options.iter().zip(scores) {
                // A discard of any kind (a tenpai hand's dama / break too), not the riichi declaration.
                if !matches!(*kind, "discard" | "dama" | "break") {
                    continue;
                }
                let t = if *id >= 34 { AKA_TILE_TYPES[usize::from(*id - 34)] } else { usize::from(*id) };
                hand_mlp[t] = hand_mlp[t].max(s);
            }
        }
        let best = hand_mlp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for x in &mut hand_mlp {
            *x = if x.is_finite() { *x - best } else { 0.0 };
        }
        Self { shanten_after, hand_mlp }
    }
}

/// The raw features of every tile type against the scored seat `r`
/// (absolute; any other seat, riichi or not), in `FEATURES` order, with the
/// tables and caps `md` (`Models::tables`).
pub fn features(md: &FoldTables, view: &MjView, r: usize, inp: &FoldInputs) -> [[f64; N_FEATURES]; 34] {
    let me = usize::from(view.seat);
    let th = &view.threats[r];
    let genbutsu = &th.safe;
    let unseen = view.unseen();
    let hand = &view.hand;
    let pat = patterns(&unseen, genbutsu, &absent_fives(th));
    let tot: [f64; 34] = std::array::from_fn(|t| pat[t].iter().sum());
    let turn = view.n_own_discards;
    let tb = md.turn_edges.iter().filter(|&&e| e <= turn).count();
    let est: [f64; 34] = std::array::from_fn(|t| if genbutsu[t] { 0.0 } else { md.rate(tot[t], tb) });
    let top: f64 = md.top_rate(tb);
    let safety: [f64; 34] = std::array::from_fn(|t| (1.0 - est[t] / top).clamp(0.0, 1.0));
    let safe_total: f64 = (0..34).map(|x| f64::from(hand[x]) * safety[x]).sum();
    let gen_total: f64 = (0..34).filter(|&x| genbutsu[x]).map(|x| f64::from(hand[x])).sum();
    let draws = (f64::from(view.drawable_count) / 4.0).ceil();
    // The two other seats: their genbutsu and their wait rates.
    let others: Vec<usize> = (0..4).filter(|&s| s != me && s != r).collect();
    let mut others_safe = [0.0; 34];
    let mut other_est = [0.0; 34];
    for &s in &others {
        let o = &view.threats[s];
        let po = patterns(&unseen, &o.safe, &absent_fives(o));
        for t in 0..34 {
            others_safe[t] += f64::from(u8::from(o.safe[t])) / others.len() as f64;
            if !o.safe[t] {
                other_est[t] += md.rate(po[t].iter().sum(), tb);
            }
        }
    }
    // r's early discards before its riichi, and its declaring tile (none
    // for a seat not in riichi).
    let early: [bool; 34] = std::array::from_fn(|x| {
        let f = th.first_discard[x];
        // (a seat not in riichi: its early discards so far)
        f != NEVER && f < md.early_discards && if th.riichi { th.decl.is_some_and(|(at, _)| f < at) } else { true }
    });
    let mut outside = [[0.0; 3]; 34];
    let mut decl_out = [0.0; 34];
    for b in [0, 9, 18] {
        let mut set = |t: usize, k: usize, from: usize| outside[b + t][k] = f64::from(u8::from(early[b + from]));
        set(0, 0, 1);
        set(8, 0, 7);
        for n in [0, 1] {
            set(n, 1, 2);
            set(8 - n, 1, 6);
        }
        for n in [1, 2] {
            set(n, 2, 3);
            set(8 - n, 2, 5);
        }
    }
    if let Some((_, d)) = th.decl.filter(|&(_, d)| d < 27) {
        let (b, n) = (usize::from(d - d % 9), i32::from(d % 9));
        let outs: &[i32] = match n {
            0..=3 => &[n - 3],
            4 => &[n - 3, n + 3],
            _ => &[n + 3],
        };
        for &o in outs.iter().filter(|&&o| (0..9).contains(&o)) {
            decl_out[b + o as usize] = 1.0;
        }
    }
    let best_shanten = inp.shanten_after.iter().flatten().min().copied();
    let flag = |x: bool| f64::from(u8::from(x));
    let riichi_dealer = flag(th.wind == 27);
    let self_dealer = flag(view.player_wind == 27);
    let dora_indicators = view.dora_indicator_types.len() as f64;
    let dora_unseen: f64 = (0..34).map(|x| f64::from(view.dora_per_tile_type[x]) * f64::from(unseen[x])).sum();
    std::array::from_fn(|t| {
        let p = &pat[t];
        let g = genbutsu[t];
        let held = hand[t] > 0;
        let num = t < 27;
        let rank = t % 9;
        let ago = if view.last_ago[t] == NEVER { md.last_ago_cap } else { f64::from(view.last_ago[t]).min(md.last_ago_cap) };
        let safe_sum = safe_total - if held { safety[t] } else { 0.0 };
        let not_gen = |x: f64| if g { 0.0 } else { x };
        [
            tot[t].ln_1p(),
            f64::from(turn).min(md.turn_cap) / md.turn_cap,
            match (inp.shanten_after[t], best_shanten) {
                (Some(s), Some(b)) => f64::from(s - b),
                _ => 0.0,
            },
            flag(t >= 27),
            match t {
                31.. => 1.0,
                27..=30 => flag(t == usize::from(view.round_wind)) + flag(t == usize::from(th.wind)),
                _ => 0.0,
            },
            safe_sum,
            other_est[t],
            p[0].ln_1p(),
            p[4],
            inp.hand_mlp[t],
            f64::from(hand[t]),
            not_gen(outside[t][0]),
            not_gen(outside[t][1]),
            not_gen(outside[t][2]),
            not_gen(decl_out[t]),
            ago / md.last_ago_cap,
            p[1].ln_1p(),
            p[2].ln_1p(),
            p[3].ln_1p(),
            others_safe[t],
            gen_total - flag(g && held),
            flag(safe_sum >= draws),
            f64::from(view.dora_per_tile_type[t]),
            flag(num && (rank == 0 || rank == 8)),
            flag(num && (rank == 1 || rank == 7)),
            flag(g),
            riichi_dealer,
            self_dealer,
            dora_indicators,
            dora_unseen,
            best_shanten.map_or(0.0, f64::from),
            flag(th.riichi),
            f64::from(th.n_melds),
            f64::from(th.meld_dora),
            riichi_dealer,
            f64::from(th.n_discards),
            f64::from(th.yakuhai_melds),
            flag(th.meld_flush),
            flag(th.meld_tanyao),
            flag(th.meld_chanta),
            f64::from(th.suit_discards[(t / 9).min(3)]),
            {
                let f = th.suit_first[(t / 9).min(3)];
                if f == NEVER { 1.0 } else { f64::from(f).min(md.turn_cap) / md.turn_cap }
            },
        ]
    })
}

/// The features against every other seat (riichi or not), in seat order
/// from this seat's shimocha: what the push / fold models read (`pf=1`).
pub fn seat_features(ctx: &Ctx) -> [(usize, [[f64; N_FEATURES]; 34]); 3] {
    let v = ctx.view;
    let inp = FoldInputs::of(ctx);
    std::array::from_fn(|k| {
        let s = (usize::from(v.seat) + k + 1) % 4;
        (s, features(ctx.models.tables(), v, s, &inp))
    })
}

/// The fold discard (`choice_by` with the fold model).
pub fn fold_choice(ctx: &Ctx) -> Option<(u8, bool)> {
    choice_by(ctx, ctx.models.fold(), &seat_features(ctx))
}

/// The discard of model `md` (the fold or the mawashi model) on the features
/// of `seat_features`: the legal tile type with the highest score (each type
/// scored by combining its seat scores, `FoldMlp::combine`; the lower type
/// on a tie), as (tile type, aka), the plain copy when both are held.
pub fn choice_by(ctx: &Ctx, md: &FoldMlp, feats: &[(usize, [[f64; N_FEATURES]; 34])]) -> Option<(u8, bool)> {
    let sit = md.sit_bias(ctx.view);
    best_discard(ctx, |t| md.combine(feats.iter().map(|(_, f)| md.score(&f[t], &sit))))
}

/// The legal discard type with the highest `score` (the lower type on a
/// tie), the plain copy when both are held.
fn best_discard(ctx: &Ctx, score: impl Fn(usize) -> f64) -> Option<(u8, bool)> {
    let mut best: Option<(u8, f64)> = None;
    for c in ctx.discard_cands() {
        let s = score(usize::from(c.tile_type));
        if best.is_none_or(|(_, b)| s > b) {
            best = Some((c.tile_type, s));
        }
    }
    let (t, _) = best?;
    let plain = ctx.discard_cands().iter().any(|c| c.tile_type == t && !c.is_aka);
    Some((t, !plain))
}
