//! The waits of a tenpai hand and their value, and the round bin of a
//! decision: parts the universal features read.

use crate::suzume::ctx::Ctx;
use crate::eval::score::Scorer;

/// The round bin of the 1-layer MLPs' outside bias
/// (0 E1, 1 E2-4, 2 S1-3, 3 all-last).
pub fn rc_round_bin(round_wind: u8, kyoku_index: u8) -> usize {
    match (round_wind, kyoku_index) {
        (27, 0) => 0,
        (27, _) => 1,
        (28, 3) => 3,
        _ => 2,
    }
}

/// The waits of a tenpai 13-tile hand and their value without riichi.
pub(crate) struct WaitValue {
    /// Unseen copies of the waits.
    pub(crate) unseen: u32,
    pub(crate) ron_points: f64,
    pub(crate) tsumo_points: f64,
    pub(crate) ron_possible: bool,
}

pub(crate) fn waits_of(ctx: &Ctx, h13: &[u8; 34]) -> u64 {
    let mut tables = ctx.tables();
    let mut mask = 0u64;
    for w in 0..34 {
        if h13[w] >= 4 {
            continue;
        }
        let mut h = *h13;
        h[w] += 1;
        if tables.shanten(&h) < 0 {
            mask |= 1 << w;
        }
    }
    mask
}

/// Dama ron and tsumo points of each wait, weighted by `unseen`.
pub(crate) fn wait_value(
    scorer: &mut Scorer,
    h13: &[u8; 34],
    akas: &[bool; 3],
    waits: u64,
    unseen: &[u8; 34],
    tsumo_too: bool,
) -> WaitValue {
    let total: u32 = (0..34).filter(|&w| waits >> w & 1 == 1).map(|w| u32::from(unseen[w])).sum();
    let (mut ron, mut tsumo, mut wsum, mut ron_possible) = (0.0, 0.0, 0.0, false);
    for w in (0..34).filter(|&w| waits >> w & 1 == 1) {
        let weight = if total > 0 { f64::from(unseen[w]) } else { 1.0 };
        let r = scorer.score4_ron(h13, akas, w, false, unseen).map_or(0.0, |s| s[0]);
        ron_possible |= r > 0.0;
        ron += weight * r;
        if tsumo_too {
            tsumo += weight * scorer.score4(h13, akas, w, false, unseen).map_or(0.0, |s| s[0]);
        }
        wsum += weight;
    }
    let n = wsum.max(1.0);
    WaitValue { unseen: total, ron_points: ron / n, tsumo_points: tsumo / n, ron_possible }
}

/// The tiles split into complete melds only (sequences and triplets; honors
/// as triplets), no pair.
fn all_melds(c: &mut [u8; 34]) -> bool {
    let Some(i) = c.iter().position(|&x| x > 0) else { return true };
    if c[i] >= 3 {
        c[i] -= 3;
        let ok = all_melds(c);
        c[i] += 3;
        if ok {
            return true;
        }
    }
    if i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        let ok = all_melds(c);
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
        return ok;
    }
    false
}

/// A tenpai 13-tile hand whose every wait is a single-tile (tanki) wait:
/// each wait `w` is held and the hand minus one `w` is 4 complete melds
/// (nobetan counts; open melds are outside `h13`, so fewer melds are
/// asked for then).
pub(crate) fn is_tanki(ctx: &Ctx, h13: &[u8; 34]) -> bool {
    let waits = waits_of(ctx, h13);
    waits != 0
        && (0..34).filter(|&w| waits >> w & 1 == 1).all(|w| {
            if h13[w] == 0 {
                return false;
            }
            let mut c = *h13;
            c[w] -= 1;
            all_melds(&mut c)
        })
}

