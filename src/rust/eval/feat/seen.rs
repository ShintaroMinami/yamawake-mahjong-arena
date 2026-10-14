//! Visible-tile counts from one seat's view.
//!
//! Hand + rivers + melds + dora indicators, with the claimed-discard double-count
//! correction (a chi/pon/daiminkan tile stays in the discarder's monotonic
//! river AND sits in the meld, so it is given back once via `called_tile`;
//! ankan tiles never left a hand).

use riichienv_core::observation::Observation;
use riichienv_core::types::MeldType;

/// `counts` is the observer's own 34-format hand counts; `dora` the
/// event-timed 34-format indicator list.
pub fn seen_counts(counts: &[i32; 34], obs: &Observation, dora: &[usize]) -> [i32; 34] {
    let mut seen = *counts;
    for i in 0..4 {
        for &tid in &obs.discards[i] {
            seen[(tid / 4) as usize] += 1;
        }
    }
    for seat_melds in &obs.melds {
        for m in seat_melds {
            for &tid in &m.tiles {
                seen[(tid / 4) as usize] += 1;
            }
            if m.meld_type != MeldType::Ankan {
                if let Some(called_tile) = m.called_tile {
                    seen[(called_tile / 4) as usize] -= 1;
                }
            }
        }
    }
    for &t in dora {
        seen[t] += 1;
    }
    for s in &mut seen {
        *s = (*s).clamp(0, 4);
    }
    seen
}
