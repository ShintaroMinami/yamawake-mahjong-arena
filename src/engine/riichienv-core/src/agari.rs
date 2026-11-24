use crate::types::{Hand, TILE_MAX};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mentsu {
    Koutsu(u8),
    Shuntsu(u8),
}

#[derive(Debug, Clone)]
pub struct Division {
    pub head: u8,
    pub body: Vec<Mentsu>,
}

pub fn is_tenpai(hand: &mut Hand) -> bool {
    // Try adding each tile 0..34
    // If is_agari becomes true, then it is tenpai.
    for i in 0..crate::types::TILE_MAX {
        if hand.counts[i] < 4 {
            hand.add(i as u8);

            // Fast checks first
            if is_kokushi(hand) {
                hand.remove(i as u8);
                return true;
            }
            if is_chiitoitsu(hand) {
                hand.remove(i as u8);
                return true;
            }

            // Pruning for standard agari:
            // Only relevant if the added tile 'i' forms a pair or mentsu.
            // This usually requires neighbors or existing count >= 2.
            let c = hand.counts[i];
            let check_standard = if i >= 27 {
                c >= 2
            } else {
                let has_p = if i % 9 > 0 {
                    hand.counts[i - 1] > 0
                } else {
                    false
                };
                let has_n = if i % 9 < 8 {
                    hand.counts[i + 1] > 0
                } else {
                    false
                };
                c >= 2 || has_p || has_n
            };

            if check_standard && is_standard_agari(hand) {
                hand.remove(i as u8);
                return true;
            }

            hand.remove(i as u8); // backtrack
        }
    }
    false
}

pub fn is_agari(hand: &mut Hand) -> bool {
    if is_kokushi(hand) {
        return true;
    }
    if is_chiitoitsu(hand) {
        return true;
    }
    is_standard_agari(hand)
}

pub fn find_divisions(hand: &Hand) -> Vec<Division> {
    let mut divisions = Vec::new();
    for i in 0..TILE_MAX {
        if hand.counts[i] >= 2 {
            let mut processing_hand = hand.clone();
            processing_hand.counts[i] -= 2;
            let mut bodies = Vec::new();
            let mut current_body = Vec::new();
            decompose_all(&mut processing_hand, 0, &mut current_body, &mut bodies);
            for body in bodies {
                divisions.push(Division {
                    head: i as u8,
                    body,
                });
            }
        }
    }
    divisions
}

fn decompose_all(
    hand: &mut Hand,
    start_idx: usize,
    current_body: &mut Vec<Mentsu>,
    results: &mut Vec<Vec<Mentsu>>,
) {
    let mut i = start_idx;
    while i < TILE_MAX && hand.counts[i] == 0 {
        i += 1;
    }

    if i == TILE_MAX {
        results.push(current_body.clone());
        return;
    }

    // Try Koutsu
    if hand.counts[i] >= 3 {
        hand.counts[i] -= 3;
        current_body.push(Mentsu::Koutsu(i as u8));
        decompose_all(hand, i, current_body, results);
        current_body.pop();
        hand.counts[i] += 3;
    }

    // Try Shuntsu
    if i < 27 {
        let is_valid_seq_start = match i {
            0..=6 => true,   // 1m-7m
            9..=15 => true,  // 1p-7p
            18..=24 => true, // 1s-7s
            _ => false,
        };

        if is_valid_seq_start && hand.counts[i + 1] > 0 && hand.counts[i + 2] > 0 {
            hand.counts[i] -= 1;
            hand.counts[i + 1] -= 1;
            hand.counts[i + 2] -= 1;
            current_body.push(Mentsu::Shuntsu(i as u8));
            decompose_all(hand, i, current_body, results);
            current_body.pop();
            hand.counts[i] += 1;
            hand.counts[i + 1] += 1;
            hand.counts[i + 2] += 1;
        }
    }
}

pub fn is_kokushi(hand: &Hand) -> bool {
    let kokushi_indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut pair_found = false;

    for &idx in &kokushi_indices {
        let c = hand.counts[idx];
        if c == 0 {
            return false;
        }
        if c == 2 {
            if pair_found {
                return false; // Two pairs? Not kokushi
            }
            pair_found = true;
        } else if c > 2 {
            return false;
        }
    }

    // Check total tiles. Should be 14.
    // Assuming the hand passed here has 14 tiles usually.
    // If not checking total, return pair_found.
    pair_found
}

pub fn is_chiitoitsu(hand: &Hand) -> bool {
    // 7 pairs
    let mut pairs = 0;
    for c in hand.counts.iter() {
        if *c == 2 {
            pairs += 1;
        } else if *c != 0 {
            return false;
        }
    }
    pairs == 7
}

pub fn is_standard_agari(hand: &mut Hand) -> bool {
    // Basic backtracking
    // 1. Find a pair (head)
    // 2. Decompose rest into 4 sequences/triplets

    for i in 0..TILE_MAX {
        if hand.counts[i] >= 2 {
            hand.counts[i] -= 2;
            if decompose(hand, 0) {
                hand.counts[i] += 2; // backtrack
                return true;
            }
            hand.counts[i] += 2; // backtrack
        }
    }
    false
}

fn decompose(hand: &mut Hand, start_idx: usize) -> bool {
    let mut i = start_idx;
    // Find first non-zero tile
    while i < TILE_MAX && hand.counts[i] == 0 {
        i += 1;
    }

    if i == TILE_MAX {
        return true; // All tiles used
    }

    // Try Koutsu (Triplet)
    if hand.counts[i] >= 3 {
        hand.counts[i] -= 3;
        if decompose(hand, i) {
            hand.counts[i] += 3; // backtrack even on success
            return true;
        }
        hand.counts[i] += 3; // backtrack
    }

    // Try Shuntsu (Sequence) - Only for number tiles
    if i < 27
        && let Some(is_valid_seq_start) = match i {
            0..=6 => Some(true),   // 1m-7m
            9..=15 => Some(true),  // 1p-7p
            18..=24 => Some(true), // 1s-7s
            _ => None,
        }
        && is_valid_seq_start
        && hand.counts[i + 1] > 0
        && hand.counts[i + 2] > 0
    {
        hand.counts[i] -= 1;
        hand.counts[i + 1] -= 1;
        hand.counts[i + 2] -= 1;
        if decompose(hand, i) {
            hand.counts[i] += 1;
            hand.counts[i + 1] += 1;
            hand.counts[i + 2] += 1;
            return true;
        }
        hand.counts[i] += 1;
        hand.counts[i + 1] += 1;
        hand.counts[i + 2] += 1;
    }

    false
}

/// The mentsu of one division held inline (no allocation). A concealed part of at most 14 tiles
/// holds at most 4 mentsu; the capacity leaves room for any hand of up to 26 tiles.
#[derive(Debug, Clone, Copy)]
pub struct Body {
    m: [Mentsu; 8],
    n: u8,
}

impl Body {
    pub const EMPTY: Body = Body { m: [Mentsu::Koutsu(0); 8], n: 0 };

    #[inline]
    fn push(&mut self, x: Mentsu) {
        self.m[self.n as usize] = x;
        self.n += 1;
    }

    #[inline]
    fn pop(&mut self) {
        self.n -= 1;
    }
}

impl std::ops::Deref for Body {
    type Target = [Mentsu];
    #[inline]
    fn deref(&self) -> &[Mentsu] {
        &self.m[..self.n as usize]
    }
}

impl<'a> IntoIterator for &'a Body {
    type Item = &'a Mentsu;
    type IntoIter = std::slice::Iter<'a, Mentsu>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.m[..self.n as usize].iter()
    }
}

/// `Division` with its body inline.
#[derive(Debug, Clone, Copy)]
pub struct Div {
    pub head: u8,
    pub body: Body,
}

/// The divisions of `find_divisions`, in the same order, held inline. A 14-tile hand has at most 5.
pub struct Divs {
    d: [Div; Divs::CAP],
    n: usize,
}

impl Divs {
    const CAP: usize = 16;

    #[inline]
    pub fn as_slice(&self) -> &[Div] {
        &self.d[..self.n]
    }
}

/// `find_divisions` without allocation (same divisions, same order).
pub fn find_divisions_inline(hand: &Hand) -> Divs {
    let mut out = Divs { d: [Div { head: 0, body: Body::EMPTY }; Divs::CAP], n: 0 };
    let mut h = hand.counts;
    for i in 0..TILE_MAX {
        if h[i] >= 2 {
            h[i] -= 2;
            let mut cur = Body::EMPTY;
            decompose_inline(&mut h, 0, i as u8, &mut cur, &mut out);
            h[i] += 2;
        }
    }
    out
}

fn decompose_inline(c: &mut [u8; TILE_MAX], start_idx: usize, head: u8, cur: &mut Body, out: &mut Divs) {
    let mut i = start_idx;
    while i < TILE_MAX && c[i] == 0 {
        i += 1;
    }
    if i == TILE_MAX {
        assert!(out.n < Divs::CAP, "more than {} divisions", Divs::CAP);
        out.d[out.n] = Div { head, body: *cur };
        out.n += 1;
        return;
    }
    if c[i] >= 3 {
        c[i] -= 3;
        cur.push(Mentsu::Koutsu(i as u8));
        decompose_inline(c, i, head, cur, out);
        cur.pop();
        c[i] += 3;
    }
    if i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        cur.push(Mentsu::Shuntsu(i as u8));
        decompose_inline(c, i, head, cur, out);
        cur.pop();
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
    }
}
