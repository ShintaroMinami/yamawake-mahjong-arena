use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

use crate::types::TILES_4P;

/// Wall state for 4-player mahjong (136 tiles).
#[derive(Debug, Clone)]
pub struct WallState {
    pub tiles: Vec<u8>,
    pub dora_indicators: Vec<u8>,
    pub rinshan_draw_count: u8,
    pub pending_kan_dora_count: u8,
    pub drawable_count: u8,
    pub wall_digest: String,
    pub salt: String,
    pub seed: Option<u64>,
    pub hand_index: u64,
    /// reljong patch: per-seat live-draw queues (duplicate eval). When
    /// `Some`, every live draw comes from the acting seat's own queue, so
    /// a call never shifts another seat's future draws. The dead wall
    /// stays in `tiles` (front-indexed) and is never split.
    pub split_queues: Option<[Vec<u8>; 4]>,
}

impl WallState {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            tiles: Vec::new(),
            dora_indicators: Vec::new(),
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            drawable_count: 0,
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
            split_queues: None,
        }
    }

    pub fn shuffle(&mut self) {
        let mut w: Vec<u8> = (0..TILES_4P as u8).collect();
        let mut rng = if let Some(episode_seed) = self.seed {
            let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::seed_from_u64(hand_seed)
        } else {
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::from_rng(&mut rand::rng())
        };

        w.shuffle(&mut rng);
        self.salt = format!("{:016x}", rng.next_u64());

        let mut hasher = Sha256::new();
        hasher.update(self.salt.as_bytes());
        for &t in &w {
            hasher.update([t]);
        }
        self.wall_digest = format!("{:x}", hasher.finalize());

        w.reverse();
        self.tiles = w;

        self.dora_indicators.clear();
        if self.tiles.len() > 5 {
            self.dora_indicators.push(self.tiles[4]);
        }
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.drawable_count = 0;
        self.split_queues = None;
    }

    pub fn load_wall(&mut self, tiles: Vec<u8>) {
        let mut t = tiles;
        t.reverse();
        self.tiles = t;
        self.dora_indicators.clear();
        if self.tiles.len() > 5 {
            self.dora_indicators.push(self.tiles[4]);
        }
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.drawable_count = 0;
        self.split_queues = None;
    }

    /// reljong patch: carve the live region into fixed per-seat queues.
    ///
    /// Call right after the deal: `tiles` then holds the 14 dead-wall
    /// tiles at the front and the live draws at the back (drawn via
    /// `pop()`). The k-th live draw is assigned to seat `(oya + k) % 4`
    /// — the seat that would draw it in a call-free game, so a call-free
    /// hand is identical to the shared-wall mode. The live region is cut
    /// off `tiles` so the front-indexed dead wall keeps its layout.
    pub fn split_live_region(&mut self, oya: u8) {
        let live = self.tiles.split_off(14);
        let mut queues: [Vec<u8>; 4] = Default::default();
        for (k, &t) in live.iter().rev().enumerate() {
            queues[(oya as usize + k) % 4].push(t);
        }
        for q in &mut queues {
            q.reverse();
        }
        self.split_queues = Some(queues);
    }

    /// reljong patch: draw for `actor` from its own queue; when heavy
    /// calling runs the actor's queue dry, borrow from the longest queue
    /// (on a tie, the highest seat index). The caller gates on
    /// `drawable_count`, which stays the single shared haitei clock.
    pub fn draw_split(&mut self, actor: u8) -> Option<u8> {
        let queues = self.split_queues.as_mut()?;
        if let Some(t) = queues[actor as usize].pop() {
            return Some(t);
        }
        let longest = (0..4).max_by_key(|&i: &usize| queues[i].len())?;
        queues[longest].pop()
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
