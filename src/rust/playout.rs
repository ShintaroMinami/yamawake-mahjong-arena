//! Play a kyoku out from a position in its middle (`docs/play_out_spec.md`).
//!
//! A position is the kyoku's mjai events from its `start_kyoku`; a sample fixes the tiles they leave hidden (the
//! hands, the live wall or a split wall's per-seat queues, the dead wall), the rest drawn from a seeded shuffle of
//! the unseen tiles. The position is rebuilt on the live engine, not on its event replay: the kyoku's whole wall is
//! laid out so that the deal and the draws give the logged tiles, and the logged moves go through `step`, so every
//! flag the engine keeps (furiten, ippatsu, first turn, pao, kuikae, the dead wall's counters) comes out as in a
//! game played to that point. A hidden seat's earlier draws are rebuilt backwards from its hand (`reconstruct`);
//! no choice made there reaches the position (its hand there is the given one). From the position, in-loop bots
//! play the kyoku out (`arena::GameSlot`), the specified seat's first decisions forced, and the engine stops at the
//! kyoku's end (`GameState::defer_next_round`).

use std::collections::{HashMap, VecDeque};

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use serde_json::{Map, Value, json};

use crate::arena::{Bot, GameSlot};

/// A tile kind: 0..34 the tile types (red fives excluded), 34 / 35 / 36 the red 5m / 5p / 5s.
type Kind = u8;
const N_KIND: usize = 37;
const RED_TYPES: [u8; 3] = [4, 13, 22];
const HONORS: [&str; 7] = ["E", "S", "W", "N", "P", "F", "C"];
/// Live draws of a kyoku: 136 tiles less the deal (52) and the dead wall (14).
const LIVE: usize = 70;

/// The kind of an mjai tile ("?" is `None`).
fn parse_kind(s: &str) -> Result<Option<Kind>, String> {
    let bad = || format!("bad mjai tile {s:?}");
    if s == "?" {
        return Ok(None);
    }
    if let Some(i) = HONORS.iter().position(|&h| h == s) {
        return Ok(Some(27 + i as u8));
    }
    let b = s.as_bytes();
    if !(b.len() == 2 || (b.len() == 3 && b[2] == b'r')) {
        return Err(bad());
    }
    let n = b[0].wrapping_sub(b'1');
    let suit = match b[1] {
        b'm' => 0,
        b'p' => 1,
        b's' => 2,
        _ => return Err(bad()),
    };
    if n > 8 || (b.len() == 3 && n != 4) {
        return Err(bad());
    }
    Ok(Some(if b.len() == 3 { 34 + suit } else { suit * 9 + n }))
}

/// The mjai name of a kind.
fn kind_name(k: Kind) -> String {
    match k {
        34..=36 => format!("5{}r", ["m", "p", "s"][usize::from(k - 34)]),
        27..=33 => HONORS[usize::from(k - 27)].to_string(),
        _ => format!("{}{}", k % 9 + 1, ["m", "p", "s"][usize::from(k / 9)]),
    }
}

fn kind_of_tid(t: u8) -> Kind {
    match t {
        16 => 34,
        52 => 35,
        88 => 36,
        _ => t / 4,
    }
}

/// The tile type (0..34) of a kind.
fn type_of(k: Kind) -> u8 {
    if k >= 34 { RED_TYPES[usize::from(k - 34)] } else { k }
}

/// The copies of each kind in a full set.
fn full_counts() -> [i32; N_KIND] {
    let mut c = [4; N_KIND];
    for (i, &t) in RED_TYPES.iter().enumerate() {
        c[usize::from(t)] = 3;
        c[34 + i] = 1;
    }
    c
}

/// The physical ids of a kind.
fn tids_of(k: Kind) -> Vec<u8> {
    if k >= 34 {
        vec![RED_TYPES[usize::from(k - 34)] * 4]
    } else if RED_TYPES.contains(&k) {
        (1..4).map(|i| k * 4 + i).collect()
    } else {
        (0..4).map(|i| k * 4 + i).collect()
    }
}

/// A splitmix64 stream: the seeded shuffle of a sample's unspecified tiles.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

// --- the position's events --------------------------------------------------------------------------------------

/// One logged event after `start_kyoku`, in the kinds the rebuild needs.
#[derive(Clone, Debug)]
enum Ev {
    Tsumo { actor: usize, tile: Option<Kind>, rinshan: bool },
    Dahai { actor: usize, tile: Kind, tsumogiri: bool },
    Reach { actor: usize },
    /// Chi, pon or daiminkan: the called tile and the hand tiles it consumed.
    Call { actor: usize, at: ActionType, tile: Kind, consumed: Vec<Kind> },
    Ankan { actor: usize, consumed: Vec<Kind> },
    Kakan { actor: usize, tile: Kind },
    Dora { tile: Kind },
}

impl Ev {
    fn actor(&self) -> Option<usize> {
        match self {
            Ev::Tsumo { actor, .. }
            | Ev::Dahai { actor, .. }
            | Ev::Reach { actor }
            | Ev::Call { actor, .. }
            | Ev::Ankan { actor, .. }
            | Ev::Kakan { actor, .. } => Some(*actor),
            Ev::Dora { .. } => None,
        }
    }
}

/// The kyoku's start (`start_kyoku`).
struct Start {
    round_wind: u8,
    oya: u8,
    honba: u8,
    kyotaku: u32,
    scores: [i32; 4],
    dora: Kind,
    tehais: [Vec<Option<Kind>>; 4],
}

fn field<'a>(e: &'a Value, k: &str) -> Result<&'a Value, String> {
    e.get(k).ok_or_else(|| format!("{} without {k:?}", e.get("type").and_then(Value::as_str).unwrap_or("event")))
}

fn uint(e: &Value, k: &str) -> Result<u64, String> {
    field(e, k)?.as_u64().ok_or_else(|| format!("{k:?} is not a non-negative integer"))
}

fn seat_of(e: &Value, k: &str) -> Result<usize, String> {
    let s = uint(e, k)?;
    if s > 3 {
        return Err(format!("{k:?} {s} is not a seat"));
    }
    Ok(s as usize)
}

fn tile(e: &Value, k: &str) -> Result<Kind, String> {
    let s = field(e, k)?.as_str().ok_or_else(|| format!("{k:?} is not a tile"))?;
    parse_kind(s)?.ok_or_else(|| format!("{k:?} is hidden (\"?\") on a public tile"))
}

fn tiles(e: &Value, k: &str) -> Result<Vec<Kind>, String> {
    field(e, k)?
        .as_array()
        .ok_or_else(|| format!("{k:?} is not a list"))?
        .iter()
        .map(|v| {
            let s = v.as_str().ok_or_else(|| format!("{k:?} holds a non-tile"))?;
            parse_kind(s)?.ok_or_else(|| format!("{k:?} holds a hidden tile"))
        })
        .collect()
}

fn parse_start(e: &Value) -> Result<Start, String> {
    if e.get("type").and_then(Value::as_str) != Some("start_kyoku") {
        return Err("the events must begin with start_kyoku".into());
    }
    let round_wind = match field(e, "bakaze")?.as_str() {
        Some("E") => 0,
        Some("S") => 1,
        Some("W") => 2,
        Some("N") => 3,
        _ => return Err("bad bakaze".into()),
    };
    let oya = match e.get("oya") {
        Some(_) => seat_of(e, "oya")? as u8,
        None => (uint(e, "kyoku")?.checked_sub(1).ok_or("kyoku 0")? % 4) as u8,
    };
    let kyotaku = e.get("kyotaku").or_else(|| e.get("kyoutaku")).and_then(Value::as_u64).unwrap_or(0) as u32;
    let sc = field(e, "scores")?.as_array().ok_or("scores is not a list")?;
    if sc.len() != 4 {
        return Err("scores must hold 4 numbers".into());
    }
    let mut scores = [0; 4];
    for (i, v) in sc.iter().enumerate() {
        scores[i] = v.as_i64().ok_or("scores must be integers")? as i32;
    }
    let th = field(e, "tehais")?.as_array().ok_or("tehais is not a list")?;
    if th.len() != 4 {
        return Err("tehais must hold 4 hands".into());
    }
    let mut tehais: [Vec<Option<Kind>>; 4] = Default::default();
    for (i, h) in th.iter().enumerate() {
        let h = h.as_array().ok_or("a tehai is not a list")?;
        if h.len() != 13 {
            return Err(format!("tehai {i} holds {} tiles, not 13", h.len()));
        }
        tehais[i] = h.iter().map(|v| parse_kind(v.as_str().unwrap_or("bad"))).collect::<Result<_, _>>()?;
    }
    Ok(Start {
        round_wind,
        oya,
        honba: uint(e, "honba")? as u8,
        kyotaku,
        scores,
        dora: tile(e, "dora_marker")?,
        tehais,
    })
}

/// The start, the moves (`Ev`) and each move's index in `events` (for messages).
fn parse_events(events: &[Value]) -> Result<(Start, Vec<Ev>, Vec<usize>), String> {
    let start = parse_start(events.first().ok_or("no events")?)?;
    let mut evs: Vec<Ev> = Vec::new();
    let mut src: Vec<usize> = Vec::new();
    for (j, e) in events.iter().enumerate().skip(1) {
        let t = e.get("type").and_then(Value::as_str).ok_or("an event without type")?;
        let ev = match t {
            "tsumo" => {
                let actor = seat_of(e, "actor")?;
                let s = field(e, "pai")?.as_str().ok_or("tsumo pai is not a tile")?;
                // A rinshan draw follows its kan (dora events aside).
                let rinshan = matches!(evs.iter().rev().find(|x| !matches!(x, Ev::Dora { .. })),
                    Some(Ev::Ankan { actor: a, .. } | Ev::Kakan { actor: a, .. }
                         | Ev::Call { actor: a, at: ActionType::Daiminkan, .. }) if *a == actor);
                Ev::Tsumo { actor, tile: parse_kind(s)?, rinshan }
            }
            "dahai" => Ev::Dahai {
                actor: seat_of(e, "actor")?,
                tile: tile(e, "pai")?,
                tsumogiri: field(e, "tsumogiri")?.as_bool().ok_or("tsumogiri is not a bool")?,
            },
            "reach" => Ev::Reach { actor: seat_of(e, "actor")? },
            "reach_accepted" | "start_game" => continue,
            "chi" | "pon" | "daiminkan" => {
                let at = match t {
                    "chi" => ActionType::Chi,
                    "pon" => ActionType::Pon,
                    _ => ActionType::Daiminkan,
                };
                Ev::Call { actor: seat_of(e, "actor")?, at, tile: tile(e, "pai")?, consumed: tiles(e, "consumed")? }
            }
            "ankan" => Ev::Ankan { actor: seat_of(e, "actor")?, consumed: tiles(e, "consumed")? },
            "kakan" => Ev::Kakan { actor: seat_of(e, "actor")?, tile: tile(e, "pai")? },
            "dora" => Ev::Dora { tile: tile(e, "dora_marker")? },
            "hora" | "ryukyoku" | "end_kyoku" | "end_game" | "start_kyoku" => {
                return Err(format!("the events go past the kyoku's end ({t})"));
            }
            _ => return Err(format!("unknown event type {t:?}")),
        };
        evs.push(ev);
        src.push(j);
    }
    Ok((start, evs, src))
}

// --- samples and the kyoku's wall -------------------------------------------------------------------------------

/// The hidden tiles a sample fixes (any part may be left to the seeded shuffle).
#[derive(Default)]
struct Sample {
    hands: [Option<Vec<Kind>>; 4],
    wall: Option<Vec<Kind>>,
    queues: Option<[Vec<Kind>; 4]>,
    rinshan: Option<Vec<Kind>>,
    dora: Option<Vec<Kind>>,
    ura: Option<Vec<Kind>>,
    /// The live tiles a kan cut off (never drawn), as `record_samples` returns them.
    undrawn: Option<Vec<Kind>>,
    seed: Option<u64>,
}

fn kinds(v: &Value, what: &str) -> Result<Vec<Kind>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|x| parse_kind(x.as_str().unwrap_or("bad"))?.ok_or_else(|| format!("{what} holds a hidden tile")))
        .collect()
}

fn parse_sample(v: &Value) -> Result<Sample, String> {
    let mut s = Sample::default();
    let Some(o) = v.as_object() else { return Err("a sample is not a dict".into()) };
    for (k, x) in o {
        match k.as_str() {
            "hands" => match x {
                Value::Array(a) => {
                    if a.len() != 4 {
                        return Err("hands as a list must hold 4 entries (null for none)".into());
                    }
                    for (i, h) in a.iter().enumerate() {
                        if !h.is_null() {
                            s.hands[i] = Some(kinds(h, "a hand")?);
                        }
                    }
                }
                Value::Object(m) => {
                    for (seat, h) in m {
                        let i: usize = seat.parse().ok().filter(|&i| i < 4).ok_or("hands keys are seats 0..3")?;
                        s.hands[i] = Some(kinds(h, "a hand")?);
                    }
                }
                _ => return Err("hands must be a list or a dict".into()),
            },
            "wall" => s.wall = Some(kinds(x, "wall")?),
            "wall_queues" => {
                let a = x.as_array().filter(|a| a.len() == 4).ok_or("wall_queues must hold 4 lists")?;
                let mut q: [Vec<Kind>; 4] = Default::default();
                for (i, y) in a.iter().enumerate() {
                    q[i] = kinds(y, "a wall queue")?;
                }
                s.queues = Some(q);
            }
            "rinshan" => s.rinshan = Some(kinds(x, "rinshan")?),
            "dora" => s.dora = Some(kinds(x, "dora")?),
            "ura" => s.ura = Some(kinds(x, "ura")?),
            "undrawn" => s.undrawn = Some(kinds(x, "undrawn")?),
            "seed" => s.seed = Some(x.as_u64().ok_or("seed must be a non-negative integer")?),
            _ => return Err(format!("unknown sample field {k:?}")),
        }
    }
    Ok(s)
}

fn count_into(c: &mut [i32; N_KIND], ks: &[Kind], sign: i32) {
    for &k in ks {
        c[usize::from(k)] += sign;
    }
}

fn remove_one(h: &mut Vec<Kind>, k: Kind, what: &str) -> Result<(), String> {
    let i = h.iter().position(|&x| x == k).ok_or_else(|| format!("{what}: {} is not in the hand", kind_name(k)))?;
    h.swap_remove(i);
    Ok(())
}

/// A hidden seat's start hand and its draws (by event index), rebuilt backwards from its hand at the position.
/// A tsumogiri discard names its draw, and so does a kan (its tile); any other draw is taken from the hand, never
/// the copy discarded right after it. Every hand on the way holds what the logged moves need (their tiles are added on the way back).
fn reconstruct(seat: usize, evs: &[Ev], now: &[Kind]) -> Result<(Vec<Kind>, HashMap<usize, Kind>), String> {
    let mut c = [0i32; N_KIND];
    count_into(&mut c, now, 1);
    let (mut named, mut avoid): (Option<Kind>, Option<Kind>) = (None, None);
    let mut draws = HashMap::new();
    for (i, ev) in evs.iter().enumerate().rev() {
        if ev.actor() != Some(seat) {
            continue;
        }
        match ev {
            Ev::Dahai { tile, tsumogiri, .. } => {
                c[usize::from(*tile)] += 1;
                (named, avoid) = if *tsumogiri { (Some(*tile), None) } else { (None, Some(*tile)) };
            }
            Ev::Tsumo { .. } => {
                let d = match named.take() {
                    Some(d) => d,
                    None => (0..N_KIND as u8)
                        .find(|&k| c[usize::from(k)] > 0 && Some(k) != avoid)
                        .or_else(|| avoid.filter(|&a| c[usize::from(a)] >= 2))
                        .ok_or_else(|| format!("seat {seat}: no tile for a draw"))?,
                };
                if c[usize::from(d)] <= 0 {
                    return Err(format!("seat {seat}: a tsumogiri of {} that the hand cannot have drawn", kind_name(d)));
                }
                c[usize::from(d)] -= 1;
                draws.insert(i, d);
                avoid = None;
            }
            Ev::Call { consumed, .. } => {
                count_into(&mut c, consumed, 1);
                (named, avoid) = (None, None);
            }
            // The draw before a kan is its tile, as a kan in riichi must be (any choice suits one out of riichi).
            Ev::Ankan { consumed, .. } => {
                count_into(&mut c, consumed, 1);
                let plain = consumed.iter().copied().find(|&k| k < 34);
                (named, avoid) = (plain.or(consumed.first().copied()), None);
            }
            Ev::Kakan { tile, .. } => {
                c[usize::from(*tile)] += 1;
                (named, avoid) = (Some(*tile), None);
            }
            Ev::Reach { .. } | Ev::Dora { .. } => {}
        }
    }
    let start: Vec<Kind> = (0..N_KIND as u8).flat_map(|k| std::iter::repeat_n(k, c[usize::from(k)].max(0) as usize)).collect();
    if start.len() != 13 || c.iter().any(|&x| x < 0) {
        return Err(format!("seat {seat}: its hand does not fit the events (a start hand of {} tiles)", start.len()));
    }
    Ok((start, draws))
}

/// A rebuilt position: its engine slot (`defer_next_round` on) at the point the events end.
pub struct Built {
    pub slot: GameSlot,
    /// The seats' scores there.
    pub scores: [i32; 4],
    /// The sample as it came out (the input form, every part given: fed back, it rebuilds the same position).
    pub sample: Value,
    /// The four hands there.
    pub hands: Value,
}

/// The tiles of the kyoku, laid out for the engine: each seat's start hand, every draw, the live wall left (or the
/// per-seat queues), the dead wall.
struct Layout {
    start: [Vec<Kind>; 4],
    /// The kind each `Ev::Tsumo` drew, by event index.
    draws: HashMap<usize, Kind>,
    wall: Vec<Kind>,
    queues: [Vec<Kind>; 4],
    rinshan: Vec<Kind>,
    dora: Vec<Kind>,
    ura: Vec<Kind>,
    /// Live tiles past the last draw a kan cut off (never drawn).
    leftover: Vec<Kind>,
    /// The seats whose hand came from the shuffle (neither the events nor the sample gave it).
    shuffled: [bool; 4],
    /// The seats' hands at the position, and which seats are hidden (their hands belong to the sample).
    now: [Vec<Kind>; 4],
    hidden: [bool; 4],
}

fn layout(start: &Start, evs: &[Ev], sample: &Sample, split: bool) -> Result<Layout, String> {
    let n_tsumo = evs.iter().filter(|e| matches!(e, Ev::Tsumo { .. })).count();
    let n_rinshan = evs.iter().filter(|e| matches!(e, Ev::Tsumo { rinshan: true, .. })).count();
    let n_dora = 1 + evs.iter().filter(|e| matches!(e, Ev::Dora { .. })).count();
    if n_tsumo > LIVE || n_rinshan > 4 || n_dora > 5 {
        return Err("more draws, rinshan draws or dora than a kyoku holds".into());
    }
    let drawable = LIVE - n_tsumo;
    // A seat is hidden when the events mask any of its tiles or the sample gives its hand.
    let mut hidden = [false; 4];
    for s in 0..4 {
        hidden[s] = sample.hands[s].is_some() || start.tehais[s].iter().any(Option::is_none);
    }
    for e in evs {
        if let Ev::Tsumo { actor, tile: None, .. } = e {
            hidden[*actor] = true;
        }
    }
    // Public tiles, and the known seats' hands played forward.
    let mut known = [0i32; N_KIND];
    known[usize::from(start.dora)] += 1;
    let mut hands: [Vec<Kind>; 4] = Default::default();
    let mut size = [13usize; 4];
    for s in 0..4 {
        if !hidden[s] {
            hands[s] = start.tehais[s].iter().map(|k| k.unwrap()).collect();
        }
    }
    let mut draws = HashMap::new();
    for (i, e) in evs.iter().enumerate() {
        match e {
            Ev::Tsumo { actor, tile, .. } => {
                size[*actor] += 1;
                if !hidden[*actor] {
                    let k = tile.unwrap();
                    hands[*actor].push(k);
                    draws.insert(i, k);
                }
            }
            Ev::Dahai { actor, tile, .. } => {
                known[usize::from(*tile)] += 1;
                size[*actor] = size[*actor].checked_sub(1).ok_or("a discard from an empty hand")?;
                if !hidden[*actor] {
                    remove_one(&mut hands[*actor], *tile, &format!("seat {actor} discards"))?;
                }
            }
            Ev::Call { actor, consumed, .. } | Ev::Ankan { actor, consumed } => {
                count_into(&mut known, consumed, 1);
                size[*actor] = size[*actor].checked_sub(consumed.len()).ok_or("a call from too small a hand")?;
                if !hidden[*actor] {
                    for &k in consumed {
                        remove_one(&mut hands[*actor], k, &format!("seat {actor} calls"))?;
                    }
                }
            }
            Ev::Kakan { actor, tile } => {
                known[usize::from(*tile)] += 1;
                size[*actor] = size[*actor].checked_sub(1).ok_or("a kakan from an empty hand")?;
                if !hidden[*actor] {
                    remove_one(&mut hands[*actor], *tile, &format!("seat {actor} kakan"))?;
                }
            }
            Ev::Dora { tile } => known[usize::from(*tile)] += 1,
            Ev::Reach { .. } => {}
        }
    }
    for s in 0..4 {
        if !hidden[s] {
            count_into(&mut known, &hands[s], 1);
        }
    }
    let mut pool = full_counts();
    for k in 0..N_KIND {
        pool[k] -= known[k];
        if pool[k] < 0 {
            return Err(format!("more than a full set of {} is in the events", kind_name(k as u8)));
        }
    }
    // Take out what the sample gives; the rest is shuffled into what it leaves out.
    let mut take = |ks: &[Kind], what: &str| -> Result<(), String> {
        for &k in ks {
            pool[usize::from(k)] -= 1;
            if pool[usize::from(k)] < 0 {
                return Err(format!("{what}: {} is not among the unseen tiles", kind_name(k)));
            }
        }
        Ok(())
    };
    let check = |got: usize, want: usize, what: &str| {
        if got == want { Ok(()) } else { Err(format!("{what} holds {got} tiles, not {want}")) }
    };
    // A wall, a queue set, the rinshan, dora and ura may be given in part: the tiles given come first.
    let most = |got: usize, max: usize, what: &str| {
        if got <= max { Ok(()) } else { Err(format!("{what} holds {got} tiles, more than {max}")) }
    };
    for s in 0..4 {
        if let Some(h) = &sample.hands[s] {
            check(h.len(), size[s], &format!("hands[{s}]"))?;
            take(h, &format!("hands[{s}]"))?;
        }
    }
    let live_given = match (split, &sample.wall, &sample.queues) {
        (false, Some(w), None) => {
            most(w.len(), drawable, "wall")?;
            take(w, "wall")?;
            w.len()
        }
        (true, None, Some(q)) => {
            most(q.iter().map(Vec::len).sum(), drawable, "wall_queues (summed)")?;
            for x in q {
                take(x, "wall_queues")?;
            }
            q.iter().map(Vec::len).sum()
        }
        (_, None, None) => 0,
        (false, _, Some(_)) => return Err("wall_queues needs split_wall".into()),
        (true, Some(_), _) => return Err("a split wall takes wall_queues, not wall".into()),
    };
    let dead = [(&sample.rinshan, 4 - n_rinshan, "rinshan"), (&sample.dora, 5 - n_dora, "dora"), (&sample.ura, 5, "ura")];
    let mut dead_open = 0;
    for (x, max, what) in dead {
        let got = x.as_ref().map_or(0, Vec::len);
        if let Some(x) = x {
            most(got, max, what)?;
            take(x, what)?;
        }
        dead_open += max - got;
    }
    let undrawn_given = sample.undrawn.as_ref().map_or(0, Vec::len);
    if let Some(u) = &sample.undrawn {
        most(u.len(), n_rinshan, "undrawn")?;
        take(u, "undrawn")?;
    }
    let mut rest: Vec<Kind> = (0..N_KIND as u8).flat_map(|k| std::iter::repeat_n(k, pool[usize::from(k)] as usize)).collect();
    let open: usize = (0..4).filter(|&s| hidden[s] && sample.hands[s].is_none()).map(|s| size[s]).sum::<usize>()
        + (drawable - live_given)
        + dead_open;
    if rest.len() != open + n_rinshan - undrawn_given {
        return Err(format!("{} unseen tiles for {} hidden places", rest.len(), open + n_rinshan - undrawn_given));
    }
    if open > 0 && sample.seed.is_none() {
        return Err("the sample leaves tiles out but has no seed".into());
    }
    Rng(sample.seed.unwrap_or(0)).shuffle(&mut rest);
    let mut it = rest.into_iter();
    let mut next = |n: usize| -> Vec<Kind> { it.by_ref().take(n).collect() };
    for s in 0..4 {
        if hidden[s] {
            hands[s] = sample.hands[s].clone().unwrap_or_else(|| next(size[s]));
        }
    }
    let mut wall = sample.wall.clone().unwrap_or_default();
    let mut queues: [Vec<Kind>; 4] = sample.queues.clone().unwrap_or_default();
    let more = next(drawable - live_given);
    if split {
        // The tiles not given go round in the call-free order from the next drawer: the seat after the last mover
        // (the dealer at the start).
        let first = evs.iter().rev().find_map(Ev::actor).map_or(usize::from(start.oya), |a| (a + 1) % 4);
        for (k, t) in more.into_iter().enumerate() {
            queues[(first + k) % 4].push(t);
        }
    } else {
        wall.extend(more);
    }
    let mut fill = |given: &Option<Vec<Kind>>, max: usize| -> Vec<Kind> {
        let mut v = given.clone().unwrap_or_default();
        let n = max - v.len();
        v.extend(next(n));
        v
    };
    let rinshan = fill(&sample.rinshan, 4 - n_rinshan);
    let dora = fill(&sample.dora, 5 - n_dora);
    let ura = fill(&sample.ura, 5);
    let mut leftover = sample.undrawn.clone().unwrap_or_default();
    let n = n_rinshan - leftover.len();
    leftover.extend(next(n));
    // The hidden seats' start hands and draws.
    let mut start_hands: [Vec<Kind>; 4] = Default::default();
    for s in 0..4 {
        if hidden[s] {
            let (h, d) = reconstruct(s, evs, &hands[s])?;
            start_hands[s] = h;
            draws.extend(d);
        } else {
            start_hands[s] = start.tehais[s].iter().map(|k| k.unwrap()).collect();
        }
    }
    let shuffled = std::array::from_fn(|s| hidden[s] && sample.hands[s].is_none());
    Ok(Layout { start: start_hands, draws, wall, queues, rinshan, dora, ura, leftover, shuffled, now: hands, hidden })
}

/// Rebuild a position on the engine: lay out the kyoku's wall, deal it and replay the logged moves through `step`.
pub fn build(events: &[Value], sample: &Value, split: bool) -> Result<Built, String> {
    let (start, evs, src) = parse_events(events)?;
    let sample = parse_sample(sample)?;
    let lay = layout(&start, &evs, &sample, split)?;
    let oya = usize::from(start.oya);
    // The wall in deal order (`WallState::load_wall`): [0, 52) the deal, [52, 122) the live draws, [122, 136) the
    // dead wall from its far end (rinshan at 135..132, dora i at 131 - 2i, ura i at 130 - 2i).
    let mut w: Vec<Kind> = Vec::with_capacity(136);
    for round in 0..4 {
        for i in 0..4 {
            let p = (oya + i) % 4;
            let r = if round < 3 { &lay.start[p][round * 4..round * 4 + 4] } else { &lay.start[p][12..13] };
            w.extend_from_slice(r);
        }
    }
    let past_live: Vec<(usize, Kind)> = evs
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Ev::Tsumo { actor, rinshan: false, .. } => Some((*actor, lay.draws[&i])),
            _ => None,
        })
        .collect();
    let past_rinshan: Vec<Kind> = evs
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Ev::Tsumo { rinshan: true, .. }).then(|| lay.draws[&i]))
        .collect();
    // Split: per seat its own past draws then its queue; the dealer's first draw opens the live region.
    let mut seat_q: [Vec<Kind>; 4] = Default::default();
    if split {
        for &(a, k) in &past_live {
            seat_q[a].push(k);
        }
        for s in 0..4 {
            seat_q[s].extend_from_slice(&lay.queues[s]);
        }
        let first = *seat_q[oya].first().ok_or("the dealer has no first draw")?;
        w.push(first);
        for s in 0..4 {
            let skip = usize::from(s == oya);
            w.extend_from_slice(&seat_q[s][skip..]);
        }
    } else {
        w.extend(past_live.iter().map(|&(_, k)| k));
        w.extend_from_slice(&lay.wall);
    }
    w.extend_from_slice(&lay.leftover);
    if w.len() != 52 + LIVE {
        return Err(format!("the live wall came out with {} tiles", w.len() - 52));
    }
    let rinshan: Vec<Kind> = past_rinshan.into_iter().chain(lay.rinshan.iter().copied()).collect();
    let shown: Vec<Kind> = std::iter::once(start.dora)
        .chain(evs.iter().filter_map(|e| if let Ev::Dora { tile } = e { Some(*tile) } else { None }))
        .chain(lay.dora.iter().copied())
        .collect();
    let mut dead = [0 as Kind; 14];
    for i in 0..4 {
        dead[13 - i] = rinshan[i];
    }
    for i in 0..5 {
        dead[9 - 2 * i] = shown[i];
        dead[8 - 2 * i] = lay.ura[i];
    }
    w.extend_from_slice(&dead);
    // Physical ids, each kind's copies in wall order.
    let mut free: Vec<VecDeque<u8>> = (0..N_KIND as u8).map(|k| tids_of(k).into()).collect();
    let tids: Vec<u8> = w
        .iter()
        .map(|&k| free[usize::from(k)].pop_front().ok_or_else(|| format!("more than a full set of {}", kind_name(k))))
        .collect::<Result<_, _>>()?;
    let mut st = GameState::new(2, false, Some(0), 0, GameRule::default_tenhou());
    st.split_wall = split;
    st.defer_next_round = true;
    st._initialize_round(start.oya, start.round_wind, start.honba, start.kyotaku, Some(tids.clone()), Some(start.scores.to_vec()));
    if split {
        // Replace the round-robin queues with the seats' own (each popped from its end).
        let mut at = 53;
        let mut q: [Vec<u8>; 4] = Default::default();
        for s in 0..4 {
            let n = seat_q[s].len() - usize::from(s == oya);
            q[s] = tids[at..at + n].iter().rev().copied().collect();
            at += n;
        }
        st.wall.split_queues = Some(q);
    }
    let mut slot = GameSlot::from_state(st)?;
    replay(&mut slot, &evs, &src, &lay.draws, &lay.shuffled)?;
    let scores = std::array::from_fn(|i| slot.state.players[i].score);
    let names = |v: &[Kind]| v.iter().map(|&k| kind_name(k)).collect::<Vec<_>>();
    let mut out = Map::new();
    let hidden_hands: Map<String, Value> =
        (0..4).filter(|&s| lay.hidden[s]).map(|s| (s.to_string(), json!(names(&lay.now[s])))).collect();
    out.insert("hands".into(), Value::Object(hidden_hands));
    if split {
        out.insert("wall_queues".into(), json!(lay.queues.iter().map(|q| names(q)).collect::<Vec<_>>()));
    } else {
        out.insert("wall".into(), json!(names(&lay.wall)));
    }
    for (k, v) in [("rinshan", &lay.rinshan), ("dora", &lay.dora), ("ura", &lay.ura), ("undrawn", &lay.leftover)] {
        out.insert(k.into(), json!(names(v)));
    }
    let hands = json!(lay.now.iter().map(|h| names(h)).collect::<Vec<_>>());
    Ok(Built { slot, scores, sample: Value::Object(out), hands })
}

/// Pass every seat asked to respond, but `except` (its own action instead).
fn respond(slot: &mut GameSlot, except: Option<(u8, Action)>) -> Result<(), String> {
    let mut acts: HashMap<u8, Action> =
        slot.state.active_players.iter().map(|&p| (p, Action::new(ActionType::Pass, None, vec![], Some(p)))).collect();
    if let Some((p, a)) = except {
        if !acts.contains_key(&p) {
            return Err(format!("seat {p} is not asked to respond"));
        }
        acts.insert(p, a);
    }
    slot.step(&acts)
}

fn waiting(slot: &GameSlot) -> bool {
    slot.state.phase == Phase::WaitResponse && !slot.state.active_players.is_empty()
}

fn consumed_kinds(a: &Action) -> Vec<Kind> {
    let mut v: Vec<Kind> = a.consume_tiles.iter().map(|&t| kind_of_tid(t)).collect();
    v.sort_unstable();
    v
}

/// The logged moves after `start_kyoku`, through the engine's `step`.
fn replay(slot: &mut GameSlot, evs: &[Ev], src: &[usize], draws: &HashMap<usize, Kind>, shuffled: &[bool; 4])
          -> Result<(), String> {
    for (i, ev) in evs.iter().enumerate() {
        let at = |what: &str| format!("events[{}] ({what})", src[i]);
        match ev {
            Ev::Tsumo { actor, .. } => {
                while waiting(slot) {
                    respond(slot, None)?;
                }
                let got = slot.state.drawn_tile.map(kind_of_tid);
                if usize::from(slot.state.current_player) != *actor || got != Some(draws[&i]) {
                    return Err(format!("{}: the engine drew {:?} for seat {}", at("tsumo"), got.map(kind_name),
                                       slot.state.current_player));
                }
            }
            Ev::Dahai { actor, tile, tsumogiri } => {
                let st = &slot.state;
                let drawn = st.drawn_tile.filter(|&t| kind_of_tid(t) == *tile);
                let pick = if *tsumogiri {
                    drawn
                } else {
                    st.players[*actor].hand.iter().copied().find(|&t| kind_of_tid(t) == *tile && Some(t) != st.drawn_tile)
                };
                let t = pick.ok_or_else(|| at(&format!("seat {actor} cannot discard {} (tsumogiri {tsumogiri})", kind_name(*tile))))?;
                slot.step(&HashMap::from([(*actor as u8, Action::new(ActionType::Discard, Some(t), vec![], Some(*actor as u8)))]))
                    .map_err(|e| format!("{}: {e}", at("dahai")))?;
            }
            Ev::Reach { actor } => {
                let a = Action::new(ActionType::Riichi, None, vec![], Some(*actor as u8));
                slot.step(&HashMap::from([(*actor as u8, a)])).map_err(|e| {
                    let why = if shuffled[*actor] {
                        format!("seat {actor} declared riichi, but the shuffle gave it a hand that cannot (give its \
                                 hand in the sample)")
                    } else {
                        e
                    };
                    format!("{}: {why}", at("reach"))
                })?;
            }
            Ev::Call { actor, at: kind, tile, consumed } => {
                let mut want = consumed.clone();
                want.sort_unstable();
                let legal = slot.state._get_legal_actions_internal(*actor as u8);
                let a = legal
                    .into_iter()
                    .find(|a| a.action_type == *kind && a.tile.map(kind_of_tid) == Some(*tile) && consumed_kinds(a) == want)
                    .ok_or_else(|| at(&format!("seat {actor} cannot make this {kind:?}")))?;
                respond(slot, Some((*actor as u8, a))).map_err(|e| format!("{}: {e}", at("call")))?;
            }
            Ev::Ankan { actor, consumed } => {
                let t = type_of(*consumed.first().ok_or_else(|| at("ankan without tiles"))?);
                let legal = slot.state._get_legal_actions_internal(*actor as u8);
                let a = legal
                    .into_iter()
                    .find(|a| a.action_type == ActionType::Ankan
                        && a.tile.or(a.consume_tiles.first().copied()).is_some_and(|c| c / 4 == t))
                    .ok_or_else(|| at(&format!("seat {actor} cannot ankan {}", kind_name(t))))?;
                slot.step(&HashMap::from([(*actor as u8, a)])).map_err(|e| format!("{}: {e}", at("ankan")))?;
            }
            Ev::Kakan { actor, tile } => {
                let legal = slot.state._get_legal_actions_internal(*actor as u8);
                let a = legal
                    .into_iter()
                    .find(|a| a.action_type == ActionType::Kakan && a.tile.map(kind_of_tid) == Some(*tile))
                    .ok_or_else(|| at(&format!("seat {actor} cannot kakan {}", kind_name(*tile))))?;
                slot.step(&HashMap::from([(*actor as u8, a)])).map_err(|e| format!("{}: {e}", at("kakan")))?;
            }
            Ev::Dora { .. } => {}
        }
    }
    Ok(())
}

// --- actions in mjai form ---------------------------------------------------------------------------------------

/// An engine action in the spec's mjai form (`drawn`: the actor's drawn tile, for `tsumogiri`).
fn to_mjai(a: &Action, drawn: Option<u8>) -> Value {
    let names = |v: &[u8]| v.iter().map(|&t| kind_name(kind_of_tid(t))).collect::<Vec<_>>();
    match a.action_type {
        ActionType::Discard => json!({"type": "dahai", "pai": kind_name(kind_of_tid(a.tile.unwrap_or(0))),
                                      "tsumogiri": a.tile == drawn}),
        ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => json!({
            "type": match a.action_type { ActionType::Chi => "chi", ActionType::Pon => "pon", _ => "daiminkan" },
            "pai": kind_name(kind_of_tid(a.tile.unwrap_or(0))), "consumed": names(&a.consume_tiles)}),
        ActionType::Ankan => json!({"type": "ankan", "consumed": names(&a.consume_tiles)}),
        ActionType::Kakan => json!({"type": "kakan", "pai": kind_name(kind_of_tid(a.tile.unwrap_or(0))),
                                    "consumed": names(&a.consume_tiles)}),
        ActionType::Tsumo | ActionType::Ron => json!({"type": "hora"}),
        ActionType::Pass => json!({"type": "none"}),
        ActionType::KyushuKyuhai => json!({"type": "ryukyoku"}),
        ActionType::Riichi => json!({"type": "reach"}),
        ActionType::Kita => json!({"type": "kita"}),
    }
}

/// The seat's legal actions now, in mjai form, a riichi spelled out per declaring discard (on a fork of the slot).
fn legal_mjai(slot: &GameSlot, seat: u8) -> Result<Vec<Value>, String> {
    let legal = slot.state._get_legal_actions_internal(seat);
    let drawn = slot.state.drawn_tile;
    let mut out: Vec<Value> = Vec::new();
    for a in &legal {
        let m = if a.action_type == ActionType::Riichi {
            let mut f = slot.fork();
            f.step(&HashMap::from([(seat, a.clone())]))?;
            for d in f.state._get_legal_actions_internal(seat) {
                let mut m = to_mjai(&d, drawn);
                m["type"] = json!("reach");
                if !out.contains(&m) {
                    out.push(m);
                }
            }
            continue;
        } else {
            to_mjai(a, drawn)
        };
        if !out.contains(&m) {
            out.push(m);
        }
    }
    Ok(out)
}

/// The legal mjai action a forced action names (fields it leaves out match any; a discard without `tsumogiri` is
/// the tsumogiri one when the drawn tile can make it).
fn from_mjai(slot: &GameSlot, seat: u8, m: &Value) -> Result<Value, String> {
    let t = m.get("type").and_then(Value::as_str).ok_or("an action without type")?;
    let legal = legal_mjai(slot, seat)?;
    let same = |l: &Value| {
        let keys = ["type", "pai", "consumed"];
        keys.iter().all(|k| m.get(*k).is_none() || l.get(*k) == m.get(*k) || (*k == "consumed" && sorted(l.get(*k)) == sorted(m.get(*k))))
            && m.get("tsumogiri").is_none_or(|x| l.get("tsumogiri") == Some(x))
    };
    let hits: Vec<&Value> = legal.iter().filter(|l| same(l)).collect();
    let pick = hits.iter().find(|l| l.get("tsumogiri") == Some(&json!(true))).or(hits.first());
    match pick {
        Some(l) => Ok((*l).clone()),
        None => Err(format!("{t} {m} is not legal here (legal: {})", Value::Array(legal))),
    }
}

fn sorted(v: Option<&Value>) -> Option<Vec<String>> {
    let mut s: Vec<String> = v?.as_array()?.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
    s.sort();
    Some(s)
}

/// The engine action of a legal mjai action of `seat` (one engine step; a reach's riichi first).
fn engine_action(slot: &GameSlot, seat: u8, m: &Value, riichi_step: bool) -> Result<Action, String> {
    let legal = slot.state._get_legal_actions_internal(seat);
    let drawn = slot.state.drawn_tile;
    if riichi_step {
        return legal.into_iter().find(|a| a.action_type == ActionType::Riichi).ok_or_else(|| "riichi is not legal".into());
    }
    let mut want = m.clone();
    if want["type"] == json!("reach") {
        want["type"] = json!("dahai");
    }
    legal
        .into_iter()
        .find(|a| {
            let l = to_mjai(a, drawn);
            l["type"] == want["type"] && l.get("pai") == want.get("pai") && sorted(l.get("consumed")) == sorted(want.get("consumed"))
                && l.get("tsumogiri") == want.get("tsumogiri")
        })
        .ok_or_else(|| format!("{m} has no engine action"))
}

// --- the entries ------------------------------------------------------------------------------------------------

/// A position's fields (`docs/play_out_spec.md` section 3).
fn position_parts(p: &Value) -> Result<(Vec<Value>, Option<u8>), String> {
    let events: Vec<Value> = p
        .get("events")
        .and_then(Value::as_array)
        .ok_or("a position needs events")?
        .iter()
        .map(|e| match e {
            Value::String(s) => serde_json::from_str(s).map_err(|e| format!("bad event line: {e}")),
            _ => Ok(e.clone()),
        })
        .collect::<Result<_, _>>()?;
    let seat = match p.get("seat") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().filter(|&s| s < 4).ok_or("seat must be 0..3")? as u8),
    };
    Ok((events, seat))
}

/// The sample `legal_actions` / `seat_options` rebuild on: the position's first sample if any, the seat's `hand`
/// over it if given, every other hidden tile from its seed (0 if none). The seat's actions depend on its own hand
/// and the public tiles only.
fn probe_sample(p: &Value, seat: u8) -> Value {
    let mut s = p.get("samples").and_then(Value::as_array).and_then(|v| v.first()).and_then(Value::as_object).cloned()
        .unwrap_or_default();
    if let Some(h) = p.get("hand") {
        let mut hands = Map::new();
        if let Some(Value::Object(m)) = s.get("hands") {
            hands = m.clone();
        } else if let Some(Value::Array(a)) = s.get("hands") {
            for (i, x) in a.iter().enumerate().filter(|(_, x)| !x.is_null()) {
                hands.insert(i.to_string(), x.clone());
            }
        }
        hands.insert(seat.to_string(), h.clone());
        s.insert("hands".into(), Value::Object(hands));
    }
    s.entry("seed").or_insert(json!(0));
    Value::Object(s)
}

fn seat_to_act(slot: &GameSlot, seat: u8) -> Result<(), String> {
    if slot.actors().contains(&seat) { Ok(()) } else { Err(format!("seat {seat} does not decide at this position")) }
}

/// `legal_actions(position)`.
pub fn legal_actions(p: &Value) -> Result<Vec<Value>, String> {
    let (events, seat) = position_parts(p)?;
    let seat = seat.ok_or("legal_actions needs the seat")?;
    let b = build(&events, &probe_sample(p, seat), false)?;
    seat_to_act(&b.slot, seat)?;
    legal_mjai(&b.slot, seat)
}

/// `seat_options` for one position: `{"legal": ..., "choice": ...}`.
pub fn seat_option(p: &Value, bot: &Bot) -> Result<Value, String> {
    let (events, seat) = position_parts(p)?;
    let seat = seat.ok_or("seat_options needs the seat")?;
    let mut b = build(&events, &probe_sample(p, seat), false)?;
    seat_to_act(&b.slot, seat)?;
    let legal = legal_mjai(&b.slot, seat)?;
    let drawn = b.slot.state.drawn_tile;
    let a = b.slot.bot_action(seat, bot)?;
    let choice = if a.action_type == ActionType::Riichi {
        b.slot.step(&HashMap::from([(seat, a)]))?;
        let d = b.slot.bot_action(seat, bot)?;
        let mut m = to_mjai(&d, drawn);
        m["type"] = json!("reach");
        m
    } else {
        to_mjai(&a, drawn)
    };
    Ok(json!({"legal": legal, "choice": choice}))
}

/// One play-out from a rebuilt position: `seat`'s next decisions forced by `forced` (mjai actions), the rest by
/// `bots`, until the kyoku ends.
fn roll(b: &Built, seat: Option<u8>, forced: &[Value], bots: &[Bot; 4], record: bool, samples: bool)
        -> Result<Value, String> {
    let mut slot = b.slot.fork();
    let k0 = slot.kyoku_events().len();
    let mut queue: VecDeque<&Value> = forced.iter().collect();
    // A forced reach between its riichi and its discard.
    let mut reach_discard: Option<Value> = None;
    let (mut n_forced, mut n_decisions) = (0usize, 0usize);
    while !slot.state.needs_initialize_next_round && !slot.state.is_done {
        let actors = slot.actors();
        if actors.is_empty() {
            return Err("the engine stopped before the kyoku's end".into());
        }
        let mut acts = HashMap::new();
        for a in actors {
            let act = if Some(a) == seat && reach_discard.is_some() {
                let m = reach_discard.take().unwrap();
                n_forced += 1;
                engine_action(&slot, a, &m, false)?
            } else if Some(a) == seat && !queue.is_empty() {
                let m = from_mjai(&slot, a, queue.pop_front().unwrap())
                    .map_err(|e| format!("forced action {}: {e}", n_forced + 1))?;
                if m["type"] == json!("reach") {
                    reach_discard = Some(m.clone());
                    engine_action(&slot, a, &m, true)?
                } else {
                    n_forced += 1;
                    engine_action(&slot, a, &m, false)?
                }
            } else {
                n_decisions += 1;
                slot.bot_action(a, &bots[usize::from(a)])?
            };
            acts.insert(a, act);
        }
        slot.step(&acts)?;
    }
    let st = &slot.state;
    let scores: Vec<i32> = (0..4).map(|i| st.players[i].score).collect();
    let delta: Vec<i32> = (0..4).map(|i| scores[i] - b.scores[i]).collect();
    let tail = &slot.kyoku_events()[k0..];
    let horas: Vec<Value> = tail
        .iter()
        .filter(|e| e["type"] == "hora")
        .map(|e| {
            let mut h = Map::new();
            for k in ["actor", "target", "han", "fu", "yaku", "yakuman", "deltas", "ura_markers"] {
                if let Some(v) = e.get(k) {
                    h.insert(k.into(), v.clone());
                }
            }
            Value::Object(h)
        })
        .collect();
    let ryukyoku = tail.iter().find(|e| e["type"] == "ryukyoku").map(|e| {
        let reason = e.get("reason").cloned().unwrap_or(Value::Null);
        let tenpai: Value = if reason == "exhaustive_draw" {
            json!((0..4).map(|i| HandEvaluator::new(st.players[i].hand.clone(), st.players[i].melds.clone()).is_tenpai())
                .collect::<Vec<_>>())
        } else {
            Value::Null
        };
        json!({"reason": reason, "deltas": e.get("deltas").cloned().unwrap_or(Value::Null), "tenpai": tenpai})
    });
    let draw = st.pending_is_draw;
    let mut out = json!({
        "end": if horas.is_empty() { "ryukyoku" } else { "hora" },
        "delta": delta,
        "scores": scores,
        "kyotaku": st.riichi_sticks,
        "honba": if st.pending_oya_won || draw { u32::from(st.honba) + 1 } else { 0 },
        "renchan": st.pending_oya_won,
        "hora": horas,
        "ryukyoku": ryukyoku,
        "forced": n_forced,
        "n_decisions": n_decisions,
    });
    if record {
        // A rinshan draw is marked: it follows its kan by the same seat (dora events aside).
        let all = slot.kyoku_events();
        let events: Vec<Value> = (k0..all.len())
            .map(|i| {
                let mut e = all[i].clone();
                if e["type"] == "tsumo" {
                    let prev = all[..i].iter().rev().find(|x| x["type"] != "dora");
                    if prev.is_some_and(|x| matches!(x["type"].as_str(), Some("ankan" | "kakan" | "daiminkan"))
                        && x["actor"] == e["actor"])
                    {
                        e["rinshan"] = json!(true);
                    }
                }
                e
            })
            .collect();
        out["events"] = Value::Array(events);
    }
    if samples {
        out["sample"] = b.sample.clone();
        out["hands"] = b.hands.clone();
    }
    Ok(out)
}

/// The forced action lists of a position: each entry an mjai dict (one action) or a list of them.
fn action_lists(p: &Value) -> Result<Vec<Vec<Value>>, String> {
    let Some(acts) = p.get("actions") else { return Ok(vec![Vec::new()]) };
    acts.as_array()
        .ok_or("actions must be a list")?
        .iter()
        .map(|a| match a {
            Value::Array(v) => Ok(v.clone()),
            Value::Object(_) => Ok(vec![a.clone()]),
            _ => Err("an action must be an mjai dict or a list of them".to_string()),
        })
        .collect()
}

/// `play_out` for one (position, sample): one result per forced action list (an error string per failure).
pub fn play_out_sample(p: &Value, sample: &Value, bots: &[Bot; 4], split: bool, record: bool, samples: bool)
        -> Vec<Result<Value, String>> {
    let parts = (|| {
        let (events, seat) = position_parts(p)?;
        let lists = action_lists(p)?;
        if seat.is_none() && lists.iter().any(|l| !l.is_empty()) {
            return Err("forced actions need the seat".to_string());
        }
        Ok((events, seat, lists))
    })();
    let (events, seat, lists) = match parts {
        Ok(x) => x,
        Err(e) => return vec![Err(e)],
    };
    match build(&events, sample, split) {
        Ok(b) => lists.iter().map(|l| roll(&b, seat, l, bots, record, samples)).collect(),
        Err(e) => lists.iter().map(|_| Err(e.clone())).collect(),
    }
}
