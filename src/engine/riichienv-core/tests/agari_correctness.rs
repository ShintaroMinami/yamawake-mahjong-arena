//! Correctness tests for agari benchmark data.
//!
//! Verifies that every benchmark case produces the expected han, fu, yaku,
//! and score values. This catches regressions in the scoring engine and
//! ensures platform-specific corner cases (composite yakuman, kazoe yakuman)
//! are handled correctly.

use riichienv_core::agari;
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::hand_evaluator_3p::HandEvaluator3P;
use riichienv_core::score;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};
use serde::Deserialize;
use std::fs;

// --- Deserialization types (mirrored from bench) ---

#[derive(Deserialize)]
struct BenchData {
    cases: Vec<AgariCase>,
}

#[derive(Deserialize)]
struct NegativeData {
    cases: Vec<NegativeCase>,
}

#[derive(Deserialize)]
struct AgariCase {
    tiles_136: Vec<u8>,
    melds: Vec<BenchMeld>,
    win_tile_136: u8,
    dora_indicators: Vec<u8>,
    ura_indicators: Vec<u8>,
    conditions: BenchConditions,
    expected: BenchExpected,
}

#[derive(Deserialize)]
struct BenchMeld {
    meld_type: String,
    tiles: Vec<u8>,
    opened: bool,
    from_who: i8,
}

#[derive(Deserialize)]
struct BenchConditions {
    tsumo: bool,
    riichi: bool,
    double_riichi: bool,
    ippatsu: bool,
    haitei: bool,
    houtei: bool,
    rinshan: bool,
    chankan: bool,
    tsumo_first_turn: bool,
    player_wind: u8,
    round_wind: u8,
    honba: u32,
    #[serde(default)]
    kita_count: u8,
    #[serde(default)]
    is_sanma: bool,
    #[serde(default = "default_num_players")]
    num_players: u8,
}

fn default_num_players() -> u8 {
    4
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct BenchExpected {
    is_win: bool,
    han: u32,
    fu: u32,
    yaku: Vec<u32>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct NegativeCase {
    counts_34: Vec<u8>,
    is_tenpai: bool,
}

// --- Helpers ---

fn parse_meld_type(s: &str) -> MeldType {
    match s {
        "chi" => MeldType::Chi,
        "pon" => MeldType::Pon,
        "daiminkan" => MeldType::Daiminkan,
        "ankan" => MeldType::Ankan,
        "kakan" => MeldType::Kakan,
        _ => panic!("Unknown meld type: {}", s),
    }
}

fn to_meld(bm: &BenchMeld) -> Meld {
    Meld::new(
        parse_meld_type(&bm.meld_type),
        bm.tiles.clone(),
        bm.opened,
        bm.from_who,
        None,
    )
}

fn to_conditions(bc: &BenchConditions) -> Conditions {
    Conditions {
        tsumo: bc.tsumo,
        riichi: bc.riichi,
        double_riichi: bc.double_riichi,
        ippatsu: bc.ippatsu,
        haitei: bc.haitei,
        houtei: bc.houtei,
        rinshan: bc.rinshan,
        chankan: bc.chankan,
        tsumo_first_turn: bc.tsumo_first_turn,
        player_wind: Wind::from(bc.player_wind),
        round_wind: Wind::from(bc.round_wind),
        riichi_sticks: 0,
        honba: bc.honba,
        kita_count: bc.kita_count,
        is_sanma: bc.is_sanma,
        num_players: bc.num_players,
    }
}

// --- Tests ---

/// Verify every 4P case: is_agari, han, fu, and yaku all match expected values.
#[test]
fn test_4p_calc_correctness() {
    let data: BenchData = {
        let s =
            fs::read_to_string("benches/data/agari_4p.json").expect("Failed to read agari_4p.json");
        serde_json::from_str(&s).expect("Failed to parse agari_4p.json")
    };
    assert!(!data.cases.is_empty(), "4P data should not be empty");

    for (i, case) in data.cases.iter().enumerate() {
        let melds: Vec<Meld> = case.melds.iter().map(to_meld).collect();
        let he = HandEvaluator::new(case.tiles_136.clone(), melds);

        // Verify is_agari
        let mut hand = he.hand.clone();
        let win_tile_34 = case.win_tile_136 / 4;
        let total: u8 = hand.counts.iter().sum::<u8>() + (he.melds.len() as u8 * 3);
        if total == 13 {
            hand.add(win_tile_34);
        }
        assert!(
            agari::is_agari(&mut hand),
            "4P case {}: is_agari should be true",
            i,
        );

        // Verify full calc: han, fu, yaku
        let conds = to_conditions(&case.conditions);
        let result = he.calc(
            case.win_tile_136,
            case.dora_indicators.clone(),
            case.ura_indicators.clone(),
            Some(conds),
        );

        assert!(
            result.is_win,
            "4P case {}: calc should return is_win=true",
            i,
        );
        assert_eq!(
            result.han, case.expected.han,
            "4P case {}: han mismatch (tiles={:?}, win={}, got_yaku={:?} expected_yaku={:?})",
            i, case.tiles_136, case.win_tile_136, result.yaku, case.expected.yaku,
        );
        assert_eq!(
            result.fu, case.expected.fu,
            "4P case {}: fu mismatch (tiles={:?}, win={})",
            i, case.tiles_136, case.win_tile_136,
        );

        let mut result_yaku = result.yaku.clone();
        let mut expected_yaku = case.expected.yaku.clone();
        result_yaku.sort();
        expected_yaku.sort();
        assert_eq!(
            result_yaku, expected_yaku,
            "4P case {}: yaku mismatch (tiles={:?}, win={})",
            i, case.tiles_136, case.win_tile_136,
        );
    }
}

/// Verify every 3P case: is_agari, han, fu, and yaku all match expected values.
#[test]
fn test_3p_calc_correctness() {
    let data: BenchData = {
        let s =
            fs::read_to_string("benches/data/agari_3p.json").expect("Failed to read agari_3p.json");
        serde_json::from_str(&s).expect("Failed to parse agari_3p.json")
    };
    assert!(!data.cases.is_empty(), "3P data should not be empty");

    for (i, case) in data.cases.iter().enumerate() {
        let melds: Vec<Meld> = case.melds.iter().map(to_meld).collect();
        let he = HandEvaluator3P::new(case.tiles_136.clone(), melds);

        let mut hand = he.hand.clone();
        let win_tile_34 = case.win_tile_136 / 4;
        let total: u8 = hand.counts.iter().sum::<u8>() + (he.melds.len() as u8 * 3);
        if total == 13 {
            hand.add(win_tile_34);
        }
        assert!(
            agari::is_agari(&mut hand),
            "3P case {}: is_agari should be true",
            i,
        );

        let conds = to_conditions(&case.conditions);
        let result = he.calc(
            case.win_tile_136,
            case.dora_indicators.clone(),
            case.ura_indicators.clone(),
            Some(conds),
        );

        assert!(
            result.is_win,
            "3P case {}: calc should return is_win=true",
            i,
        );
        assert_eq!(
            result.han, case.expected.han,
            "3P case {}: han mismatch (tiles={:?}, win={}, got_yaku={:?} expected_yaku={:?})",
            i, case.tiles_136, case.win_tile_136, result.yaku, case.expected.yaku,
        );
        assert_eq!(
            result.fu, case.expected.fu,
            "3P case {}: fu mismatch (tiles={:?}, win={})",
            i, case.tiles_136, case.win_tile_136,
        );

        let mut result_yaku = result.yaku.clone();
        let mut expected_yaku = case.expected.yaku.clone();
        result_yaku.sort();
        expected_yaku.sort();
        assert_eq!(
            result_yaku, expected_yaku,
            "3P case {}: yaku mismatch (tiles={:?}, win={})",
            i, case.tiles_136, case.win_tile_136,
        );
    }
}

/// Verify negative hands have correct tile count.
#[test]
fn test_negative_data_integrity() {
    let data: NegativeData = {
        let s = fs::read_to_string("benches/data/hands_negative.json")
            .expect("Failed to read hands_negative.json");
        serde_json::from_str(&s).expect("Failed to parse hands_negative.json")
    };
    assert!(!data.cases.is_empty(), "Negative data should not be empty");

    for (i, case) in data.cases.iter().enumerate() {
        let total: u8 = case.counts_34.iter().sum();
        assert!(
            total == 13 || total == 14,
            "Negative case {} should have 13 or 14 tiles, got {}",
            i,
            total,
        );
    }
}

/// Verify calculate_score produces correct values for all scoring tiers,
/// including double/triple yakuman (MjSoul platform corner cases).
#[test]
fn test_calculate_score_correctness() {
    // (han, fu, is_oya, is_tsumo, honba, num_players, expected_pay_ron, expected_tsumo_oya, expected_tsumo_ko)
    type ScoreCase = (u8, u8, bool, bool, u32, u8, u32, u32, u32);
    let cases: Vec<ScoreCase> = vec![
        // Basic scoring tiers
        (1, 30, false, false, 0, 4, 1000, 0, 0),
        (1, 30, false, true, 0, 4, 0, 500, 300),
        (3, 30, true, false, 0, 4, 5800, 0, 0),
        // Mangan
        (5, 0, false, false, 0, 4, 8000, 0, 0),
        (5, 0, true, true, 0, 4, 0, 0, 4000),
        (5, 0, false, true, 0, 4, 0, 4000, 2000),
        // Haneman / Baiman / Sanbaiman
        (6, 0, false, false, 0, 4, 12000, 0, 0),
        (8, 0, false, false, 0, 4, 16000, 0, 0),
        (11, 0, false, false, 0, 4, 24000, 0, 0),
        // Single yakuman
        (13, 0, false, false, 0, 4, 32000, 0, 0),
        (13, 0, true, false, 0, 4, 48000, 0, 0),
        (13, 0, false, true, 0, 4, 0, 16000, 8000),
        (13, 0, true, true, 0, 4, 0, 0, 16000),
        // Double yakuman (MjSoul: kokushi_13, suanko_tanki, junsei_chuuren, daisuushii)
        (26, 0, false, false, 0, 4, 64000, 0, 0),
        (26, 0, true, false, 0, 4, 96000, 0, 0),
        (26, 0, false, true, 0, 4, 0, 32000, 16000),
        (26, 0, true, true, 0, 4, 0, 0, 32000),
        // Double yakuman with honba
        (26, 0, false, false, 2, 4, 64600, 0, 0),
        (26, 0, true, true, 2, 4, 0, 200, 32200),
        // Triple yakuman (e.g. chinrouto + suanko_tanki)
        (39, 0, false, false, 0, 4, 96000, 0, 0),
        (39, 0, true, true, 0, 4, 0, 0, 48000),
        // Quadruple yakuman (e.g. suanko + tsuiiso + daisuushii)
        (52, 0, false, false, 0, 4, 128000, 0, 0),
        // Quintuple yakuman (e.g. suanko_tanki + tsuiiso + daisuushii)
        (65, 0, false, false, 0, 4, 160000, 0, 0),
        // 3-player scoring
        (13, 0, false, false, 0, 3, 32000, 0, 0),
        (13, 0, false, true, 0, 3, 0, 16000, 8000),
        (26, 0, false, false, 0, 3, 64000, 0, 0),
        (26, 0, true, true, 0, 3, 0, 0, 32000),
    ];

    for (i, &(han, fu, is_oya, is_tsumo, honba, num_players, exp_ron, exp_oya, exp_ko)) in
        cases.iter().enumerate()
    {
        let result = score::calculate_score(han, fu, is_oya, is_tsumo, honba, num_players);
        assert_eq!(
            result.pay_ron, exp_ron,
            "Score case {}: pay_ron mismatch (han={}, fu={}, oya={}, tsumo={}, honba={}, np={})",
            i, han, fu, is_oya, is_tsumo, honba, num_players,
        );
        assert_eq!(
            result.pay_tsumo_oya, exp_oya,
            "Score case {}: pay_tsumo_oya mismatch (han={}, fu={}, oya={}, tsumo={}, honba={}, np={})",
            i, han, fu, is_oya, is_tsumo, honba, num_players,
        );
        assert_eq!(
            result.pay_tsumo_ko, exp_ko,
            "Score case {}: pay_tsumo_ko mismatch (han={}, fu={}, oya={}, tsumo={}, honba={}, np={})",
            i, han, fu, is_oya, is_tsumo, honba, num_players,
        );
    }
}
