"""Play statistics from mjai logs, the match's stats and time, and the report."""

import json

import numpy as np
import pytest

from yamawake import RulebotAgent, TsumogiriAgent, lineup, run_match
from yamawake.playstats import play_counts, ratio
from yamawake.report import report


def _log(*events: dict) -> list[str]:
    return [json.dumps(e) for e in events]


def test_win_points_are_the_kyoku_delta_and_a_double_ron_deals_in_once():
    log = _log(
        {"type": "start_kyoku", "honba": 2, "kyotaku": 1, "oya": 0},
        {"type": "reach", "actor": 1},
        {"type": "reach_accepted", "actor": 1},
        {"type": "pon", "actor": 2},
        {"type": "pon", "actor": 2},
        {"type": "hora", "actor": 1, "target": 3, "deltas": [0, 8000 + 600 + 2000, 0, -8600]},
        {"type": "hora", "actor": 2, "target": 3, "deltas": [0, 0, 3900, -3900]},
    )
    c = play_counts([log])
    assert c["kyoku"][0].tolist() == [1, 1, 1, 1]
    assert c["win"][0].tolist() == [0, 1, 1, 0]
    assert c["win_pts"][0].tolist() == [0, 8000 + 600 + 2000 - 1000, 3900, 0]
    assert c["deal_in"][0].tolist() == [0, 0, 0, 1]           # one kyoku, two rons
    assert c["deal_pts"][0].tolist() == [0, 0, 0, 8600 + 3900]
    assert c["call"][0].tolist() == [0, 0, 1, 0]              # two pons, one kyoku
    assert c["riichi"][0].tolist() == [0, 1, 0, 0]


def test_a_riichi_dealt_in_on_its_tile_still_counts_and_tsumo_has_no_deal_in():
    log = _log(
        {"type": "start_kyoku", "honba": 0, "kyotaku": 0, "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "hora", "actor": 1, "target": 0, "deltas": [-2000, 2000, 0, 0]},
        {"type": "start_kyoku", "honba": 1, "kyotaku": 0, "oya": 1},
        {"type": "hora", "actor": 2, "target": 2, "deltas": [-1100, -2100, 4300, -1100]},
    )
    c = play_counts([log])
    assert c["riichi"][0].tolist() == [1, 0, 0, 0]
    assert c["deal_in"][0].tolist() == [1, 0, 0, 0]
    assert c["win_pts"][0].tolist() == [0, 2000, 4300, 0]



def test_an_accepted_riichi_that_deals_in_counts_the_payment_only():
    log = _log(
        {"type": "start_kyoku", "honba": 0, "kyotaku": 0, "oya": 0},
        {"type": "reach", "actor": 2},
        {"type": "reach_accepted", "actor": 2},
        {"type": "hora", "actor": 0, "target": 2, "deltas": [3900 + 1000, 0, -3900, 0]},
    )
    c = play_counts([log])
    assert c["win_pts"][0].tolist() == [4900, 0, 0, 0]
    assert c["deal_pts"][0].tolist() == [0, 0, 3900, 0]


def test_ratio_pools_and_its_se_vanishes_for_a_constant_ratio():
    assert ratio(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == (0.5, 0.0)
    assert ratio(np.zeros(3), np.zeros(3)) == (0.0, 0.0)


def test_match_stats_time_and_report():
    r = run_match(lineup(TsumogiriAgent(), RulebotAgent(), "2v2"), walls=4, play_stats=True)
    stats = r.play_stats()
    assert [s["name"] for s in stats] == ["tsumogiri", "rulebot"]
    assert stats[0]["win"] == 0.0 and stats[0]["call"] == 0.0 and stats[0]["riichi"] == 0.0
    assert stats[1]["adj_eff"] == pytest.approx(
        stats[1]["win"] * stats[1]["avg_win"] - stats[1]["deal_in"] * stats[1]["avg_deal"])
    times = r.time_stats()
    assert [t["name"] for t in times] == ["tsumogiri", "rulebot", "arena"]
    assert all(t["decisions"] > 0 for t in times[:2]) and times[2]["seconds"] > 0
    text = report(r)
    for section in ("Standings", "Play per kyoku", "adj. eff", "Time", "paired over 4 walls"):
        assert section in text
    assert "play_stats" in r.to_json() and "time_stats" in r.to_json()
