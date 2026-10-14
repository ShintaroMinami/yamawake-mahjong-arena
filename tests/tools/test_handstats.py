"""handstats: tenpai turn, value at tenpai and outcomes from mjai logs."""
from __future__ import annotations

import gzip
import json

import pytest

from yamawake import RulebotAgent, run_match
from yamawake.handstats import (GAME_METRICS, collect, collect_all, compare, main, pair,
                                 pair_games, parse_game, summarize, summarize_games,
                                 tenpai_value)

FILLER = ["1p", "1p", "2p", "3p", "7p", "8p", "9p", "W", "W", "S", "S", "C", "C"]


def _kyoku(kyoku, honba, oya, dora, tehai0, body):
    head = {"type": "start_kyoku", "bakaze": "E", "kyoku": kyoku, "honba": honba,
            "kyotaku": 0, "oya": oya, "dora_marker": dora,
            "scores": [25000] * 4, "tehais": [tehai0, FILLER, FILLER, FILLER]}
    return [head, *body, {"type": "end_kyoku"}]


def _tsumo(a, p):
    return {"type": "tsumo", "actor": a, "pai": p}


def _dahai(a, p, tsumogiri=False):
    return {"type": "dahai", "actor": a, "pai": p, "tsumogiri": tsumogiri}


CRAFTED = [
    {"type": "start_game", "names": ["me", "rb", "rb", "rb"]},
    # E1: 1-shanten deal; riichi tenpai on the 2nd draw (waits 4m-7m, 4m is
    # dora and one 4m is out), then ron on 7m.
    *_kyoku(1, 0, 0, "3m", ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                            "2s", "2s", "5m", "9p"], [
        _tsumo(0, "N"), _dahai(0, "N", True),
        _tsumo(1, "F"), _dahai(1, "4m"),
        _tsumo(2, "F"), _dahai(2, "F", True),
        _tsumo(3, "F"), _dahai(3, "F", True),
        _tsumo(0, "6m"), {"type": "reach", "actor": 0}, _dahai(0, "9p"),
        {"type": "reach_accepted", "actor": 0},
        _tsumo(1, "7m"), _dahai(1, "7m", True),
        {"type": "hora", "actor": 0, "target": 1, "deltas": [3900, -2900, 0, 0],
         "ura_markers": []}]),
    # E1-1: yakuless dama tenpai on the deal; deals in on the first discard.
    *_kyoku(1, 1, 0, "N", ["1m", "2m", "3m", "4p", "5p", "6p", "1s", "1s", "1s",
                           "9s", "9s", "5m", "6m"], [
        _tsumo(0, "W"), _dahai(0, "W", True),
        {"type": "hora", "actor": 2, "target": 0, "deltas": [-1300, 0, 1300, 0],
         "ura_markers": []}]),
    # E2: pon of haku before any draw, tenpai at turn 0 (open), exhaustive draw.
    *_kyoku(2, 0, 1, "N", ["1m", "1m", "5p", "6p", "7p", "2s", "3s", "4s", "P", "P",
                           "9m", "9m", "E"], [
        _tsumo(1, "C"), _dahai(1, "P"),
        {"type": "pon", "actor": 0, "target": 1, "pai": "P", "consumed": ["P", "P"]},
        _dahai(0, "E"),
        {"type": "ryukyoku", "reason": "exhaustive_draw",
         "deltas": [3000, -1000, -1000, -1000]}]),
    {"type": "end_game"},
]


@pytest.fixture
def crafted(tmp_path):
    path = tmp_path / "arena-1-000000.json.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(json.dumps(e) for e in CRAFTED) + "\n")
    return path


def test_crafted_riichi_tenpai(crafted):
    k1, k2, k3 = parse_game(crafted, "me")
    assert (k1.tenpai_turn, k1.riichi_at_tenpai, k1.closed_at_tenpai) == (2, True, True)
    # 4m: riichi pinfu dora, 3 han 30 fu oya ron 5800 / tsumo 4 han 20 fu 2600 all;
    # 7m: riichi pinfu, 2900 / 1300 all. Weights 3 (one 4m out) and 4.
    assert k1.ev_ron == pytest.approx((3 * 5800 + 4 * 2900) / 7)
    assert k1.ev_tsumo == pytest.approx((3 * 7800 + 4 * 3900) / 7)
    assert not k1.yakuless and not k1.dead_waits
    assert (k1.won, k1.win_ron, k1.win_tsumo, k1.deal_in) == (True, 2900, None, None)
    assert k1.riichi and not k1.called and k1.delta == 2900


def test_crafted_yakuless_and_deal_in(crafted):
    k2 = parse_game(crafted, "me")[1]
    assert (k2.honba, k2.tenpai_turn, k2.riichi_at_tenpai) == (1, 0, False)
    # No yaku by ron; menzen tsumo 1 han 30 fu oya = 500 all.
    assert k2.yakuless and k2.ev_ron == 0 and k2.ev_tsumo == pytest.approx(1500)
    assert (k2.won, k2.deal_in, k2.delta) == (False, 1300, -1300)


def test_crafted_call_and_ryukyoku(crafted):
    k3 = parse_game(crafted, "me")[2]
    assert (k3.tenpai_turn, k3.closed_at_tenpai, k3.called) == (0, False, True)
    # Haku shanpon 1m/9m as a non-oya: 1 han 30 fu ron 1000, 40 fu tsumo 400/700.
    assert k3.ev_ron == pytest.approx(1000) and k3.ev_tsumo == pytest.approx(1500)
    assert k3.exhaustive and k3.ryukyoku_tenpai and k3.delta == 3000


def test_crafted_summary(crafted):
    s = summarize(collect(crafted.parent, "me"))
    assert s["kyoku"] == 3 and s["tenpai_rate"]["mean"] == 1.0
    assert s["tenpai_turn"]["mean"] == pytest.approx(2 / 3)
    assert s["tenpai_curve"][0] == pytest.approx(2 / 3) and s["tenpai_curve"][1] == 1.0
    assert s["win_rate"]["mean"] == pytest.approx(1 / 3)
    assert s["yakuless"]["mean"] == pytest.approx(1 / 3)
    assert parse_game(crafted, "nobody") == []


def test_crafted_riichi_wins_and_deal_ins(crafted):
    k1, k2, k3 = parse_game(crafted, "me")
    # E1: first riichi on the 2nd draw, waits 4m (one out) / 7m: 7 unseen, good
    # shape; ron on the very next discard (ippatsu, no ura indicator).
    assert (k1.riichi_first, k1.riichi_turn, k1.riichi_waits, k1.riichi_good) == (True, 2, 7, True)
    assert (k1.win_turn, k1.ippatsu, k1.ura, k1.oya, k1.renchan) == (2, True, False, True, True)
    assert "Riichi" in k1.win_yaku and "Pinfu" in k1.win_yaku and k1.win_limit == "below_mangan"
    # E1-1: deals in from a closed (dama) tenpai to a closed, unriichi'd winner.
    assert (k2.deal_in_to, k2.deal_in_as, k2.renchan, k2.dama) == ("dama", "closed", False, True)
    # E2: one call, not the oya.
    assert (k3.n_calls, k3.oya, k3.renchan, k3.dama) == (1, False, None, None)
    _, games = collect_all(crafted.parent, "me")
    assert [(g.rank, g.score) for g in games] == [(1, 28000)]


def test_dead_waits():
    visible = [0] * 34
    visible[3] = visible[6] = 4                      # every 4m and 7m seen
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2s", "2s", "5m", "6m"]
    ron, tsumo, yakuless, dead = tenpai_value(hand, [], visible, ["N"], 1, 0, True)
    assert (ron, tsumo, dead) == (None, None, True) and not yakuless


@pytest.fixture(scope="module")
def rulebot_logs(tmp_path_factory):
    out = tmp_path_factory.mktemp("rb")
    rb = RulebotAgent()
    run_match([rb] * 4, walls=2, flight=8, record_logs=out)
    return out


def test_real_run_is_sane(rulebot_logs):
    ks = collect(rulebot_logs, "rulebot")
    assert len(ks) >= 8 * 4 * 4                       # 8 hanchan, >= 4 kyoku, 4 seats
    assert all(k.tenpai_turn is not None for k in ks if k.won)
    s = summarize(ks)
    assert 0.2 < s["tenpai_rate"]["mean"] < 0.9
    assert 3 < s["tenpai_turn"]["mean"] < 15
    assert 1000 < s["ev_ron"]["mean"] < 15000
    assert s["ev_tsumo"]["mean"] > s["ev_ron"]["mean"]
    assert 0.1 < s["win_rate"]["mean"] < 0.4
    # Over all four seats the kyoku deltas cancel, but for riichi sticks left
    # on the table when a game ends.
    total = sum(k.delta for k in ks)
    assert total <= 0 and total % 1000 == 0


def test_pairing_with_itself(rulebot_logs, capsys, tmp_path):
    ks = collect(rulebot_logs, "rulebot")
    pairs, bad = pair(ks, ks)
    assert len(pairs) == len(ks) and bad == 0
    c = compare(pairs)
    assert c["tenpai_rate"]["diff"] == 0 and c["tenpai_rate"]["a"] == c["tenpai_rate"]["b"]
    main([str(rulebot_logs), "--name", "rulebot", "--vs", str(rulebot_logs),
          "--json", str(tmp_path / "hs.json")])
    assert "paired:" in capsys.readouterr().out
    assert json.loads((tmp_path / "hs.json").read_text())["paired"]["pairs"] == len(ks)


def test_real_run_reads_the_engines_win_scoring(rulebot_logs, tmp_path):
    """The engine's `hora` carries han / fu / yaku; re-scoring a log without
    them (stripped here) gives the same han, fu and yaku."""
    import gzip
    import json
    ks, games = collect_all(rulebot_logs, "rulebot")
    wins = [k for k in ks if k.won]
    assert wins and all(k.win_han and k.win_yaku for k in wins)
    for f in sorted(rulebot_logs.glob("*.json.gz")):
        lines = [json.loads(x) for x in gzip.open(f, "rt")]
        for e in lines:
            for key in ("han", "fu", "yaku", "yakuman"):
                e.pop(key, None)
        with gzip.open(tmp_path / f.name, "wt") as out:
            out.write("\n".join(json.dumps(e) for e in lines) + "\n")
    old = [k for k in collect(tmp_path, "rulebot") if k.won]
    assert [(k.win_han, k.win_fu, k.win_yaku) for k in old] == [(k.win_han, k.win_fu, k.win_yaku) for k in wins]
    assert sorted(g.rank for g in games) == sorted([1, 2, 3, 4] * (len(games) // 4))
    g = summarize_games(games)
    assert g["rank"]["mean"] == pytest.approx(2.5)
    c = compare(pair_games(games, games), GAME_METRICS)
    assert c["pairs"] == len(games) and c["rank"]["diff"] == 0
    s = summarize(ks)
    assert 0 < s["riichi_win"]["mean"] < 1 and s["yaku"]["wins"] == len(wins)
