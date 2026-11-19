"""The play limits `win` / `call` (`agents.PLAY_LIMITS`) of rulebot and suzume: a limited seat never wins or calls."""
from __future__ import annotations

import gzip
import json

import pytest

from yamawake import RulebotAgent, build_agent, run_match

CALLS = ("chi", "pon", "daiminkan")


def _acts(out, names):
    """Per player name: how many hora and calls its seats made over the recorded games."""
    n = {name: {"hora": 0, "call": 0} for name in names}
    for f in out.glob("arena-*.json.gz"):
        events = [json.loads(line) for line in gzip.open(f, "rt")]
        seats = events[0]["names"]
        for e in events:
            if e["type"] == "hora":
                n[seats[e["actor"]]]["hora"] += 1
            elif e["type"] in CALLS:
                n[seats[e["actor"]]]["call"] += 1
    return n


@pytest.mark.parametrize("bot", ["rulebot", "rulebot:kobalab", "suzume", "suzume-rla225"])
def test_a_limited_seat_never_wins_or_calls(tmp_path, bot):
    specs = [bot, f"{bot}:win=0", f"{bot}:call=0", f"{bot}:win=0:call=0"]
    run_match([build_agent(s) for s in specs], walls=3, record_logs=tmp_path)
    n = _acts(tmp_path, specs)
    assert n[bot]["hora"] > 0 and n[bot]["call"] > 0
    assert n[specs[1]]["hora"] == 0 and n[specs[1]]["call"] > 0
    assert n[specs[2]]["hora"] > 0 and n[specs[2]]["call"] == 0
    assert n[specs[3]] == {"hora": 0, "call": 0}


@pytest.mark.parametrize("bot", ["rulebot", "suzume"])
def test_nobody_winning_still_plays_every_game_out(tmp_path, bot):
    """Four seats that never win: every kyoku is a draw, through riichi tsumogiri past declined tsumo and ron."""
    a = build_agent(f"{bot}:win=0")
    r = run_match([a, a, a, a], walls=2, record_logs=tmp_path)
    assert r.scores.shape[0] == 8
    events = [json.loads(line) for f in tmp_path.glob("arena-*.json.gz") for line in gzip.open(f, "rt")]
    assert not any(e["type"] == "hora" for e in events)
    assert any(e["type"] == "reach" for e in events) and any(e["type"] == "ryukyoku" for e in events)


def test_the_limits_take_0_or_1():
    assert build_agent("rulebot:kobalab:call=0").config["call"] is False
    assert RulebotAgent({"win": "true"}).config == {"win": True}
    for spec in ("rulebot:win=2", "rulebot:kan=0", "rulebot:bogus", "suzume:call=yes"):
        with pytest.raises(ValueError):
            build_agent(spec)
