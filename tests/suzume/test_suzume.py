"""suzume: seating and options, short matches, recorded logs and traces, the switches, the rule list.

The traces, switches and rules are the legacy sets' (`legacy-suzume-mlp3`); the default `suzume` is an A1 set.
"""
from __future__ import annotations

import gzip
import json

import pytest

from yamawake import SuzumeAgent, RulebotAgent, build_agent, lineup, suzume_rules, run_match
from yamawake.suzume import KNOBS, effective_config, main


def _events(path):
    return [json.loads(line) for line in gzip.open(path, "rt")]


def test_suzume_x4_completes_and_records(default_logs):
    logs = sorted(default_logs.glob("arena-*.json.gz"))
    assert len(logs) == 8
    names = {rule["name"] for rule in suzume_rules()} | {"fallback"}
    seen = set()
    for log in logs:
        events = _events(log)
        assert events[-1]["type"] == "end_game"
        trace = json.loads(log.with_name(log.name.replace(".json.gz", ".suzume.json")).read_text())
        assert trace and {rule for _, _, rule in trace} <= names
        seen |= {rule for _, _, rule in trace}
        # A traced forced discard is followed by that seat's dahai.
        for i, seat, rule in trace:
            ev = next((e for e in events[i:] if e.get("actor") == seat), None)
            if rule == "discard_forced":
                assert ev["type"] == "dahai", (i, seat, rule, ev)
    assert "universal_mlp" in seen


LEGACY = "legacy-suzume-mlp3"


def _legacy(config: dict | None = None, name: str = LEGACY) -> SuzumeAgent:
    return SuzumeAgent(config, name=name, models=LEGACY)


@pytest.mark.parametrize("make, traces", [(SuzumeAgent, 0), (_legacy, 8)])
def test_suzume_vs_rulebot_1v3(tmp_path, make, traces):
    a = make()
    r = run_match(lineup(a, RulebotAgent()), walls=2, flight=8, record_logs=tmp_path)
    assert r.protocol["rotations"] == 4 and r.games == 8
    assert sum(s.avg_rank for s in r.slots) == pytest.approx(10.0)
    head = json.loads(gzip.open(sorted(tmp_path.glob("arena-*.json.gz"))[0], "rt").readline())
    assert sorted(head["names"]) == sorted(["rulebot", "rulebot", "rulebot", a.name])
    assert len(list(tmp_path.glob("*.suzume.json"))) == traces


@pytest.fixture(scope="module")
def default_logs(tmp_path_factory):
    """Four legacy suzume bots (default options) on 2 standard walls, recorded."""
    out = tmp_path_factory.mktemp("default")
    run_match([_legacy()] * 4, walls=2, record_logs=out)
    return out


@pytest.mark.parametrize("option, events", [("riichi", {"reach"}), ("call", {"chi", "pon"})])
def test_switch_off_never_acts(tmp_path, default_logs, option, events):
    off = _legacy({option: 0}, name=f"{option}0")
    run_match([off] * 4, walls=2, record_logs=tmp_path)
    assert not any(e["type"] in events for log in tmp_path.glob("arena-*.json.gz") for e in _events(log))
    # The default bot does act on the same walls.
    assert any(e["type"] in events for log in default_logs.glob("arena-*.json.gz") for e in _events(log))


def test_seat_spec_and_options():
    assert set(KNOBS) == {"kan", "call", "riichi", "fold", "fold_shanten", "pf"}
    assert effective_config() == {"kan": True, "call": True, "riichi": True, "fold": True, "fold_shanten": 1, "pf": True}
    a = build_agent("legacy-suzume-mlp3:call=0:pf=0")
    assert a.name.startswith("legacy-suzume-mlp3:") and a.lane == "suzume"
    assert effective_config(a.config) == {"kan": True, "call": False, "riichi": True, "fold": True, "fold_shanten": 1, "pf": False}
    # --opt options reach legacy suzume bots; the spec's own pairs win; others are ignored.
    b = build_agent("legacy-suzume-mlp3:kan=1", kan="0", riichi="0", device="cpu")
    assert b.config == {"kan": "1", "riichi": "0", "models": "legacy-suzume-mlp3"}
    for bad in ({"ron_rate": 0.3}, {"dp_max_shanten": 2}, {"universal_mode": 1}, {"disable": "kan_mlp"},
                {"kan": 2}, {"fold_shanten": 9}, {"bogus": 1}):
        with pytest.raises(ValueError):
            _legacy(bad)
    # The current sets play on the A1 lane and take only the A1 knobs (--opt suzume options are ignored there).
    for name in ("suzume", "suzume-rla225"):
        c = build_agent(name)
        assert c.lane == "a1" and c.config == {"a1": name}
        with pytest.raises(ValueError):
            build_agent(f"{name}:call=0")
        assert build_agent(name, call="0").config == {"a1": name}
    k = build_agent("suzume:aggression=-0.5:riichi_tendency=1.25", call_tendency="0.5")
    assert k.config == {"a1": "suzume", "aggression": "-0.5", "riichi_tendency": "1.25", "call_tendency": "0.5"}


def test_rule_list(capsys):
    rules = suzume_rules()
    assert [r["name"] for r in rules] == ["win_always", "kyushu_always", "pf_discard", "fold_discard", "kan_mlp", "universal_mlp", "discard_forced"]
    assert {r["mode"] for r in rules} == {"push", "fold"}
    main(["list"])
    out = capsys.readouterr().out
    assert "universal_mlp" in out and "fold_shanten=1" in out



