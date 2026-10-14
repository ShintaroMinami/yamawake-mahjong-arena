"""The rulebot's configs and kan (its play is pinned by tests/golden)."""
from __future__ import annotations

import gzip
import json

import pytest

from yamawake import RulebotAgent, SelfPlayLoop, build_agent, run_match
from yamawake.rulebot import LATEST, STANDARD


def test_configs_and_seat_specs():
    assert STANDARD == dict(LATEST, call_ev_sp=True, call_sp_flat_pass=True, kan=True)
    assert build_agent("rulebot").config == STANDARD
    latest = build_agent("rulebot:latest")
    assert latest.config == LATEST and latest.describe()["name"] == "rulebot:latest"
    with pytest.raises(KeyError, match="no_such_knob"):
        SelfPlayLoop([1], 2, [[(0, "rulebot", {"no_such_knob": 1})]])


def test_the_standard_rulebot_kans_but_never_daiminkan(tmp_path):
    """STANDARD declares ankan / kakan (never daiminkan); with `kan` off, none."""
    def kans(config, out):
        rb = RulebotAgent(config)
        run_match([rb, rb, rb, rb], walls=10, flight=40, record_logs=out)
        seen = {"ankan": 0, "kakan": 0, "daiminkan": 0}
        for f in out.glob("*.gz"):
            for line in gzip.open(f, "rt"):
                t = json.loads(line)["type"]
                if t in seen:
                    seen[t] += 1
        return seen
    on = kans(STANDARD, tmp_path / "on")
    assert on["ankan"] > 0 and on["kakan"] > 0 and on["daiminkan"] == 0
    assert kans(dict(STANDARD, kan=False), tmp_path / "off") == {"ankan": 0, "kakan": 0, "daiminkan": 0}
