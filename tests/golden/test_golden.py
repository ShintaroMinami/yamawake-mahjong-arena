"""The standard rulebot replays its recorded golden games bit for bit (`cases.py`)."""

from __future__ import annotations

import json

import pytest
from cases import CASES, HERE, play


@pytest.mark.parametrize("name", list(CASES))
def test_golden_case_replays(name):
    want = json.loads((HERE / f"{name}.json").read_text())
    got = play(name)
    assert got["scores"] == want["scores"], f"{name}: scores changed (re-record if intended)"
    assert got["seat_of_slot"] == want["seat_of_slot"]
    bad = [i for i, (a, b) in enumerate(zip(got["log_sha256"], want["log_sha256"])) if a != b]
    assert not bad and len(got["log_sha256"]) == len(want["log_sha256"]), \
        f"{name}: games {bad[:5]} play differently (re-record if intended)"
