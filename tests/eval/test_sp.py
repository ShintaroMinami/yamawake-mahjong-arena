"""The single-player DP answers exactly as reljong v0.1's spdp_rs did."""
from __future__ import annotations

import json
from pathlib import Path

from yamawake import run_sp
from sp_cases import digest, payloads

FIXTURE = Path(__file__).resolve().parent / "run_sp_v01.json"


def test_run_sp_matches_the_v01_digest():
    assert digest(run_sp) == json.loads(FIXTURE.read_text())


def test_run_sp_candidates_are_sane():
    cands = run_sp(payloads(1)[0])
    discards = [c for c in cands if c["tile"] is not None and c["tile"] >= 0]
    assert discards
    for c in cands:
        assert all(0.0 <= p <= 1.0 + 1e-9 for p in c["win"])
