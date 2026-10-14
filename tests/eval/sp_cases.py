"""Deterministic run_sp payloads and a digest of the solver's answers.

Shared by tests/eval/test_sp.py and the one-off that recorded the reference digest
with reljong v0.1's `spdp_rs.run_sp` (tests/eval/run_sp_v01.json):

    cd $RELJONG_V01/code && uv run --frozen python \\
        <yamawake>/tests/eval/sp_cases.py spdp_rs > <yamawake>/tests/eval/run_sp_v01.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys


def payloads(n: int = 40, seed: int = 7) -> list[dict]:
    """`n` closed-hand payloads: 14 random tiles, one dora, varied turns."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        deck = [t for t in range(34) for _ in range(4)]
        rng.shuffle(deck)
        hand = [0] * 34
        for t in deck[:14]:
            hand[t] += 1
        ind = deck[14]
        seen = list(hand)
        seen[ind] += 1
        out.append({
            "hand": hand, "reds_in_hand": [False] * 3, "seen_counts": seen,
            "reds_seen": [False] * 3, "draws_left": rng.choice([4, 8, 12, 17]),
            "own_turn": True, "last_draw_is_haitei": rng.random() < 0.5,
            "double_riichi_possible": False, "riichi_when_tenpai": True, "is_menzen": True,
            "jikaze": 27 + rng.randrange(4), "bakaze": 27, "dora_indicators": [ind],
            "meld_dora_han": 0, "melds": [], "collect_tiles": [],
        })
    return out


def digest(run_sp) -> dict:
    """Solve every payload; sha256 over the exact (repr) answers, and counts."""
    h = hashlib.sha256()
    cands = 0
    for p in payloads():
        for c in run_sp(p):
            cands += 1
            h.update(repr(sorted(c.items())).encode())
    return {"payloads": len(payloads()), "candidates": cands, "sha256": h.hexdigest()}


if __name__ == "__main__":
    mod = __import__(sys.argv[1], fromlist=["run_sp"])
    print(json.dumps(digest(mod.run_sp), indent=1))
