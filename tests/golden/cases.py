"""Golden games: the standard rulebot on fixed walls, recorded once.

Every case plays a small match on the standard walls and keeps each game's
final scores and the sha256 of its mjai log (`<case>.json` here). The test
(`test_golden.py`) replays the cases and asks for the same bytes: any change
in the arena, the engine or the rulebot that alters a single decision fails
it. After an intended change of play, re-record:

    uv run python tests/golden/cases.py            # every case
    uv run python tests/golden/cases.py rulebot_x4 # some
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from yamawake import build_agent, lineup, run_match

HERE = Path(__file__).resolve().parent

#: name -> (player specs as `lineup` takes them, mode, walls, single kyoku).
#: All-same-bot cases play one wall per game's worth (their rotations replay
#: the same game).
CASES: dict[str, tuple[tuple[str, ...], str | None, int, bool]] = {
    "rulebot_x4": (("rulebot",) * 4, None, 50, False),
    "tsumogiri_1v3_rulebot": (("tsumogiri", "rulebot"), "1v3", 25, False),
}


def play(name: str) -> dict:
    """Play one case: its games' scores (by seat) and log digests."""
    specs, mode, walls, single = CASES[name]
    agents = [build_agent(s) for s in specs]
    players = lineup(*agents, mode=mode) if mode else agents
    with tempfile.TemporaryDirectory() as tmp:
        r = run_match(players, walls=walls, single_kyoku=single, record_logs=tmp)
        logs = sorted(Path(tmp).glob("arena-*.json.gz"))
        digests = [hashlib.sha256(gzip.open(f, "rb").read()).hexdigest() for f in logs]
    return {"seats": list(specs), "mode": mode, "walls": walls, "single_kyoku": single,
            "scores": r.scores.tolist(), "seat_of_slot": r.seat_of_slot.tolist(),
            "log_sha256": digests}


def record(names: list[str]) -> None:
    """Write `<case>.json` for each case."""
    for name in names:
        g = play(name)
        head = {k: v for k, v in g.items() if k not in ("scores", "seat_of_slot", "log_sha256")}
        rows = [f'  "{k}": [\n' + ",\n".join(f"    {json.dumps(x)}" for x in g[k]) + "\n  ]"
                for k in ("scores", "seat_of_slot", "log_sha256")]
        text = "{\n" + "".join(f"  {json.dumps(k)}: {json.dumps(v)},\n" for k, v in head.items())
        (HERE / f"{name}.json").write_text(text + ",\n".join(rows) + "\n}\n")
        print(f"recorded {name}")


if __name__ == "__main__":
    record(sys.argv[1:] or list(CASES))
