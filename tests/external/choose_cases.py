"""suzume's decisions as the Python player makes them, recorded for the Rust entry (`yamawake::external`).

Each spec plays two hanchan on the standard walls with one `SuzumePlayer` per seat. At every decision a fresh
player's `choose` and `act` are taken; a few decisions of each kind (the choice's type and basis, and whether a call
or a riichi was legal) are kept with the seat's observation (`serialize_to_base64()`) and its kyoku events so far.
For a riichi, the case also keeps the tile the seat's player discards after the declaration.

`test_choose_cases.py` asks the Python player, and `tests/external.rs` (`cargo test`) the Rust entry, for the same
answers. After an intended change of play, re-record:

    uv run python tests/external/choose_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

from yamawake import run_match
from yamawake.agents import LANE_OBSERVATION
from yamawake.external import SuzumePlayer

HERE = Path(__file__).resolve().parent
PATH = HERE / "choose_cases.json"

#: The specs, as both entries take them (`Suzume::from_spec` in Rust).
SPECS = ("suzume", "suzume:aggression=0.5:call_shift=0.3:riichi_shift=-0.2", "suzume-rla225:call=0")
#: Kept decisions per kind and spec.
PER_KIND = 2
WALLS = 2


def _act(a) -> list:
    return [int(a.action_type), a.tile, sorted(a.consume_tiles)]


class _Recorder:
    """Plays `spec` at every seat and keeps a few decisions of each kind (`kyokus`, `cases`)."""

    lane = LANE_OBSERVATION

    def __init__(self, spec: str) -> None:
        self.spec, self.name = spec, f"recorder-{spec}"
        self._players: dict[tuple[int, int], SuzumePlayer] = {}
        self._kyoku: dict[tuple[int, int], int] = {}        # (game, seat) -> index in kyokus of its kyoku
        self._riichi: dict[tuple[int, int], dict] = {}      # (game, seat) -> the case of its riichi
        self._seen: dict[tuple, int] = {}
        self.kyokus: list[list[dict]] = []
        self.cases: list[dict] = []

    def describe(self) -> dict:
        return {"kind": "recorder", "name": self.name}

    def act_batch(self, decisions: list) -> list:
        out = []
        for d in decisions:
            key = (d.game, d.seat)
            p = self._players.get(key) or self._players.setdefault(key, SuzumePlayer(self.spec))
            a = p.act(d.obs, d.kyoku_events)
            out.append(a)
            if key in self._riichi:
                self._riichi.pop(key)["riichi_tile"] = a.tile
            events = list(d.kyoku_events)
            k = self._kyoku.get(key)
            if k is None or self.kyokus[k][0] != events[0]:
                k = self._kyoku[key] = len(self.kyokus)
                self.kyokus.append(events)
            self.kyokus[k] = events
            c = SuzumePlayer(self.spec).choose(d.obs, events)
            types = {m["type"] for m in c["legal"]}
            kind = (c["choice"]["type"], c["basis"], bool(types & {"chi", "pon", "daiminkan"}), "reach" in types)
            if self._seen.get(kind, 0) >= PER_KIND:
                continue
            self._seen[kind] = self._seen.get(kind, 0) + 1
            case = {"spec": self.spec, "kyoku": k, "n_events": len(events), "obs": d.obs.serialize_to_base64(),
                    "act": _act(SuzumePlayer(self.spec).act(d.obs, events)), "choose": c, "riichi_tile": None}
            if c["choice"]["type"] == "reach":
                self._riichi[key] = case
            self.cases.append(case)
        return out


def record() -> dict:
    """Play every spec and gather its cases (kyoku event lists shared, each case a prefix of one)."""
    kyokus, cases = [], []
    for spec in SPECS:
        r = _Recorder(spec)
        run_match([r] * 4, walls=WALLS, rotate=False)
        for c in r.cases:
            cases.append({**c, "kyoku": c["kyoku"] + len(kyokus)})
        kyokus.extend(r.kyokus)
    used = sorted({c["kyoku"] for c in cases})
    renumber = {k: i for i, k in enumerate(used)}
    upto = {k: max(c["n_events"] for c in cases if c["kyoku"] == k) for k in used}
    return {"kyokus": [kyokus[k][:upto[k]] for k in used],
            "cases": [{**c, "kyoku": renumber[c["kyoku"]]} for c in cases]}


def load() -> dict:
    """The recorded cases."""
    return json.loads(PATH.read_text())


if __name__ == "__main__":
    data = record()
    PATH.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"{len(data['cases'])} cases, {len(data['kyokus'])} kyoku event lists -> {PATH}")
