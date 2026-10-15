"""Basic play statistics per game and seat, read from the mjai logs.

    counts = play_counts(logs)          # {"kyoku", "win", ...}, each (games, 4) by seat

Per kyoku a seat counts at most once for `win`, `deal_in` (a double ron
too), `call` (any chi, pon or daiminkan) and `riichi` (a declaration, one
dealt in on its tile included). `win_pts` is the winner's kyoku delta: its
`hora` delta (honba and the riichi sticks collected) less its own accepted
riichi stick; `deal_pts` the ron payment (honba included, both rons of a
double ron; the seat's own riichi stick not counted), as `handstats` counts
them. A tsumo has no one dealing in.
"""

from __future__ import annotations

import json

import numpy as np

COUNTS = ("kyoku", "win", "deal_in", "call", "riichi", "win_pts", "deal_pts")
_CALLS = {"chi", "pon", "daiminkan"}


def _type(line: str) -> str:
    """An mjai line's event type, read without parsing the JSON."""
    i = line.index('"', line.index(":", line.rfind('"type"') + 6)) + 1
    return line[i:line.index('"', i)]


def play_counts(logs: list[list[str]]) -> dict[str, np.ndarray]:
    """Sum the play statistics of every game and seat.

    Args:
        logs: One full-information mjai log per game (lines of JSON).

    Returns:
        `{name: (games, 4) int64}` by seat for each of `COUNTS`.
    """
    out = {k: np.zeros((len(logs), 4), dtype=np.int64) for k in COUNTS}
    for g, log in enumerate(logs):
        called, dealt, paid = set(), set(), set()
        for line in log:
            t = _type(line)
            if t in ("tsumo", "dahai"):
                continue
            if t == "start_kyoku":
                called, dealt, paid = set(), set(), set()
                out["kyoku"][g] += 1
            elif t == "reach_accepted":
                paid.add(json.loads(line)["actor"])
            elif t == "reach":
                out["riichi"][g, json.loads(line)["actor"]] += 1
            elif t in _CALLS:
                a = json.loads(line)["actor"]
                if a not in called:
                    called.add(a)
                    out["call"][g, a] += 1
            elif t == "hora":
                e = json.loads(line)
                a, tg, d = e["actor"], e["target"], e["deltas"]
                out["win"][g, a] += 1
                out["win_pts"][g, a] += d[a] - 1000 * (a in paid)
                if tg != a:
                    if tg not in dealt:
                        dealt.add(tg)
                        out["deal_in"][g, tg] += 1
                    out["deal_pts"][g, tg] -= d[tg]
    return out


def ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Pooled ratio sum(y) / sum(x) and its standard error over clusters.

    Args:
        y: Numerator per cluster (e.g. wins per wall).
        x: Denominator per cluster (e.g. kyoku per wall).

    Returns:
        `(ratio, se)`; the se is the ratio estimator's, 0 with fewer than two
        clusters or no denominator.
    """
    sx = float(x.sum())
    if sx == 0:
        return 0.0, 0.0
    r = float(y.sum()) / sx
    n = len(x)
    if n < 2:
        return r, 0.0
    resid = y - r * x
    return r, float(np.sqrt((resid ** 2).sum() / (n * (n - 1))) / (sx / n))
