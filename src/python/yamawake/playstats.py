"""Basic play statistics per game and seat, read from the mjai logs.

    counts = play_counts(logs)          # {"kyoku", "win", ...}, each (games, 4) by seat

Per kyoku a seat counts at most once for `win`, `deal_in` (a double ron
too), `call` (any chi, pon or daiminkan) and `riichi` (a declaration, one
dealt in on its tile included). `win_pts` and `deal_pts`
are the points of the hand itself: the honba bonus and the riichi sticks are
taken out (both go to the first winner of a double ron only, so a second
winner's delta is its hand alone). A tsumo has no one dealing in.
"""

from __future__ import annotations

import json

import numpy as np

COUNTS = ("kyoku", "win", "deal_in", "call", "riichi", "win_pts", "deal_pts")
_CALLS = ('"chi"', '"pon"', '"daiminkan"')


def play_counts(logs: list[list[str]]) -> dict[str, np.ndarray]:
    """Sum the play statistics of every game and seat.

    Args:
        logs: One full-information mjai log per game (lines of JSON).

    Returns:
        `{name: (games, 4) int64}` by seat for each of `COUNTS`.
    """
    out = {k: np.zeros((len(logs), 4), dtype=np.int64) for k in COUNTS}
    for g, log in enumerate(logs):
        honba = sticks = 0
        called, dealt, first_win = set(), set(), True
        for line in log:
            if '"start_kyoku"' in line:
                e = json.loads(line)
                honba, sticks, called, dealt, first_win = e["honba"], e["kyotaku"], set(), set(), True
                out["kyoku"][g] += 1
            elif '"reach_accepted"' in line:
                sticks += 1
            elif '"reach"' in line:
                out["riichi"][g, json.loads(line)["actor"]] += 1
            elif any(c in line for c in _CALLS) and '"type"' in line:
                e = json.loads(line)
                if e["type"] in ("chi", "pon", "daiminkan") and e["actor"] not in called:
                    called.add(e["actor"])
                    out["call"][g, e["actor"]] += 1
            elif '"hora"' in line:
                e = json.loads(line)
                a, t, d = e["actor"], e["target"], e["deltas"]
                bonus = honba * 300 if first_win else 0
                out["win"][g, a] += 1
                out["win_pts"][g, a] += d[a] - bonus - (sticks * 1000 if first_win else 0)
                if t != a:
                    if t not in dealt:
                        dealt.add(t)
                        out["deal_in"][g, t] += 1
                    out["deal_pts"][g, t] += -d[t] - bonus
                first_win = False
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
