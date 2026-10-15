"""The printable report of a match: standings, play statistics and time.

    print(report(result))

Sections, each a table with one row per slot (or side, `MatchResult.table`):
the standings (rank, points and rank shares, or the kyoku delta of a
single-kyoku match), the 2v2 pairing, the play statistics per kyoku and the
hand points (when the match gathered `play_stats`), and the time each player
and the arena itself took.
"""

from __future__ import annotations

import numpy as np

from yamawake.match import MatchResult


def _table(headers: list[str], rows: list[list[str]], left: int = 2) -> list[str]:
    """Lines of a table: the first `left` columns left-aligned, the rest right."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  " + "  ".join(c.ljust(w) if i < left else c.rjust(w)
                                for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return [line(headers), *(line(r) for r in rows)]


def _pm(v: float, se: float, fmt: str) -> str:
    return f"{v:{fmt}} ± {se:{fmt.lstrip('+')}}"


def _rank_share(r: MatchResult, slots: list[int]) -> np.ndarray:
    """How often the slots placed 1st..4th, pooled over their games."""
    rows = np.arange(len(r.scores))
    order = np.argsort(np.argsort(-r.scores + np.arange(4) * 1e-6, axis=1), axis=1) + 1
    got = np.concatenate([order[rows, r.seat_of_slot[:, j]] for j in slots])
    return np.array([(got == k).mean() for k in (1, 2, 3, 4)])


def report(r: MatchResult) -> str:
    """The match as readable tables.

    Args:
        r: A finished match (`run_match`).

    Returns:
        The report, ready to print.
    """
    p = r.protocol
    balanced = int(p["rotations"]) == 4
    label = "slot" if balanced else "side"
    unit = "hanchan" if r.kyoku is None else "single-kyoku games"
    out = [f"{r.games} {unit} ({p['walls']} walls x {p['rotations']}), "
           f"wall_base {p['wall_base']}, split_wall {p['split_wall']}", ""]

    out.append("Standings")
    if r.kyoku is None:
        shown = r.slots if balanced else r.sides()
        rows = [[str(s.slot), s.name, _pm(s.avg_rank, s.se, ".4f"), _pm(s.avg_pt, s.se_pt, "+.3f"),
                 *(f"{x:.4f}" for x in _rank_share(r, js))] for s, js in zip(shown, r._groups())]
        out += _table([label, "player", "avg rank", "avg pt", "1st", "2nd", "3rd", "4th"], rows)
    else:
        rank = {s.slot: s.avg_rank for s in (r.slots if balanced else r.sides())}
        rows = [[str(k["slot"]), k["name"], _pm(k["delta"], k["delta_se"], "+.0f"),
                 f"{rank[k['slot']]:.4f}"] for k in r.kyoku_stats()]
        out += _table([label, "player", "kyoku delta", "avg rank"], rows)
    if not balanced:
        d = r.paired()
        sigma = f"{abs(d['rank']) / d['rank_se']:.1f} sigma" if d["rank_se"] else "no spread"
        line = (f"  paired over {d['walls']} walls, slots 0,2 minus 1,3: "
                f"rank {_pm(d['rank'], d['rank_se'], '+.4f')} ({sigma}), "
                f"pt {_pm(d['pt'], d['pt_se'], '+.4f')}")
        if "delta" in d:
            line += f", kyoku delta {_pm(d['delta'], d['delta_se'], '+.0f')}"
        out.append(line)

    if r.play is not None:
        stats = r.play_stats()
        out += ["", "Play per kyoku"]
        out += _table([label, "player", "win", "deal-in", "call", "riichi"],
                      [[str(s["slot"]), s["name"],
                        *(_pm(s[k], s[f"{k}_se"], ".4f") for k in ("win", "deal_in", "call", "riichi"))]
                       for s in stats])
        out += ["", "Hand points"]
        out += _table([label, "player", "avg win", "avg deal-in", "adj. efficiency"],
                      [[str(s["slot"]), s["name"], _pm(s["avg_win"], s["avg_win_se"], ".0f"),
                        _pm(s["avg_deal"], s["avg_deal_se"], ".0f"),
                        _pm(s["adj_eff"], s["adj_eff_se"], "+.0f")] for s in stats])
        out.append("  adj. efficiency = win x avg win - deal-in x avg deal-in "
                   "(points per kyoku; honba and riichi sticks excluded)")

    times = r.time_stats()
    if times:
        total = sum(t["seconds"] for t in times) or 1.0
        rows = []
        for t in times:
            slot = "" if t["slot"] is None else str(t["slot"])
            n = t["decisions"]
            rows.append([slot, t["name"] if t["slot"] is not None else "arena (engine, loop)",
                         f"{n:,}" if n else "-", f"{t['seconds']:.1f}",
                         f"{t['ms_per_decision']:.3f}" if n else "-",
                         f"{100 * t['seconds'] / total:.0f}%"])
        out += ["", "Time"]
        out += _table([label, "player", "decisions", "seconds", "ms/decision", "share"], rows)
        out.append(f"  total: {r.seconds:.1f} s wall, {r.cpu_seconds:.1f} s cpu, "
                   f"{r.games / max(r.seconds, 1e-9) * 60:,.0f} {unit}/min")
    return "\n".join(out)
