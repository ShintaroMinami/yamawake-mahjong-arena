"""The printable report of a match: standings, play statistics and time.

    print(report(result))

Three tables, one row per slot (or side, `MatchResult.table`): the standings
(rank, points and rank shares, or the kyoku delta of a single-kyoku match,
with the 2v2 pairing under it), the play statistics per kyoku (when the
match gathered `play_stats`), and the time each player and the arena itself
took. Standard errors over walls sit in parentheses under their values.
"""

from __future__ import annotations

from yamawake.match import ROTATIONS, MatchResult

RULE = "─"


def _table(title: str, headers: list[str], rows: list[list[str]], left: int = 2,
           foot: list[list[str]] | None = None, notes: list[str] | None = None) -> list[str]:
    """Lines of a ruled table: the first `left` columns left-aligned, the rest right.

    Args:
        title: The line above the table.
        headers: Column names.
        rows: Body rows (an se row leaves its label cells blank).
        left: How many leading columns align left.
        foot: Rows under a rule after the body (e.g. a total).
        notes: Lines under the closing rule.
    """
    body = rows + (foot or [])
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  " + "  ".join(c.ljust(w) if i < left else c.rjust(w)
                                for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    rule = "  " + RULE * (sum(widths) + 2 * (len(widths) - 1))
    out = [title, rule, line(headers), rule, *(line(r) for r in rows)]
    if foot:
        out += [rule, *(line(r) for r in foot)]
    out.append(rule)
    return out + [f"  {n}" for n in notes or []]


def _se(x: float, fmt: str) -> str:
    return f"({x:{fmt}})"


def _standings(r: MatchResult, label: str, balanced: bool) -> list[str]:
    p = r.protocol
    unit = "hanchan" if r.kyoku is None else "single-kyoku games"
    walls_at = (f"wall_base {p['wall_base']}" if "wall_base" in p
                else f"wall_seed {p['wall_seed']}, from wall {p['wall_start']}")
    title = (f"Standings  ({r.games:,} {unit} = {p['walls']:,} walls x {p['rotations']}, "
             f"{walls_at})")
    shown = r.slots if balanced else r.sides()
    rows = []
    if r.kyoku is None:
        heads = [label, "player", "avg rank", "avg pt", "1st", "2nd", "3rd", "4th"]
        shares = r.rank_share(r._groups())[1]
        for s, share in zip(shown, shares):
            rows.append([str(s.slot), s.name, f"{s.avg_rank:.4f}", f"{s.avg_pt:+.3f}",
                         *(f"{100 * x:.1f}%" for x in share)])
            rows.append(["", "", _se(s.se, ".4f"), _se(s.se_pt, ".3f"), "", "", "", ""])
    else:
        heads = [label, "player", "kyoku delta", "avg rank"]
        rank = {s.slot: s.avg_rank for s in shown}
        for k in r.kyoku_stats():
            rows.append([str(k["slot"]), k["name"], f"{k['delta']:+,.0f}", f"{rank[k['slot']]:.4f}"])
            rows.append(["", "", _se(k["delta_se"], ",.0f"), ""])
    notes = []
    if not balanced:
        d = r.paired()
        sigma = f"{abs(d['rank']) / d['rank_se']:.1f} sigma" if d["rank_se"] else "no spread"
        notes.append(f"paired over {d['walls']:,} walls, slots 0,2 minus 1,3:  rank {d['rank']:+.4f} "
                     f"({d['rank_se']:.4f}, {sigma})   pt {d['pt']:+.4f} ({d['pt_se']:.4f})"
                     + (f"   kyoku delta {d['delta']:+,.0f} ({d['delta_se']:,.0f})"
                        if "delta" in d else ""))
    return _table(title, heads, rows, notes=notes)


def _play(r: MatchResult, label: str) -> list[str]:
    rows = []
    for s in r.play_stats():
        rows.append([str(s["slot"]), s["name"],
                     *(f"{100 * s[k]:.2f}%" for k in ("win", "deal_in", "call", "riichi")),
                     f"{s['avg_win']:,.0f}", f"{s['avg_deal']:,.0f}", f"{s['adj_eff']:+,.0f}"])
        rows.append(["", "",
                     *(_se(100 * s[f"{k}_se"], ".2f") for k in ("win", "deal_in", "call", "riichi")),
                     _se(s["avg_win_se"], ",.0f"), _se(s["avg_deal_se"], ",.0f"),
                     _se(s["adj_eff_se"], ",.0f")])
    return _table("Play per kyoku",
                  [label, "player", "win", "deal-in", "call", "riichi",
                   "avg win", "avg deal-in", "adj. eff"], rows,
                  notes=["call: chi, pon or daiminkan; riichi: declared. avg win: kyoku delta "
                         "of a won kyoku (honba, sticks, own stick paid); avg deal-in: the ron payment.",
                         "adj. eff = win x avg win - deal-in x avg deal-in (points per kyoku)."])


def _time(r: MatchResult, label: str) -> list[str]:
    times = r.time_stats()
    used = sum(t["seconds"] for t in times) or 1.0
    rows = []
    for t in times:
        n = t["decisions"]
        rows.append(["" if t["slot"] is None else str(t["slot"]),
                     t["name"] if t["slot"] is not None else "arena (engine, loop)",
                     f"{n:,}" if n else "-", f"{t['seconds']:,.1f}",
                     f"{t['ms_per_decision']:.3f}" if n else "-",
                     f"{100 * t['seconds'] / used:.0f}%"])
    unit = "hanchan" if r.kyoku is None else "games"
    foot = [["", "total", "", f"{used:,.1f}", "", "100%"]]
    return _table("Time", [label, "player", "decisions", "seconds", "ms/decision", "share"],
                  rows, foot=foot,
                  notes=[f"wall clock {r.seconds:,.1f} s, process cpu {r.cpu_seconds:,.1f} s, "
                         f"{r.games / max(r.seconds, 1e-9) * 60:,.0f} {unit}/min. "
                         "Seconds are summed over threads."])


def _label(r: MatchResult) -> tuple[str, bool]:
    """The row label and whether rows are slots (four rotations) or sides."""
    balanced = int(r.protocol["rotations"]) == ROTATIONS
    return ("slot" if balanced else "side"), balanced


def standings(r: MatchResult) -> str:
    """The standings table alone (`MatchResult.table`).

    Args:
        r: A finished match (`run_match`).

    Returns:
        The table, ready to print.
    """
    return "\n".join(_standings(r, *_label(r)))


def report(r: MatchResult) -> str:
    """The match as ruled tables.

    Args:
        r: A finished match (`run_match`).

    Returns:
        The report, ready to print.
    """
    label, balanced = _label(r)
    parts = [_standings(r, label, balanced)]
    if r.play is not None:
        parts.append(_play(r, label))
    if r.timing is not None:
        parts.append(_time(r, label))
    return "\n\n".join("\n".join(p) for p in parts)
