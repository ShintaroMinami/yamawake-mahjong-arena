"""Run an arena match: fixed walls, four seat rotations (a 2v2 two).

    python -m yamawake.arena --players <a> <b> <c> <d> [--walls 1000]
    yamawake-arena --mode 1v3 --players tsumogiri rulebot

A player is one of:

    rulebot | rulebot:kobalab      the ported rulebot, decided inside Rust
    suzume[:KEY=VALUE...]         the per-decision MLP bot, decided inside Rust
                                   (`python -m yamawake.suzume list`)
    tsumogiri                      the floor baseline, decided inside Rust
    tsumogiri-python               the same moves made in Python (observation lane)
    <anything a plugin registers>  see `yamawake.agents.build_agent`

Options for plugin players (e.g. a model's device) go through --opt KEY=VALUE;
suzume takes its options from it too (`--opt kan=0`), or per player
from the spec (`suzume:call=0:riichi=0`).

Examples:

    # one tsumogiri against three rulebots, the first 50 standard walls
    python -m yamawake.arena --mode 1v3 --players tsumogiri rulebot --walls 50

    # suzume against three rulebots
    python -m yamawake.arena --mode 1v3 --players suzume rulebot

    # every routine run uses the standard walls (`yamawake.walls.standard_walls`);
    # --wall-base fresh draws a new set, or pass a recorded base to replay one
    python -m yamawake.arena --players ... --wall-base fresh

    # East 1 only: each game is one kyoku (honba 0, 25,000 each, seat 0
    # dealing) on the hanchan's East 1 wall, free of score and placement
    # effects; reports the kyoku delta and win / deal-in rates
    python -m yamawake.arena --mode 1v3 --players suzume tsumogiri --single-kyoku
"""

from __future__ import annotations

import argparse
import sys
import time

from yamawake.agents import IN_LOOP_LANES, build_agent
from yamawake.match import lineup, run_match
from yamawake.report import report


def _opts(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"--opt takes KEY=VALUE, got {p!r}")
        out[k] = v
    return out


def _clock(seconds: float) -> str:
    """`mm:ss`, or `h:mm:ss` from an hour."""
    m, sec = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class _Progress:
    """A one-line progress bar on stderr (a line per 10% when not a terminal).

    Finished games advance it a flight at a time; with Python players, their
    answered decisions (against ~`per_game` a game, shared by the four seats)
    also advance it within a flight.
    """

    def __init__(self, per_game: int, py_players: int, width: int = 30) -> None:
        self.per_game, self.py_players, self.width = per_game, py_players, width
        self.t0 = time.time()
        self.tty = sys.stderr.isatty()
        self.shown, self.last = -1, 0.0

    def __call__(self, games_done: int, games_total: int, decisions: int) -> None:
        frac = games_done / games_total
        if self.py_players:
            expect = games_total * self.per_game * self.py_players / 4
            frac = max(frac, min(0.99, decisions / expect))
        pct = int(100 * frac)
        now = time.time()
        if self.tty:
            if pct < 100 and now - self.last < 0.2:     # redraw at most 5 times a second
                return
        elif pct < self.shown + 10:
            return
        self.shown, self.last = pct, now
        elapsed = now - self.t0
        eta = _clock(elapsed / frac * (1 - frac)) if frac > 0 else "?"
        bar = "#" * int(self.width * frac)
        line = (f"[{bar:<{self.width}}] {pct:3d}%  {max(games_done, int(frac * games_total))}/{games_total} games  "
                f"[{_clock(elapsed)}<{eta}]")
        print(("\r" + line) if self.tty else line, end="" if self.tty else "\n",
              file=sys.stderr, flush=True)

    def close(self) -> None:
        """End the bar's line."""
        if self.tty and self.shown >= 0:
            print(file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    """The command line entry point (`yamawake-arena`)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", nargs="+", required=True,
                    help="four players, or two with --mode")
    ap.add_argument("--mode", choices=("1v3", "2v2"), default=None,
                    help="expand two players into the standard lineup")
    ap.add_argument("--walls", type=int, default=None,
                    help="wall matrices; the standard set's first 1,000 by default")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="option passed to every player's factory (plugin players)")
    ap.add_argument("--wall-base", default=None,
                    help="'fresh' for new walls, or a recorded base to replay; "
                         "default is the standard set")
    ap.add_argument("--wall-seed", type=int, default=None,
                    help="play seeded walls: walls --wall-start .. of this seed "
                         "(0 <= seed < 2**32); excludes --wall-base")
    ap.add_argument("--wall-start", type=int, default=0,
                    help="first wall index of --wall-seed; pass the walls played so far "
                         "to add new ones")
    ap.add_argument("--flight", type=int, default=256)
    ap.add_argument("--record-logs", default=None, help="write the games as a corpus")
    ap.add_argument("--out", default=None, help="append the result to this jsonl")
    ap.add_argument("--save-scores", default=None,
                    help="write the raw per-game scores to this .npz, so a later "
                         "question about the games does not cost the match again")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip the play statistics (win, deal-in, call, riichi, hand points)")
    ap.add_argument("--single-kyoku", action="store_true",
                    help="every game is East 1 only (honba 0, 25,000 each) and ends "
                         "with it; reports the kyoku delta and win / deal-in rates")
    ap.add_argument("--no-rotate", action="store_true",
                    help="play each wall once instead of through every seat rotation "
                         "(the seating cycles over the walls); games are independent")
    args = ap.parse_args(argv)

    if args.mode:
        if len(args.players) != 2:
            raise SystemExit("--mode takes exactly two players: challenger opponent")
        specs = lineup(*args.players, mode=args.mode)
    else:
        if len(args.players) != 4:
            raise SystemExit("--players takes four players (or two with --mode)")
        specs = args.players
    built: dict[str, object] = {}
    opts = _opts(args.opt)
    try:
        players = [built[sp] if sp in built else built.setdefault(sp, build_agent(sp, **opts))
                 for sp in specs]
    except ValueError as e:
        raise SystemExit(str(e))
    progress = _Progress(per_game=60 if args.single_kyoku else 566,
                         py_players=sum(1 for a in players
                                        if getattr(a, "lane", None) not in IN_LOOP_LANES))
    if args.wall_seed is not None and args.wall_base is not None:
        raise SystemExit("pass --wall-base or --wall-seed, not both")
    if args.wall_start and args.wall_seed is None:
        raise SystemExit("--wall-start needs --wall-seed")
    result = run_match(players, walls=args.walls, wall_base=args.wall_base,
                       wall_seed=args.wall_seed, wall_start=args.wall_start,
                       flight=args.flight, record_logs=args.record_logs,
                       out=args.out,
                       save_scores=args.save_scores, progress=progress,
                       single_kyoku=args.single_kyoku, play_stats=not args.no_stats,
                       rotate=not args.no_rotate)
    progress.close()
    print(report(result), flush=True)

if __name__ == "__main__":
    main()
