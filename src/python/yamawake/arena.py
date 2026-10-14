"""Run an arena match: fixed walls, four seat rotations (a 2v2 two).

    python -m yamawake.arena --players <a> <b> <c> <d> [--walls 1000]
    yamawake-arena --mode 1v3 --players tsumogiri rulebot

A player is one of:

    rulebot | rulebot:latest       the ported rulebot, decided inside Rust
    suzume[:KEY=VALUE...]         the per-decision MLP bot, decided inside Rust
                                   (`python -m yamawake.suzume list`)
    tsumogiri                      the floor baseline
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

from yamawake.agents import build_agent
from yamawake.match import lineup, rotations_of, run_match
from yamawake.walls import standard_walls


def _opts(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"--opt takes KEY=VALUE, got {p!r}")
        out[k] = v
    return out


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
    ap.add_argument("--flight", type=int, default=256)
    ap.add_argument("--record-logs", default=None, help="write the games as a corpus")
    ap.add_argument("--out", default=None, help="append the result to this jsonl")
    ap.add_argument("--save-scores", default=None,
                    help="write the raw per-game scores to this .npz, so a later "
                         "question about the games does not cost the match again")
    ap.add_argument("--single-kyoku", action="store_true",
                    help="every game is East 1 only (honba 0, 25,000 each) and ends "
                         "with it; reports the kyoku delta and win / deal-in rates")
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
    # Progress: decisions answered by the Python players, against ~566 a hanchan
    # (~60 a single kyoku) shared by the four players (in-Rust bots are not
    # counted).
    import time as _time
    t0 = _time.time()
    py_players = sum(1 for a in players
                      if getattr(a, "lane", None) not in ("rulebot", "suzume"))
    rotations = rotations_of(players)
    n_games = (args.walls or standard_walls()["default_walls"]) * rotations
    per_game = 60 if args.single_kyoku else 566
    expect = max(1, int(n_games * per_game * py_players / 4))
    shown = [0]

    def progress(decisions, games_started):
        pct = min(99, int(100 * decisions / expect))
        if pct >= shown[0] + 5:
            shown[0] = pct - pct % 5
            print(f"[{(_time.time() - t0) / 60:5.1f} min] ~{pct}% ({decisions}/{expect} "
                  f"decisions, {games_started}/{n_games} games started)", flush=True)

    result = run_match(players, walls=args.walls, wall_base=args.wall_base,
                       flight=args.flight, record_logs=args.record_logs,
                       out=args.out,
                       save_scores=args.save_scores, progress=progress,
                       single_kyoku=args.single_kyoku)
    print(result.table(), flush=True)
    if args.mode == "2v2":
        d = result.paired()
        sigma = (f"{abs(d['rank']) / d['rank_se']:.1f} sigma" if d["rank_se"]
                 else "no spread")
        names, share, se = result.rank_share()
        print(f"{'side':<26}" + "".join(f"{f'{r}位':>12}" for r in (1, 2, 3, 4)))
        for i, n in enumerate(names):
            print(f"{n[:26]:<26}" + "".join(
                f"{share[i, r]:8.4f}+-{se[i, r]:.3f}" for r in range(4)))
        print(f"paired over {d['walls']} walls: slots 0,2 minus 1,3 = "
              f"{d['rank']:+.4f} +/- {d['rank_se']:.4f} rank, "
              f"{d['pt']:+.4f} +/- {d['pt_se']:.4f} pt "
              f"({sigma}; negative rank favours slots 0,2)", flush=True)
        if "delta" in d:
            print(f"kyoku delta, slots 0,2 minus 1,3 = {d['delta']:+.0f} "
                  f"+/- {d['delta_se']:.0f} points", flush=True)


if __name__ == "__main__":
    main()
