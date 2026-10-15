"""Running a match: one call, one protocol, one result.

    from yamawake import RulebotAgent, TsumogiriAgent, run_match

    result = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(),
                        RulebotAgent()], walls=1000)
    print(result.table())

The protocol is fixed rather than passed piecemeal: `walls` wall matrices, each
played through four seat rotations (a 2v2 through two), with split draw queues, so a slot sees
every seat on identical walls and a seat's draws do not move when the seating
does. Everything needed to replay the match -- the wall base, the protocol and
each agent's identity -- comes back in the result.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yamawake.agents import LANE_SUZUME
from yamawake.driver import HANCHAN, SINGLE_KYOKU, play_games
from yamawake.playstats import play_counts, ratio
from yamawake.rules import DEFAULT_RULE, rule_label
from yamawake.walls import FRESH, blob_for_wall, fresh_wall_ids, standard_walls

ROTATIONS = 4          # seat shifts per wall; a 2v2 needs only two (`_rotations`)
START_SCORE = 25_000   # every seat's score at the start of a game
# The standard evaluation walls (`walls.standard_walls()`, 5,000 of them). A run
# takes the FIRST `walls`, so a 1,000-wall run is a prefix of a 5,000-wall one and
# the two compare directly; everything routine is measured there, so two results
# differ by their players and nothing else. `wall_base=FRESH` draws a set that
# has never been used, for a check that must not reuse these.


@dataclass
class SlotResult:
    """One seat-slot's standing over the match."""

    slot: int
    name: str
    avg_rank: float
    se: float
    avg_pt: float
    se_pt: float


@dataclass
class MatchResult:
    """Everything the match produced, and everything needed to repeat it."""

    slots: list[SlotResult]
    games: int
    protocol: dict
    agents: list[dict]
    scores: np.ndarray = field(repr=False)          # (games, 4) final points
    seat_of_slot: np.ndarray = field(repr=False)    # (games, 4) slot -> seat
    seconds: float = 0.0
    rule: Any = field(default=DEFAULT_RULE, repr=False)   # settles `sides`/`paired`
    # Single-kyoku matches only: {"win", "deal_in"}, (games, 4) bool by SEAT.
    kyoku: dict | None = field(default=None, repr=False)
    # With `play_stats`: `playstats.play_counts`, (games, 4) by SEAT.
    play: dict | None = field(default=None, repr=False)
    # Per agent (`protocol["slot_agent"]` index), then the arena: [decisions, seconds].
    timing: list | None = None
    cpu_seconds: float = 0.0        # the whole process's CPU time over the match

    def _groups(self) -> list[list[int]]:
        """The slots of each table row: one per slot with four rotations,
        else the slots of each side."""
        sa = self.protocol["slot_agent"]
        if int(self.protocol["rotations"]) == ROTATIONS:
            return [[j] for j in range(4)]
        return [[j for j, k in enumerate(sa) if k == a] for a in dict.fromkeys(sa)]

    def play_stats(self) -> list[dict]:
        """Per slot (or side) play statistics, pooled per kyoku.

        `win`, `deal_in`, `call` and `riichi` are shares of kyoku;
        `avg_win` the mean points of a won hand and `avg_deal` of a hand
        dealt into (all a double ron paid; honba and riichi sticks taken
        out, `playstats`);
        `adj_eff` is `win * avg_win - deal_in * avg_deal`, points per kyoku.
        Each comes with its standard error over walls (`*_se`).

        Raises:
            ValueError: The match was not played with `play_stats`.
        """
        if self.play is None:
            raise ValueError("play_stats needs a match run with play_stats=True")
        rot = int(self.protocol["rotations"])
        walls = len(self.scores) // rot
        rows = np.arange(len(self.scores))

        def per_wall(key: str, js: list[int]) -> np.ndarray:
            v = sum(self.play[key][rows, self.seat_of_slot[:, j]] for j in js)
            return v.reshape(walls, rot).sum(1).astype(float)

        out = []
        for js in self._groups():
            c = {k: per_wall(k, js) for k in self.play}
            row = {"slot": js[0], "name": self.slots[js[0]].name}
            for key, (y, x) in {"win": ("win", "kyoku"), "deal_in": ("deal_in", "kyoku"),
                                "call": ("call", "kyoku"), "riichi": ("riichi", "kyoku"),
                                "avg_win": ("win_pts", "win"),
                                "avg_deal": ("deal_pts", "deal_in")}.items():
                row[key], row[f"{key}_se"] = ratio(c[y], c[x])
            row["adj_eff"], row["adj_eff_se"] = ratio(c["win_pts"] - c["deal_pts"], c["kyoku"])
            out.append(row)
        return out

    def time_stats(self) -> list[dict]:
        """Each agent's decisions and time, then the arena's own.

        Agents: one row per distinct agent (`slot` its lowest slot). In-loop
        bots are timed in Rust and summed over threads (CPU time);
        observation-lane agents over their `act_batch` calls (wall-clock).
        The last row, `name` "arena" (`slot` None), is the arena's own
        time: the engine and the loop in Rust, summed over threads, and the
        Python side's turning rows into decisions.
        """
        if self.timing is None:
            return []
        sa = self.protocol["slot_agent"]
        out = []
        for a, (n, sec) in enumerate(self.timing[:-1]):
            j = sa.index(a)
            out.append({"slot": j, "name": self.slots[j].name, "decisions": int(n),
                        "seconds": float(sec), "ms_per_decision": 1e3 * sec / n if n else 0.0})
        out.append({"slot": None, "name": "arena", "decisions": 0,
                    "seconds": float(self.timing[-1][1]), "ms_per_decision": 0.0})
        return out

    def kyoku_stats(self) -> list[dict]:
        """Per slot (or side) outcome of a single-kyoku match.

        One kyoku's rank says little, so the kyoku is the unit here: `delta`
        is the mean final score minus 25,000 (riichi deposits included; a
        deposit still on the table when the kyoku ends in a draw is lost),
        `win` the share of kyoku the slot won (a `hora` by it) and `deal_in`
        the share it dealt into (the target of another seat's ron). Each
        comes with its standard error over walls (`*_se`). Rows follow
        `table`: one per slot with four rotations, else one per side.

        Raises:
            ValueError: The match was not played with `single_kyoku`.
        """
        if self.kyoku is None:
            raise ValueError("kyoku_stats needs a single_kyoku match")
        rot = int(self.protocol["rotations"])
        walls = len(self.scores) // rot
        rows = np.arange(len(self.scores))
        sa = self.protocol["slot_agent"]
        groups = ([[j] for j in range(4)] if rot == ROTATIONS else
                  [[j for j, k in enumerate(sa) if k == a] for a in dict.fromkeys(sa)])
        metrics = {"delta": self.scores - START_SCORE,
                   "win": self.kyoku["win"], "deal_in": self.kyoku["deal_in"]}
        out = []
        for js in groups:
            row = {"slot": js[0], "name": self.slots[js[0]].name}
            for key, metric in metrics.items():
                m = np.mean([metric[rows, self.seat_of_slot[:, j]] for j in js],
                            axis=0).reshape(walls, rot).mean(1)
                row[key] = float(m.mean())
                row[f"{key}_se"] = (float(m.std(ddof=1) / np.sqrt(walls))
                                    if walls > 1 else 0.0)
            out.append(row)
        return out

    def sides(self) -> list[SlotResult]:
        """One row per agent, pooling the slots it fills.

        A 2v2 runs two rotations, so a SIDE sees all four seats but neither of
        its two slots does: slot 0 sits at seats 0 and 1, slot 2 at 2 and 3.
        Their separate averages differ by seat luck alone and mean nothing on
        their own, so the side is the unit. `slot` is the agent's lowest slot.
        """
        rot = int(self.protocol["rotations"])
        walls = len(self.scores) // rot
        rows = np.arange(len(self.scores))
        order = np.argsort(np.argsort(-self.scores + np.arange(4) * 1e-6,
                                      axis=1), axis=1) + 1
        pts = _pt_of(self.scores, self.rule)

        out = []
        for a in dict.fromkeys(self.protocol["slot_agent"]):
            js = [j for j, k in enumerate(self.protocol["slot_agent"]) if k == a]
            per = []
            for metric in (order, pts):
                m = np.mean([metric[rows, self.seat_of_slot[:, j]] for j in js], axis=0)
                m = m.reshape(walls, rot).mean(1)
                per.append((float(m.mean()),
                            float(m.std(ddof=1) / np.sqrt(walls)) if walls > 1 else 0.0))
            out.append(SlotResult(js[0], self.slots[js[0]].name,
                                  per[0][0], per[0][1], per[1][0], per[1][1]))
        return out

    def rank_share(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """How often each side placed 1st, 2nd, 3rd and 4th.

        Pooled over the slots a side fills and the rotations, so the four
        shares of one side sum to 1. The standard error comes from the spread
        over walls, the unit the seating is balanced in.

        Returns:
            `(names, share, se)` with `share` and `se` shaped (sides, 4).
        """
        rot = int(self.protocol["rotations"])
        walls = len(self.scores) // rot
        rows = np.arange(len(self.scores))
        order = np.argsort(np.argsort(-self.scores + np.arange(4) * 1e-6,
                                      axis=1), axis=1) + 1
        names, share, se = [], [], []
        for a in dict.fromkeys(self.protocol["slot_agent"]):
            js = [j for j, k in enumerate(self.protocol["slot_agent"]) if k == a]
            got = np.stack([order[rows, self.seat_of_slot[:, j]] for j in js], axis=1)
            names.append(self.slots[js[0]].name)
            per_wall = np.stack(
                [(got == r).reshape(walls, rot * len(js)).mean(1) for r in (1, 2, 3, 4)],
                axis=1)                                  # (walls, 4)
            share.append(per_wall.mean(0))
            se.append(per_wall.std(0, ddof=1) / np.sqrt(walls) if walls > 1
                      else np.zeros(4))
        return names, np.stack(share), np.stack(se)

    def table(self) -> str:
        """The standings as a printable table, one row per slot (or side)."""
        # With fewer than four rotations a slot is not seat-balanced on its
        # own, only the side is, so report sides (`sides`).
        balanced = int(self.protocol["rotations"]) == ROTATIONS
        shown = self.slots if balanced else self.sides()
        label = "slot" if balanced else "side"
        if self.kyoku is None:
            head = (f"{label:>4} {'name':<26} {'avg_rank':>9} {'se':>7} "
                    f"{'avg_pt':>8} {'se_pt':>7}")
            rows = [f"{s.slot:>4} {s.name[:26]:<26} {s.avg_rank:9.4f} {s.se:7.4f} "
                    f"{s.avg_pt:8.3f} {s.se_pt:7.3f}" for s in shown]
        else:
            # One kyoku: the point delta and how it ended; rank is a side note.
            rank = {s.slot: s.avg_rank for s in shown}
            head = (f"{label:>4} {'name':<26} {'delta':>8} {'se':>6} {'win':>6} "
                    f"{'deal_in':>7} {'avg_rank':>9}")
            rows = [f"{r['slot']:>4} {r['name'][:26]:<26} {r['delta']:8.0f} "
                    f"{r['delta_se']:6.0f} {r['win']:6.3f} {r['deal_in']:7.3f} "
                    f"{rank[r['slot']]:9.4f}" for r in self.kyoku_stats()]
        p = self.protocol
        unit = "hanchan" if self.kyoku is None else "single-kyoku games"
        tail = (f"{self.games} {unit} ({p['walls']} walls x {p['rotations']}), "
                f"wall_base {p['wall_base']}, split_wall {p['split_wall']}, "
                f"{self.seconds:.0f}s, {self.games / max(self.seconds, 1e-9) * 60:.0f}/min")
        return "\n".join([head, *rows, tail])

    def paired(self, a: tuple[int, ...] = (0, 2),
               b: tuple[int, ...] = (1, 3)) -> dict:
        """Slots `a` minus slots `b`, paired inside each wall (D21).

        A 2v2 seats one side at slots 0 and 2 and the other at 1 and 3, and
        every wall is played through all its rotations, so within a wall the
        deal is the same for both sides and the luck cancels. The difference
        is therefore taken per wall, not per game, and the spread over walls
        is what the standard error is built from -- far tighter than comparing
        two separate matches' aggregates.

        Args:
            a: Slot indices of the challenger side.
            b: Slot indices of the other side.

        Returns:
            `{"rank", "rank_se", "pt", "pt_se", "walls"}`, plus `"delta"`,
            `"delta_se"` (kyoku point delta) for a single-kyoku match. A
            negative `rank` favours `a`, since rank 1 is first place.
        """
        rot = int(self.protocol["rotations"])
        rows = np.arange(len(self.scores))
        order = np.argsort(np.argsort(-self.scores + np.arange(4) * 1e-6,
                                      axis=1), axis=1) + 1
        pts = _pt_of(self.scores, self.rule)
        walls = len(self.scores) // rot

        def side(metric, slots):
            per = [metric[rows, self.seat_of_slot[:, j]] for j in slots]
            return np.mean(per, axis=0).reshape(walls, rot).mean(1)

        out = {"walls": walls}
        metrics = [("rank", order), ("pt", pts)]
        if self.kyoku is not None:
            metrics.append(("delta", self.scores - START_SCORE))
        for key, metric in metrics:
            d = side(metric, a) - side(metric, b)
            out[key] = float(d.mean())
            out[f"{key}_se"] = float(d.std(ddof=1) / np.sqrt(walls)) if walls > 1 else 0.0
        return out

    def to_json(self) -> dict:
        """The result without the raw arrays, for a jsonl row.

        A single-kyoku match adds `kyoku_stats` (`kyoku_stats()`).
        """
        row = {"slots": [asdict(s) for s in self.slots], "games": self.games,
               "protocol": self.protocol, "agents": self.agents,
               "seconds": round(self.seconds, 1)}
        if self.kyoku is not None:
            row["kyoku_stats"] = self.kyoku_stats()
        if self.play is not None:
            row["play_stats"] = self.play_stats()
        if self.timing is not None:
            row["time_stats"] = self.time_stats()
        return row


def _slot_agents(players: list) -> tuple[list, list[int]]:
    """The distinct agents and each slot's index among them (one agent
    object may fill several slots)."""
    agents, slot_agent = [], []
    for s in players:
        if s not in agents:
            agents.append(s)
        slot_agent.append(agents.index(s))
    return agents, slot_agent


def rotations_of(players: list) -> int:
    """The seat rotations `run_match` plays for this seating (`_rotations`)."""
    return _rotations(_slot_agents(players)[1])


def _rotations(slot_agent: list[int]) -> int:
    """Seat shifts that produce a distinct game, for this seating.

    A 2v2 is `[A, B, A, B]`, which repeats after a shift of two: the same
    agent is back at every seat, and with split walls and deterministic agents
    the core replays the game bit for bit. Two rotations already give each
    SIDE all four seats, so the other two are pure cost -- a 2v2 over N walls
    is 2N hanchan, not 4N.

    Everything else takes all four. Self-play included: there every game is
    identical, and it is the four rotations that put each slot in every seat,
    which is what makes its `avg_rank` exactly 2.5.
    """
    if len(set(slot_agent)) == 1:
        return ROTATIONS
    if all(slot_agent[j] == slot_agent[(j + 2) % 4] for j in range(4)):
        return 2
    return ROTATIONS


def lineup(challenger, opponent, mode: str = "1v3") -> list:
    """The two standard seatings, as a `players` list.

    "1v3" is one challenger against three opponents; "2v2" seats them
    alternately, which halves the variance when both sides are models.
    """
    if mode == "1v3":
        return [challenger, opponent, opponent, opponent]
    if mode == "2v2":
        return [challenger, opponent, challenger, opponent]
    raise ValueError(f"mode must be 1v3 or 2v2, got {mode!r}")


def _pt_of(scores: np.ndarray, rule) -> np.ndarray:
    """Settled points per seat (uma and oka applied)."""
    return np.array([rule.final_pt([int(x) for x in row]) for row in scores],
                    dtype=np.float64)


def run_match(players: list, walls: int | None = None,
              wall_base: int | str | None = None,
              split_wall: bool = True, game_mode: int = HANCHAN,
              rule=None, flight: int = 256,
              record_logs: str | Path | None = None,
              out: str | Path | None = None,
              save_scores: str | Path | None = None, progress=None,
              single_kyoku: bool = False, play_stats: bool = False) -> MatchResult:
    """Play `walls` x 4 hanchan (x 2 for a 2v2) and return every slot's standing.

    Args:
        players: Four agents (`agents`); slot j is `players[j]`.
        walls: Wall matrices, each played through the four rotations (two
            for a 2v2); the
            standard set's count by default.
        wall_base: Which walls. The standard set by default
            (`walls.standard_walls()`), a recorded base to replay one, or
            `FRESH` to draw a new set.
        split_wall: Per-seat fixed draw queues.
        game_mode: 2 hanchan, 1 east-only, 0 a single kyoku (`single_kyoku`).
        rule: Uma/oka rule (`rules.UmaRule`; anything with `final_pt`),
            `rules.DEFAULT_RULE` by default.
        flight: Games in flight in the core.
        record_logs: Write every game as `<dir>/<id>.json.gz` in the corpus
            format, with the agents' names in `start_game`. Games with a
            suzume player also get
            `<id>.suzume.json`: `[mjai log index, seat, deciding rule]` per
            suzume decision.
        out: Append the result (without the raw arrays) to this jsonl.
        save_scores: Write the raw `scores` and `seat_of_slot` to this `.npz`.
            The jsonl keeps aggregates only, so without this any later question
            about the games -- a rank distribution, a different split -- costs
            the whole match again. A single-kyoku match adds `win` and
            `deal_in` (by seat, like `scores`).
        single_kyoku: Every game is one kyoku and nothing else: East 1,
            honba 0, no deposits, 25,000 each, seat 0 dealing, and the game
            ends with that kyoku whatever the result (riichienv's game mode
            0; `game_mode=0` means the same). The kyoku is played on the
            wall East 1 of a hanchan gets, with the same rotations, so it is
            exactly a hanchan's first kyoku free of placement effects later
            on. `scores` holds the kyoku's final scores (delta = score -
            25,000), the result adds who won and who dealt in
            (`MatchResult.kyoku`, also saved by `save_scores`), and `table`,
            the jsonl row and `paired` report the kyoku delta and win /
            deal-in rates (`MatchResult.kyoku_stats`).
        play_stats: Also gather the play statistics (win, deal-in, call and
            riichi rates, hand points; `MatchResult.play_stats`). They are
            read from the games' logs, which costs some speed.
        progress: Optional `progress(games_done, games_total, decisions)`,
            called as the games advance (`driver.play_games`); nothing is
            printed without it.

    Returns:
        A `MatchResult`.
    """
    if len(players) != 4:
        raise ValueError("a match takes exactly four players")
    if single_kyoku and game_mode not in (HANCHAN, SINGLE_KYOKU):
        raise ValueError(f"single_kyoku plays game mode {SINGLE_KYOKU}; "
                         f"got game_mode={game_mode}")
    single_kyoku = single_kyoku or game_mode == SINGLE_KYOKU
    if single_kyoku:
        game_mode = SINGLE_KYOKU
    rule = rule or DEFAULT_RULE

    agents, slot_agent = _slot_agents(players)

    std = standard_walls()
    walls = walls or std["default_walls"]
    if wall_base is None:
        base = std["wall_base"]
        if walls > std["walls"]:
            raise ValueError(f"the standard set holds {std['walls']} walls; "
                             f"asked for {walls}")
    elif wall_base == FRESH:
        base, _ = fresh_wall_ids(1)
    else:
        base = int(wall_base)
    blobs = [blob_for_wall(base + w) for w in range(walls)]

    rotations = _rotations(slot_agent)
    seeds, maps, game_blobs, seat_of_slot = [], [], [], []
    for w in range(walls):
        for r in range(rotations):
            seeds.append(base + w)
            # Rotation r seats slot j at seat (j + r) % 4.
            seat_of_slot.append([(j + r) % 4 for j in range(4)])
            maps.append(tuple(slot_agent[(s - r) % 4] for s in range(4)))
            game_blobs.append(blobs[w])

    t0, c0 = time.time(), time.process_time()
    timing = [[0, 0.0] for _ in range(len(agents) + 1)]     # agents, then the arena
    has_suzume = any(getattr(a, "lane", None) == LANE_SUZUME for a in agents)
    traces: list | None = [] if record_logs is not None and has_suzume else None
    scores, logs = play_games(seeds, maps, agents, wall_blobs=game_blobs,
                              split_wall=split_wall, game_mode=game_mode,
                              flight=flight,
                              want_logs=record_logs is not None or single_kyoku or play_stats,
                              progress=progress,
                              traces=traces, timing=timing)
    seconds, cpu_seconds = time.time() - t0, time.process_time() - c0

    scores = np.array(scores, dtype=np.int64)                 # (games, 4) by SEAT
    seat_of_slot = np.array(seat_of_slot, dtype=np.int64)     # (games, 4) slot -> seat
    pts = _pt_of(scores, rule)
    order = np.argsort(np.argsort(-scores + np.arange(4) * 1e-6, axis=1), axis=1) + 1
    slots = []
    for j in range(4):
        seat = seat_of_slot[:, j]
        rows = np.arange(len(scores))
        rank = order[rows, seat].reshape(walls, rotations).mean(1)
        pt = pts[rows, seat].reshape(walls, rotations).mean(1)
        se = float(rank.std(ddof=1) / np.sqrt(walls)) if walls > 1 else 0.0
        se_pt = float(pt.std(ddof=1) / np.sqrt(walls)) if walls > 1 else 0.0
        slots.append(SlotResult(j, players[j].describe().get("name", f"slot{j}"),
                                float(rank.mean()), se, float(pt.mean()), se_pt))

    protocol = {"walls": walls, "rotations": rotations,
                "slot_agent": list(slot_agent), "wall_base": base,
                "split_wall": split_wall, "game_mode": game_mode,
                "rule": rule_label(rule)}
    kyoku = None
    if single_kyoku:
        protocol["single_kyoku"] = True
        kyoku = _kyoku_outcomes(logs)
    result = MatchResult(slots, len(scores), protocol,
                         [s.describe() for s in players], scores, seat_of_slot,
                         seconds, rule, kyoku,
                         play_counts(logs) if play_stats else None, timing, cpu_seconds)
    if record_logs is not None:
        _write_logs(Path(record_logs), logs, seat_of_slot, players, base, traces)
    if save_scores is not None:
        Path(save_scores).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_scores, scores=scores, seat_of_slot=seat_of_slot,
                 protocol=json.dumps(protocol),
                 agents=json.dumps(result.agents), **(kyoku or {}))
    if out is not None:
        with open(out, "a") as f:
            f.write(json.dumps(result.to_json()) + "\n")
    return result


def _kyoku_outcomes(logs: list[list[str]]) -> dict[str, np.ndarray]:
    """Who won and who dealt in, per game and seat, from single-kyoku logs.

    Also checks every game was exactly the one kyoku the mode promises.

    Returns:
        `{"win", "deal_in"}`, each (games, 4) bool by seat. A double ron
        marks both winners; a tsumo marks no one as dealing in.

    Raises:
        RuntimeError: A game is not one kyoku of East 1, honba 0, no
            deposits, 25,000 each, seat 0 dealing.
    """
    win = np.zeros((len(logs), 4), dtype=bool)
    deal_in = np.zeros((len(logs), 4), dtype=bool)
    want = {"bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [START_SCORE] * 4}
    for g, log in enumerate(logs):
        starts = 0
        for line in log:
            if '"start_kyoku"' in line:
                e = json.loads(line)
                if any(e.get(k) != v for k, v in want.items()):
                    raise RuntimeError(f"game {g} is not a single East 1 kyoku: {line[:200]}")
                starts += 1
            elif '"hora"' in line:
                e = json.loads(line)
                win[g, e["actor"]] = True
                if e["target"] != e["actor"]:
                    deal_in[g, e["target"]] = True
        if starts != 1:
            raise RuntimeError(f"game {g} has {starts} kyoku, not one")
    return {"win": win, "deal_in": deal_in}


def _write_logs(out_dir: Path, logs: list[list[str]], seat_of_slot: np.ndarray,
                players: list, wall_base: int, traces: list | None = None) -> None:
    """Write the games as gzipped mjai, with the players' names in `start_game`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [s.describe().get("name", f"slot{j}") for j, s in enumerate(players)]
    for g, log in enumerate(logs):
        by_seat = [""] * 4
        for j in range(4):
            seat = int(seat_of_slot[g, j])
            by_seat[seat] = names[j]
        log = list(log)
        log[0] = json.dumps({"type": "start_game", "names": by_seat})
        log_id = f"arena-{wall_base}-{g:06d}"
        with gzip.open(out_dir / f"{log_id}.json.gz", "wt") as f:
            f.write("\n".join(log) + "\n")
        if traces is not None and traces[g]:
            (out_dir / f"{log_id}.suzume.json").write_text(
                json.dumps([list(t) for t in traces[g]]))
