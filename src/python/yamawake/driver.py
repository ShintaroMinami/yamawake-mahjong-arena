"""Drive `_core.SelfPlayLoop` for a set of games, feeding whatever sits there.

A rulebot or suzume bot never reaches Python; every other player gets a `Decision` built
by `context.py` from the event stream the core hands out.

`play_games` is the whole surface: a list of games (seeds, seat maps, optional
wall blobs) in, final scores and -- if asked -- each game's mjai log out. Code
that needs its own control of the ticks (e.g. online self-play overlapping
encoding with inference) drives `SelfPlayLoop` directly the same way:

    loop = SelfPlayLoop(seeds, game_mode, bot_seats, wall_tables=blobs,
                        split_wall=True)
    while (batch := loop.collect()) is not None:      # GIL released inside
        specs = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            trackers[g, s].feed(obs.new_events())
            specs.append(mjai_action_spec(choose(decision_of(g, s, obs, trackers[g, s]))))
        loop.apply(specs)
"""

from __future__ import annotations

import time

from typing import Callable, Optional

import riichienv as rv

from yamawake._core import SelfPlayLoop
from yamawake.agents import IN_LOOP_LANES, LANE_OBSERVATION
from yamawake.context import SeatTracker, decision_of

HANCHAN = 2
# riichienv's single-kyoku game: East 1, honba 0, 25,000 each, and the game
# ends with that kyoku's end whatever the result (no renchan, no next kyoku).
SINGLE_KYOKU = 0
ActionSpec = tuple[int, Optional[int], list[int]]


def mjai_action_spec(a: rv.Action) -> ActionSpec:
    """`rv.Action` -> the `(action_type, tile, consume_tiles)` the loop matches literally.

    The physical tile is the agent's choice (a tsumogiri and a tedashi of the
    same kind differ), so the spec names it exactly.
    """
    return (int(a.action_type), a.tile, list(a.consume_tiles))


def play_games(seeds: list[int], seat_maps: list[tuple[int, ...]], agents: list,
               wall_blobs: list[bytes] | None = None, split_wall: bool = True,
               game_mode: int = HANCHAN, flight: int = 256, want_logs: bool = False,
               progress: Callable[[int, int, int], None] | None = None,
               traces: list | None = None, timing: list | None = None,
               on_logs: Callable[[list[list[str]]], None] | None = None
               ) -> tuple[list[list[int]], list[list[str]]]:
    """Play every game to the end.

    Args:
        seeds: One per game.
        seat_maps: Per game, `seat_maps[g][s]` is the index in `agents` of the
            agent sitting at absolute seat `s`.
        agents: The distinct agents; each declares its `lane`.
        wall_blobs: Per game, a packed wall table (`walls.blob_for_wall`), or
            None for seeded walls.
        split_wall: Give every seat its own fixed draw queue.
        game_mode: riichienv code -- 2 hanchan, 1 east-only, 0 one kyoku
            (`SINGLE_KYOKU`).
        flight: Games in flight at once.
        want_logs: Also return each game's full-information mjai log.
        progress: Optional `progress(games_done, games_total, decisions)`,
            called after every flight of games and every tick within one;
            `decisions` counts the Python-side decisions answered so far
            (rulebot and suzume bots decide inside Rust and are not counted).
        traces: If a list, one entry per game is appended to it: the game's
            suzume decisions as `(mjai log index, seat, deciding rule)`
            (empty for games without a suzume bot).
        timing: If a list of one `[decisions, seconds]` per agent plus one
            for the arena, each agent's decisions and the time it spent on
            them are added to its entry: in-loop bots timed in Rust (summed
            over threads), observation-lane agents over their `act_batch`
            calls. The last entry gets the arena's own time: the loop's
            work in Rust less the in-loop bots', plus the Python side's
            turning rows into decisions.
        on_logs: Optional `on_logs(flight_logs)`, called with each flight's
            mjai logs as the flight ends (whether or not `want_logs` keeps
            them), so a caller can reduce them without holding every log.

    Returns:
        `(scores, logs)`: final points per game (4 each), and the logs (empty
        when `want_logs` is False).
    """
    lanes = [getattr(a, "lane", LANE_OBSERVATION) for a in agents]
    for ai, lane in enumerate(lanes):
        if lane != LANE_OBSERVATION and lane not in IN_LOOP_LANES:
            raise ValueError(f"agent {ai} declares an unknown lane {lane!r}")
    scores: list[list[int]] = []
    logs: list[list[str]] = []

    answered = 0
    for lo in range(0, len(seeds), flight):
        hi = min(lo + flight, len(seeds))
        maps = seat_maps[lo:hi]
        # The in-loop bots' seats: (seat, "rulebot" / "suzume" / "tsumogiri", config).
        bot_seats = [[(s, lanes[m[s]], agents[m[s]].config) for s in range(4)
                      if lanes[m[s]] in IN_LOOP_LANES] for m in maps]
        loop = SelfPlayLoop(
            seeds[lo:hi], game_mode, bot_seats,
            wall_tables=None if wall_blobs is None else wall_blobs[lo:hi],
            split_wall=split_wall)
        # One tracker per (game, seat) on the observation lane: the core hands
        # out each seat's new events once, and never again.
        trackers = {(g, s): SeatTracker(getattr(agents[m[s]], "reducer_factory", None))
                    for g, m in enumerate(maps) for s in range(4)
                    if lanes[m[s]] == LANE_OBSERVATION}

        while True:
            batch = loop.collect()
            if batch is None:
                break
            loop.apply(_observation_specs(batch, maps, agents, trackers, timing))
            if progress is not None:
                answered += batch["n"]
                progress(lo, len(seeds), answered)

        scores.extend(loop.scores())
        if timing is not None:
            timing[-1][1] += sum(loop.work_times())
            for m, per_seat in zip(maps, loop.bot_times()):
                for s, (n, sec) in enumerate(per_seat):
                    timing[m[s]][0] += n
                    timing[m[s]][1] += sec
                    timing[-1][1] -= sec
        if progress is not None:
            progress(hi, len(seeds), answered)
        if want_logs or on_logs is not None:
            flight_logs = [loop.mjai_log(g) for g in range(hi - lo)]
            if on_logs is not None:
                on_logs(flight_logs)
            if want_logs:
                logs.extend(flight_logs)
        if traces is not None:
            traces.extend(loop.suzume_trace(g) for g in range(hi - lo))
    return scores, logs


def _observation_specs(batch: dict, maps: list[tuple[int, ...]], agents: list,
                       trackers: dict, timing: list | None = None) -> list[ActionSpec]:
    """Action specs for the tick's rows, in row order.

    Every row is turned into a `Decision` first, then grouped by agent so each
    agent decides its whole tick in one call.
    """
    t_start = time.perf_counter()
    agent_time = 0.0
    m = batch["n"]
    decisions, owners = [], []
    for i in range(m):
        g, s = batch["games"][i], batch["seats"][i]
        obs = rv.Observation.deserialize_from_base64(batch["obs"][i])
        tracker = trackers[(g, s)]
        tracker.feed(obs.new_events())
        owner = maps[g][s]
        decisions.append(decision_of(g, s, obs, tracker,
                                     getattr(agents[owner], "annotate", None)))
        owners.append(owner)

    specs: list = [None] * m
    for ai, agent in enumerate(agents):
        rows = [i for i, o in enumerate(owners) if o == ai]
        if not rows:
            continue
        t0 = time.perf_counter()
        actions = agent.act_batch([decisions[i] for i in rows])
        if timing is not None:
            dt = time.perf_counter() - t0
            timing[ai][0] += len(rows)
            timing[ai][1] += dt
            agent_time += dt
        if len(actions) != len(rows):
            raise ValueError(f"agent {ai} returned {len(actions)} actions for "
                             f"{len(rows)} decisions")
        for i, action in zip(rows, actions):
            specs[i] = mjai_action_spec(action)
    if timing is not None:
        timing[-1][1] += time.perf_counter() - t_start - agent_time
    return specs
