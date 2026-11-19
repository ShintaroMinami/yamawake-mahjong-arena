"""A kyoku played out from a position in its middle by in-loop bots (`docs/play_out_spec.md`).

    from yamawake import legal_actions, play_out, seat_options

    position = {"events": events, "seat": 2,           # the kyoku's mjai events from start_kyoku
                "actions": legal_actions({"events": events, "seat": 2}),
                "samples": [{"seed": 0}, {"seed": 1}]}  # hidden tiles: given, or shuffled from the seed
    results = play_out([position])                      # results[0][a][s]: the kyoku's end for action a, sample s

The position is rebuilt on the engine itself (its wall laid out for the logged deal and draws, the logged moves
replayed through `step`), then the bots play to the kyoku's end inside Rust, the (position, sample) pairs in
parallel with the GIL released (`src/rust/playout.rs`).
"""

from __future__ import annotations

import json
from typing import Any

from yamawake._core import playout_legal, playout_play_out, playout_seat_options
from yamawake.agents import IN_LOOP_LANES, build_agent

__all__ = ["legal_actions", "play_out", "seat_options"]


def _bot(spec: str) -> tuple[str, dict]:
    """An in-loop bot's (kind, config) from a player spec (`agents.build_agent`)."""
    agent = build_agent(spec)
    if agent.lane not in IN_LOOP_LANES:
        raise ValueError(f"{spec!r} is not an in-loop bot (suzume, rulebot, tsumogiri)")
    return agent.lane, dict(agent.config)


def _position(p: dict) -> dict:
    """A position with its events as dicts (JSON lines are parsed)."""
    if not isinstance(p, dict) or "events" not in p:
        raise ValueError("a position is a dict with events")
    events = [json.loads(e) if isinstance(e, str) else e for e in p["events"]]
    return {**p, "events": events}


def legal_actions(position: dict) -> list[dict]:
    """The seat's legal actions at the position, in mjai form.

    Args:
        position: `{"events", "seat"}`, and `"hand"` (the seat's hand) when the events mask its tiles.

    Returns:
        Every legal action: `dahai` per (tile, tsumogiri), `reach` per declaring discard, `chi` per pair used,
        `pon`, kans, `hora`, `none`, `ryukyoku` (kyushu kyuhai).

    Raises:
        ValueError: The position does not rebuild, or the seat does not decide there.
    """
    return json.loads(playout_legal(json.dumps(_position(position))))


def seat_options(positions: list[dict], player: str = "suzume", threads: int | None = None) -> list[dict]:
    """Per position, the seat's legal actions and the bot's choice.

    Args:
        positions: As for `legal_actions`.
        player: The in-loop bot playing the seat (a player spec, e.g. `suzume:aggression=0.5`).
        threads: Threads to use (`None`: every core).

    Returns:
        `{"legal": legal_actions(...), "choice": the bot's action}` per position; a riichi is the `reach` with its
        discard.

    Raises:
        ValueError: A position fails (its index and why).
    """
    kind, config = _bot(player)
    out = json.loads(playout_seat_options(json.dumps([_position(p) for p in positions]), kind, config, threads))
    for i, r in enumerate(out):
        if "error" in r:
            raise ValueError(f"position {i}: {r['error']}")
    return out


def play_out(positions: list[dict], players: str | list[str] = "suzume", split_wall: bool = False,
             threads: int | None = None, record_events: bool = False, on_error: str = "raise") -> list:
    """Play each position's kyoku out, for every forced action list and every sample.

    Args:
        positions: `{"events", "seat", "actions", "samples"}` (`docs/play_out_spec.md` section 3): `actions` a list
            whose entries are an mjai action or a list of them (the seat's next decisions; `[]` forces none;
            omitted: one entry `[]`), `samples` a list of hidden-tile dicts (omitted: one `{}`).
        players: One in-loop bot spec for all four seats, or four.
        split_wall: Draw from per-seat queues (`wall_queues`) as the arena's split walls do.
        threads: Threads to use (`None`: every core).
        record_events: Also return each play-out's mjai events.
        on_error: `"raise"` (ValueError naming the position, action and sample) or `"record"` (`{"error": why}`
            in that result's place).

    Returns:
        `results[i][a][s]`: `end`, `delta`, `scores`, `kyotaku`, `honba`, `renchan`, `hora`, `ryukyoku`, `forced`,
        `n_decisions` (and `events`).
    """
    if on_error not in ("raise", "record"):
        raise ValueError(f"on_error is 'raise' or 'record', got {on_error!r}")
    specs = [players] * 4 if isinstance(players, str) else list(players)
    if len(specs) != 4:
        raise ValueError("players is one spec or four")
    bots = [_bot(s) for s in specs]
    out: list[Any] = json.loads(playout_play_out(json.dumps([_position(p) for p in positions]), bots,
                                                 split_wall, threads, record_events))
    if on_error == "raise":
        for i, per_action in enumerate(out):
            for a, per_sample in enumerate(per_action):
                for s, r in enumerate(per_sample):
                    if "error" in r:
                        raise ValueError(f"position {i}, action {a}, sample {s}: {r['error']}")
    return out
