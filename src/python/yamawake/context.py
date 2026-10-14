"""Turning the loop's observation rows into decisions an agent can use.

The core hands a player on the observation lane a base64 `rv.Observation`
carrying its seat's INCREMENTAL masked mjai events -- everything since that
seat last moved, and never again. What an encoder needs beyond the observation is therefore
tracked here, per (game, seat):

    kyoku_events   the events of the current kyoku, accumulated as they arrive
    drawable_count the live wall, counted from the `tsumo` events
    reducer        optional: the caller's own per-event reducer (e.g. a river
                   record), restarted at every `start_kyoku`

`SeatTracker` does the accumulation; `decision_of` assembles the `Decision`,
and its `annotate` hook lets a caller attach anything derived from it (e.g. an
action mask in its own id space) under `Decision.extra`. Both are pure
bookkeeping over the event stream, so a game replayed elsewhere produces the
same context.

    tracker = SeatTracker(reducer_factory=KawaRecord)      # any `.apply(event)` object
    tracker.feed(obs.new_events())
    d = decision_of(g, s, obs, tracker, annotate=lambda d: {"mask": my_mask(d.obs)})
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import riichienv as rv

# The live wall holds 70 tiles after the deal; `drawable_count` is what is left
# AFTER the draw the decision is being made on, so the first turn reads 69.
# A kan's replacement draw is a `tsumo` event too, and it takes a live tile in
# exactly the same way.
LIVE_WALL = 70

#: A per-event reducer: a zero-argument factory of objects with `.apply(event: dict)`.
ReducerFactory = Callable[[], Any]
#: A decision hook: `annotate(decision) -> dict`, merged into `Decision.extra`.
Annotate = Callable[["Decision"], dict]


@dataclass(frozen=True)
class Decision:
    """One seat's turn to move, with the context an encoder needs.

    `obs.legal_actions()` lists the moves. `kyoku_events` is this kyoku's mjai
    stream as this seat saw it (others' draws masked). `reducer` is the tracker's
    reducer of that stream (None without a `reducer_factory`); `extra` holds what an
    `annotate` hook attached. `dora_indicator_types` are the dora indicators as
    tile types (0..33), unlike the 136-ids of `obs.dora_indicators`.
    """

    game: int
    seat: int
    obs: rv.Observation
    kyoku_events: list[dict]
    drawable_count: int
    reacting_to: int | None
    dora_indicator_types: tuple[int, ...]
    reducer: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class SeatTracker:
    """The event history of one (game, seat), reduced as it arrives.

    Args:
        reducer_factory: Optional; builds a fresh reducer at the start and at every
            `start_kyoku`, and every event (the `start_kyoku` included) is
            passed to its `.apply(event)`.
    """

    reducer_factory: Optional[ReducerFactory] = None
    events: list[dict] = field(default_factory=list)
    drawable_count: int = LIVE_WALL
    reducer: Any = None

    def __post_init__(self) -> None:
        if self.reducer is None and self.reducer_factory is not None:
            self.reducer = self.reducer_factory()

    def feed(self, new_events: list[str]) -> None:
        """Absorb the seat's new mjai lines (in order)."""
        for line in new_events:
            event = json.loads(line)
            if event["type"] == "start_kyoku":
                self.events = []
                self.drawable_count = LIVE_WALL
                if self.reducer_factory is not None:
                    self.reducer = self.reducer_factory()
            if event["type"] == "tsumo":
                self.drawable_count -= 1
            self.events.append(event)
            if self.reducer is not None:
                self.reducer.apply(event)

    @property
    def reacting_to(self) -> int | None:
        """Whose discard or kakan this seat is answering, if any."""
        for event in reversed(self.events):
            if event["type"] in ("dahai", "kakan"):
                return int(event["actor"])
            if event["type"] in ("tsumo", "start_kyoku"):
                return None
        return None


def decision_of(game: int, seat: int, obs: rv.Observation, tracker: SeatTracker,
                annotate: Optional[Annotate] = None) -> Decision:
    """Assemble a `Decision` from a tracked seat and its current observation.

    Args:
        game: The game's index in its loop.
        seat: The absolute seat deciding.
        obs: The seat's current observation.
        tracker: The seat's tracker, already fed `obs.new_events()`.
        annotate: Optional hook; its dict is merged into `Decision.extra`.

    Returns:
        The `Decision`.
    """
    reacting = tracker.reacting_to
    if reacting == seat:
        reacting = None                     # one's own kakan is not a reaction
    d = Decision(
        game=game, seat=seat, obs=obs,
        kyoku_events=tracker.events,
        drawable_count=tracker.drawable_count,
        reacting_to=reacting,
        dora_indicator_types=tuple(tid // 4 for tid in obs.dora_indicators),
        reducer=tracker.reducer,
    )
    if annotate is not None:
        d.extra.update(annotate(d))
    return d
