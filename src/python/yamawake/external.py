"""suzume outside the arena: one seat of a game run by someone else.

    from yamawake.external import SuzumePlayer

    player = SuzumePlayer("suzume")             # any suzume player spec
    action = player.act(obs, kyoku_events)      # riichienv Observation -> Action
    reply = player.act_mjai(obs, kyoku_events)  # the same move as an mjai dict

The environment hands the seat a riichienv `Observation` and the kyoku's
mjai events as the seat saw them (from its `start_kyoku`; others' draws may be
masked). One `SuzumePlayer` serves one seat: it keeps the kyoku memo the
arena keeps per seat (the riichi tile, the push / fold history) and starts
it over at every new `start_kyoku`, so it plays the moves the in-loop bot
plays. The decision itself is the in-loop bot's, in Rust (`_core.SuzumeSeat`).

`SuzumeObservationAgent` is the same player on the arena's observation lane
(a Python agent, one `SuzumePlayer` per game and seat): it plays the games the
in-loop `suzume` plays, through the path an outside environment takes.
"""

from __future__ import annotations

import json

import riichienv as rv

from yamawake.agents import LANE_OBSERVATION, build_agent
from yamawake.context import Decision


class SuzumePlayer:
    """One seat of suzume, fed observations and the kyoku's mjai events."""

    def __init__(self, spec: str = "suzume") -> None:
        """Args: spec: a suzume player spec, e.g. `suzume-mlp1:call=0`.

        Raises:
            ValueError: The spec is not a suzume bot.
        """
        from yamawake._core import SuzumeSeat
        agent = build_agent(spec)
        if getattr(agent, "lane", None) != "suzume":
            raise ValueError(f"{spec!r} is not a suzume bot")
        self.name = agent.name
        self.config = dict(agent.config)
        self._seat = SuzumeSeat(self.config)
        self.last_rule: str | None = None

    def act(self, obs: rv.Observation, kyoku_events: list) -> rv.Action:
        """The seat's move.

        Args:
            obs: The seat's riichienv observation.
            kyoku_events: The kyoku's mjai events so far, from its
                `start_kyoku` (dicts or JSON lines).

        Returns:
            One of `obs.legal_actions()`; `last_rule` names the suzume rule
            that chose it.
        """
        lines = [e if isinstance(e, str) else json.dumps(e) for e in kyoku_events]
        t, tile, consume, self.last_rule = self._seat.decide(obs.serialize_to_base64(), lines)
        want = (t, tile, sorted(consume))
        for a in obs.legal_actions():
            if (int(a.action_type), a.tile, sorted(a.consume_tiles)) == want:
                return a
        raise RuntimeError(f"suzume chose {want}, not a legal action here")

    def act_mjai(self, obs: rv.Observation, kyoku_events: list) -> dict:
        """The seat's move as an mjai action (`act`, then `Action.to_mjai`)."""
        ev = self.act(obs, kyoku_events).to_mjai()
        ev = dict(json.loads(ev) if isinstance(ev, str) else ev)
        ev.setdefault("actor", obs.player_id)
        if ev.get("type") == "dahai":
            ev["tsumogiri"] = obs.drawn_tile is not None and ev.get("pai") == _pai(obs.drawn_tile)
        return ev

    def reset(self) -> None:
        """Forget the kyoku memo (a new game on the same seat)."""
        self._seat.reset()


def _pai(tile: int) -> str:
    """The mjai name of a 136-id tile (red fives as 5mr / 5pr / 5sr)."""
    t, red = tile // 4, tile in (16, 52, 88)
    if t >= 27:
        return ["E", "S", "W", "N", "P", "F", "C"][t - 27]
    return f"{t % 9 + 1}{'mps'[t // 9]}{'r' if red else ''}"


class SuzumeObservationAgent:
    """suzume on the observation lane: a `SuzumePlayer` per game and seat."""

    lane = LANE_OBSERVATION

    def __init__(self, spec: str = "suzume") -> None:
        """Args: spec: a suzume player spec."""
        self.spec = spec
        self.name = SuzumePlayer(spec).name + "-python"
        self._players: dict[tuple[int, int], SuzumePlayer] = {}

    def act_batch(self, decisions: list[Decision]) -> list[rv.Action]:
        """Each seat's move from its own player."""
        out = []
        for d in decisions:
            p = self._players.get((d.game, d.seat))
            if p is None:
                p = self._players[(d.game, d.seat)] = SuzumePlayer(self.spec)
            out.append(p.act(d.obs, d.kyoku_events))
        return out

    def describe(self) -> dict:
        """`{"kind": "suzume-python", "name": <name>, "spec": <spec>}`."""
        return {"kind": "suzume-python", "name": self.name, "spec": self.spec}
