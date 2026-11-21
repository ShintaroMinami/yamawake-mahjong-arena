"""suzume outside the arena: `yamawake.external` plays the in-loop bot's moves."""

import json

import pytest
import riichienv as rv

from yamawake import RulebotAgent, build_agent, decide, lineup, run_match
from yamawake._core import SelfPlayLoop
from yamawake.context import SeatTracker
from yamawake.external import SuzumeObservationAgent, SuzumePlayer


@pytest.mark.parametrize("spec", ["suzume", "legacy-suzume-mlp1"])
def test_the_observation_lane_suzume_plays_the_in_loop_games(spec):
    """Through Python's riichienv the observation loses the live wall; the
    player restores it from the events and keeps the kyoku memo, so the games
    are the in-loop bot's to the point."""
    a = run_match(lineup(build_agent(spec), RulebotAgent(), "1v3"), walls=3)
    b = run_match(lineup(SuzumeObservationAgent(spec), RulebotAgent(), "1v3"), walls=3)
    assert (a.scores == b.scores).all()


def test_a_round_tripped_observation_decides_as_the_raw_one():
    """`decide` on Python's re-serialized observation equals the raw one's."""
    loop = SelfPlayLoop([1], 2, [[]])
    trackers, seen = {}, 0
    while (batch := loop.collect()) is not None and seen < 200:
        specs = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            t = trackers.setdefault((g, s), SeatTracker())
            t.feed(obs.new_events())
            for kind in ("rulebot", "suzume"):
                assert decide(kind, b64, t.events) == decide(kind, obs.serialize_to_base64(), t.events)
            specs.append(tuple(decide("suzume", b64, t.events)["action"]))
            seen += 1
        loop.apply(specs)


def test_act_mjai_answers_with_a_legal_mjai_move():
    loop = SelfPlayLoop([1], 2, [[]])
    batch = loop.collect()
    obs = rv.Observation.deserialize_from_base64(batch["obs"][0])
    t = SeatTracker()
    t.feed(obs.new_events())
    p = SuzumePlayer("suzume")
    ev = p.act_mjai(obs, t.events)
    legal = [json.loads(a.to_mjai()) if isinstance(a.to_mjai(), str) else dict(a.to_mjai())
             for a in obs.legal_actions()]
    assert ev["type"] == "dahai" and ev["actor"] == obs.player_id and "tsumogiri" in ev
    assert any(x.get("pai") == ev["pai"] for x in legal) and p.last_rule
    with pytest.raises(ValueError):
        SuzumePlayer("rulebot")


def _engine(obs: rv.Observation, m: dict) -> tuple:
    """The (action_type, tile, consume_tiles) of the legal action mjai move `m` names (a `reach` names the riichi)."""
    for a in obs.legal_actions():
        x = a.to_mjai()
        x = dict(json.loads(x) if isinstance(x, str) else x)
        if m["type"] == "reach":
            if x["type"] == "reach":
                return int(a.action_type), a.tile, list(a.consume_tiles)
            continue
        if x["type"] != m["type"] or ("pai" in m and x.get("pai") != m["pai"]):
            continue
        if sorted(x.get("consumed", [])) != sorted(m.get("consumed", [])):
            continue
        if m["type"] == "dahai" and (a.tile == obs.drawn_tile) != m["tsumogiri"]:
            continue
        return int(a.action_type), a.tile, list(a.consume_tiles)
    raise AssertionError(f"{m} is not legal")


def _games(seed: int, answer, limit: int = 1500) -> None:
    """Drive a two-game lockstep loop whose seats all answer from Python: `answer(game, seat, obs, events)` gives
    each decision's (action_type, tile, consume_tiles)."""
    loop = SelfPlayLoop([seed], 2, [[]])
    trackers, seen = {}, 0
    while (batch := loop.collect()) is not None and seen < limit:
        rows = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            t = trackers.setdefault((g, s), SeatTracker())
            t.feed(obs.new_events())
            rows.append(answer(g, s, obs, t.events))
            seen += 1
        loop.apply(rows)


def test_choose_answers_a_riichi_with_its_discard_in_one_call():
    """`choose` (docs/play_out_spec.md section 5): `choice` is `legal[choice_index]`, the best score when the model
    decided, and a riichi is the `reach` with its discard; asked again after the declaration, the seat plays it."""
    players, pending, riichis = {}, {}, []

    def answer(g, s, obs, events):
        p = players.setdefault((g, s), SuzumePlayer("suzume"))
        if (g, s) in pending:
            want = pending.pop((g, s))
            m = p.act_mjai(obs, events)
            assert (m["type"], m["pai"], m["tsumogiri"]) == ("dahai", want["pai"], want["tsumogiri"])
            riichis.append(want)
            return _engine(obs, {**want, "type": "dahai"})
        r = p.choose(obs, events)
        assert r["legal"][r["choice_index"]] == r["choice"]
        if r["basis"] == "model":
            assert r["scores"][r["choice_index"]] == max(r["scores"])
        if r["choice"]["type"] == "reach":
            pending[(g, s)] = r["choice"]
        return _engine(obs, r["choice"])

    _games(11, answer)
    assert riichis, "no riichi in the games"


def test_a_riichi_not_declared_is_forgotten():
    """A riichi the seat chose but that was not declared (another discard played in its place) leaves no tile in
    the memo: its next decision is a fresh player's."""
    players, overridden, checked = {}, set(), [0]

    def answer(g, s, obs, events):
        p = players.setdefault((g, s), SuzumePlayer("suzume"))
        r = p.choose(obs, events)
        if (g, s) in overridden:
            overridden.discard((g, s))
            assert r["choice"] == SuzumePlayer("suzume").choose(obs, events)["choice"] and r["basis"] != "riichi_tile"
            checked[0] += 1
        if r["choice"]["type"] == "reach":
            # Play a discard of another tile without declaring.
            other = next((m for m in r["legal"] if m["type"] == "dahai" and m["pai"] != r["choice"]["pai"]), None)
            if other is not None:
                overridden.add((g, s))
                return _engine(obs, other)
        return _engine(obs, r["choice"])

    _games(12, answer)
    assert checked[0] > 0, "no riichi overridden"
