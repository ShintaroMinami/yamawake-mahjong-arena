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
