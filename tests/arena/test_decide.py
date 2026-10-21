"""decide: one rulebot / suzume decision on a position, as the loop makes it."""
from __future__ import annotations

import pytest
import riichienv as rv

from yamawake import (PythonTsumogiriAgent, RulebotAgent, SeatTracker, SelfPlayLoop, decide,
                      mjai_action_spec, suzume_rules)


def test_decide_answers_a_position_with_a_legal_action():
    """A suzume bot answers a live observation with a legal action and its rule."""
    loop = SelfPlayLoop([7], 0, [[]], split_wall=False)
    batch = loop.collect()
    rules = {r["name"] for r in suzume_rules()} | {"fallback", "a1"}
    for b64 in batch["obs"]:
        obs = rv.Observation.deserialize_from_base64(b64)
        tr = SeatTracker()
        tr.feed(obs.new_events())
        legal = {(int(a.action_type), a.tile, tuple(a.consume_tiles)) for a in obs.legal_actions()}
        for spec in ("suzume", "legacy-suzume-mlp1:call=0"):
            d = decide(spec, b64, tr.events)
            action_type, tile, consume = d["action"]
            assert (action_type, tile, tuple(consume)) in legal and d["rule"] in rules
        assert "debug" in decide("rulebot", b64, tr.events, debug=True)
        assert decide("tsumogiri", b64, tr.events)["action"] == _tsumogiri_spec(obs)
    with pytest.raises(ValueError):
        decide("tsumogiri-python", batch["obs"][0], [])


def _tsumogiri_spec(obs) -> tuple:
    a = PythonTsumogiriAgent().act_batch([type("D", (), {"obs": obs})()])[0]
    return (int(a.action_type), a.tile, list(a.consume_tiles))


def test_decide_replays_the_in_loop_rulebot():
    """A game whose four players answer through `decide("rulebot")` is the in-loop rulebot game."""
    seeds = [11]
    loop = SelfPlayLoop(seeds, 2, [[]], split_wall=False)
    trackers = {}
    while (batch := loop.collect()) is not None:
        specs = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            tr = trackers.setdefault((g, s), SeatTracker())
            tr.feed(obs.new_events())
            want = decide("rulebot", b64, tr.events)["action"]
            action = next(a for a in obs.legal_actions()
                          if (int(a.action_type), a.tile, tuple(a.consume_tiles))
                          == (want[0], want[1], tuple(want[2])))
            specs.append(mjai_action_spec(action))
        loop.apply(specs)
    rb = RulebotAgent()
    ref = SelfPlayLoop(seeds, 2, [[(s, "rulebot", rb.config) for s in range(4)]], split_wall=False)
    assert ref.collect() is None
    assert loop.scores() == ref.scores()
