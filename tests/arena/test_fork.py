"""Forks of the lockstep loop (`SelfPlayLoop.fork`, `fork_results`, `release`) through the Python surface."""
from __future__ import annotations

import pytest
import riichienv as rv

from yamawake import SelfPlayLoop, mjai_action_spec


def _move(b64: str) -> rv.Action:
    """A move that depends on the observation alone: win when possible, else tsumogiri, else pass."""
    obs = rv.Observation.deserialize_from_base64(b64)
    legal = obs.legal_actions()
    kinds = {rv.ActionType.Tsumo, rv.ActionType.Ron}
    for a in legal:
        if a.action_type in kinds:
            return a
    for a in legal:
        if a.action_type == rv.ActionType.Discard and a.tile == obs.drawn_tile:
            return a
    for a in legal:
        if a.action_type == rv.ActionType.Pass:
            return a
    return legal[0]


def _step(loop: SelfPlayLoop, batch: dict) -> None:
    loop.apply([mjai_action_spec(_move(o)) for o in batch["obs"]])


def test_forks_play_in_the_loop_and_report_their_kyoku():
    loop = SelfPlayLoop([7, 8], 2, split_wall=True)
    for _ in range(30):
        batch = loop.collect()
        _step(loop, batch)
    batch = loop.collect()
    g, s, b64 = batch["games"][0], batch["seats"][0], batch["obs"][0]
    legal = rv.Observation.deserialize_from_base64(b64).legal_actions()
    discards = [mjai_action_spec(a) for a in legal if a.action_type == rv.ActionType.Discard][:3]
    forks = loop.fork(g, s, discards)
    assert forks == [2, 3, 4][:len(discards)]
    with pytest.raises(ValueError):
        loop.fork(g, (s + 1) % 4, discards)          # no decision of that seat
    _step(loop, batch)
    results = {}
    while len(results) < len(forks):
        for f, r in zip(forks, loop.fork_results(forks, events=True, release=False)):
            if r is not None:
                results[f] = r
        if len(results) < len(forks):
            batch = loop.collect()
            assert batch is not None
            _step(loop, batch)
    for f, d in zip(forks, discards):
        r = results[f]
        assert {"end", "delta", "scores", "kyotaku", "honba", "renchan", "hora", "ryukyoku", "parent", "seat",
                "action", "events"} <= set(r)
        assert (r["parent"], r["seat"], tuple(r["action"])) == (g, s, (d[0], d[1], sorted(d[2])))
        assert r["events"][0]["type"] == "dahai" and r["events"][0]["actor"] == s
    # Results again with release (the default): the forks are freed and reused.
    assert all(r is not None for r in loop.fork_results(forks))
    with pytest.raises(ValueError):
        loop.fork_results(forks)                      # released
    batch = loop.collect()
    assert not set(forks) & set(batch["games"])
    again = loop.fork(batch["games"][0], batch["seats"][0], [mjai_action_spec(_move(batch["obs"][0]))])
    assert again[0] in forks
    _step(loop, batch)
    loop.release(again)                               # abandoned before its kyoku ends
    while (batch := loop.collect()) is not None:
        _step(loop, batch)
    assert loop.done()
