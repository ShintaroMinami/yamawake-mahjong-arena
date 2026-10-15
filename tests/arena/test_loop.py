"""The low-level surface: the lockstep loop contract, the context hooks, the player registry."""
from __future__ import annotations

import json

import pytest
import riichienv as rv

from yamawake import (PythonTsumogiriAgent, SeatTracker, SelfPlayLoop, TsumogiriAgent, build_agent,
                       decision_of, mjai_action_spec, play_games, register_agent,
                       register_resolver)
from yamawake.agents import LANE_OBSERVATION, LANE_TSUMOGIRI, RulebotAgent
from yamawake.context import LIVE_WALL
from yamawake.rulebot import LATEST, STANDARD
from yamawake.walls import blob_for_wall


def _drive(loop: SelfPlayLoop, trackers: dict, choose) -> int:
    """Run a loop to the end the way online self-play does; returns ticks."""
    ticks = 0
    while (batch := loop.collect()) is not None:
        assert set(batch) == {"n", "games", "seats", "obs"}
        assert batch["n"] == len(batch["games"]) == len(batch["seats"]) == len(batch["obs"])
        specs = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            tr = trackers.setdefault((g, s), SeatTracker())
            tr.feed(obs.new_events())
            specs.append(mjai_action_spec(choose(decision_of(g, s, obs, tr))))
        loop.apply(specs)
        ticks += 1
    return ticks


def test_driving_the_loop_directly_matches_play_games():
    seeds = [11, 12, 13]
    blobs = [blob_for_wall(s) for s in seeds]
    rb = [[(1, "rulebot", STANDARD), (3, "rulebot", STANDARD)] for _ in seeds]
    loop = SelfPlayLoop(seeds, 2, rb, wall_tables=blobs, split_wall=True)
    assert _drive(loop, {}, lambda d: PythonTsumogiriAgent().act_batch([d])[0]) > 0
    assert loop.done()
    want, logs = play_games(seeds, [(0, 1, 0, 1)] * 3, [PythonTsumogiriAgent(), RulebotAgent()],
                            wall_blobs=blobs, want_logs=True)
    assert loop.scores() == want
    assert loop.mjai_log(2) == logs[2]
    assert all(sum(s) <= 100000 for s in want)


def test_apply_checks_its_answers():
    loop = SelfPlayLoop([5], 2)
    batch = loop.collect()
    with pytest.raises(ValueError, match="decisions for"):
        loop.apply([])
    with pytest.raises(ValueError, match="not legal"):
        loop.apply([(0, 200, [])] * batch["n"])
    with pytest.raises(ValueError, match="unapplied"):
        loop.collect()


def test_an_all_rulebot_game_finishes_in_one_collect():
    loop = SelfPlayLoop([3, 4], 2, [[(s, "rulebot", LATEST) for s in range(4)]] * 2,
                        wall_tables=[blob_for_wall(3), blob_for_wall(4)], split_wall=True)
    assert loop.collect() is None and loop.done()
    assert all(sum(s) <= 100000 for s in loop.scores())


class _Reducer:
    """A toy per-event reducer: counts the kyoku's discards."""

    def __init__(self) -> None:
        self.events, self.discards = 0, 0

    def apply(self, event: dict) -> None:
        self.events += 1
        self.discards += event["type"] == "dahai"


class _Watcher(PythonTsumogiriAgent):
    """Tsumogiri, with both context hooks, checking each decision as it comes.

    A decision's `kyoku_events` and `reducer` are the tracker's live objects, so
    they are read at decision time, not afterwards.
    """

    reducer_factory = _Reducer

    def __init__(self) -> None:
        self.n, self.first_draws = 0, []

    @staticmethod
    def annotate(d) -> dict:
        return {"n_legal": len(d.obs.legal_actions())}

    def act_batch(self, decisions):
        for d in decisions:
            self.n += 1
            assert isinstance(d.reducer, _Reducer)
            assert d.reducer.events == len(d.kyoku_events)        # restarted per kyoku
            assert d.reducer.discards == sum(e["type"] == "dahai" for e in d.kyoku_events)
            assert d.extra == {"n_legal": len(d.obs.legal_actions())}
            assert 0 <= d.drawable_count < LIVE_WALL
            assert d.kyoku_events[0]["type"] == "start_kyoku"
            if sum(e["type"] == "tsumo" for e in d.kyoku_events) == 1 and d.reacting_to is None:
                self.first_draws.append(d.drawable_count)
        return super().act_batch(decisions)


def test_the_reducer_and_annotate_hooks_reach_every_decision():
    w = _Watcher()
    play_games([21], [(0, 0, 0, 0)], [w], wall_blobs=[blob_for_wall(21)])
    assert w.n > 100
    # The dealer's first decision of a kyoku is on the first draw: 69 left.
    assert w.first_draws and set(w.first_draws) == {LIVE_WALL - 1}


def test_a_tracker_without_a_reducer_keeps_none():
    tr = SeatTracker()
    tr.feed([json.dumps({"type": "start_kyoku"}), json.dumps({"type": "tsumo", "actor": 0})])
    assert tr.reducer is None and tr.drawable_count == LIVE_WALL - 1 and len(tr.events) == 2


def test_registry_names_and_resolvers():
    assert build_agent("tsumogiri").describe()["kind"] == "tsumogiri"
    assert build_agent("tsumogiri").lane == LANE_TSUMOGIRI
    assert build_agent("tsumogiri-python").lane == LANE_OBSERVATION
    with pytest.raises(ValueError, match="unknown player"):
        build_agent("nobody-registered-this")

    got = {}
    register_agent("echo", lambda spec, **o: got.setdefault(spec, (spec, o)))
    build_agent("echo:x", device="cpu")
    assert got["echo:x"] == ("echo:x", {"device": "cpu"})
    register_resolver(lambda spec, **o: PythonTsumogiriAgent() if spec.endswith(".pth") else None)
    assert build_agent("some/model.pth").lane == LANE_OBSERVATION
