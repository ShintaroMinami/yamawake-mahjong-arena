"""The Python player gives the recorded answers (`choose_cases.py`); `tests/external.rs` asks the Rust entry."""

from __future__ import annotations

import pytest
import riichienv as rv
from choose_cases import load

from yamawake.external import SuzumePlayer

DATA = load()


def _same_choice(got: dict, want: dict) -> None:
    """Equal, the scores to 1e-9 (the models' sums may round differently on another platform)."""
    assert {k: v for k, v in got.items() if k != "scores"} == {k: v for k, v in want.items() if k != "scores"}
    if want["scores"] is None:
        assert got["scores"] is None
    else:
        assert [s is None for s in got["scores"]] == [s is None for s in want["scores"]]
        assert [s for s in got["scores"] if s is not None] == pytest.approx(
            [s for s in want["scores"] if s is not None], abs=1e-9)


@pytest.mark.parametrize("i", range(len(DATA["cases"])))
def test_python_player_gives_the_recorded_answer(i):
    case = DATA["cases"][i]
    events = DATA["kyokus"][case["kyoku"]][:case["n_events"]]
    obs = rv.Observation.deserialize_from_base64(case["obs"])
    _same_choice(SuzumePlayer(case["spec"]).choose(obs, events), case["choose"])
    a = SuzumePlayer(case["spec"]).act(obs, events)
    assert [int(a.action_type), a.tile, sorted(a.consume_tiles)] == case["act"]


def test_the_cases_cover_each_kind_of_answer():
    kinds = {(c["choose"]["choice"]["type"], c["choose"]["basis"]) for c in DATA["cases"]}
    assert {("dahai", "model"), ("reach", "model"), ("chi", "model"), ("pon", "model"), ("none", "model"),
            ("hora", "win"), ("dahai", "one_discard")} <= kinds
    assert all(c["riichi_tile"] is not None for c in DATA["cases"] if c["choose"]["choice"]["type"] == "reach")
    assert {c["spec"] for c in DATA["cases"]} == {"suzume", "suzume:aggression=0.5:call_shift=0.3:riichi_shift=-0.2",
                                                  "suzume-rla225:call=0"}
