"""suzume from style coordinates (`yamawake.style`): calibration and the player spec."""

from __future__ import annotations

import pytest

from yamawake import build_agent
from yamawake.style import calibrate, default_style, predict

# (call_rate, riichi_rate, push_z) -> options from the paper's style reproduction (scipy least_squares,
# paper/code_cog2027/analysis/bots/style_match.py settings(), 2026-10-31)
REFERENCE = [
    ((0.364, 0.670, -0.78), (-0.3786, 0.2525, -0.0109)),   # the Houou median
    ((0.331, 0.719, -0.36), (-0.2224, 0.1433, 0.1064)),
    ((0.100, 0.400, -3.00), (-0.6444, -0.7547, -0.3939)),
    ((0.600, 0.900, 3.00), (0.2626, 0.8500, 0.5081)),
]


@pytest.mark.parametrize("target, expected", REFERENCE)
def test_calibrate_matches_the_paper(target, expected):
    res = calibrate(*target)
    assert res["reached"]
    assert list(res["options"].values()) == pytest.approx(expected, abs=2e-3)
    assert [res["predicted"][k] for k in ("call_rate", "riichi_rate", "push_z")] == pytest.approx(target, abs=0.01)


def test_default_style_is_all_zero():
    assert default_style()["push_z"] == pytest.approx(0, abs=1e-3)
    assert calibrate(*default_style().values())["options"] == pytest.approx(
        {"aggression": 0, "call_shift": 0, "riichi_shift": 0}, abs=2e-3)
    assert predict(0, 0, 0) == default_style()


def test_unreachable_or_extrapolated_target_is_flagged():
    assert not calibrate(0.364, 0.670, -6)["reached"]
    res = calibrate(0.95, 0.5, 0)
    assert res["extrapolated"] and res["options"]["call_shift"] > 1
    assert not calibrate(*REFERENCE[0][0])["extrapolated"]
    with pytest.raises(ValueError):
        calibrate(1.2, 0.5, 0)


def test_spec_takes_style_coordinates():
    k = build_agent("suzume:call_rate=0.364:riichi_rate=0.670:push_z=-0.78")
    assert k.config["a1"] == "suzume"
    assert [k.config[o] for o in ("aggression", "call_shift", "riichi_shift")] == pytest.approx(
        REFERENCE[0][1], abs=2e-3)
    assert k.describe()["style"]["reached"]
    # a coordinate left out stays at default suzume's
    k = build_agent("suzume:push_z=1")
    assert k.describe()["style"]["target"]["call_rate"] == pytest.approx(default_style()["call_rate"])
    assert k.config["aggression"] > 0


def test_spec_rejects_mixed_or_other_sets():
    with pytest.raises(ValueError, match="not both"):
        build_agent("suzume:call_rate=0.4:aggression=0.5")
    with pytest.raises(ValueError):
        build_agent("suzume-rla225:call_rate=0.4")
    with pytest.warns(UserWarning, match="out of reach"):
        build_agent("suzume:push_z=-6")
    with pytest.warns(UserWarning, match="extrapolation"):
        build_agent("suzume:call_rate=0.95")
