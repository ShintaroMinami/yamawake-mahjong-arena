"""The arena command line."""
from __future__ import annotations

import json

import numpy as np
import pytest

from yamawake.arena import main


def test_cli_runs_a_2v2_and_writes_its_outputs(tmp_path, capsys):
    main(["--mode", "2v2", "--players", "tsumogiri", "rulebot", "--walls", "2",
          "--out", str(tmp_path / "r.jsonl"), "--save-scores", str(tmp_path / "s.npz")])
    text = capsys.readouterr().out
    assert "4 hanchan = 2 walls x 2" in text and "paired over 2 walls" in text and "/min" in text
    row = json.loads((tmp_path / "r.jsonl").read_text())
    assert row["protocol"]["rule"] == "RewardRule" and row["games"] == 4
    assert np.load(tmp_path / "s.npz")["scores"].shape == (4, 4)
    with pytest.raises(SystemExit):
        main(["--players", "tsumogiri", "rulebot", "rulebot"])


def test_cli_seats_suzume_with_an_option(capsys):
    main(["--mode", "1v3", "--players", "legacy-suzume-mlp1", "tsumogiri", "--walls", "1", "--opt", "kan=0"])
    out = capsys.readouterr().out
    assert "legacy-suzume-mlp1" in out and "tsumogiri" in out


def test_cli_plays_ordinary_random_walls(tmp_path, capsys):
    """--no-rotate --no-split --wall-base fresh: one game per new wall, shared draws."""
    main(["--mode", "1v3", "--players", "tsumogiri", "rulebot", "--walls", "3",
          "--no-rotate", "--no-split", "--wall-base", "fresh", "--out", str(tmp_path / "r.jsonl")])
    assert "no split wall" in capsys.readouterr().out
    p = json.loads((tmp_path / "r.jsonl").read_text())["protocol"]
    assert p["split_wall"] is False and p["rotate"] is False and p["rotations"] == 1
    assert p["wall_base"] != 0
