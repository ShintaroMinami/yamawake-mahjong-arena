"""Directory model sets (`models=dir:<path>`) and the feature dumps (`SuzumeFeatures`)."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
import riichienv as rv

from yamawake import RulebotAgent, SuzumeAgent, run_match
from yamawake._core import SuzumeFeatures, suzume_config
from yamawake.agents import LANE_OBSERVATION
from yamawake.external import SuzumePlayer

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / "params" / "hand" / "flat_mlp3"
PUSH = ["discard_P_12", "discard_P_3", "discard_P_t", "riichi_P", "call_P", "kan"]
TABLE_KEYS = ["wait_table", "pattern_edges", "turn_edges", "turn_cap", "last_ago_cap", "early_discards"]


def _full_set(d: Path) -> Path:
    """The default set (`suzume`: hand flat_mlp3, state normal_mlp3) as a directory."""
    shutil.copytree(HAND, d)
    shutil.copy(ROOT / "params" / "state" / "normal_mlp3.json", d / "state.json")
    return d


def _push_set(d: Path, tables: bool = False) -> Path:
    """Only the push models of flat_mlp3 (and the fold tables, with `tables`)."""
    d.mkdir()
    for name in PUSH:
        shutil.copy(HAND / f"{name}.json", d / f"{name}.json")
    if tables:
        fold = json.loads((HAND / "discard_F.json").read_text())
        (d / "fold_tables.json").write_text(json.dumps({k: fold[k] for k in TABLE_KEYS}))
    return d


def _logs(tmp_path: Path, name: str, agent) -> list[bytes]:
    out = tmp_path / name
    run_match([agent, RulebotAgent(), agent, RulebotAgent()], walls=2, flight=8, record_logs=out)
    return [gzip.decompress(p.read_bytes()) for p in sorted(out.glob("arena-*.json.gz"))]


def test_a_directory_copy_of_the_default_set_plays_the_same_games(tmp_path):
    d = _full_set(tmp_path / "full")
    assert _logs(tmp_path, "dir", SuzumeAgent(models=f"dir:{d}")) == _logs(tmp_path, "builtin", SuzumeAgent())


def test_a_push_only_set_plays_only_without_push_fold_and_fold(tmp_path):
    d = _push_set(tmp_path / "push")
    with pytest.raises(ValueError, match="pf=0"):
        suzume_config({"models": f"dir:{d}"})
    with pytest.raises(ValueError, match="fold=0"):
        suzume_config({"models": f"dir:{d}", "pf": 0})
    agent = SuzumeAgent({"pf": 0, "fold": 0}, models=f"dir:{d}")
    assert _logs(tmp_path, "dir", agent) == _logs(tmp_path, "builtin", SuzumeAgent({"pf": 0, "fold": 0}))


def test_a_bad_directory_set_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        suzume_config({"models": f"dir:{tmp_path}", "pf": 0, "fold": 0})
    d = _push_set(tmp_path / "half")
    shutil.copy(HAND / "riichi_M.json", d / "riichi_M.json")
    with pytest.raises(ValueError, match="all or none"):
        suzume_config({"models": f"dir:{d}", "pf": 0, "fold": 0})


def _discard_id(action) -> int:
    """The universal option id of a discard action (34 + suit for a red five)."""
    t = action.tile
    return 34 + (t // 4) // 9 if t in (16, 52, 88) else t // 4


class _Checker:
    """Plays suzume (`pf=0`, `fold=0`) on the observation lane and checks each
    decision against the dumps of the same position."""

    lane = LANE_OBSERVATION

    def __init__(self, models: str) -> None:
        self.spec = {"pf": 0, "fold": 0, "models": models}
        self.players: dict = {}
        self.features = SuzumeFeatures(self.spec)
        self.full = SuzumeFeatures({"models": "suzume"})
        self.checked = {"universal_mlp": 0, "fold": 0, "kan": 0}

    def act_batch(self, decisions):
        out = []
        for d in decisions:
            key = (d.game, d.seat)
            if key not in self.players:
                self.players[key] = SuzumePlayer("suzume:pf=0:fold=0")
            lines = [json.dumps(e) for e in d.kyoku_events]
            a = self.players[key].act(d.obs, d.kyoku_events)
            obs = d.obs.serialize_to_base64()
            u = self.features.universal(obs, lines)
            assert len(u["rows"]) == len(u["options"]) and all(len(r) == len(u["names"]) == 76 for r in u["rows"])
            m = self.features.universal_mlp(obs, lines)
            declaring = any(e.get("type") == "reach" and e.get("actor") == d.seat for e in d.kyoku_events[-1:])
            if self.players[key].last_rule == "universal_mlp" and not declaring and m is not None and m["model"] != "call" and a.action_type == rv.ActionType.DISCARD:
                kind, oid = m["options"][m["choice"]]
                if kind != "riichi":
                    assert oid == _discard_id(a), (kind, oid, a)
                    self.checked["universal_mlp"] += 1
            k = self.features.universal_kan(obs, lines)
            if k is not None:
                assert k["options"][0][0] in ("discard", "dama", "break") and k["options"][1][1] >= 42
                self.checked["kan"] += 1
            f = self.full.fold_features(obs, lines)
            if f is not None:
                x = f["features"]
                assert len(x) == 3 and all(len(s) == 34 and len(s[0]) == len(f["names"]) == 42 for s in x)
                assert set(f["legal"]) == {t for t, s in enumerate(f["shanten_after"]) if s is not None}
                self.checked["fold"] += 1
            out.append(a)
        return out

    def describe(self) -> dict:
        return {"kind": "checker", "name": "checker"}


def test_the_dumps_match_the_decisions(tmp_path):
    d = _push_set(tmp_path / "push", tables=True)
    checker = _Checker(f"dir:{d}")
    run_match([checker, RulebotAgent(), checker, RulebotAgent()], walls=1, flight=4)
    assert checker.checked["universal_mlp"] > 100 and checker.checked["fold"] > 100


def test_the_fold_features_need_tables(tmp_path):
    d = _push_set(tmp_path / "push")
    f = SuzumeFeatures({"models": f"dir:{d}", "pf": 0, "fold": 0})
    with pytest.raises(ValueError, match="fold tables"):
        f.fold_features("", [])


def test_fold_tables_give_the_fold_models_features(tmp_path):
    """A push-only set with the fold model's tables dumps the same fold
    features as the full set (same push models, same tables)."""
    d = _push_set(tmp_path / "push", tables=True)
    a = SuzumeFeatures({"models": f"dir:{d}", "pf": 0, "fold": 0})
    b = SuzumeFeatures({"models": "suzume-mlp3-flat"})
    seen = []

    class Grab:
        lane = LANE_OBSERVATION

        def __init__(self):
            self.player = {}

        def act_batch(self, decisions):
            out = []
            for dd in decisions:
                p = self.player.setdefault((dd.game, dd.seat), SuzumePlayer("suzume"))
                lines = [json.dumps(e) for e in dd.kyoku_events]
                obs = dd.obs.serialize_to_base64()
                fa, fb = a.fold_features(obs, lines), b.fold_features(obs, lines)
                assert fa == fb
                seen.append(fa is not None)
                out.append(p.act(dd.obs, dd.kyoku_events))
            return out

        def describe(self):
            return {"kind": "grab", "name": "grab"}

    g = Grab()
    run_match([g, RulebotAgent(), g, RulebotAgent()], walls=1, flight=4)
    assert sum(seen) > 100
