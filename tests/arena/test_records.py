"""yamawake records (docs/records.md): exact round trips, the Tenhou-style subset, concatenation."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from yamawake import RulebotAgent, TsumogiriAgent, blob_for_wall, lineup, play_games, run_match
from yamawake.records import OPS, RecordWriter, Records, convert, final_scores

NAMES = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def games():
    """20 rulebot hanchan: (final scores, mjai lines) each."""
    n = 20
    scores, logs = play_games(list(range(n)), [(0, 0, 0, 0)] * n, [RulebotAgent()],
                              wall_blobs=[blob_for_wall(i) for i in range(n)], want_logs=True)
    return list(zip(scores, logs))


def _write(tmp_path, games, name="rec"):
    w = RecordWriter()
    for scores, log in games:
        w.add_game([json.loads(line) for line in log], scores=scores, names=NAMES)
    return Records(w.save(tmp_path / name))


def test_engine_logs_round_trip_exactly(tmp_path, games):
    r = _write(tmp_path, games)
    assert len(r) == len(games)
    for g, (scores, log) in enumerate(games):
        lines = r.lines(g)
        assert lines[0] == json.dumps({"names": NAMES, "type": "start_game"}, separators=(",", ":"))
        assert lines[1:] == log[1:]
        assert r.scores(g) == list(scores) == final_scores(r.events(g))
    counts = np.bincount([OPS.index(json.loads(line)["type"]) for _, log in games for line in log
                          if json.loads(line)["type"] in OPS], minlength=len(OPS))
    assert r.op_counts().tolist() == counts.tolist()
    assert r["events"].shape == (r.kyoku_event_start[-1], 2) and "win_yaku" in r.a


def test_tenhou_style_logs_keep_what_they_have(tmp_path, games):
    """No han / fu / yaku, no draw reason, no ankan pai, a null seed: the detail arrays are left out
    and decoding adds only the ankan pai (its first consumed tile) and a tsumo win's tsumo flag."""
    stripped = []
    for _, log in games:
        evs = []
        for e in map(json.loads, log):
            if e["type"] == "start_game":
                e = {"type": "start_game", "names": NAMES, "seed": None}
            elif e["type"] == "hora":
                e = {k: v for k, v in e.items() if k not in ("han", "fu", "yaku", "yakuman", "tsumo")}
            elif e["type"] == "ryukyoku":
                e.pop("reason")
            elif e["type"] == "ankan":
                assert e.pop("pai") == e["consumed"][0]
            evs.append(e)
        stripped.append(evs)
    assert any(e["type"] == "ankan" for evs in stripped for e in evs)
    w = RecordWriter()
    for evs in stripped:
        w.add_game(evs)
    r = Records(w.save(tmp_path / "tenhou"))
    assert "win_han" not in r.a and "win_yaku" not in r.a
    for g, evs in enumerate(stripped):
        want = []
        for e in evs:
            e = dict(e)
            if e["type"] == "start_game":
                del e["seed"]
            elif e["type"] == "ankan":
                e["pai"] = e["consumed"][0]
            elif e["type"] == "hora" and e["actor"] == e["target"]:
                e["tsumo"] = True
            want.append(e)
        assert r.events(g) == want


def test_directories_concatenate_array_by_array(tmp_path, games):
    a, b, both = _write(tmp_path, games[:7], "a"), _write(tmp_path, games[7:], "b"), _write(tmp_path, games, "ab")
    assert a.players == b.players == both.players
    for name, arr in both.a.items():
        assert np.array_equal(np.concatenate([a[name], b[name]]), arr), name


def test_the_writer_refuses_what_it_would_lose(tmp_path, games):
    evs = [json.loads(line) for line in games[0][1]]
    w = RecordWriter()
    bad = [dict(e, extra=1) if e["type"] == "dahai" else e for e in evs]
    with pytest.raises(ValueError, match="does not keep"):
        w.add_game(bad)
    with pytest.raises(ValueError, match="reason"):
        w.add_game([dict(e, reason="fanpai") if e["type"] == "ryukyoku" else e for e in evs])
    assert len(w) == 0                         # a refused game leaves nothing behind
    w.add_game(evs)
    w.save(tmp_path / "one")
    with pytest.raises(FileExistsError):
        w.save(tmp_path / "one")


def test_run_match_records_the_same_games_as_mjai(tmp_path):
    players = lineup(TsumogiriAgent(), RulebotAgent())
    m = run_match(players, walls=2, record_logs=tmp_path / "mjai")
    r = run_match(players, walls=2, record_logs=tmp_path / "rec", record_format="records")
    rec = Records(tmp_path / "rec")
    assert len(rec) == r.games == m.games
    assert rec.meta["source"]["run_match"]["walls"] == 2
    files = sorted((tmp_path / "mjai").glob("*.json.gz"))
    for g, path in enumerate(files):
        with gzip.open(path, "rt") as f:
            lines = f.read().splitlines()
        assert rec.lines(g)[1:] == lines[1:]
        assert rec.names(g) == json.loads(lines[0])["names"]
        assert rec.scores(g) == r.scores[g].tolist()
    with pytest.raises(FileExistsError):
        run_match(players, walls=1, record_logs=tmp_path / "rec", record_format="records")


def test_convert_reads_a_mjai_directory(tmp_path):
    players = lineup(TsumogiriAgent(), RulebotAgent())
    r = run_match(players, walls=1, record_logs=tmp_path / "mjai")
    assert convert(tmp_path / "mjai", tmp_path / "rec") == {"games": r.games, "skipped": 0}
    rec = Records(tmp_path / "rec")
    assert [i.decode() for i in rec["game_id"]] == [p.name.split(".")[0]
                                                   for p in sorted((tmp_path / "mjai").glob("*.json.gz"))]
    assert [rec.scores(g) for g in range(len(rec))] == r.scores.tolist()


def test_the_log_tools_read_records_as_they_read_mjai(tmp_path):
    from yamawake.handstats import collect_all
    from yamawake.records import read_games
    from yamawake.tenhou6 import convert_file, convert_records
    players = lineup(TsumogiriAgent(), RulebotAgent())
    run_match(players, walls=2, record_logs=tmp_path / "mjai")
    run_match(players, walls=2, record_logs=tmp_path / "rec", record_format="records")
    assert list(read_games(tmp_path / "rec")) == list(read_games(tmp_path / "mjai"))
    name = players[1].describe()["name"]
    assert collect_all(tmp_path / "rec", name) == collect_all(tmp_path / "mjai", name)
    first = sorted((tmp_path / "mjai").glob("*.json.gz"))[0]
    assert convert_records(tmp_path / "rec", 0) == convert_file(first)
