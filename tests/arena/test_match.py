"""run_match: the duplicate protocol, its result tables and what it records."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from yamawake import (FRESH, SuzumeAgent, RulebotAgent, TsumogiriAgent, lineup, play_games,
                       run_match, standard_walls)
from yamawake.rules import DEFAULT_RULE, UmaRule, rank_of
from yamawake.walls import blob_for_wall


@pytest.fixture(scope="module")
def one_vs_three(tmp_path_factory):
    """Tsumogiri against three rulebots on 3 standard walls, recorded."""
    out = tmp_path_factory.mktemp("1v3")
    r = run_match(lineup(TsumogiriAgent(), RulebotAgent()), walls=3, flight=12,
                  record_logs=out / "logs", save_scores=out / "scores.npz")
    return r, out


@pytest.fixture(scope="module")
def two_vs_two():
    """Tsumogiri pairs against rulebot pairs on 6 standard walls."""
    return run_match(lineup(TsumogiriAgent(), RulebotAgent(), mode="2v2"), walls=6, flight=24)


def test_identical_agents_average_exactly_2_5():
    """The duplicate protocol's own check: no seat can be lucky."""
    rb = RulebotAgent()
    result = run_match([rb, rb, rb, rb], walls=2)
    assert result.games == 8
    for slot in result.slots:
        assert slot.avg_rank == pytest.approx(2.5) and slot.se == pytest.approx(0.0)
    # Points are equal across slots too, but they do not sum to zero: riichi
    # sticks left on the table when a game ends are paid by nobody.
    pts = [slot.avg_pt for slot in result.slots]
    assert max(pts) - min(pts) < 1e-9


def test_1v3_plays_four_rotations_and_reports_every_slot(one_vs_three):
    r, _ = one_vs_three
    assert r.protocol["rotations"] == 4 and r.games == 12 and r.protocol["split_wall"]
    assert sum(s.avg_rank for s in r.slots) == pytest.approx(10.0)
    assert r.slots[0].avg_rank > 3.0                  # tsumogiri is the floor
    lines = r.table().splitlines()
    assert lines[0].startswith("Standings") and lines[2].split()[0] == "slot"
    assert len(lines) == 4 + 2 * 4 + 1                # title, rules, head; value + se rows


def test_the_standard_walls_are_the_default_and_a_base_replays(one_vs_three):
    """Routine runs share walls, so two results differ by their players only."""
    r, _ = one_vs_three
    assert r.protocol["wall_base"] == standard_walls()["wall_base"]
    rb = RulebotAgent()
    first = run_match([TsumogiriAgent(), rb, rb, rb], walls=2, wall_base=FRESH)
    assert first.protocol["wall_base"] != r.protocol["wall_base"]
    again = run_match([TsumogiriAgent(), rb, rb, rb], walls=2,
                      wall_base=first.protocol["wall_base"])
    assert np.array_equal(first.scores, again.scores)


def test_recorded_games_are_mjai_with_the_seat_names(one_vs_three):
    _, out = one_vs_three
    files = sorted((out / "logs").glob("arena-*.json.gz"))
    assert len(files) == 12
    events = [json.loads(line) for line in gzip.open(files[0], "rt")]
    assert events[0]["type"] == "start_game"
    assert sorted(events[0]["names"]) == ["rulebot", "rulebot", "rulebot", "tsumogiri"]
    assert events[1]["type"] == "start_kyoku" and events[-1]["type"] == "end_game"


def test_save_scores_round_trips(one_vs_three):
    r, out = one_vs_three
    with np.load(out / "scores.npz") as z:
        assert np.array_equal(z["scores"], r.scores)
        assert np.array_equal(z["seat_of_slot"], r.seat_of_slot)
        assert json.loads(str(z["protocol"]))["walls"] == 3
        assert len(json.loads(str(z["agents"]))) == 4


def test_2v2_plays_two_rotations_and_reports_sides(two_vs_two):
    """[A, B, A, B] repeats after a shift of two; a side, not a slot, is the row."""
    r = two_vs_two
    assert r.protocol["rotations"] == 2 and r.games == 12
    # Both sides still see every seat: slots 0,2 sit at 0,2 then at 1,3.
    assert {tuple(sorted(r.seat_of_slot[g, j] for j in (0, 2))) for g in range(2)} == {(0, 2), (1, 3)}
    sides = r.sides()
    assert len(sides) == 2 and [s.slot for s in sides] == [0, 1]
    assert sum(s.avg_rank for s in sides) == pytest.approx(5.0)
    assert sides[0].avg_rank == pytest.approx((r.slots[0].avg_rank + r.slots[2].avg_rank) / 2)
    assert sides[0].avg_rank - sides[1].avg_rank == pytest.approx(r.paired()["rank"])
    lines = r.table().splitlines()
    assert lines[2].split()[0] == "side" and "paired over" in lines[-1]
    assert len(lines) == 4 + 2 * 2 + 2                # two sides; closing rule, paired note


def test_rank_share_sums_to_one_per_side(two_vs_two):
    """A side's four shares are its seats' finishes, so they add up to 1."""
    r = two_vs_two
    names, share, se = r.rank_share()
    assert names == [s.name for s in r.sides()] and share.shape == se.shape == (2, 4)
    assert np.allclose(share.sum(1), 1.0)
    assert share[0, 0] == 0.0 and share[1, 0] > 0     # tsumogiri never wins a seat
    implied = (share * np.arange(1, 5)).sum(1)
    assert implied == pytest.approx([s.avg_rank for s in r.sides()])


def test_paired_diff_matches_raw_ranks():
    """A 2v2's difference is taken per wall, where both sides saw the same deal."""
    rb = RulebotAgent()
    r = run_match([TsumogiriAgent(), rb, TsumogiriAgent(), rb], walls=6, flight=12)
    d = r.paired()
    # Tsumogiri hardly ever beats the rulebot: close to a flat +2 rank gap.
    assert d["walls"] == 6 and 1.5 < d["rank"] <= 2.0 and d["pt"] < 0
    rows = np.arange(len(r.scores))
    order = np.argsort(np.argsort(-r.scores + np.arange(4) * 1e-6, axis=1), axis=1) + 1

    def side(js):
        return np.mean([order[rows, r.seat_of_slot[:, j]] for j in js], axis=0)

    assert d["rank"] == pytest.approx(float((side((0, 2)) - side((1, 3))).mean()))
    a = TsumogiriAgent()
    d = run_match([a, a, a, a], walls=2, flight=8).paired()
    assert d["rank"] == pytest.approx(0.0) and d["rank_se"] == 0.0 and d["pt"] == pytest.approx(0.0)


def test_2v2_shift_of_two_replays_the_game():
    """Why two rotations are enough -- and the guard if that ever stops holding.

    Shifting a `[A, B, A, B]` seating by two puts the same agent back at every
    seat. With split walls the draws are pinned to the seat, so the core plays
    the identical hanchan. If a change ever made this false, a 2v2 would be
    quietly losing half its data, so it is asserted here rather than assumed.
    """
    a, b = TsumogiriAgent(), RulebotAgent()
    slot_agent = [0, 1, 0, 1]
    seeds, maps, blobs = [], [], []
    for w in range(4):
        for rot in range(4):
            seeds.append(1000 + w)
            maps.append(tuple(slot_agent[(seat - rot) % 4] for seat in range(4)))
            blobs.append(blob_for_wall(1000 + w))
    scores, _ = play_games(seeds, maps, [a, b], wall_blobs=blobs, split_wall=True, flight=16)
    s = np.array(scores).reshape(4, 4, 4)          # (wall, rotation, seat)
    assert np.array_equal(s[:, 2], s[:, 0]) and np.array_equal(s[:, 3], s[:, 1])
    assert not np.array_equal(s[:, 1], s[:, 0])    # the +1 shift is a real game


def test_uma_rule_matches_the_v01_default():
    assert rank_of([25000, 30000, 30000, 15000]).tolist() == [2, 0, 1, 3]
    assert DEFAULT_RULE.uma_eff.tolist() == [110.0, 45.0, 0.0, -135.0]
    pt = DEFAULT_RULE.final_pt([40000, 30000, 20000, 10000])
    assert pt == pytest.approx([1.2, 0.45, -0.1, -1.55]) and np.isclose(pt.sum(), 0.0, atol=1e-6)
    assert UmaRule(uma=(20.0, 10.0, -10.0, -20.0)).uma_eff[0] == pytest.approx(40.0)


def _kyoku_starts(path) -> tuple[list[dict], list[str]]:
    events = [json.loads(line) for line in gzip.open(path, "rt")]
    return [e for e in events if e["type"] == "start_kyoku"], [e["type"] for e in events]


@pytest.mark.parametrize("players", ["rulebot4", "suzume1v3"])
def test_single_kyoku_games_are_one_east_1_kyoku(tmp_path, players):
    """Every game is East 1, honba 0, 25,000 each, seat 0 dealing, and ends there."""
    tg = TsumogiriAgent()
    agents = [RulebotAgent()] * 4 if players == "rulebot4" else [SuzumeAgent(), tg, tg, tg]
    r = run_match(agents, walls=3, single_kyoku=True, record_logs=tmp_path,
                  save_scores=tmp_path / "s.npz")
    assert r.games == 12 and r.protocol["game_mode"] == 0 and r.protocol["single_kyoku"]
    files = sorted(tmp_path.glob("arena-*.json.gz"))
    assert len(files) == 12
    for f in files:
        starts, types = _kyoku_starts(f)
        assert len(starts) == 1 and types[0] == "start_game"
        assert types[-2:] == ["end_kyoku", "end_game"] and types.count("end_kyoku") == 1
        s = starts[0]
        assert (s["bakaze"], s["kyoku"], s["honba"], s["kyotaku"], s["oya"]) == ("E", 1, 0, 0, 0)
        assert s["scores"] == [25000] * 4
    stats = r.kyoku_stats()
    assert len(stats) == 4 and all({"delta", "win", "deal_in"} <= set(row) for row in stats)
    rows = np.arange(r.games)
    with np.load(tmp_path / "s.npz") as z:
        assert np.array_equal(z["scores"], r.scores) and np.array_equal(z["win"], r.kyoku["win"])
        d = z["scores"] - 25000
        assert stats[0]["delta"] == pytest.approx(d[rows, r.seat_of_slot[:, 0]].mean())
    if players == "suzume1v3":
        assert not r.kyoku["win"][rows[:, None], r.seat_of_slot[:, 1:]].any()  # tsumogiri


def test_single_kyoku_is_the_hanchans_east_1(tmp_path):
    """Same wall, same seating: the kyoku replays a hanchan's first kyoku event for event."""
    rb = RulebotAgent()
    players = [TsumogiriAgent(), rb, rb, rb]
    run_match(players, walls=2, single_kyoku=True, record_logs=tmp_path / "e1")
    run_match(players, walls=2, record_logs=tmp_path / "hc")
    for f in sorted((tmp_path / "e1").glob("arena-*.json.gz")):
        a = gzip.open(f, "rt").read().splitlines()
        b = gzip.open(tmp_path / "hc" / f.name, "rt").read().splitlines()
        cut = [next(i for i, line in enumerate(x) if '"end_kyoku"' in line) for x in (a, b)]
        assert a[:cut[0]] == b[:cut[1]]


def test_the_rust_and_python_tsumogiri_play_the_same_games():
    from yamawake import PythonTsumogiriAgent
    rb = RulebotAgent()
    for seating in (lambda t: [t, rb, rb, rb], lambda t: [t, t, t, t]):
        rust = run_match(seating(TsumogiriAgent()), walls=4)
        py = run_match(seating(PythonTsumogiriAgent()), walls=4)
        assert (rust.scores == py.scores).all()
