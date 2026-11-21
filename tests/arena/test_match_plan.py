"""match_plan: run_match's games, to run in parts of one's own (requested by tide)."""

import numpy as np
import pytest

from yamawake import RulebotAgent, TsumogiriAgent, lineup, match_plan, play_games, run_match


@pytest.mark.parametrize("mode", ["1v3", "2v2"])
def test_the_plan_is_run_matchs_seating(mode):
    players = lineup(TsumogiriAgent(), RulebotAgent(), mode)
    plan = match_plan(players, walls=3, wall_base=7)
    cycle = 4 if mode == "1v3" else 2
    assert (plan.seat_cycle, plan.rotations, len(plan.games)) == (cycle, cycle, 3 * cycle)
    assert [(g.wall, g.rotation, g.seed) for g in plan.games] == [(w, r, 7 + w) for w in range(3) for r in range(cycle)]
    for g in plan.games:
        assert g.seat_of_slot == [(j + g.rotation) % 4 for j in range(4)]
        assert all(g.seat_agent[g.seat_of_slot[j]] == plan.slot_agent[j] for j in range(4))
    once = match_plan(players, walls=5, wall_base=7, rotate=False)
    assert [(g.wall, g.rotation) for g in once.games] == [(w, w % cycle) for w in range(5)] and once.rotations == 1
    with pytest.raises(ValueError):
        match_plan(players, wall_base=7, wall_seed=1)


@pytest.mark.parametrize("mode,rotate", [("1v3", True), ("2v2", True), ("1v3", False)])
def test_the_plan_played_in_parts_gives_run_matchs_scores(mode, rotate):
    players = lineup(TsumogiriAgent(), RulebotAgent(), mode)
    r = run_match(players, walls=3, wall_base=11, rotate=rotate)
    plan = match_plan(players, walls=3, wall_base=11, rotate=rotate)
    half = len(plan.games) // 2
    parts = [play_games(**plan.play_args(gs))[0] for gs in (plan.games[:half], plan.games[half:])]
    assert (np.array(parts[0] + parts[1]) == r.scores).all()
    assert (np.array([g.seat_of_slot for g in plan.games]) == r.seat_of_slot).all()
    assert {k: r.protocol[k] for k in plan.protocol} == plan.protocol
