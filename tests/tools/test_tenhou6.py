"""The mjai -> tenhou.net/6 conversion."""

from __future__ import annotations

import json

import pytest

from yamawake.tenhou6 import convert_file, load_mjai, tile_code, to_tenhou6


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Two arena hanchan recorded as mjai: the logs the converter is for."""
    from yamawake import RulebotAgent, TsumogiriAgent, run_match

    out = tmp_path_factory.mktemp("logs")
    rb = RulebotAgent()
    run_match([TsumogiriAgent(), rb, rb, rb], walls=1, record_logs=out)
    return sorted(out.glob("*.json.gz"))[:2]


def test_tile_codes():
    assert [tile_code(f"{n}m") for n in range(1, 10)] == list(range(11, 20))
    assert [tile_code(f"{n}p") for n in range(1, 10)] == list(range(21, 30))
    assert [tile_code(f"{n}s") for n in range(1, 10)] == list(range(31, 40))
    assert [tile_code(t) for t in "ESWNPFC"] == list(range(41, 48))
    assert [tile_code(t) for t in ("5mr", "5pr", "5sr")] == [51, 52, 53]


def _game(*events: dict) -> dict:
    """A one-kyoku game around `events`, so the converter has something to do."""
    start = {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
             "kyotaku": 0, "oya": 0, "dora_marker": "1m",
             "scores": [25000] * 4, "tehais": [["1m"] * 13 for _ in range(4)]}
    end = [{"type": "ryukyoku", "reason": "exhaustive_draw",
            "deltas": [0, 0, 0, 0]}, {"type": "end_kyoku"},
           {"type": "end_game"}]
    return to_tenhou6([{"type": "start_game", "names": list("ABCD")},
                       start, *events, *end])


def _rows(obj: dict, seat: int) -> tuple[list, list]:
    """(draws, discards) of one seat in the only kyoku."""
    k = obj["log"][0]
    return k[5 + 3 * seat], k[6 + 3 * seat]


def test_chi_is_always_kamicha_first():
    # Actor 1 chis a 5p from actor 0 with 6p7p.
    obj = _game({"type": "chi", "actor": 1, "target": 0, "pai": "5pr",
                 "consumed": ["6p", "7p"]})
    assert _rows(obj, 1)[0] == ["c522627"]


@pytest.mark.parametrize("target,expected", [
    (0, "p414141"),        # kamicha of actor 1
    (3, "41p4141"),        # toimen
    (2, "4141p41"),        # shimocha
])
def test_pon_letter_marks_the_discarder(target, expected):
    obj = _game({"type": "pon", "actor": 1, "target": target, "pai": "E",
                 "consumed": ["E", "E"]})
    assert _rows(obj, 1)[0] == [expected]


@pytest.mark.parametrize("target,expected", [
    (0, "m39393939"),
    (3, "39m393939"),
    (2, "393939m39"),
])
def test_daiminkan_letter_marks_the_discarder(target, expected):
    obj = _game({"type": "daiminkan", "actor": 1, "target": target, "pai": "9s",
                 "consumed": ["9s", "9s", "9s"]})
    draws, discards = _rows(obj, 1)
    assert draws == [expected]
    # A daiminkan costs the turn's discard, which tenhou writes as a 0.
    assert discards == [0]


def test_ankan_and_kakan_in_discard_row():
    obj = _game({"type": "ankan", "actor": 2, "consumed": ["1p"] * 4})
    assert _rows(obj, 2) == ([], ["212121a21"])

    # Actor 2 pons off its toimen (seat 0), then adds the fourth tile.
    obj = _game({"type": "pon", "actor": 2, "target": 0, "pai": "1p",
                 "consumed": ["1p", "1p"]},
                {"type": "kakan", "actor": 2, "pai": "1p",
                 "consumed": ["1p"] * 3})
    draws, discards = _rows(obj, 2)
    assert draws == ["21p2121"]            # the pon stays where it was
    assert discards == ["21k212121"]       # the added tile follows the k


def test_kakan_finds_its_pon_with_a_red_five():
    """The pon holds a red five, the added tile is a normal one."""
    obj = _game({"type": "pon", "actor": 0, "target": 3, "pai": "5p",
                 "consumed": ["5pr", "5p"]},
                {"type": "kakan", "actor": 0, "pai": "5p",
                 "consumed": ["5pr", "5p", "5p"]})
    draws, discards = _rows(obj, 0)
    assert draws == ["p255225"]
    assert discards == ["k25255225"]


def test_riichi_and_tsumogiri_markers():
    obj = _game({"type": "reach", "actor": 3},
                {"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": True},
                {"type": "reach_accepted", "actor": 3},
                {"type": "dahai", "actor": 3, "pai": "9s", "tsumogiri": False})
    assert _rows(obj, 3)[1] == ["r60", 39]


def test_scores_carry_between_kyokus(recorded):
    """Settling kyoku i must reproduce the scores the log records for i+1.

    This is the end-to-end check on the result rows: the deltas, the riichi
    sticks each declaration puts on the table, and who collects them.
    """
    for path in recorded:
        events = load_mjai(path)
        starts = [i for i, e in enumerate(events) if e["type"] == "start_kyoku"]
        ends = [i for i, e in enumerate(events) if e["type"] == "end_kyoku"]
        assert len(starts) == len(ends) >= 2

        for i in range(len(ends) - 1):
            partial = to_tenhou6(events[:ends[i] + 1] + [{"type": "end_game"}])
            got = partial["sc"][0::2]
            nxt = events[starts[i + 1]]
            want = list(nxt["scores"])
            # A truncated game hands the sticks still on the table to the leader;
            # in the real game they stayed there, so put them back.
            if nxt["kyotaku"]:
                want[max(range(4), key=lambda s: (want[s], -s))] += 1000 * nxt["kyotaku"]
            assert got == want, f"kyoku {i} -> {i + 1}"


def test_log_shape_for_the_viewer(recorded):
    for path in recorded:
        obj = convert_file(path)
        assert json.loads(json.dumps(obj))            # serializable
        assert len(obj["name"]) == 4 and len(obj["sc"]) == 8
        assert sum(obj["sc"][0::2]) == 100000
        for k in obj["log"]:
            assert len(k) == 17
            assert len(k[0]) == 3 and len(k[1]) == 4
            for seat in range(4):
                assert len(k[4 + 3 * seat]) == 13      # haipai, before the draw
            assert k[16] and isinstance(k[16][0], str)


def _win(deltas: list[int], actor: int, target: int, oya: int = 0,
         honba: int = 0) -> str:
    """The score string a single win produces."""
    start = {"type": "start_kyoku", "bakaze": "E", "kyoku": oya + 1,
             "honba": honba, "kyotaku": 0, "oya": oya, "dora_marker": "1m",
             "scores": [25000] * 4, "tehais": [["1m"] * 13 for _ in range(4)]}
    obj = to_tenhou6([{"type": "start_game", "names": list("ABCD")}, start,
                      {"type": "hora", "actor": actor, "target": target,
                       "deltas": deltas, "ura_markers": []},
                      {"type": "end_kyoku"}, {"type": "end_game"}])
    return obj["log"][0][16][2][3]


@pytest.mark.parametrize("deltas,actor,target,oya,honba,want", [
    # 30fu 4han is 2000-3900, NOT a mangan, even though 2000 is a mangan's
    # non-dealer payment -- both payments have to match.
    ([-3900, -2000, -2000, 7900], 3, 3, 0, 0, "2000-3900点"),
    ([-4000, -2000, -2000, 8000], 3, 3, 0, 0, "満貫2000-4000点"),
    ([12000, -4000, -4000, -4000], 0, 0, 0, 0, "満貫4000点∀"),
    ([8000, 0, -8000, 0], 0, 2, 1, 0, "満貫8000点"),
    ([0, 0, -2600, 2600], 3, 2, 0, 0, "2600点"),
    # One honba adds 300 to a ron and 100 to each tsumo payment; the hand
    # itself is still a 2600 and a 2000-3900.
    ([0, 0, -2900, 2900], 3, 2, 0, 1, "2600点"),
    ([-4000, -2100, -2100, 8200], 3, 3, 0, 1, "2000-3900点"),
])
def test_score_string_from_deltas(deltas, actor, target,
                                                     oya, honba, want):
    assert _win(deltas, actor, target, oya, honba) == want
