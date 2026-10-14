"""Convert an mjai log into the tenhou.net/6 JSON replay format.

The arena records its games as mjai (`match._write_logs`, the corpus
format). That is the right thing to train on and the wrong thing to *watch*:
to look at a hand you want the tenhou replay viewer, which reads the
"tenhou.net/6" JSON dialect. This module is the bridge.

    from yamawake.tenhou6 import convert_file, dump
    dump(convert_file("game.json.gz"), "game.json")

Then open https://tenhou.net/6/ and paste the file, or serve it and use
`https://tenhou.net/6/#json={...}`.

## The format, briefly

One object per game::

    {"title": [...], "name": [...], "rule": {...}, "sc": [...], "log": [kyoku, ...]}

Each kyoku is a flat array::

    [[kyoku_index, honba, riichi_sticks],
     [starting scores x4],
     [dora indicators], [ura indicators],
     haipai_0, draws_0, discards_0,      # by ABSOLUTE seat, seat 0 first
     haipai_1, draws_1, discards_1,
     haipai_2, draws_2, discards_2,
     haipai_3, draws_3, discards_3,
     result]

Tiles are two-digit numbers: 11-19 man, 21-29 pin, 31-39 sou, 41-47 ESWN
haku hatsu chun, 51/52/53 the red fives. A discard of 60 means "the tile just
drawn" (tsumogiri) and an `"r"` prefix marks the riichi declaration.

Calls are strings mixed into the *draw* row (chi, pon, daiminkan -- they
replace a draw) or the *discard* row (ankan, kakan -- they happen mid-turn).
One letter names the call and its POSITION says which player the tile came
from: 0 = kamicha, 1 = toimen, 2 = shimocha (3 for daiminkan and ankan, which
have four tiles)::

    chi        "c252627"      always kamicha, letter first
    pon        "p414141" / "41p4141" / "4141p41"
    daiminkan  "m39393939" / "39m393939" / "222222m22"
    kakan      "k37373737" / "31k313131" / "3737k3737"   (the added tile follows k)
    ankan      "121212a12"

## What is not recoverable

The mjai log records the score deltas of a win, not its yaku, han or fu, so
the result line carries the points only (plus the limit name when the payment
is exactly one). `riichienv`'s replay can recompute a hand's value, but on
these logs it does not agree with the deltas the engine itself produced --
it reported haitei/houtei on five of nine hands in a spot check -- so it is
not used here.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

TSUMOGIRI = 60                       # tenhou's "discarded what I just drew"
_SUIT = {"m": 10, "p": 20, "s": 30}
_HONOR = {"E": 41, "S": 42, "W": 43, "N": 44, "P": 45, "F": 46, "C": 47}

# Ryukyoku reasons as riichienv names them, in the viewer's words.
_RYUKYOKU = {
    "exhaustive_draw": "流局",
    "nagashimangan": "流し満貫",
    "kyushu_kyuhai": "九種九牌",
    "sanchaho": "三家和了",
    "sufuurenta": "四風連打",
    "suukansansen": "四槓散了",
    "suucha_riichi": "四家立直",
}
# Payment -> limit name, for the score string, keyed by (is_oya, is_tsumo).
# Only payments a limit hand alone can produce are listed: a non-dealer tsumo
# needs BOTH payments, because 30fu 4han pays 2000-3900 and its 2000 would
# otherwise read as a mangan.
_LIMIT: dict[tuple[bool, bool], dict] = {
    (False, False): {8000: "満貫", 12000: "跳満", 16000: "倍満",
                     24000: "三倍満", 32000: "役満"},
    (True, False): {12000: "満貫", 18000: "跳満", 24000: "倍満",
                    36000: "三倍満", 48000: "役満"},
    (False, True): {(2000, 4000): "満貫", (3000, 6000): "跳満",
                    (4000, 8000): "倍満", (6000, 12000): "三倍満",
                    (8000, 16000): "役満"},
    (True, True): {4000: "満貫", 6000: "跳満", 8000: "倍満",
                   12000: "三倍満", 16000: "役満"},
}
TENHOU_DISPLAY_UMA = (20.0, 10.0, -10.0, -20.0)
RETURN_POINTS = 30000


def tile_code(pai: str) -> int:
    """One mjai tile name as a tenhou tile number ("5mr" -> 51, "C" -> 47)."""
    if pai in _HONOR:
        return _HONOR[pai]
    if pai.endswith("r"):                       # 5mr / 5pr / 5sr
        return 50 + _SUIT[pai[1]] // 10
    return _SUIT[pai[1]] + int(pai[0])


def _deaka(code: int) -> int:
    """51/52/53 -> 15/25/35; anything else unchanged."""
    return 10 * (code % 10) + 5 if code // 10 == 5 else code


def _makeaka(code: int) -> int:
    """15/25/35 -> 51/52/53; anything else unchanged (honors are not fives)."""
    return 50 + code // 10 if code < 40 and code % 10 == 5 else code


def _rel(actor: int, target: int) -> int:
    """Where `target` sits from `actor`: 0 kamicha, 1 toimen, 2 shimocha."""
    return (actor - target + 3) % 4


class _Kyoku:
    """One kyoku under construction, in tenhou's row layout."""

    def __init__(self, event: dict) -> None:
        round_wind = "ESW".index(event["bakaze"])
        self.kyoku_index = 4 * round_wind + event["kyoku"] - 1
        self.honba = int(event["honba"])
        self.riichi_sticks = int(event["kyotaku"])
        self.oya = int(event["oya"])
        self.scores = [int(s) for s in event["scores"]]
        self.dora_indicators = [tile_code(event["dora_marker"])]
        self.ura_indicators: list[int] = []
        self.haipai = [[tile_code(t) for t in h] for h in event["tehais"]]
        self.draws: list[list[Any]] = [[] for _ in range(4)]
        self.discards: list[list[Any]] = [[] for _ in range(4)]
        self.result: list[Any] = []
        self.deltas = [0, 0, 0, 0]
        self.riichi_stage = [False] * 4
        self.in_riichi = [False] * 4  # riichi sticks laid down during this kyoku
        self.first_win = True

    def rows(self) -> list[Any]:
        """The kyoku as the viewer wants it: header, then seat by seat."""
        out: list[Any] = [[self.kyoku_index, self.honba, self.riichi_sticks], self.scores,
                          self.dora_indicators, self.ura_indicators]
        for s in range(4):
            out += [self.haipai[s], self.draws[s], self.discards[s]]
        out.append(self.result)
        return out

    # -- the events that build a hand -------------------------------------

    def dahai(self, event: dict) -> None:
        actor = event["actor"]
        sym: Any = TSUMOGIRI if event.get("tsumogiri") else tile_code(event["pai"])
        if self.riichi_stage[actor]:
            self.riichi_stage[actor] = False
            sym = f"r{sym}"
        self.discards[actor].append(sym)

    def chi(self, event: dict) -> None:
        # Chi is only ever from kamicha, so the letter is always first.
        c = [tile_code(t) for t in event["consumed"]]
        self.draws[event["actor"]].append(f"c{tile_code(event['pai'])}{c[0]}{c[1]}")

    def pon(self, event: dict) -> None:
        tiles = [str(tile_code(t)) for t in event["consumed"]]
        tiles.insert(_rel(event["actor"], event["target"]), f"p{tile_code(event['pai'])}")
        self.draws[event["actor"]].append("".join(tiles))

    def daiminkan(self, event: dict) -> None:
        tiles = [str(tile_code(t)) for t in event["consumed"]]
        idx = _rel(event["actor"], event["target"])
        tiles.insert(3 if idx == 2 else idx, f"m{tile_code(event['pai'])}")
        self.draws[event["actor"]].append("".join(tiles))
        self.discards[event["actor"]].append(0)     # tenhou marks the lost turn

    def ankan(self, event: dict) -> None:
        t = [str(tile_code(x)) for x in event["consumed"]]
        self.discards[event["actor"]].append(f"{t[0]}{t[1]}{t[2]}a{t[3]}")

    def kakan(self, event: dict) -> None:
        # The pon string stays in the draw row; its copy in the discard row
        # gets the "p" swapped for "k" plus the tile just added.
        added = tile_code(event["pai"])
        keys = (f"p{_deaka(added)}", f"p{_makeaka(added)}")
        for meld_code in self.draws[event["actor"]]:
            if isinstance(meld_code, str) and any(k in meld_code for k in keys):
                self.discards[event["actor"]].append(
                    meld_code.replace("p", f"k{added}", 1))
                return
        raise ValueError(f"kakan on {event['pai']} with no matching pon")

    # -- the events that end one ------------------------------------------

    def hora(self, event: dict) -> None:
        deltas = [int(d) for d in event["deltas"]]
        self.deltas = [a + b for a, b in zip(self.deltas, deltas)]
        if event.get("ura_markers") and not self.ura_indicators:
            self.ura_indicators = [tile_code(t) for t in event["ura_markers"]]
        actor, target = int(event["actor"]), int(event["target"])
        text = self._score_text(deltas, actor, target)
        if not self.result:
            self.result = ["和了"]
        self.result += [deltas, [actor, target, actor, text]]
        self.first_win = False

    def _score_text(self, deltas: list[int], actor: int, target: int) -> str:
        """The winning hand's value, read back out of the deltas.

        The viewer only requires a string ending in 点, 飜) or 役満); the mjai
        log does not carry han or fu, so this is the payment and, when the
        payment names a limit hand exactly, its name.
        """
        # Honba and the riichi sticks ride on the first win of the kyoku only.
        hb = self.honba if self.first_win else 0
        is_oya = actor == self.oya
        if actor == target:                                   # tsumo
            ko = next(-deltas[s] - 100 * hb for s in range(4)
                      if s != actor and s != self.oya)
            if is_oya:
                body, key = f"{ko}点∀", ko
            else:
                po = -deltas[self.oya] - 100 * hb
                body, key = f"{ko}-{po}点", (ko, po)
        else:                                                 # ron
            key = -deltas[target] - 300 * hb
            body = f"{key}点"
        return _LIMIT[(is_oya, actor == target)].get(key, "") + body

    def ryukyoku(self, event: dict) -> None:
        deltas = [int(d) for d in event["deltas"]]
        self.deltas = deltas
        self.result = [_RYUKYOKU.get(event.get("reason", ""), "流局"), deltas]


def to_tenhou6(events: list[dict], *, names: list[str] | None = None,
               ref: str = "", title: str = "", rule_disp: str = "般南喰赤",
               final_scores: list[int] | None = None) -> dict:
    """Build the tenhou.net/6 object for one game's mjai events.

    Args:
        events: The game's mjai events, in order, starting at `start_game`.
        names: Player names, by seat; the log's own `start_game` names by default.
        ref: Free-form game id, shown by the viewer.
        title: Second title line; the first is the rule display.
        rule_disp: Rule string in the title bar.
        final_scores: Final points per seat. Computed from the log when they
            are not given.

    Returns:
        A JSON-serializable dict.

    Raises:
        ValueError: The events do not parse as one complete game.
    """
    if not events or events[0].get("type") != "start_game":
        raise ValueError("an mjai log starts with start_game")
    players = list(names or events[0].get("names") or [f"seat{i}" for i in range(4)])
    log: list[Any] = []
    k: _Kyoku | None = None
    scores, riichi_sticks = [25000] * 4, 0

    calls = {"chi": "chi", "pon": "pon", "daiminkan": "daiminkan",
             "ankan": "ankan", "kakan": "kakan"}
    for event in events[1:]:
        t = event["type"]
        if t == "start_kyoku":
            k = _Kyoku(event)
            scores = list(k.scores)
            continue
        if k is None:
            if t in ("end_game", "start_game"):
                continue
            raise ValueError(f"{t} outside a kyoku")
        if t == "tsumo":
            k.draws[event["actor"]].append(tile_code(event["pai"]))
        elif t == "dahai":
            k.dahai(event)
        elif t == "reach":
            k.riichi_stage[event["actor"]] = True
        elif t == "reach_accepted":
            k.in_riichi[event["actor"]] = True
        elif t in calls:
            getattr(k, calls[t])(event)
        elif t == "dora":
            k.dora_indicators.append(tile_code(event["dora_marker"]))
        elif t == "hora":
            k.hora(event)
        elif t == "ryukyoku":
            k.ryukyoku(event)
        elif t == "end_kyoku":
            log.append(k.rows())
            # A hora's deltas already hand the winner the sticks on the table,
            # but never charge the 1000 each declaration cost; a draw charges
            # nothing and carries them. So settle the declarations here.
            scores = [s + d - 1000 * k.in_riichi[i]
                      for i, (s, d) in enumerate(zip(k.scores, k.deltas))]
            riichi_sticks = k.riichi_sticks + sum(k.in_riichi) - sum(k.deltas) // 1000
            k = None
        # end_game and anything else carry nothing we draw.

    if riichi_sticks > 0:                   # tenhou gives what is left to the leader
        scores[max(range(4), key=lambda s: (scores[s], -s))] += 1000 * riichi_sticks
    final = list(final_scores) if final_scores is not None else scores
    return {"ver": "2.3", "ref": ref, "ratingc": "PF4",
            "title": [rule_disp, title],
            "name": players, "rule": {"disp": rule_disp, "aka51": 1, "aka52": 1,
                                    "aka53": 1},
            "dan": [""] * 4, "rate": [0] * 4, "sx": ["C"] * 4,
            "sc": _sc(final), "log": log}


def _sc(scores: list[int]) -> list[float]:
    """[points, result] per seat, with tenhou's displayed uma and 30000 back."""
    # Ties go to the earlier seat, as tenhou settles them.
    order = sorted(range(4), key=lambda s: (-scores[s], s))
    rank = {s: r for r, s in enumerate(order)}
    out: list[float] = []
    for s in range(4):
        pt = (scores[s] - RETURN_POINTS) / 1000.0 + TENHOU_DISPLAY_UMA[rank[s]]
        out += [scores[s], round(pt, 1)]
    return out


def load_mjai(path: str | Path) -> list[dict]:
    """Read an mjai log (plain or gzip) into its events."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_file(path: str | Path, **kwargs) -> dict:
    """Convert one mjai log file; `ref` defaults to its name."""
    kwargs.setdefault("ref", Path(path).name.split(".")[0])
    return to_tenhou6(load_mjai(path), **kwargs)


def dump(obj: dict, path: str | Path) -> Path:
    """Write one converted game as compact JSON the viewer can read."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    return out
