"""Statistics of a player from recorded arena logs: speed, value, riichi, calls, wins, deal-ins, results.

    python -m yamawake.handstats LOGDIR --name suzume
    python -m yamawake.handstats LOGDIR --name suzume --vs OTHER --vs-name rulebot \\
        --json out.json

Reads the gzipped mjai games `--record-logs` writes (`arena-<wall_base>-NNNNNN
.json.gz`) and, for every kyoku of every seat whose player name (`start_game`) is
`--name`, measures how fast the hand got to tenpai, what it was worth there,
how it riichi'd, called, won and dealt in, and how each hanchan ended. Post
hoc only: nothing about the arena changes.

## Definitions

- **Turn.** The seat's own draws in the kyoku so far (every `tsumo`,
  rinshan included). Tenpai on the deal is turn 0; tenpai reached by the
  discard after the n-th draw is turn n, and so is tenpai reached by the
  discard after a call (a call is not a draw).
- **Tenpai.** Shanten 0 (`riichienv.calculate_shanten`, which also counts
  chiitoitsu and kokushi) of the 13 - 3 x melds concealed tiles after the
  seat's discard, or of the deal. The first such moment is the *first
  tenpai*; a kyoku with none is censored (it counts as not tenpai, also when
  another seat ended the kyoku early).
- **Value at tenpai.** At the first tenpai, for each wait (34 tile types,
  `HandEvaluator.get_waits`), the points the hand scores if it wins on that
  tile, by ron and by tsumo, averaged with weights `4 - visible` (copies the
  seat cannot see: not in its hand, a discard, a meld or a dora indicator).
  Riichi counts iff the discard that reached tenpai was the seat's riichi
  declaration (so a dama tenpai that riichis later is valued as dama).
  Dora and aka count, ura does not; no ippatsu, haitei or similar; honba
  and riichi sticks are 0. A wait that has no yaku scores 0. A tenpai whose
  waits are all dead (no unseen copies) has no value and is reported apart.
- **Yakuless.** No wait wins by ron without riichi: a closed hand can then
  only win by riichi or menzen tsumo.
- **Outcomes.** Win points are the seat's `hora` delta minus the riichi
  sticks it collected (honba kept); deal-in points the ron payment (both
  rons of a double ron). Riichi is a `reach` by the seat, a call any chi,
  pon or daiminkan. Ryukyoku tenpai is shanten 0 at an exhaustive draw.
  The kyoku delta is the seat's deltas over the kyoku's `hora` and
  `ryukyoku` events minus 1000 per accepted riichi.
- **Riichi.** First (senseki) when no other seat had an accepted riichi
  before it, else a chase (oikake). Its turn is the seat's draws at the
  declaration; its waits are counted on the declaring discard (unseen
  copies over the wait types; good shape = two or more wait types). After a
  riichi the kyoku ends in the seat's win, its deal-in, an exhaustive draw
  or something else.
- **Dama.** A first tenpai that is closed and never riichi'd in the kyoku.
- **Calls.** A called kyoku has a chi, pon or daiminkan by the seat; its
  call count is those calls.
- **Wins.** Han, fu, the yaku and the limit (below mangan, mangan, haneman,
  baiman, sanbaiman, yakuman, counted yakuman included), as the arena's
  engine scored them (its `hora` events carry `han`, `fu`, `yaku`,
  `yakuman`). A log without them (an older engine) is re-scored with
  riichienv's evaluator from the tracked hand (ura dora from the log's
  `ura_markers`; riichi, double riichi, ippatsu, haitei / houtei, rinshan,
  chankan and tenhou / chiihou from the play).
  The win turn is the seat's draws at the win.
- **Deal-ins.** By the winner's state (riichi, open hand, else dama) and the
  seat's own (riichi, open, else closed). Another seat's tsumo costs the
  seat its payment (`tsumo_by_other`); a ron between two other seats is a
  side ron (yokoido).
- **Renchan.** A kyoku the seat deals that the next kyoku is dealt by it
  again (the last kyoku of a game is left out).
- **Hanchan.** The seat's final rank (ties to the lower seat, as the arena
  ranks) and score: the last `start_kyoku` scores plus the riichi deposits
  and deltas after it (riichi sticks left on the table at the end are not
  given out). Bust: a final score below 0.

## Paired comparison

Two runs on the same walls and seatings (same `--wall-base`, same lineup
positions) write the same file names, and a seat's draws depend only on the
wall. A kyoku is paired by (file, round wind, kyoku, honba, seat) and kept
only when the seat's deal (haipai) is identical on both sides. Each metric
is compared on the pairs where both sides define it (e.g. a first-tenpai
turn needs tenpai on both), and both sides' absolute means are printed with
the paired difference. Hanchan results pair by (file, seat). Standard errors
are clustered by game file.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import riichienv as rv
# The core evaluator, not riichienv.hand's wrapper: it takes the 14-tile hand
# and a `Wind`-typed Conditions directly, at a fraction of the cost.
from riichienv._riichienv import Conditions, HandEvaluator, Meld, MeldType, Wind

from yamawake.context import LIVE_WALL
from yamawake.tenhou6 import load_mjai

MAX_TURN = 18                       # the tenpai-rate curve runs over turns 1..MAX_TURN
WINDS = (Wind.East, Wind.South, Wind.West, Wind.North)
ROUND_WIND = {"E": 0, "S": 1, "W": 2, "N": 3}
HONORS = "ESWNPFC"
MELD_TYPE = {"chi": MeldType.Chi, "pon": MeldType.Pon, "daiminkan": MeldType.Daiminkan,
             "ankan": MeldType.Ankan, "kakan": MeldType.Kakan}


def tile_type(pai: str) -> int:
    """34-type index of an mjai tile ("5mr" -> 4, "E" -> 27)."""
    if pai[0] in HONORS and len(pai) == 1:
        return 27 + HONORS.index(pai)
    return "mps".index(pai[1]) * 9 + int(pai[0]) - 1


def to_tids(groups: Iterable[list[str]]) -> list[list[int]]:
    """136 ids for mjai tiles, distinct across all groups (hand, then melds).

    A red five is its type's copy 0 (riichienv's aka id); other copies take
    the next free id, so a hand never holds the same id twice.
    """
    used = [0] * 34                      # non-red copies taken per type
    out = []
    for g in groups:
        ids = []
        for pai in g:
            t = tile_type(pai)
            if pai.endswith("r"):
                ids.append(4 * t)
            else:
                five = t < 27 and t % 9 == 4
                ids.append(4 * t + used[t] + (1 if five else 0))
                used[t] += 1
        out.append(ids)
    return out


@dataclass
class KyokuStat:
    """One seat's kyoku: speed, value at tenpai and outcome.

    `tenpai_turn` is None when the seat never reached tenpai; the value fields
    are None without tenpai, or (`ev_ron`, `ev_tsumo`) when every wait is dead.
    Point fields are None unless that event happened.
    """

    game: str                      # log file stem, e.g. arena-0-000012
    round_wind: int
    kyoku: int
    honba: int
    seat: int
    haipai: tuple[str, ...]        # sorted deal
    tenpai_turn: int | None = None
    riichi_at_tenpai: bool | None = None
    closed_at_tenpai: bool | None = None
    ev_ron: float | None = None
    ev_tsumo: float | None = None
    yakuless: bool | None = None
    dead_waits: bool | None = None
    won: bool = False
    win_ron: int | None = None
    win_tsumo: int | None = None
    deal_in: int | None = None
    riichi: bool = False
    called: bool = False
    exhaustive: bool = False
    ryukyoku_tenpai: bool | None = None
    delta: int = 0
    oya: bool = False
    renchan: bool | None = None
    riichi_turn: int | None = None
    riichi_first: bool | None = None
    riichi_waits: int | None = None
    riichi_good: bool | None = None
    dama: bool | None = None
    n_calls: int = 0
    win_turn: int | None = None
    win_han: int | None = None
    win_fu: int | None = None
    win_limit: str | None = None
    win_yaku: tuple[str, ...] | None = None
    win_rescored: int | None = None    # the re-scored points (no honba, no sticks)
    ippatsu: bool | None = None
    ura: bool | None = None
    deal_in_to: str | None = None      # "riichi", "open" or "dama"
    deal_in_as: str | None = None      # "riichi", "open" or "closed"
    tsumo_by_other: int | None = None
    side_ron: bool = False

    @property
    def key(self) -> tuple:
        """The pairing key: same file, kyoku, honba and seat."""
        return (self.game, self.round_wind, self.kyoku, self.honba, self.seat)


@dataclass
class GameStat:
    """One seat's hanchan result."""

    game: str
    seat: int
    rank: int                      # 1..4
    score: int                     # final score

    @property
    def key(self) -> tuple:
        """The pairing key: same file and seat."""
        return (self.game, self.seat)


def tenpai_value(hand: list[str], melds: list[tuple[str, list[str], int]],
                 visible: list[int], dora_indicators: list[str], player_wind: int,
                 round_wind: int, riichi: bool) -> tuple[float | None, float | None, bool, bool]:
    """Expected points of a 13-tile tenpai hand over its live waits.

    Args:
        hand: The concealed tiles (mjai).
        melds: `(mjai type, tiles, from seat)` per meld.
        visible: Copies of each tile type the seat can see (hand included).
        dora_indicators: Indicators turned so far (mjai).
        player_wind: 0..3, the seat's wind (0 = the oya).
        round_wind: 0..3.
        riichi: Score with riichi.

    Returns:
        `(ev_ron, ev_tsumo, yakuless, dead_waits)`; the EVs are None when no
        wait has an unseen copy.
    """
    groups = to_tids([hand, *[m[1] for m in melds]])
    hand_ids = groups[0]
    taken = {i for g in groups for i in g}
    rv_melds = [Meld(MELD_TYPE[kind], ids, kind != "ankan", who)
                for (kind, _, who), ids in zip(melds, groups[1:])]
    dora = to_tids([dora_indicators])[0]
    waits = HandEvaluator(hand_ids, rv_melds).get_waits()
    oya = player_wind == 0
    total = ron = tsumo = 0.0
    yakuless = True
    for w in waits:
        win = next((i for i in range(4 * w, 4 * w + 4)
                    if i not in taken and not (i == 4 * w and w < 27 and w % 9 == 4)),
                   None)
        if win is None:                      # every non-red copy is in the hand
            continue
        tiles = sorted(hand_ids + [win])

        def calc(is_tsumo: bool, with_riichi: bool):
            cond = Conditions(tsumo=is_tsumo, riichi=with_riichi,
                              player_wind=WINDS[player_wind], round_wind=WINDS[round_wind])
            return HandEvaluator(tiles, rv_melds).calc(win, dora, [], cond)

        dama = calc(False, False)
        yakuless &= not dama.is_win
        n = max(0, 4 - visible[w])
        if n == 0:
            continue
        r = calc(False, True) if riichi else dama
        t = calc(True, riichi)
        total += n
        ron += n * (r.ron_agari if r.is_win else 0)
        if t.is_win:
            tsumo += n * (3 * t.tsumo_agari_ko if oya
                          else t.tsumo_agari_oya + 2 * t.tsumo_agari_ko)
    if total == 0:
        return None, None, yakuless, True
    return ron / total, tsumo / total, yakuless, False


#: Win limits, lowest first (`limit_of`).
LIMITS = ("below_mangan", "mangan", "haneman", "baiman", "sanbaiman", "yakuman")
YAKU_IPPATSU, YAKU_URA = 30, 33
LIVE_DRAWS = LIVE_WALL              # tsumo events (rinshan included) in a full kyoku


def limit_of(han: int, fu: int, yakuman: bool) -> str:
    """The limit class of a win's han and fu (`LIMITS`)."""
    if yakuman or han >= 13:
        return "yakuman"
    if han >= 11:
        return "sanbaiman"
    if han >= 8:
        return "baiman"
    if han >= 6:
        return "haneman"
    if han >= 5 or (han == 4 and fu >= 40) or (han == 3 and fu >= 70):
        return "mangan"
    return "below_mangan"


def win_value(hand: list[str], melds: list[tuple[str, list[str], int]], win: str,
              dora_indicators: list[str], ura_indicators: list[str], player_wind: int,
              round_wind: int, **conditions) -> tuple[int, int, bool, tuple[int, ...], int] | None:
    """Han, fu, yakuman, yaku ids and points of a won hand.

    Args:
        hand: The concealed tiles, the winning tile included (mjai).
        melds: `(mjai type, tiles, from seat)` per meld.
        win: The winning tile (mjai; the last copy of it in `hand`).
        dora_indicators, ura_indicators: Turned indicators (mjai).
        player_wind: 0..3, the seat's wind (0 = the oya).
        round_wind: 0..3.
        **conditions: riichienv `Conditions` flags (`tsumo`, `riichi`,
            `double_riichi`, `ippatsu`, `haitei`, `houtei`, `rinshan`,
            `chankan`, `tsumo_first_turn`).

    Returns:
        `(han, fu, yakuman, yaku ids, points)` with no honba or sticks, or
        None if the evaluator finds no win.
    """
    groups = to_tids([hand, *[m[1] for m in melds]])
    hand_ids = groups[0]
    win_id = hand_ids[len(hand) - 1 - hand[::-1].index(win)]
    rv_melds = [Meld(MELD_TYPE[kind], ids, kind != "ankan", who)
                for (kind, _, who), ids in zip(melds, groups[1:])]
    cond = Conditions(player_wind=WINDS[player_wind], round_wind=WINDS[round_wind], **conditions)
    r = HandEvaluator(sorted(hand_ids), rv_melds).calc(
        win_id, to_tids([dora_indicators])[0], to_tids([ura_indicators])[0], cond)
    if not r.is_win:
        return None
    if conditions.get("tsumo"):
        pts = 3 * r.tsumo_agari_ko if player_wind == 0 else r.tsumo_agari_oya + 2 * r.tsumo_agari_ko
    else:
        pts = r.ron_agari
    return r.han, r.fu, bool(r.yakuman), tuple(r.yaku), pts


_YAKU_NAMES: dict[int, str] = {}


def yaku_name(yaku_id: int) -> str:
    """A yaku's English name (riichienv's)."""
    if not _YAKU_NAMES:
        _YAKU_NAMES.update({y.id: y.name_en for y in rv.get_all_yaku()})
    return _YAKU_NAMES.get(yaku_id, str(yaku_id))


class _Seat:
    """One seat's hand and state during one kyoku (every seat is followed;
    only the measured ones carry a `KyokuStat`)."""

    def __init__(self, tehai: list[str], stat: KyokuStat | None) -> None:
        self.stat = stat
        self.hand = list(tehai)
        self.melds: list[tuple[str, list[str], int]] = []
        self.draws = 0
        self.declared = False           # its reach awaits the declaring discard
        self.riichi = False             # reach accepted
        self.double_riichi = False
        self.ippatsu = False
        self.rinshan = False            # its next tsumo is a replacement draw
        self.last_draw: str | None = None

    @property
    def open(self) -> bool:
        return any(m[0] != "ankan" for m in self.melds)

    def remove(self, tiles: list[str]) -> None:
        if self.stat is None:           # only a measured seat's hand is needed
            return
        for p in tiles:
            self.hand.remove(p)

    def shanten(self) -> int:
        return rv.calculate_shanten(to_tids([self.hand])[0])

    def state(self) -> str:
        return "riichi" if self.riichi else "open" if self.open else "closed"


def parse_game(path: str | Path, name: str) -> list[KyokuStat]:
    """Every kyoku of every seat called `name` in one recorded game.

    Args:
        path: An `arena-*.json.gz` (or plain jsonl) mjai log.
        name: The player name to measure, as in the log's `start_game`.

    Returns:
        One `KyokuStat` per (kyoku, matching seat), in play order.
    """
    return _parse(Path(path), name)[0]


def _parse(path: Path, name: str) -> tuple[list[KyokuStat], list[GameStat]]:
    events = load_mjai(path)
    game = path.name.split(".")[0]
    names = events[0].get("names", [])
    measured = [s for s, n in enumerate(names) if n == name]
    out: list[KyokuStat] = []
    if not measured:
        return out, []
    seats: list[_Seat] = []
    public = [0] * 34                   # discards, meld tiles, dora indicators
    doras: list[str] = []
    oya = round_wind = draws = 0
    scores = [25000] * 4
    any_call = False                    # a call or kan so far this kyoku
    riichi_seen = False                 # some seat's riichi accepted this kyoku
    last_discard: tuple[int, str] | None = None
    last_kakan: tuple[int, str] | None = None
    prev_oya: dict[int, KyokuStat] = {}  # the measured oya's kyoku awaiting the next deal

    def visible(s: _Seat) -> list[int]:
        v = list(public)
        for p in s.hand:
            v[tile_type(p)] += 1
        return v

    def check_tenpai(s: _Seat, riichi: bool) -> None:
        st = s.stat
        if st is None or st.tenpai_turn is not None or s.shanten() != 0:
            return
        st.tenpai_turn = s.draws
        st.closed_at_tenpai = all(m[0] == "ankan" for m in s.melds)
        st.riichi_at_tenpai = riichi
        st.ev_ron, st.ev_tsumo, st.yakuless, st.dead_waits = tenpai_value(
            s.hand, s.melds, visible(s), doras, (st.seat - oya) % 4, round_wind, riichi)

    def riichi_waits(s: _Seat) -> None:
        groups = to_tids([s.hand, *[m[1] for m in s.melds]])
        rv_melds = [Meld(MELD_TYPE[kind], ids, kind != "ankan", who)
                    for (kind, _, who), ids in zip(s.melds, groups[1:])]
        waits = HandEvaluator(groups[0], rv_melds).get_waits()
        v = visible(s)
        s.stat.riichi_waits = sum(max(0, 4 - v[w]) for w in waits)
        s.stat.riichi_good = len(waits) >= 2

    def score_win(e: dict, s: _Seat) -> None:
        actor, target = e["actor"], e["target"]
        st = s.stat
        st.win_turn = s.draws
        if "yaku" in e:                  # the engine's own scoring (yamawake's fork)
            set_win(st, s, e["han"], e["fu"], e.get("yakuman", False), e["yaku"])
            return
        tsumo = target == actor
        if tsumo:
            hand, win = s.hand, s.last_draw
        else:
            win = last_kakan[1] if last_kakan and last_kakan[0] == target else last_discard[1]
            hand = [*s.hand, win]
        conditions = dict(
            tsumo=tsumo, riichi=s.riichi, double_riichi=s.double_riichi, ippatsu=s.ippatsu,
            haitei=tsumo and draws == LIVE_DRAWS and not s.rinshan,
            houtei=not tsumo and draws == LIVE_DRAWS and not last_kakan,
            rinshan=tsumo and s.rinshan, chankan=bool(not tsumo and last_kakan),
            tsumo_first_turn=tsumo and s.draws == 1 and not any_call)
        r = win_value(hand, s.melds, win, doras, e.get("ura_markers") or [],
                      (st.seat - oya) % 4, round_wind, **conditions)
        if r is None:
            return
        han, fu, yakuman, yaku, st.win_rescored = r
        set_win(st, s, han, fu, yakuman, yaku)

    def set_win(st: KyokuStat, s: _Seat, han: int, fu: int, yakuman: bool, yaku) -> None:
        st.win_han, st.win_fu = han, fu
        st.win_limit = limit_of(han, fu, yakuman)
        st.win_yaku = tuple(sorted({yaku_name(y) for y in yaku}))
        if s.riichi:
            st.ippatsu = YAKU_IPPATSU in yaku
            st.ura = YAKU_URA in yaku

    def end_ippatsu() -> None:
        for s2 in seats:
            s2.ippatsu = False

    for e in events:
        t = e["type"]
        actor = e.get("actor")
        s = seats[actor] if actor is not None and seats else None
        if t == "start_kyoku":
            oya, round_wind = e["oya"], ROUND_WIND[e["bakaze"]]
            scores = list(e["scores"])
            for seat, st in prev_oya.items():
                st.renchan = oya == seat
            prev_oya = {}
            public = [0] * 34
            doras = [e["dora_marker"]]
            public[tile_type(doras[0])] += 1
            draws, any_call, riichi_seen = 0, False, False
            last_discard = last_kakan = None
            seats = []
            for seat in range(4):
                st = None
                if seat in measured:
                    st = KyokuStat(game, round_wind, e["kyoku"], e["honba"], seat,
                                   tuple(sorted(e["tehais"][seat])), oya=seat == oya)
                    out.append(st)
                    if seat == oya:
                        prev_oya[seat] = st
                seats.append(_Seat(e["tehais"][seat], st))
                check_tenpai(seats[seat], False)
        elif t == "tsumo":
            draws += 1
            s.hand.append(e["pai"])
            s.draws += 1
            s.last_draw = e["pai"]
            last_kakan = None
        elif t == "dahai":
            public[tile_type(e["pai"])] += 1
            s.remove([e["pai"]])
            s.rinshan = False
            if not s.declared:
                s.ippatsu = False
            last_discard, last_kakan = (actor, e["pai"]), None
            check_tenpai(s, s.declared)
            if s.declared and s.stat is not None:
                riichi_waits(s)
            s.declared = False
        elif t == "reach":
            s.declared = True
            if s.stat is not None:
                s.stat.riichi = True
                s.stat.riichi_turn = s.draws
                s.stat.riichi_first = not riichi_seen
        elif t == "reach_accepted":
            s.riichi = s.ippatsu = True
            s.double_riichi = s.draws == 1 and not any_call
            riichi_seen = True
            scores[actor] -= 1000
            if s.stat is not None:
                s.stat.delta -= 1000
        elif t in ("chi", "pon", "daiminkan"):
            any_call = True
            end_ippatsu()
            for p in e["consumed"]:
                public[tile_type(p)] += 1
            s.remove(e["consumed"])
            s.melds.append((t, [*e["consumed"], e["pai"]], e["target"]))
            s.rinshan = t == "daiminkan"
            if s.stat is not None:
                s.stat.called = True
                s.stat.n_calls += 1
        elif t == "ankan":
            any_call = True
            end_ippatsu()
            for p in e["consumed"]:
                public[tile_type(p)] += 1
            s.remove(e["consumed"])
            s.melds.append((t, list(e["consumed"]), actor))
            s.rinshan = True
        elif t == "kakan":
            any_call = True
            end_ippatsu()
            public[tile_type(e["pai"])] += 1
            s.remove([e["pai"]])
            k = next(i for i, m in enumerate(s.melds) if m[0] == "pon"
                     and tile_type(m[1][0]) == tile_type(e["pai"]))
            s.melds[k] = ("kakan", [*s.melds[k][1], e["pai"]], s.melds[k][2])
            s.rinshan = True
            last_kakan = (actor, e["pai"])
        elif t == "dora":
            doras.append(e["dora_marker"])
            public[tile_type(e["dora_marker"])] += 1
        elif t == "hora":
            d = e["deltas"]
            target = e["target"]
            for seat in range(4):
                scores[seat] += d[seat]
                st = seats[seat].stat
                if st is not None:
                    st.delta += d[seat]
            if s.stat is not None:
                pts = d[actor] - sum(d)           # without the riichi sticks taken
                s.stat.won = True
                if target == actor:
                    s.stat.win_tsumo = pts
                else:
                    s.stat.win_ron = pts
                score_win(e, s)
            for seat in measured:
                st = seats[seat].stat
                if seat == actor:
                    continue
                if target == actor:
                    st.tsumo_by_other = (st.tsumo_by_other or 0) - d[seat]
                elif seat == target:
                    st.deal_in = (st.deal_in or 0) - d[seat]
                    if st.deal_in_to is None:
                        w = seats[actor]
                        st.deal_in_to = "riichi" if w.riichi else "open" if w.open else "dama"
                        st.deal_in_as = seats[seat].state()
                else:
                    st.side_ron = True
        elif t == "ryukyoku":
            d = e.get("deltas") or [0, 0, 0, 0]
            exhaustive = e.get("reason", "exhaustive_draw") == "exhaustive_draw"
            for seat in range(4):
                scores[seat] += d[seat]
                st = seats[seat].stat
                if st is None:
                    continue
                st.delta += d[seat]
                if exhaustive:
                    st.exhaustive = True
                    st.ryukyoku_tenpai = seats[seat].shanten() == 0
    for st in out:
        if st.closed_at_tenpai:
            st.dama = not st.riichi
    order = sorted(range(4), key=lambda x: (-scores[x], x))
    games = [GameStat(game, seat, order.index(seat) + 1, scores[seat]) for seat in measured]
    return out, games


def _files(log_dir: str | Path) -> list[Path]:
    files = sorted(Path(log_dir).glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no *.json.gz logs in {log_dir}")
    return files


def collect(log_dir: str | Path, name: str) -> list[KyokuStat]:
    """`parse_game` over every recorded game in a directory (sorted by file)."""
    return [k for f in _files(log_dir) for k in _parse(f, name)[0]]


def collect_all(log_dir: str | Path, name: str) -> tuple[list[KyokuStat], list[GameStat]]:
    """Every kyoku and every hanchan result of player `name` in a directory."""
    ks: list[KyokuStat] = []
    gs: list[GameStat] = []
    for f in _files(log_dir):
        k, g = _parse(f, name)
        ks += k
        gs += g
    return ks, gs


# --- metrics ---------------------------------------------------------------

def _f(x) -> float | None:
    return None if x is None else float(x)


def _b(cond: bool | None, x: bool | None = True) -> float | None:
    """1/0 for `x` where `cond` holds, else None (a rate over those kyoku)."""
    return None if not cond or x is None else float(bool(x))


def _limit_at_least(k: KyokuStat, limit: str) -> float | None:
    if k.win_limit is None:
        return None
    return float(LIMITS.index(k.win_limit) >= LIMITS.index(limit))


#: (section, [(key, label, value of one kyoku or None where undefined)]).
#: Rates are 0/1; a rate over a subset is None outside it.
SECTIONS: list[tuple[str, list[tuple[str, str, Callable[[KyokuStat], float | None]]]]] = [
    ("speed and value", [
        ("tenpai_rate", "tenpai rate", lambda k: float(k.tenpai_turn is not None)),
        *[(f"tenpai_by_{t}", f"  tenpai by turn {t}",
           (lambda t: lambda k: float(k.tenpai_turn is not None and k.tenpai_turn <= t))(t))
          for t in (3, 6, 9, 12, 15, 18)],
        ("tenpai_turn", "first tenpai turn", lambda k: _f(k.tenpai_turn)),
        ("ev_ron", "value at tenpai, ron", lambda k: k.ev_ron),
        ("ev_tsumo", "value at tenpai, tsumo", lambda k: k.ev_tsumo),
        ("riichi_at_tenpai", "riichi at first tenpai", lambda k: _f(k.riichi_at_tenpai)),
        ("closed_at_tenpai", "closed at first tenpai", lambda k: _f(k.closed_at_tenpai)),
        ("yakuless", "yakuless at tenpai", lambda k: _f(k.yakuless)),
        ("dead_waits", "dead waits at tenpai", lambda k: _f(k.dead_waits)),
        ("dama", "dama (closed tenpai)", lambda k: _f(k.dama)),
    ]),
    ("outcomes", [
        ("win_rate", "win rate", lambda k: float(k.won)),
        ("win_points", "points on win", lambda k: _f(k.win_ron if k.win_ron is not None
                                                     else k.win_tsumo)),
        ("win_ron", "  by ron", lambda k: _f(k.win_ron)),
        ("win_tsumo", "  by tsumo", lambda k: _f(k.win_tsumo)),
        ("deal_in_rate", "deal-in rate", lambda k: float(k.deal_in is not None)),
        ("deal_in_points", "deal-in points", lambda k: _f(k.deal_in)),
        ("tsumo_by_other_rate", "another's tsumo", lambda k: float(k.tsumo_by_other is not None)),
        ("tsumo_by_other_points", "  points paid", lambda k: _f(k.tsumo_by_other)),
        ("side_ron_rate", "side ron (others)", lambda k: float(k.side_ron)),
        ("exhaustive_rate", "exhaustive draw", lambda k: float(k.exhaustive)),
        ("ryukyoku_tenpai", "  tenpai at the draw", lambda k: _f(k.ryukyoku_tenpai)),
        ("riichi_rate", "riichi rate", lambda k: float(k.riichi)),
        ("call_rate", "call rate", lambda k: float(k.called)),
        ("renchan", "renchan (as oya)", lambda k: _f(k.renchan)),
        ("delta", "kyoku delta", lambda k: float(k.delta)),
    ]),
    ("riichi", [
        ("riichi_first", "first (not a chase)", lambda k: _b(k.riichi, k.riichi_first)),
        ("riichi_turn", "turn", lambda k: _f(k.riichi_turn)),
        ("riichi_waits", "unseen waits", lambda k: _f(k.riichi_waits)),
        ("riichi_good", "good shape (2+ waits)", lambda k: _b(k.riichi, k.riichi_good)),
        ("riichi_win", "then win", lambda k: _b(k.riichi, k.won)),
        ("riichi_deal_in", "then deal in", lambda k: _b(k.riichi, k.deal_in is not None)),
        ("riichi_draw", "then exhaustive draw", lambda k: _b(k.riichi, k.exhaustive)),
        ("riichi_win_points", "points on win", lambda k: _f(k.win_ron if k.win_ron is not None
                                                            else k.win_tsumo) if k.riichi else None),
        ("ippatsu", "ippatsu (of wins)", lambda k: _f(k.ippatsu)),
        ("ura", "ura dora (of wins)", lambda k: _f(k.ura)),
    ]),
    ("calls", [
        ("calls", "calls per called kyoku", lambda k: float(k.n_calls) if k.called else None),
        ("call_win", "then win", lambda k: _b(k.called, k.won)),
        ("call_deal_in", "then deal in", lambda k: _b(k.called, k.deal_in is not None)),
        ("call_win_points", "points on win", lambda k: _f(k.win_ron if k.win_ron is not None
                                                          else k.win_tsumo) if k.called else None),
    ]),
    ("wins", [
        ("win_turn", "win turn", lambda k: _f(k.win_turn)),
        ("tsumo_share", "by tsumo", lambda k: _b(k.won, k.win_tsumo is not None)),
        ("win_han", "han", lambda k: _f(k.win_han)),
        *[(f"limit_{x}", f"{x.replace('_', ' ')}+", (lambda x: lambda k: _limit_at_least(k, x))(x))
          for x in LIMITS[1:]],
    ]),
    ("deal-ins", [
        ("deal_in_to_riichi", "to a riichi", lambda k: _b(k.deal_in_to, k.deal_in_to == "riichi")),
        ("deal_in_to_open", "to an open hand", lambda k: _b(k.deal_in_to, k.deal_in_to == "open")),
        ("deal_in_to_dama", "to a dama hand", lambda k: _b(k.deal_in_to, k.deal_in_to == "dama")),
        ("deal_in_as_riichi", "while in riichi", lambda k: _b(k.deal_in_as, k.deal_in_as == "riichi")),
        ("deal_in_as_open", "while open", lambda k: _b(k.deal_in_as, k.deal_in_as == "open")),
        ("deal_in_as_closed", "while closed", lambda k: _b(k.deal_in_as, k.deal_in_as == "closed")),
    ]),
]
METRICS = [m for _, ms in SECTIONS for m in ms]

#: Hanchan metrics (key, label, value of one GameStat).
GAME_METRICS: list[tuple[str, str, Callable[[GameStat], float]]] = [
    ("rank", "average rank", lambda g: float(g.rank)),
    *[(f"rank_{r}", f"  rank {r}", (lambda r: lambda g: float(g.rank == r))(r)) for r in range(1, 5)],
    ("score", "final score", lambda g: float(g.score)),
    ("bust", "bust (below 0)", lambda g: float(g.score < 0)),
]


def cluster_mean(values: Iterable[float], clusters: Iterable) -> tuple[float, float]:
    """Mean and its standard error, clustered (e.g. by game file).

    The ratio estimator's linearized variance: kyoku of one game share its
    seating and score history, so they are not independent draws.
    """
    v = np.asarray(list(values), dtype=np.float64)
    if len(v) == 0:
        return math.nan, math.nan
    c = list(clusters)
    m = float(v.mean())
    _, idx = np.unique(np.array(c, dtype=object).astype(str), return_inverse=True)
    resid = np.bincount(idx, weights=v - m)
    g = len(resid)
    if g < 2:
        return m, 0.0
    return m, float(math.sqrt(g / (g - 1) * np.sum(resid ** 2)) / len(v))


def summarize(stats: list[KyokuStat]) -> dict:
    """Every metric's mean and SE over one side's kyoku, plus the tenpai curve."""
    out: dict = {"kyoku": len(stats), "games": len({k.game for k in stats})}
    for key, _, fn in METRICS:
        pairs = [(v, k.game) for k in stats if (v := fn(k)) is not None]
        m, se = cluster_mean([p[0] for p in pairs], [p[1] for p in pairs])
        out[key] = {"mean": m, "se": se, "n": len(pairs)}
    turns = [k.tenpai_turn for k in stats if k.tenpai_turn is not None]
    out["tenpai_turn"]["median"] = float(np.median(turns)) if turns else math.nan
    n = max(len(stats), 1)
    out["tenpai_curve"] = [sum(1 for t in turns if t <= x) / n for x in range(1, MAX_TURN + 1)]
    wins = [k for k in stats if k.win_yaku is not None]
    counts: dict[str, int] = {}
    for k in wins:
        for y in k.win_yaku:
            counts[y] = counts.get(y, 0) + 1
    out["yaku"] = {"wins": len(wins),
                   "share": {y: c / len(wins) for y, c in
                             sorted(counts.items(), key=lambda x: (-x[1], x[0]))}}
    return out


def summarize_games(games: list[GameStat]) -> dict:
    """Every hanchan metric's mean and SE over one side's games."""
    out: dict = {"hanchan": len(games)}
    for key, _, fn in GAME_METRICS:
        m, se = cluster_mean([fn(g) for g in games], [g.game for g in games])
        out[key] = {"mean": m, "se": se, "n": len(games)}
    return out


def pair(a: list[KyokuStat], b: list[KyokuStat]) -> tuple[list[tuple[KyokuStat, KyokuStat]], int]:
    """Kyoku of `a` matched to `b` by key, kept only with the same deal.

    Returns:
        `(pairs, mismatched)`: the pairs, and how many keys matched with a
        different haipai (dropped).
    """
    by_key = {k.key: k for k in b}
    pairs, bad = [], 0
    for k in a:
        o = by_key.get(k.key)
        if o is None:
            continue
        if o.haipai != k.haipai:
            bad += 1
            continue
        pairs.append((k, o))
    return pairs, bad


def pair_games(a: list[GameStat], b: list[GameStat]) -> list[tuple[GameStat, GameStat]]:
    """Hanchan results of `a` matched to `b` by (file, seat)."""
    by_key = {g.key: g for g in b}
    return [(g, by_key[g.key]) for g in a if g.key in by_key]


def compare(pairs: list[tuple], metrics: list | None = None) -> dict:
    """Both sides' means and the paired difference, metric by metric.

    Each metric uses the pairs where both sides define it. `metrics` is
    `METRICS` (kyoku pairs, the default) or `GAME_METRICS` (hanchan pairs,
    `pair_games`).
    """
    out: dict = {"pairs": len(pairs)}
    for key, _, fn in metrics or METRICS:
        rows = [(x, y, p.game) for p, q in pairs
                if (x := fn(p)) is not None and (y := fn(q)) is not None]
        g = [r[2] for r in rows]
        ma, sa = cluster_mean([r[0] for r in rows], g)
        mb, sb = cluster_mean([r[1] for r in rows], g)
        md, sd = cluster_mean([r[0] - r[1] for r in rows], g)
        out[key] = {"a": ma, "a_se": sa, "b": mb, "b_se": sb,
                    "diff": md, "diff_se": sd, "n": len(rows)}
    return out


# --- printing ----------------------------------------------------------------

_POINTS = {"ev_ron", "ev_tsumo", "win_points", "win_ron", "win_tsumo", "deal_in_points",
           "tsumo_by_other_points", "riichi_win_points", "call_win_points", "delta", "score"}
_COUNTS = {"tenpai_turn", "riichi_turn", "riichi_waits", "calls", "win_turn", "win_han", "rank"}


def _num(key: str, m: float) -> str:
    if math.isnan(m):
        return "-"
    if key in _POINTS:
        return f"{m:.0f}"
    if key in _COUNTS:
        return f"{m:.2f}"
    return f"{m:.3f}"


def _se(key: str, se: float) -> str:
    return "" if math.isnan(se) else f"({_num(key, se)})"


def _rows(sections: list, row) -> list[str]:
    lines = []
    for title, metrics in sections:
        lines.append(f" {title}")
        lines += [row(key, label) for key, label, _ in metrics]
    return lines


def format_summary(name: str, s: dict, games: dict | None = None, top_yaku: int = 12) -> str:
    """One side's report as text."""
    lines = [f"{name}: {s['kyoku']} kyoku in {s['games']} games"]

    def row(key: str, label: str, r: dict) -> str:
        extra = f"  median {r['median']:.0f}" if key == "tenpai_turn" and r["n"] else ""
        return f"  {label:<26} {_num(key, r['mean']):>8} {_se(key, r['se']):<9} n={r['n']}{extra}"

    if games is not None:
        lines.append(" hanchan")
        lines += [row(key, label, games[key]) for key, label, _ in GAME_METRICS]
    body = _rows(SECTIONS, lambda key, label: row(key, label, s[key]))
    curve = "  tenpai by turn " + " ".join(f"{t}:{x:.2f}" for t, x in enumerate(s["tenpai_curve"], 1))
    at = body.index(f" {SECTIONS[1][0]}")
    lines += [*body[:at], curve, *body[at:]]
    y = s["yaku"]
    lines.append(f" yaku (share of {y['wins']} wins)")
    lines += [f"  {k:<26} {v:>8.3f}" for k, v in list(y["share"].items())[:top_yaku]]
    return "\n".join(lines)


def format_compare(name_a: str, name_b: str, c: dict, mismatched: int = 0,
                   games: dict | None = None) -> str:
    """The paired table: both sides' absolute values, then the difference."""
    w = min(max(18, len(name_a), len(name_b)), 28)
    lines = [f"paired: {c['pairs']} kyoku (same file, kyoku, honba, seat and haipai)"
             + (f"; {mismatched} dropped for a different haipai" if mismatched else ""),
             f"  {'metric':<26} {name_a[:w]:>{w}} {name_b[:w]:>{w}} {'A-B':>16} {'n':>7}"]

    def row(key: str, label: str, r: dict) -> str:
        return (f"  {label:<26} "
                f"{_num(key, r['a']) + ' ' + _se(key, r['a_se']):>{w}} "
                f"{_num(key, r['b']) + ' ' + _se(key, r['b_se']):>{w}} "
                f"{_num(key, r['diff']) + ' ' + _se(key, r['diff_se']):>16} {r['n']:>7}")

    if games is not None:
        lines.append(f" hanchan ({games['pairs']} paired by file and seat)")
        lines += [row(key, label, games[key]) for key, label, _ in GAME_METRICS]
    lines += _rows(SECTIONS, lambda key, label: row(key, label, c[key]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """`python -m yamawake.handstats LOGDIR --name NAME [--vs DIR --vs-name NAME]`."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", help="a --record-logs directory")
    ap.add_argument("--name", required=True, help="player name to measure (start_game names)")
    ap.add_argument("--vs", default=None, help="a second --record-logs directory, same walls")
    ap.add_argument("--vs-name", default=None, help="player name in --vs (default: --name)")
    ap.add_argument("--json", default=None, help="write the numbers to this file")
    args = ap.parse_args(argv)

    t0 = time.time()
    a, ga = collect_all(args.log_dir, args.name)
    if not a:
        raise SystemExit(f"no player named {args.name!r} in {args.log_dir}")
    out: dict = {"a": {"dir": args.log_dir, "name": args.name, **summarize(a),
                       "hanchan": summarize_games(ga)}}
    print(format_summary(args.name, out["a"], out["a"]["hanchan"]))
    if args.vs:
        vs_name = args.vs_name or args.name
        b, gb = collect_all(args.vs, vs_name)
        if not b:
            raise SystemExit(f"no player named {vs_name!r} in {args.vs}")
        out["b"] = {"dir": args.vs, "name": vs_name, **summarize(b),
                    "hanchan": summarize_games(gb)}
        print(format_summary(vs_name, out["b"], out["b"]["hanchan"]))
        pairs, bad = pair(a, b)
        out["paired"] = {**compare(pairs), "mismatched_haipai": bad,
                         "hanchan": compare(pair_games(ga, gb), GAME_METRICS)}
        print(format_compare(args.name, vs_name, out["paired"], bad, out["paired"]["hanchan"]))
    out["seconds"] = round(time.time() - t0, 2)
    print(f"{out['seconds']:.1f}s", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
