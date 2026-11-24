"""yamawake records: hanchan logs as numpy arrays in one directory (`docs/records.md`).

Every event is 2 bytes, `(op << 4) | (aux << 2) | actor` and an argument byte; the calls'
tiles, the wins and the draws sit in typed tables in event order; per-game and per-kyoku
counts give every offset by a cumulative sum, so two directories concatenate array by array.
The mjai events come back exactly as yamawake's engine writes them.

    w = RecordWriter()
    w.add_game(events, scores=final)            # mjai dicts, start_game .. end_game
    w.save("out/")
    r = Records("out/")
    r.lines(0) == engine_lines                  # sorted keys, (",", ":") separators
    op = r["events"][:, 0] >> 4                 # every event of the corpus at once

    python -m yamawake.records convert SRC_DIR OUT_DIR     # *.json.gz mjai logs -> records
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from array import array
from collections.abc import Iterator
from pathlib import Path

import numpy as np

FORMAT, VERSION = "yamawake-records", 1
OPS = ("tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach", "reach_accepted", "dora",
       "hora", "ryukyoku")
OP = {name: i for i, name in enumerate(OPS)}
#: Tiles each call puts in `meld_tiles`.
MELD_TILES = {"chi": 2, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 3}
_MELD_OF_OP = np.array([MELD_TILES.get(name, 0) for name in OPS] + [0] * 4, dtype=np.int64)
REASONS = (None, "exhaustive_draw", "kyushu_kyuhai", "sufuurenta", "suucha_riichi", "suukansansen", "sanchaho",
           "nagashimangan")
REASON = {name: i for i, name in enumerate(REASONS) if name}
WINDS = ("E", "S", "W", "N")
NAMES = tuple(f"{n}{s}" for s in "mps" for n in range(1, 10)) + ("E", "S", "W", "N", "P", "F", "C")
RED, TSUMOGIRI, NO_TILE = 0x40, 0x80, 0xFF
MAX_URA, MAX_WINS = 5, 3
YAKUMAN_ID = 35                         # riichienv's yaku ids from 35 up are yakuman
_TILE = {name: i for i, name in enumerate(NAMES)} | {f"5{s}r": 9 * k + 4 | RED for k, s in enumerate("mps")}
_STR = [NAMES[b & 0x3F] + ("r" if b & RED else "") if (b & 0x3F) < 34 else "?" for b in range(256)]
#: The keys each event type may carry (beyond "type"); others would be lost, so they are refused.
_KEYS = {
    "start_game": {"names", "seed"}, "end_game": set(), "end_kyoku": set(),
    "start_kyoku": {"bakaze", "dora_marker", "honba", "kyoku", "kyotaku", "oya", "scores", "tehais"},
    "tsumo": {"actor", "pai"}, "dahai": {"actor", "pai", "tsumogiri"},
    "chi": {"actor", "target", "pai", "consumed"}, "pon": {"actor", "target", "pai", "consumed"},
    "daiminkan": {"actor", "target", "pai", "consumed"}, "ankan": {"actor", "pai", "consumed"},
    "kakan": {"actor", "pai", "consumed"}, "reach": {"actor"}, "reach_accepted": {"actor"},
    "dora": {"dora_marker"}, "ryukyoku": {"deltas", "reason"},
    "hora": {"actor", "target", "deltas", "ura_markers", "han", "fu", "yaku", "yakuman", "tsumo"},
}


def tile_byte(pai: str) -> int:
    """An mjai tile as its byte, e.g. "5mr" -> 0x44, "C" -> 33."""
    try:
        return _TILE[pai]
    except KeyError:
        raise ValueError(f"not a tile: {pai!r}") from None


def tile_str(b: int) -> str:
    """A tile byte as its mjai name (bit 7, a discard's tsumogiri, ignored)."""
    return _STR[b & 0x7F]


def dumps(ev: dict) -> str:
    """An event as yamawake's engine writes it: sorted keys, `(",", ":")` separators."""
    return json.dumps(ev, separators=(",", ":"), sort_keys=True)


def _hundreds(values, what: str) -> list[int]:
    if len(values) != 4 or any(v % 100 or not -32768 <= v // 100 <= 32767 for v in values):
        raise ValueError(f"{what} {values}: not four multiples of 100 within int16")
    return [v // 100 for v in values]


def final_scores(events: list[dict], leftover_to_top: bool = True) -> list[int]:
    """A game's final scores from its log: the last kyoku's start, its deltas, less its riichi sticks.

    The sticks left on the table at the end go to first place (the earlier seat on a tie), as
    Tenhou, Mahjong Soul and yamawake's engine settle a game of rounds; `leftover_to_top=False`
    leaves them to nobody (the engine's one-kyoku mode).
    """
    scores, sticks = None, 0
    for e in events:
        t = e["type"]
        if t == "start_kyoku":
            scores, sticks = list(e["scores"]), e["kyotaku"]
        elif t in ("hora", "ryukyoku"):
            scores = [a + b for a, b in zip(scores, e["deltas"])]
            if t == "hora":
                sticks = 0
        elif t == "reach_accepted":
            scores[e["actor"]] -= 1000
            sticks += 1
    if scores is None:
        raise ValueError("no kyoku")
    if leftover_to_top and sticks:
        scores[max(range(4), key=lambda s: (scores[s], -s))] += 1000 * sticks
    return scores


class RecordWriter:
    """Collects games and writes them as one records directory."""

    def __init__(self) -> None:
        self.players: dict[str, int] = {}
        self.g_n_kyoku, self.g_player, self.g_scores = array("H"), array("i"), array("h")
        self.extra: dict[str, list] | None = None          # per extra field, its value per game
        self.k_info, self.k_scores, self.k_haipai = bytearray(), array("h"), bytearray()
        self.k_n_events, self.k_n_meld, self.k_n_wins = array("H"), bytearray(), bytearray()
        self.events, self.melds = bytearray(), bytearray()
        self.w_deltas, self.w_han, self.w_fu, self.w_yaku, self.w_ura = (array("h"), bytearray(), bytearray(),
                                                                         array("Q"), bytearray())
        self.d_deltas = array("h")

    def __len__(self) -> int:
        return len(self.g_n_kyoku)

    def add_game(self, events: list[dict], scores: list[int] | None = None, names: list[str] | None = None,
                 **extra) -> None:
        """Add one game, its mjai events from start_game to end_game.

        Args:
            events: The game's events (dicts).
            scores: Its final scores; by default `final_scores(events)`.
            names: The seats' player names; by default `start_game`'s `names` (else empty names).
            extra: Per-game values for `game_<name>` arrays (the same names for every game).

        Raises:
            ValueError: The log does not fit the format; nothing of the game is kept.
        """
        if not events or events[0]["type"] != "start_game" or events[-1]["type"] != "end_game":
            raise ValueError("a game runs from start_game to end_game")
        head = events[0]
        _check_keys(head)
        if head.get("seed") is not None:
            raise ValueError("start_game carries a seed the format does not keep")
        names = list(names or head.get("names") or [""] * 4)
        if len(names) != 4:
            raise ValueError(f"not four player names: {names}")
        if self.extra is not None and set(extra) != set(self.extra):
            raise ValueError(f"extra fields {sorted(extra)} differ from the earlier games' {sorted(self.extra)}")
        kyokus, cur = [], None
        for e in events[1:-1]:
            if e["type"] == "start_kyoku":
                if cur is not None:
                    raise ValueError("start_kyoku inside a kyoku")
                cur = [e]
            elif cur is None:
                raise ValueError(f"{e['type']} outside a kyoku")
            else:
                cur.append(e)
                if e["type"] == "end_kyoku":
                    kyokus.append(cur)
                    cur = None
        if cur is not None or not kyokus:
            raise ValueError("an unfinished kyoku, or no kyoku")
        encoded = [_encode_kyoku(k) for k in kyokus]               # all or nothing
        final = _hundreds(final_scores(events) if scores is None else list(scores), "final scores")
        for k in encoded:
            self._append(*k)
        self.g_n_kyoku.append(len(kyokus))
        self.g_player.extend(self.players.setdefault(n, len(self.players)) for n in names)
        self.g_scores.extend(final)
        if self.extra is None:
            self.extra = {key: [] for key in extra}
        for key, v in extra.items():
            self.extra[key].append(v)

    def _append(self, info, scores, haipai, events, melds, wins, draw) -> None:
        self.k_info += info
        self.k_scores.extend(scores)
        self.k_haipai += haipai
        self.k_n_events.append(len(events) // 2)
        self.k_n_meld.append(len(melds))
        self.k_n_wins.append(len(wins))
        self.events += events
        self.melds += melds
        for deltas, han, fu, yaku, ura in wins:
            self.w_deltas.extend(deltas)
            self.w_han.append(han)
            self.w_fu.append(fu)
            self.w_yaku.append(yaku)
            self.w_ura += ura
        if draw is not None:
            self.d_deltas.extend(draw)

    def arrays(self) -> dict[str, np.ndarray]:
        """Every array of the records, by file name (without `.npy`)."""
        out = {
            "game_n_kyoku": np.frombuffer(self.g_n_kyoku, np.uint16),
            "game_player": np.frombuffer(self.g_player, np.int32).reshape(-1, 4),
            "game_scores": np.frombuffer(self.g_scores, np.int16).reshape(-1, 4),
            "kyoku_info": np.frombuffer(self.k_info, np.uint8).reshape(-1, 4),
            "kyoku_scores": np.frombuffer(self.k_scores, np.int16).reshape(-1, 4),
            "kyoku_haipai": np.frombuffer(self.k_haipai, np.uint8).reshape(-1, 4, 13),
            "kyoku_n_events": np.frombuffer(self.k_n_events, np.uint16),
            "kyoku_n_meld_tiles": np.frombuffer(self.k_n_meld, np.uint8),
            "kyoku_n_wins": np.frombuffer(self.k_n_wins, np.uint8),
            "events": np.frombuffer(self.events, np.uint8).reshape(-1, 2),
            "meld_tiles": np.frombuffer(self.melds, np.uint8),
            "win_deltas": np.frombuffer(self.w_deltas, np.int16).reshape(-1, 4),
            "win_ura": np.frombuffer(self.w_ura, np.uint8).reshape(-1, MAX_URA),
            "draw_deltas": np.frombuffer(self.d_deltas, np.int16).reshape(-1, 4),
        }
        if any(self.w_han):         # logs without han, fu and yaku (e.g. Tenhou's) leave the three out
            out.update(win_han=np.frombuffer(self.w_han, np.uint8), win_fu=np.frombuffer(self.w_fu, np.uint8),
                       win_yaku=np.frombuffer(self.w_yaku, np.uint64))
        for key, values in (self.extra or {}).items():
            out[f"game_{key}"] = np.array([v.encode() for v in values] if values and isinstance(values[0], str)
                                          else values)
        return out

    def save(self, out_dir: str | Path, meta: dict | None = None) -> Path:
        """Write the records to `out_dir`, a new or empty directory.

        Raises:
            FileExistsError: `out_dir` holds files (a stale array would be read as part of the records).
        """
        out = Path(out_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"{out} is not empty")
        out.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        for name, a in arrays.items():
            np.save(out / f"{name}.npy", np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")))
        players = sorted(self.players, key=self.players.get)
        (out / "players.json").write_text(json.dumps(players, ensure_ascii=False))
        info = {"format": FORMAT, "version": VERSION, "games": len(self), "kyokus": len(arrays["kyoku_n_events"]),
                "events": len(arrays["events"]), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **(meta or {})}
        (out / "meta.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        return out


def _check_keys(e: dict) -> None:
    allowed = _KEYS.get(e.get("type"))
    if allowed is None:
        raise ValueError(f"unexpected event type {e.get('type')!r}")
    extra = set(e) - allowed - {"type"}
    if extra:
        raise ValueError(f"{e['type']} carries keys the format does not keep: {sorted(extra)}")


def _encode_kyoku(evs: list[dict]) -> tuple:
    """(info, scores, haipai, event bytes, meld bytes, wins, draw deltas or None) of one kyoku."""
    head = evs[0]
    _check_keys(head)
    if head["oya"] != head["kyoku"] - 1 or head["bakaze"] not in WINDS or not 1 <= head["kyoku"] <= 4:
        raise ValueError(f"start_kyoku {head['bakaze']}{head['kyoku']} with oya {head['oya']}")
    if not (0 <= head["honba"] <= 255 and 0 <= head["kyotaku"] <= 255):
        raise ValueError("honba or kyotaku beyond a byte")
    tehais = head["tehais"]
    if len(tehais) != 4 or any(len(h) != 13 for h in tehais):
        raise ValueError("haipai is not 4 x 13")
    info = bytes((WINDS.index(head["bakaze"]) * 4 + head["kyoku"] - 1, head["honba"], head["kyotaku"],
                  tile_byte(head["dora_marker"])))
    scores = _hundreds(head["scores"], "scores")
    haipai = bytes(tile_byte(p) for h in tehais for p in h)
    out, melds, wins, draw = bytearray(), bytearray(), [], None
    for e in evs[1:-1]:
        _check_keys(e)
        t = e["type"]
        if t not in OP:
            raise ValueError(f"unexpected {t} inside a kyoku")
        if wins and t != "hora" or draw is not None:
            raise ValueError(f"{t} after the kyoku's end")
        a = e.get("actor", 0)
        aux = (e["target"] - a) % 4 if t in ("chi", "pon", "daiminkan", "hora") else 0
        arg = 0
        if t in ("tsumo", "chi", "pon", "daiminkan", "kakan"):
            arg = tile_byte(e["pai"])
        elif t == "dahai":
            if not isinstance(e["tsumogiri"], bool):
                raise ValueError("tsumogiri is not a bool")
            arg = tile_byte(e["pai"]) | (TSUMOGIRI if e["tsumogiri"] else 0)
        elif t == "ankan":
            arg = tile_byte(e["pai"] if "pai" in e else e["consumed"][0])
        elif t == "dora":
            arg = tile_byte(e["dora_marker"])
        elif t == "ryukyoku":
            reason = e.get("reason")
            if reason is not None and reason not in REASON:
                raise ValueError(f"unknown ryukyoku reason {reason!r}")
            arg = REASON.get(reason, 0)
            draw = _hundreds(e["deltas"], "ryukyoku deltas")
        elif t == "hora":
            wins.append(_encode_win(e, aux))
            if len(wins) > MAX_WINS:
                raise ValueError("more than three wins in a kyoku")
        if t in MELD_TILES:
            if len(e["consumed"]) != MELD_TILES[t]:
                raise ValueError(f"{t} consumes {e['consumed']}")
            melds += bytes(tile_byte(p) for p in e["consumed"])
        out += bytes(((OP[t] << 4) | (aux << 2) | a, arg))
    if not wins and draw is None:
        raise ValueError("a kyoku without a hora or a ryukyoku")
    if len(out) // 2 > 0xFFFF or len(melds) > 0xFF:
        raise ValueError("too many events or call tiles in a kyoku")
    return info, scores, haipai, out, melds, wins, draw


def _encode_win(e: dict, aux: int) -> tuple:
    yaku = e.get("yaku", [])
    if any(not 0 <= y < 64 for y in yaku) or len(set(yaku)) != len(yaku):
        raise ValueError(f"yaku ids {yaku}: not distinct ids below 64")
    if "yakuman" in e and e["yakuman"] != any(y >= YAKUMAN_ID for y in yaku):
        raise ValueError(f"yakuman {e['yakuman']} disagrees with the yaku ids {yaku}")
    if "tsumo" in e and (e["tsumo"] is not True or aux != 0):
        raise ValueError("tsumo must be true and only on a tsumo win")
    han, fu = e.get("han", 0), e.get("fu", 0)
    detail = {"han", "fu", "yaku", "yakuman"} & set(e)
    if detail and (len(detail) != 4 or not 1 <= han <= 255 or not 0 <= fu <= 255 or list(yaku) != sorted(yaku)):
        raise ValueError(f"hora han {han} fu {fu} yaku {yaku}: all four keys, han 1-255, fu 0-255, ids ascending")
    ura = e["ura_markers"]
    if len(ura) > MAX_URA:
        raise ValueError(f"{len(ura)} ura markers")
    mask = 0
    for y in yaku:
        mask |= 1 << y
    return (_hundreds(e["deltas"], "hora deltas"), han, fu, mask,
            bytes(tile_byte(p) for p in ura) + bytes([NO_TILE] * (MAX_URA - len(ura))))


class Records:
    """A records directory, memory-mapped.

    `r[name]` is an array (e.g. `r["events"]`); `game_kyoku_start`, `kyoku_event_start`,
    `kyoku_meld_start`, `kyoku_win_start` and `kyoku_draw_start` (each one longer than its
    table) are the offsets, summed from the counts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta.get("format") != FORMAT or self.meta.get("version") != VERSION:
            raise ValueError(f"{self.path}: not {FORMAT} version {VERSION}")
        self.players = json.loads((self.path / "players.json").read_text())
        self.a = {p.stem: np.load(p, mmap_mode="r") for p in sorted(self.path.glob("*.npy"))}
        a = self.a

        def starts(counts) -> np.ndarray:
            out = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=out[1:])
            return out

        self.game_kyoku_start = starts(a["game_n_kyoku"])
        self.kyoku_event_start = starts(a["kyoku_n_events"])
        self.kyoku_meld_start = starts(a["kyoku_n_meld_tiles"])
        self.kyoku_win_start = starts(a["kyoku_n_wins"])
        self.kyoku_draw_start = starts(np.asarray(a["kyoku_n_wins"]) == 0)
        for name, n, total in (("kyoku", self.game_kyoku_start[-1], len(a["kyoku_info"])),
                               ("event", self.kyoku_event_start[-1], len(a["events"])),
                               ("meld tile", self.kyoku_meld_start[-1], len(a["meld_tiles"])),
                               ("win", self.kyoku_win_start[-1], len(a["win_deltas"])),
                               ("draw", self.kyoku_draw_start[-1], len(a["draw_deltas"]))):
            if n != total:
                raise ValueError(f"{self.path}: the counts sum to {n} {name}s, the table holds {total}")

    def __len__(self) -> int:
        return len(self.a["game_n_kyoku"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.a[name]

    def kyokus(self, g: int) -> range:
        """Game g's kyoku rows."""
        return range(int(self.game_kyoku_start[g]), int(self.game_kyoku_start[g + 1]))

    def names(self, g: int) -> list[str]:
        """Game g's player names by seat."""
        return [self.players[i] for i in self.a["game_player"][g]]

    def scores(self, g: int) -> list[int]:
        """Game g's final scores."""
        return [int(s) * 100 for s in self.a["game_scores"][g]]

    def game_id(self, g: int) -> str:
        """Game g's id: its `game_id`, else the name `run_match` gives its mjai file
        (`arena-<wall tag>-<g:06d>`), else `<g:06d>`."""
        if "game_id" in self.a:
            return self.a["game_id"][g].decode()
        p = self.meta.get("source", {}).get("run_match")
        if p is None:
            return f"{g:06d}"
        tag = p["wall_base"] if "wall_base" in p else f"s{p['wall_seed']}-{p['wall_start']}"
        return f"arena-{tag}-{g:06d}"

    def kyoku_events(self, k: int) -> list[dict]:
        """Kyoku row k's mjai events, start_kyoku to end_kyoku."""
        a = self.a
        info = a["kyoku_info"][k].tolist()
        oya = info[0] % 4
        out = [{"bakaze": WINDS[info[0] // 4], "dora_marker": _STR[info[3]], "honba": info[1],
                "kyoku": oya + 1, "kyotaku": info[2], "oya": oya,
                "scores": [s * 100 for s in a["kyoku_scores"][k].tolist()],
                "tehais": [[_STR[b] for b in h] for h in a["kyoku_haipai"][k].tolist()], "type": "start_kyoku"}]
        events = a["events"][self.kyoku_event_start[k]:self.kyoku_event_start[k + 1]].tolist()
        melds = a["meld_tiles"][self.kyoku_meld_start[k]:self.kyoku_meld_start[k + 1]].tolist()
        w, d, m = int(self.kyoku_win_start[k]), int(self.kyoku_draw_start[k]), 0
        for head, arg in events:
            t, aux, actor = OPS[head >> 4], (head >> 2) & 3, head & 3
            e = {"type": t}
            if t == "dora":
                e["dora_marker"] = _STR[arg]
            elif t == "ryukyoku":
                e["deltas"] = [x * 100 for x in a["draw_deltas"][d].tolist()]
                if arg:
                    e["reason"] = REASONS[arg]
                d += 1
            else:
                e["actor"] = actor
                if t == "hora":
                    e["target"] = (actor + aux) % 4
                    e["deltas"] = [x * 100 for x in a["win_deltas"][w].tolist()]
                    e["ura_markers"] = [_STR[b] for b in a["win_ura"][w].tolist() if b != NO_TILE]
                    han = int(a["win_han"][w]) if "win_han" in a else 0
                    if han:
                        yaku = [i for i in range(64) if int(a["win_yaku"][w]) >> i & 1]
                        e.update(han=han, fu=int(a["win_fu"][w]), yaku=yaku,
                                 yakuman=any(y >= YAKUMAN_ID for y in yaku))
                    if aux == 0:
                        e["tsumo"] = True
                    w += 1
                elif t not in ("reach", "reach_accepted"):
                    e["pai"] = _STR[arg]
                    if t == "dahai":
                        e["tsumogiri"] = bool(arg & TSUMOGIRI)
                    elif t in MELD_TILES:
                        n = MELD_TILES[t]
                        e["consumed"] = [_STR[b] for b in melds[m:m + n]]
                        m += n
                        if aux:
                            e["target"] = (actor + aux) % 4
            out.append(e)
        out.append({"type": "end_kyoku"})
        return out

    def events(self, g: int) -> list[dict]:
        """Game g's mjai events, start_game to end_game."""
        out = [{"names": self.names(g), "type": "start_game"}]
        for k in self.kyokus(g):
            out += self.kyoku_events(k)
        return out + [{"type": "end_game"}]

    def lines(self, g: int) -> list[str]:
        """Game g's mjai lines as yamawake's engine writes them (`start_game` with the names)."""
        return [dumps(e) for e in self.events(g)]

    def op_counts(self) -> np.ndarray:
        """(12,) events of each op over the whole records (`OPS` order), without a Python loop."""
        return np.bincount(np.asarray(self.a["events"][:, 0]) >> 4, minlength=len(OPS))[:len(OPS)]


def _read_mjai(path: Path) -> list[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_records(path: str | Path) -> bool:
    """`path` is a records directory (its meta.json names the format)."""
    meta = Path(path) / "meta.json"
    return meta.is_file() and json.loads(meta.read_text()).get("format") == FORMAT


def read_games(path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    """(game id, mjai events) of every game a `--record-logs` directory holds, in order.

    Reads a records directory, or else the directory's `*.json.gz` mjai logs (sorted; the id is the
    file name without its suffixes). Both give the same ids and events for the same games.

    Raises:
        FileNotFoundError: The directory holds neither.
    """
    if is_records(path):
        r = Records(path)
        for g in range(len(r)):
            yield r.game_id(g), r.events(g)
        return
    files = sorted(Path(path).glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no records and no *.json.gz logs in {path}")
    for f in files:
        yield f.name.split(".")[0], _read_mjai(f)


def convert(src: str | Path, out: str | Path, pattern: str = "*.json.gz") -> dict:
    """Write the mjai logs `src/**/pattern` (sorted by path) as one records directory.

    Each game's id (the file name without its suffixes) goes to `game_id`; its final scores are
    `final_scores` of its log. Logs that do not fit the format are skipped and listed in the meta.
    """
    src = Path(src)
    w, skipped = RecordWriter(), {}
    paths = sorted(src.rglob(pattern))
    for path in paths:
        gid = path.name.split(".")[0]
        try:
            w.add_game(_read_mjai(path), id=gid)
        except (ValueError, KeyError, TypeError, EOFError, OSError) as e:     # JSONDecodeError is a ValueError
            skipped[str(path.relative_to(src))] = f"{type(e).__name__}: {e}"
    meta = {"source": {"dir": str(src), "pattern": pattern, "logs": len(paths), "skipped": skipped}}
    w.save(out, meta)
    return {"games": len(w), "skipped": len(skipped)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m yamawake.records", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="mjai logs (gzip or plain jsonl) -> a records directory")
    c.add_argument("src")
    c.add_argument("out")
    c.add_argument("--pattern", default="*.json.gz")
    args = ap.parse_args(argv)
    print(convert(args.src, args.out, args.pattern))


if __name__ == "__main__":
    main()
