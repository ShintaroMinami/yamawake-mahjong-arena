"""play_out / legal_actions / seat_options (`docs/play_out_spec.md`): a cut arena game plays on exactly as it did."""
from __future__ import annotations

import gzip
import json
import random

import pytest

from yamawake import build_agent, legal_actions, lineup, play_out, run_match, seat_options


def _kyokus(out):
    """Every recorded kyoku's events, start_kyoku to its last event before end_kyoku."""
    ks = []
    for f in sorted(out.glob("arena-*.json.gz")):
        cur = None
        for line in gzip.open(f, "rt"):
            e = json.loads(line)
            if e["type"] == "start_kyoku":
                cur = [e]
            elif cur is not None:
                if e["type"] == "end_kyoku":
                    ks.append(cur)
                    cur = None
                else:
                    cur.append(e)
    return ks


def _rinshan(ev, i):
    j = i - 1
    while ev[j]["type"] == "dora":
        j -= 1
    return ev[j]["type"] in ("ankan", "kakan", "daiminkan") and ev[j].get("actor") == ev[i]["actor"]


def _logged(ev, j):
    """The logged action at j in the spec's form, with its seat (None if j is no decision's action)."""
    e, t = ev[j], ev[j]["type"]
    prev = ev[j - 1]
    own = prev["type"] == "tsumo" and prev.get("actor") == e.get("actor")
    react = prev["type"] in ("dahai", "kakan") and t in ("chi", "pon", "daiminkan", "hora")
    if not (own or react):
        return None
    if t == "dahai":
        return e["actor"], {"type": "dahai", "pai": e["pai"], "tsumogiri": e["tsumogiri"]}
    if t == "reach":
        return e["actor"], {"type": "reach", "pai": ev[j + 1]["pai"], "tsumogiri": ev[j + 1]["tsumogiri"]}
    if t in ("chi", "pon", "daiminkan", "kakan"):
        return e["actor"], {"type": t, "pai": e["pai"], "consumed": e["consumed"]}
    if t == "ankan":
        return e["actor"], {"type": "ankan", "consumed": e["consumed"]}
    if t == "hora":
        return e["actor"], {"type": "hora"}
    return None


def _future(ev, j, split):
    """The true hidden future after j, as far as the log shows it."""
    live, queues, rinshan, dora, ura = [], [[], [], [], []], [], [], []
    for i in range(j, len(ev)):
        e = ev[i]
        if e["type"] == "tsumo":
            (rinshan if _rinshan(ev, i) else live).append(e["pai"])
            if not _rinshan(ev, i):
                queues[e["actor"]].append(e["pai"])
        elif e["type"] == "dora":
            dora.append(e["dora_marker"])
        elif e["type"] == "hora" and e.get("ura_markers") and not ura:
            ura = e["ura_markers"]
    s = {"rinshan": rinshan, "dora": dora, "ura": ura, "seed": 0}
    s["wall_queues" if split else "wall"] = queues if split else live
    return s


def _hands(ev):
    h = [list(t) for t in ev[0]["tehais"]]
    for e in ev[1:]:
        t, a = e["type"], e.get("actor")
        if t == "tsumo":
            h[a].append(e["pai"])
        elif t in ("dahai", "kakan"):
            h[a].remove(e["pai"])
        elif t in ("chi", "pon", "daiminkan", "ankan"):
            for c in e["consumed"]:
                h[a].remove(c)
    return h


def _masked(ev, seat):
    """The events with the other seats' deal and draws as "?", and their true hands."""
    out = [dict(ev[0], tehais=[t if s == seat else ["?"] * 13 for s, t in enumerate(ev[0]["tehais"])])]
    out += [dict(e, pai="?") if e["type"] == "tsumo" and e["actor"] != seat else e for e in ev[1:]]
    h = _hands(ev)
    return out, {str(s): h[s] for s in range(4) if s != seat}


def _cuts(ks, n, seed=0):
    rng = random.Random(seed)
    cuts = [(ev, j) for ev in ks for j in range(1, len(ev)) if _logged(ev, j)]
    return rng.sample(cuts, min(n, len(cuts)))


@pytest.fixture(scope="module", params=[("rulebot", False), ("rulebot", True), ("suzume", False), ("suzume", True)],
                ids=["rulebot-shared", "rulebot-split", "suzume-shared", "suzume-split"])
def games(request, tmp_path_factory):
    bot, split = request.param
    out = tmp_path_factory.mktemp(f"{bot}{int(split)}")
    a = build_agent(bot)
    run_match([a, a, a, a], walls=2, split_wall=split, record_logs=out)
    return bot, split, _kyokus(out)


def test_a_cut_game_plays_on_as_it_did(games):
    """The true hidden tiles and the logged action: the rest of the kyoku comes out event for event, with the
    other seats' tiles masked in half the cases."""
    bot, split, ks = games
    cases = _cuts(ks, 60)
    positions = []
    for k, (ev, j) in enumerate(cases):
        seat, act = _logged(ev, j)
        p = {"events": ev[:j], "seat": seat, "actions": [act], "samples": [_future(ev, j, split)]}
        if k % 2:
            p["events"], p["samples"][0]["hands"] = _masked(ev[:j], seat)
        positions.append(p)
    res = play_out(positions, bot, split_wall=split, record_events=True)
    for (ev, j), r in zip(cases, res):
        r = r[0][0]
        # The play-out marks rinshan draws; the arena's log does not.
        got = [{k: v for k, v in e.items() if k != "rinshan"} for e in r["events"]]
        assert got == ev[j:]
        assert [i for i, e in enumerate(r["events"]) if e.get("rinshan")] == \
               [i - j for i in range(j, len(ev)) if ev[i]["type"] == "tsumo" and _rinshan(ev, i)]
        assert r["forced"] == 1 and r["end"] == ("hora" if any(e["type"] == "hora" for e in ev[j:]) else "ryukyoku")


def test_the_logged_move_is_legal_and_suzume_chooses_it(games):
    bot, split, ks = games
    if split:
        pytest.skip("legal actions do not depend on the wall")
    positions = []
    for ev, j in _cuts(ks, 40, seed=1):
        seat, act = _logged(ev, j)
        p = {"events": ev[:j], "seat": seat}
        legal = legal_actions(p)
        norm = lambda m: {**m, "consumed": sorted(m["consumed"])} if "consumed" in m else m  # noqa: E731
        assert norm(act) in [norm(m) for m in legal]
        positions.append((p, act))
    opts = seat_options([p for p, _ in positions], bot)
    for (p, act), o in zip(positions, opts):
        assert o["choice"] in o["legal"]
        if bot == "suzume":
            # The game was suzume's own: its choice at the cut is the move it made (red / tsumogiri included).
            c, a = o["choice"], act
            assert (c["type"], c.get("pai"), sorted(c.get("consumed", []))) == \
                   (a["type"], a.get("pai"), sorted(a.get("consumed", [])))


@pytest.fixture(scope="module")
def kyoku(tmp_path_factory):
    out = tmp_path_factory.mktemp("one")
    run_match(lineup(build_agent("suzume"), build_agent("rulebot"), "2v2"), walls=1, record_logs=out)
    return max(_kyokus(out), key=len)


def test_samples_by_seed_and_by_prefix(kyoku):
    ev = kyoku
    j = next(i for i in range(10, len(ev)) if _logged(ev, i) and ev[i]["type"] == "dahai")
    seat, act = _logged(ev, j)
    base = {"events": ev[:j], "seat": seat, "actions": [act, []]}
    r = play_out([{**base, "samples": [{"seed": 5}, {"seed": 5}, {"seed": 6}]}], record_events=True)[0]
    assert r[0][0] == r[0][1] and r[1][0] == r[1][1]
    assert r[0][0]["events"] != r[0][2]["events"]
    # A given wall prefix comes first: the next draw is its first tile.
    first = "C" if ev[j]["pai"] != "C" else "E"
    unseen = play_out([{**base, "samples": [{"wall": [first], "seed": 1}]}], record_events=True, on_error="record")
    got = unseen[0][0][0]
    if "error" not in got:  # the tile may be all visible already
        assert next(e for e in got["events"] if e["type"] == "tsumo")["pai"] == first
    assert play_out([{**base, "samples": [{"seed": 2}]}], threads=1) == play_out([{**base, "samples": [{"seed": 2}]}])


def test_bad_positions_are_errors_with_their_place(kyoku):
    ev = kyoku
    j = next(i for i in range(10, len(ev)) if _logged(ev, i) and ev[i]["type"] == "dahai")
    seat, _ = _logged(ev, j)
    base = {"events": ev[:j], "seat": seat}
    with pytest.raises(ValueError, match="position 0, action 0, sample 0: forced action 1:.*not legal"):
        play_out([{**base, "actions": [{"type": "pon", "pai": "1m", "consumed": ["1m", "1m"]}], "samples": [{"seed": 0}]}])
    with pytest.raises(ValueError, match=r"hands\[\d\] holds 1 tiles, not 1[34]"):
        play_out([{**base, "samples": [{"hands": {str((seat + 1) % 4): [ev[0]["tehais"][(seat + 1) % 4][0]]}, "seed": 0}]}])
    with pytest.raises(ValueError, match="more than"):
        play_out([{**base, "samples": [{"wall": ["1m"] * 80, "seed": 0}]}])
    with pytest.raises(ValueError, match="needs split_wall"):
        play_out([{**base, "samples": [{"wall_queues": [[], [], [], []], "seed": 0}]}])
    with pytest.raises(ValueError, match="no seed"):
        play_out([{**base, "samples": [{}]}])
    with pytest.raises(ValueError, match="past the kyoku's end"):
        legal_actions({"events": ev + [{"type": "end_kyoku"}], "seat": seat})
    rec = play_out([{**base, "samples": [{}]}], on_error="record")
    assert "no seed" in rec[0][0][0]["error"]
    other = next(s for s in range(4) if s != seat and s not in [e.get("actor") for e in ev[j - 1:j]])
    with pytest.raises(ValueError, match="does not decide"):
        legal_actions({**base, "seat": other})


def _full_set():
    from collections import Counter
    c = Counter()
    for suit in "mps":
        for n in range(1, 10):
            c[f"{n}{suit}"] = 3 if n == 5 else 4
        c[f"5{suit}r"] = 1
    for h in "ESWNPFC":
        c[h] = 4
    return c


def _visible(ev):
    """The public tiles of the events: discards, the hand tiles of melds, the dora indicators shown."""
    from collections import Counter
    c = Counter([ev[0]["dora_marker"]])
    for e in ev[1:]:
        t = e["type"]
        if t == "dahai" or t == "kakan":
            c[e["pai"]] += 1
        elif t in ("chi", "pon", "daiminkan", "ankan"):
            c.update(e["consumed"])
        elif t == "dora":
            c[e["dora_marker"]] += 1
    return c


@pytest.mark.parametrize("split", [False, True], ids=["shared", "split"])
def test_a_recorded_sample_rebuilds_the_same_play_out(kyoku, split):
    """`record_samples` returns the sample as it came out: with the public tiles it makes up the full set, and fed
    back without a seed it plays the same kyoku; rinshan draws are marked."""
    from collections import Counter
    ev = kyoku
    positions = []
    for j in (5, len(ev) // 3, len(ev) // 2):
        while not _logged(ev, j):
            j += 1
        seat, act = _logged(ev, j)
        masked, _ = _masked(ev[:j], seat)
        for events in (ev[:j], masked):
            positions.append({"events": events, "seat": seat, "actions": [act, []],
                              "samples": [{"seed": 11}, {"seed": 12}]})
    res = play_out(positions, split_wall=split, record_events=True, record_samples=True)
    again = []
    for p, r in zip(positions, res):
        samples = [r[0][s]["sample"] for s in range(2)]
        assert all(r[a][s]["sample"] == samples[s] for a in range(2) for s in range(2))
        again.append({**p, "samples": samples})
        tiles = _visible(p["events"]) + Counter(sum(r[0][0]["hands"], []))
        smp = samples[0]
        for k in ("rinshan", "dora", "ura", "undrawn"):
            tiles.update(smp[k])
        tiles.update(sum(smp["wall_queues"], []) if split else smp["wall"])
        assert tiles == _full_set()
    assert play_out(again, split_wall=split, record_events=True, record_samples=True) == res
    for r in res:
        for x in (y for a in r for y in a):
            evs = x["events"]
            for i, e in enumerate(evs):
                if e.get("rinshan"):
                    prev = next(y for y in reversed(evs[:i]) if y["type"] != "dora")
                    assert prev["type"] in ("ankan", "kakan", "daiminkan") and prev["actor"] == e["actor"]
