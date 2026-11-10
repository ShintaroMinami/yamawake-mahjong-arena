# rulebot: a fast rule-based bot

[日本語](rulebot.md) | **English**

rulebot aims to play fast and deterministically, as a baseline for mahjong bots.

- Code: [`src/rust/rulebot/`](../src/rust/rulebot/) (decisions), [`src/rust/eval/`](../src/rust/eval/) (the hand-value DP, scores and shanten; shared with suzume), [`src/python/yamawake/rulebot.py`](../src/python/yamawake/rulebot.py) (Python side: the contents of the two settings `STANDARD` and `KOBALAB`)
- Sources: [`docs/THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

## 1. Decision flow

A single decision (`rulebot::react`) is made in the following order.

1. Win if it can.
2. Concealed/added kan (setting `kan`): a port of majiang-ai's `select_gang` (`kan.rs`). No kan when an opponent has declared riichi and the hand is not ready. From a ready hand to 2-shanten, the kan with the highest hand evaluation after the kan (`kobalab_ev`) among those that do not lower it from before the kan. At 3-shanten or more, the first kan that does not change the yaku-aware shanten number. No open kan.
3. When riichi is possible, decide whether to declare it (`should_riichi`: stay dama if a dama ron scores 6400 points or more even on the lowest-scoring wait, and so on).
4. Discard (`discard.rs`).
5. Call (`call.rs`).

## 2. Discards

With no threat, it picks a discard that advances the hand, by the following steps.
- At 2-shanten or closer: the tile with the highest expected value from the hand-value DP (`sp_adapter.rs`, `eval::dp`).
- Farther than that, or when the DP has no value, it compares
    1. the yaku-aware shanten number (`yaku_shanten.rs`, majiang-ai's `Player#xiangting`)
    2. the number of improving tiles
    3. keeping dora
    4. choosing isolated tiles
    in that order.

Threats are opponents' riichi and calls that look ready (many calls for the turn, and a visible value of 3 or more).
With a threat, the discard is decided in two tiers (`threatened_discard`).

- 2-shanten or more: fold. Discard the safest tile (among equally safe tiles, the one that costs the hand least).
- 1-shanten and ready hand: decide whether to push or fold on a single axis of hand value and danger (`push_or_fold`: subtract danger × a coefficient from each discard's DP expected value, add the safest tile as a candidate with value 0, and pick the highest; the coefficient is 50 at 1-shanten and 5 at a ready hand). When the DP has no value, at 1-shanten, the safest tile among those that keep the shanten number; at a ready hand, the same hand-advancing discard as without a threat.

Danger is estimated by enumerating wait shapes (`danger.rs`: as in majiang-ai's base danger, weighted by the number of unseen tiles, with dora and the dealer weighted more). Against several threats, the largest danger is used, multiplied by 1.6 for each additional threat.

## 3. Calls

It follows majiang-ai's `select_fulou`. At 3-shanten or more, it makes calls that lower the yaku-aware shanten number (no calls when there is a threat). At 2-shanten or closer, it compares the hand value after calling with that after passing, and makes the highest-valued call among those that beat passing. With a threat, it makes no call whose value after the call is under 750 points (250 points when ready). Hand value is measured by the hand-value DP in the standard setting (`call_ev_sp`; the value of passing by `call_sp_flat_pass`, which does not assume riichi), and by `kobalab_ev` (a port of majiang-ai's `eval_shoupai` / `eval_fulou`) in `KOBALAB`.

## 4. Settings

The settings are fields of `RulebotConfig` ([`src/rust/rulebot/config.rs`](../src/rust/rulebot/config.rs)), only those that tell the two settings apart (`call_ev_sp`, `call_sp_flat_pass`, `kan`). The default is the same as `STANDARD` (player `rulebot`): all three on (it declares concealed/added kan). `KOBALAB` has all three off. Values both settings share (50, 5, 1.6, 6400 above, and so on) are constants in the same file. The two settings are kept on the Python side ([`src/python/yamawake/rulebot.py`](../src/python/yamawake/rulebot.py)).

| Setting | Contents |
|---|---|
| `STANDARD` (player name `rulebot`) | The standard setting. Replaces `KOBALAB`'s call evaluation with the hand-value DP and declares concealed/added kan (since 2026-10-13) |
| `KOBALAB` (player name `rulebot:kobalab`) | Uses `kobalab_ev` (a port of majiang-ai's evaluation) for calls and makes no kan. The setting `STANDARD` was built from |

From Python, seat it with `RulebotAgent(setting)`. Fields left out keep the Rust default (on), and an unknown field is an error.

## 5. Deciding a single position

`decide("rulebot", observation, the mjai events of the current kyoku, debug=True)` gives the same decision as in a game, along with what it was based on (shanten, threats, hand value and so on) ([`docs/arena.en.md`](arena.en.md), section 3).

## 6. Tests

- golden ([`tests/golden/`](../tests/golden/)): checks that games of the standard rulebot (rulebot ×4, and 1 tsumogiri vs 3 rulebot) follow the recorded scores and game records.
- Kan ([`tests/rulebot/`](../tests/rulebot/)): checks that the standard setting declares concealed and added kan but no open kan, and that with `kan` off it makes no kan.
- Hand-value DP ([`tests/eval/test_sp.py`](../tests/eval/test_sp.py)): checks that it gives the same answers as the recorded ones ([`tests/eval/run_sp_v01.json`](../tests/eval/run_sp_v01.json)).
