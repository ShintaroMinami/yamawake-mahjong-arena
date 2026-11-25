# Vendored riichienv-core 0.4.8 (fork)

`riichienv-core/` is the crates.io source of
[riichienv-core](https://github.com/smly/RiichiEnv) 0.4.8, licensed under
Apache-2.0 (`riichienv-core/LICENSE`, taken from the riichienv 0.4.8 wheel;
the crate ships no license file of its own). yamawake-mahjong-arena links it statically.

The fork changes twelve files (`riichienv-core.patch`, against the crates.io
source, made with `diff -u` per file with `a/` and `b/` labels). It adds
`shanten::HandShanten` (marked "yamawake patch"): a hand's Nyanten suit keys
and chiitoitsu/kokushi tallies, giving the shanten with one tile added or
removed by rehashing one suit, or with every tile at once
(`lowering_draws`, `keeping_discards`), with the same values as
`calc_shanten_from_counts` (yamawake's DP and suzume use it). For the same
users it adds `HandEvaluator::verdict`, what `calc` says of a win (han, fu,
yakuman) without building the score and the yaku list, removes a division
search `calc` did not use, and stops `calculate_yaku` filling `yaku_names`,
which nothing read (the han, fu and yaku ids are unchanged); `verdict` runs
`calculate_yaku` without allocating (the divisions, their mentsu and the yaku
ids held in fixed arrays, `agari::find_divisions_inline`: the same search in
the same order, 2026-11-25); its hand-level
checks (tanyao, the flushes, honroutou, the all-honor / all-terminal /
all-green yakuman) read one bit mask of the tile types taken per evaluation
instead of a pass over the counts each (the same results, pinned by a
fingerprint test in `src/rust/eval/yaku_check.rs`). The riichi
check of the legal actions skips its per-discard tenpai tests when the hand's
shanten is 1 or more (no discard can leave such a hand tenpai; the same
actions, tested in `src/rust/eval/shanten.rs`). And it adds
three opt-in controls to `GameState`, all bit-identical to stock when off:
two for the wall and one for the end of a kyoku.

- `wall_table`: explicit walls keyed by (kyoku ordinal, honba), so renchan can
  never shift which wall a game-progress coordinate is played on; missing keys
  fall back to the seeded shuffle.
- `split_wall`: each seat draws only from its own fixed queue of the kyoku's
  live wall (the dead wall stays shared; a seat whose queue runs dry borrows
  from the longest), so calls never shift another seat's draws.
- `defer_next_round`: stop at the end of a kyoku instead of dealing the next
  one in the same `step`; the finished kyoku's hands, scores and sticks stay
  in place, and the next `step` deals on. yamawake's `play_out` plays a kyoku
  out from a position in its middle with it (`docs/play_out_spec.md`).

It also fills in what a consumer would otherwise rebuild from the events
(marked "yamawake patch"):

- `Observation` carries `drawable_count` (the live wall left) and
  `last_discarder` (the seat of `last_discard`); `last_discard` itself is
  fixed to hold the tile (stock 0.4.8 read the stored seat as the tile). Both
  new fields default when absent, so the stock wheel still reads the JSON.
- `hora` events carry the win's `han`, `fu`, `yaku` (ids, ascending) and
  `yakuman`, as scored by the engine.

And it makes the game flow follow Tenhou's and Mahjong Soul's ranked rules
where stock does not (marked "yamawake patch"; tested in `src/tests.rs`,
`game_flow_tests`):

- Five `GameRule` flags, all off giving the stock flow:
  `abortive_draw_plays_on` (an abortive draw in the all-last never ends the
  game by the dealer's lead), `nagashi_renchan_on_tenpai` (a nagashi mangan
  keeps the deal when the dealer is tenpai, not when the dealer made it),
  `extension_renchan_plays_on` (a kept deal in the West round ends the game
  only with the dealer on top with 30,000 or more, not on anyone's 30,000),
  `extension_multi_ron_renchan_plays_on` (the same for a deal kept by a
  multiple ron that includes the dealer) and `leftover_sticks_to_top` (the
  riichi sticks left at the end of a game of rounds go to first place).
  `default_tenhou()` turns all five on, `default_mjsoul()` all but
  `extension_renchan_plays_on`.
- Under either preset: a triple-ron abortive draw on a riichi declaration
  tile leaves that riichi unaccepted (no stick), and a `ryukyoku`'s `deltas`
  are its payments only (stock kept the riichi sticks in them unless noten
  payments overwrote them).

With these, 20,000 evenly spaced Tenhou Houou logs (2016-2020) play again move
for move on walls built from the logs (`docs/arena.md`).

And it fixes four bugs of the MJAI replay's win contexts
(`replay/mjai_replay.rs` and `replay/mod.rs`, marked "yamawake patch"; the
first three are the change of smly/RiichiEnv#268 for #267): the live wall left
starts at the start-of-round count (stock counted it down twice, so mid-round
wins were taken for haitei / houtei), a kan dora marker is counted once, an
ankan takes its own tiles out of the hand (stock took 1m), and the rinshan
flag of a kan or kita stays through the replacement draw (stock cleared it
there, so MJAI replays never had rinshan kaihou; the 3-player engine also
takes a kita's replacement draw as rinshan). yamawake builds the core without
its `python` feature, so the replay is not compiled in. With the four fixes,
all 2,671 wins of 300 Tenhou Houou hanchan replay with their recorded han, fu
and yaku (checked on riichienv built with the same changes).

Everything else is unchanged. The Python side uses the stock `riichienv==0.4.8`
wheel: observations cross from this core to the wheel serialized (base64), so
the two must stay on the same version.
