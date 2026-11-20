# Vendored riichienv-core 0.4.8 (fork)

`riichienv-core/` is the crates.io source of
[riichienv-core](https://github.com/smly/RiichiEnv) 0.4.8, licensed under
Apache-2.0 (`riichienv-core/LICENSE`, taken from the riichienv 0.4.8 wheel;
the crate ships no license file of its own). yamawake-mahjong-arena links it statically.

The fork changes seven files (`riichienv-core.patch`, against the crates.io
source, made with `diff -u` per file with `a/` and `b/` labels). It adds
`shanten::HandShanten` (marked "yamawake patch"): a hand's Nyanten suit keys
and chiitoitsu/kokushi tallies, giving the shanten with one tile added or
removed by rehashing one suit, or with every tile at once
(`lowering_draws`, `keeping_discards`), with the same values as
`calc_shanten_from_counts` (yamawake's DP and suzume use it). For the same
users it adds `HandEvaluator::verdict`, what `calc` says of a win (han, fu,
yakuman) without building the score and the yaku list, removes a division
search `calc` did not use, and stops `calculate_yaku` filling `yaku_names`,
which nothing read (the han, fu and yaku ids are unchanged); its hand-level
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
- `hora` events carry the win's `han`, `fu`, `yaku` (ids) and `yakuman`, as
  scored by the engine.

Everything else is unchanged. The Python side uses the stock `riichienv==0.4.8`
wheel: observations cross from this core to the wheel serialized (base64), so
the two must stay on the same version.
