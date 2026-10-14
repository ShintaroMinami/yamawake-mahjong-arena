# Vendored riichienv-core 0.4.8 (fork)

`riichienv-core/` is the crates.io source of
[riichienv-core](https://github.com/smly/RiichiEnv) 0.4.8, licensed under
Apache-2.0 (`riichienv-core/LICENSE`, taken from the riichienv 0.4.8 wheel;
the crate ships no license file of its own). yamawake-mahjong-arena links it statically.

The fork changes four files (`riichienv-core.patch`, against the crates.io
source, made with `diff -u` per file with `a/` and `b/` labels). It adds
`shanten::HandShanten` (marked "yamawake patch"): a hand's Nyanten suit keys
and chiitoitsu/kokushi tallies, giving the shanten with one tile added or
removed by rehashing one suit, with the same values as
`calc_shanten_from_counts` (yamawake's DP and suzume use it). And it adds
two opt-in wall controls to `GameState`, both bit-identical to stock when
off:

- `wall_table`: explicit walls keyed by (kyoku ordinal, honba), so renchan can
  never shift which wall a game-progress coordinate is played on; missing keys
  fall back to the seeded shuffle.
- `split_wall`: each seat draws only from its own fixed queue of the kyoku's
  live wall (the dead wall stays shared; a seat whose queue runs dry borrows
  from the longest), so calls never shift another seat's draws.

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
