# Third-party notices

yamawake-mahjong-arena is MIT-licensed (`LICENSE`). It contains or derives from the
following third-party work.

## riichienv-core 0.4.8 (RiichiEnv)

- Source: https://github.com/smly/RiichiEnv (crate `riichienv-core` 0.4.8), by smly
- License: Apache License 2.0 -- full text in `src/engine/riichienv-core/LICENSE`
  (taken from the riichienv 0.4.8 distribution; the crate ships no NOTICE file)
- Use: vendored and modified under `src/engine/riichienv-core/`. Changes: split
  per-seat draw queues and an explicit (kyoku, honba) wall table, both opt-in,
  and an incremental shanten type (`shanten::HandShanten`) over its own
  tables (`src/engine/README.md`, `src/engine/riichienv-core.patch`). The Python
  package `riichienv` (same license) is a runtime dependency.

## majiang-ai (電脳麻将 / Majiang)

- Source: https://github.com/kobalab/majiang-ai, by Satoshi Kobayashi (kobalab)
- License: MIT (text below)
- Use: `src/rust/rulebot/kobalab_ev.rs` is a port of majiang-ai's `eval_shoupai` /
  `eval_fulou`; the rulebot's call logic follows its `select_fulou`,
  `src/rust/rulebot/kan.rs` (ankan / kakan) is a port of its `select_gang`, and
  `src/rust/rulebot/yaku_shanten.rs` follows its `Player#xiangting`. The wait-shape
  danger weights (`src/rust/rulebot/danger.rs`, `base_danger` and the tanki /
  shanpon / kanchan / penchan / ryanmen scheme) and the call-ways weights
  (`src/rust/rulebot/discard.rs`, `call_ways`) follow majiang-ai's `suanpai.js`.

```
MIT License

Copyright (c) 2021 Satoshi Kobayashi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Nyanten

- Source: https://github.com/Cryolite/nyanten, by Cryolite
- License: MIT for the files used here (text below). The repository's GPLv3
  files (`nyanten/test/correctness.cpp`, `nyanten/test/performance_baseline.cpp`)
  are not used or included.
- Use: the shanten lookup tables in the vendored riichienv-core
  (`src/engine/riichienv-core/src/shanten.rs`, and the tables at
  `src/engine/riichienv-core/src/data/nyanten_*.bin`) are
  generated from Nyanten's tables by riichienv.

```
MIT License

Copyright (c) 2024 Cryolite

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Method credit (no code included)

- The single-player DP (`src/rust/eval/dp.rs`, `src/rust/eval/score.rs`) implements the method
  of tomohxx's *Mahjong Algorithm* book, ch. 6 "win probability"
  (https://tomohxx.github.io/mahjong-algorithm-book/probability/),
  following the explanations of nekobean's mahjong-cpp
  (https://github.com/nekobean/mahjong-cpp, GPL-3.0).
  Some implementation details follow libriichi (Mortal,
  https://github.com/Equim-chan/Mortal, AGPL-3.0).
  yamawake-mahjong-arena contains no code from any of them.

