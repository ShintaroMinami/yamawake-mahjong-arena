# yamawake-mahjong-arena

[日本語](README.md) | **English**

### Mahjong on duplicated walls

In duplicate bridge every table plays the same deals. This keeps results from
hinging on the luck of the cards, so they measure the difference in skill
between players more accurately. yamawake-mahjong-arena brings the same idea to
measuring mahjong skill.
Every kyoku's wall is fixed, and each seat draws from its own part of the wall
(*split fixed walls*). This way calls and dealer repeats never shift the
scheduled draws: they hold to the end of the game.
The split fixed walls are assigned to the seats and rotated in turn.
Every player plays every split fixed wall once, so differences in results come
only from the players' choices.

### Main components

- **arena**: the split-fixed-wall arena; a Rust game loop runs N games in
  parallel (see `docs/arena.md`).
- **rulebot**: a fast rule-based bot, useful as a baseline (see
  `docs/rulebot.md`).
- **suzume**: a distilled model of a reinforcement-learned mahjong AI; a light
  bot built on small per-decision MLPs (see `docs/suzume.md`).

On one CPU it runs about 1,180 hanchan/min with four rulebots and about 370
hanchan/min with four suzume bots. The docs under `docs/` are in Japanese.

## Strength of suzume

TODO: suzume's strength measured in matches against
- tsumogiri
- 電脳麻将 (Majiang)
- mortal-298k
- etc.

## Install

### From a release wheel (no Rust toolchain needed)

Each [release](https://github.com/ShintaroMinami/yamawake-mahjong-arena/releases)
carries wheels for Linux x86_64/aarch64 (manylinux2014) and macOS arm64,
CPython 3.11-3.13, with the Rust core precompiled; `riichienv` and `numpy`
come from PyPI.

    gh release download -R ShintaroMinami/yamawake-mahjong-arena -p '*.whl' -D wheels
    pip install --find-links wheels yamawake

### From source

Needs Python >= 3.11, a Rust toolchain and [uv](https://docs.astral.sh/uv/).

    git clone https://github.com/ShintaroMinami/yamawake-mahjong-arena.git
    cd yamawake-mahjong-arena
    uv sync                # builds the extension (maturin) into .venv
    uv run pytest          # tests

Without uv, `pip install git+https://github.com/ShintaroMinami/yamawake-mahjong-arena.git`
also works given a Rust toolchain.

The package is `yamawake` and the command is `yamawake-arena`.

## Usage

    uv run yamawake-arena --mode 1v3 --players tsumogiri rulebot --walls 50
    uv run python -m yamawake.arena --mode 2v2 --players suzume rulebot
    uv run python -m yamawake.handstats logs/ --name suzume     # stats of --record-logs games

    from yamawake import RulebotAgent, TsumogiriAgent, run_match
    r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()], walls=50)
    print(r.table())

| Player | Description |
|---|---|
| `tsumogiri` | a bot that discards every draw |
| `rulebot`, `rulebot:latest` | the fast rule-based bot |
| `suzume` (= `suzume-mlp3`), `suzume-mlp3-flat`, `suzume-mlp3-avoid4`, `suzume-mlp1`, `suzume-mlp1-flat` | light, reasonably strong bots; options as in `suzume-mlp1:call=0` |

## Playing your own agent

An agent is any object with `lane = "observation"`, `act_batch(decisions)` and
`describe()`. `act_batch` receives, in one list, the decisions of every game
waiting on that agent, so a model can batch its inference. Each decision `d`
carries `d.obs` (a `riichienv.Observation`; `d.obs.legal_actions()` lists the
legal moves), the kyoku's mjai-format events `d.kyoku_events`, and more. Pick
one legal action per decision and return them in the same order.

    import riichienv as rv
    from yamawake import RulebotAgent, lineup, run_match

    class MyAgent:
        lane = "observation"

        def act_batch(self, decisions):
            actions = []
            for d in decisions:
                legal = d.obs.legal_actions()
                # win when possible, else the first legal action
                win = [a for a in legal if a.action_type in (rv.ActionType.TSUMO, rv.ActionType.RON)]
                actions.append(win[0] if win else legal[0])
            return actions

        def describe(self):
            return {"kind": "mine", "name": "my-agent"}

    r = run_match(lineup(MyAgent(), RulebotAgent(), "2v2"), walls=100)   # MyAgent x 2 vs rulebot x 2
    print(r.table())

Registering a name with `register_agent("my-agent", lambda spec, **opts: MyAgent())`
lets you name it within the same process, as in
`yamawake.arena.main(["--players", "my-agent", "rulebot", ...])`. To use it from
the `yamawake-arena` command, put the registering function in your package's
`yamawake.agents` entry-point group.

For the low-level loop, single-position decisions (`decide`), log statistics
(`handstats`) and the Tenhou replay converter (`tenhou6`), see `docs/arena.md`.

## Acknowledgements

This project draws on the repositories and writings below. Our heartfelt
thanks to their authors and developers for the excellent software and
insights they share.

- **[RiichiEnv](https://github.com/smly/RiichiEnv)** by smly (Apache-2.0) is the
  simulator's foundation. yamawake-mahjong-arena's game engine is a vendored fork of
  riichienv-core 0.4.8 that adds per-seat draw queues and an explicit
  (kyoku, honba) wall table (`src/engine/README.md`), and the Python side uses
  the `riichienv` package.
- **[電脳麻将 (Majiang)](https://github.com/kobalab/Majiang)** by Satoshi Kobayashi (kobalab;
  [majiang-ai](https://github.com/kobalab/majiang-ai), MIT, and the wider
  電脳麻将 project): the rulebot's design references it.
  `src/rust/rulebot/kobalab_ev.rs` is a port of majiang-ai's `eval_shoupai` /
  `eval_fulou`, the call logic follows its `select_fulou` and the kan logic
  its `select_gang`.
- The algorithm for valuing hands (dynamic programming, DP) implements the method of
  [tomohxx](https://github.com/tomohxx)'s *Mahjong Algorithm* book
  ([ch. 6, win probability](https://tomohxx.github.io/mahjong-algorithm-book/probability/)),
  following the explanations of [nekobean](https://github.com/nekobean)'s
  [mahjong-cpp](https://github.com/nekobean/mahjong-cpp). Some implementation
  details follow libriichi ([Mortal](https://github.com/Equim-chan/Mortal)).
- Also used, with thanks: through riichienv-core, the shanten tables of
  [Nyanten](https://github.com/Cryolite/nyanten) (MIT).

See `docs/THIRD_PARTY_NOTICES.md` for their full license texts.

## License

This repository is licensed under MIT (`LICENSE`). The vendored riichienv-core
is Apache-2.0 (`src/engine/riichienv-core/LICENSE`).
