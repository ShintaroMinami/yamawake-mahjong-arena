# arena: a mahjong arena on split fixed walls

[日本語](arena.md) | **English**

- Code: [`src/rust/arena/`](../src/rust/arena/) (the game loop), [`src/python/yamawake/`](../src/python/yamawake/) (the Python interface)
- Game engine: a fork of riichienv-core 0.4.8 ([`src/engine/README.md`](../src/engine/README.md))

## 1. How it works

- **Split walls**: each seat draws from its own wall. Calls do not shift any seat's draws.
- **Fixed walls**: a 136-tile wall is fixed for each kyoku × honba (East 1 to North 4 × 12). Dealer repeats do not change the wall used in each kyoku.
- **Seat rotation**: one wall is played with the seats rotated, so that every player plays the same wall once from each seat. In 1-vs-3 it is played four times with the seats shifted; in 2-vs-2 (`[A, B, A, B]`) twice (the other two rotations would repeat the same games).
- **Standard walls**: by default, walls are generated from `wall_base 0` and the first 1,000 are used (configurable). If the same predefined walls are always used, differences in results reflect only differences between the players.

The arena runs N games in parallel. rulebot and suzume decide inside Rust; Python is called only on a Python player's turn.

Apart from the decision itself, each decision of a Python player costs about 20 µs for passing data back and forth (converting the observation built in Rust for Python and reading the result back, and assembling the per-decision record). This is about twice the arena's own time per decision (about 10 µs). With an ordinary model such as suzume (about 0.16 ms per decision) it is not noticeable, but when you evaluate a very fast algorithm, this overhead can affect the overall speed. For example, the tsumogiri bot chooses the same moves either way, yet four `tsumogiri` deciding inside Rust play about 3,070 hanchan/min, whereas four `tsumogiri-python` deciding in Python play about 1,570 hanchan/min (one thread of an Apple M4 Pro, 20 walls, no seat rotation). The arena row of the Time table includes this overhead.

## 2. Command line

``` bash
uv run yamawake-arena --mode 2v2 --players suzume rulebot
uv run yamawake-arena --players rulebot rulebot:kobalab rulebot rulebot \
    --walls 1000 --out results.jsonl --save-scores scores.npz --record-logs logs/
```

| Option | Meaning |
|---|---|
| `--players` | the four players. Together with `--mode`, two are enough |
| `--mode 1v3\|2v2` | arranges the two players as `[A, B, B, B]` or `[A, B, A, B]` |
| `--walls N` | number of walls (default: the first 1,000 standard walls) |
| `--wall-base fresh\|Integer` | `fresh` draws new walls. Passing a value reproduces those walls |
| `--wall-seed S` / `--wall-start K` | plays `--walls` walls starting from the K-th wall of seed S (S and K at least 0 and below 2³²; default K = 0). To add walls, pass the number already played as K. Walls from different seeds do not overlap. Cannot be combined with `--wall-base` |
| `--opt KEY=VALUE` | options passed to the players (rulebot settings and so on) |
| `--single-kyoku` | plays only East 1 in every game (see below) |
| `--no-rotate` | plays only one hanchan per wall, without rotating the seats (see below) |
| `--no-split` | does not use split walls: all seats draw from one wall, as in ordinary mahjong (see below) |
| `--game-rule tenhou\|mjsoul` | the table rule; Tenhou's by default (see below) |
| `--out FILE` | appends a summary of the results to a jsonl file |
| `--save-scores FILE` | writes each game's final scores and seat order to an npz file |
| `--record-logs DIR` | writes the game logs of all games in mjai format (gzip) |
| `--record-format mjai\|records` | the format of `--record-logs` (since 0.17.0): `records` writes one directory of arrays ([`records.md`](records.md), Japanese; to a new or empty directory) |
| `--no-stats` | skips statistics such as the win rate (they are computed from the game logs, which makes the run about 3% slower) |
| `--flight N` | number of games run concurrently (default 256) |

Progress is shown as a bar on standard error, and the results go to standard output as the tables below. The numbers in parentheses under the values are standard errors with walls as the unit.

| Table | Contents |
|---|---|
| Standings | mean rank, points, and the rates of finishing 1st to 4th. In 2-vs-2, also the differences paired by wall |
| Play per kyoku | per-kyoku win rate, deal-in rate, call rate (kyoku with a chi, pon or open kan), riichi rate (kyoku with a declaration), mean win points (the mean kyoku balance over kyoku won, including honba and deposits, minus one's own riichi stick), mean deal-in points (points paid on deal-ins, including honba, not counting one's own riichi stick), and adjusted efficiency (win rate × mean win points − deal-in rate × mean deal-in points, in points per kyoku) |
| Time | each player's number of decisions, time, and time per decision, and the time of the arena itself (the game engine and the loop). For the built-in bots and the arena, the sum of per-thread times; for Python players, the time spent in `act_batch` |

The call and riichi rates in Play per kyoku use all kyoku as the denominator. The style coordinates `call_rate` and `riichi_rate` ([`suzume.en.md`](suzume.en.md)) use the kyoku in which a call or riichi was available, so their values differ.

Points are settled with 25,000 starting points, a 30,000 return, rank points of +90 / +45 / 0 / −135, and an oka of 20 (`rules.DEFAULT_RULE`).

`--game-rule` (`run_match(game_rule=...)`, since 0.17.0) chooses the rule for the yaku and the game flow. Since 0.17.0 the default (`tenhou`) game flow is Tenhou's, and the riichi sticks left at the end go to first place (before, to nobody). The settlement (the points above) is the same under both.

| Rule | Contents |
|---|---|
| `tenhou` (default) | Tenhou's ranked games |
| `mjsoul` | Mahjong Soul's ranked games (four players, East-South) |

The two rules differ as follows.

| Item | `tenhou` | `mjsoul` |
|---|---|---|
| Double yakuman (thirteen-wait kokushi, single-wait suuankou, pure chuuren, daisuushii) | no | yes |
| Triple ron | abortive draw | a triple win (the winner nearest the discarder takes the honba and the riichi sticks) |
| Kokushi robbing a closed kan | no | yes |
| Pao on a composite yakuman (daisangen, daisuushii) | liable for all of it | only for the pao yakuman |
| A kept deal in the West round | ends the game only with the dealer on top with 30,000 or more | anyone's 30,000 ends the game (except a deal kept by a double or triple ron that includes the dealer) |

The two share the following.

- In the all-last, a dealer win or a dealer tenpai at the draw ends the game with the dealer on top with 30,000 or more.
- An abortive draw (nine terminals and so on) in the all-last plays on even with the dealer on top.
- A nagashi mangan keeps the deal when the dealer is tenpai.
- With nobody at 30,000 at the end of South 4 the game goes into the West round, and it ends with West 4 at the latest.
- The riichi sticks left at the end of the game go to first place (on a tie, the seat nearer the first dealer).

The `tenhou` game flow was checked by playing 20,000 evenly spaced Tenhou game logs (Houou table, East-South with red fives, 2016 to 2020) again move by move on walls built from the logs (every move, every kyoku's start, the dora and the final scores agree). `mjsoul` follows Mahjong Soul's official rules (<https://mahjongsoul.com/news/46>). They do not state what an abortive draw does in the all-last; since the automatic stop covers only a dealer win and a dealer tenpai, the game plays on.

With `--single-kyoku`, everyone starts with 25,000 points at 0 honba and plays the same wall as East 1 of a hanchan, and the game ends after that one kyoku. It is meant for comparing hand building without the influence of the score situation, and reports the kyoku balance and the win and deal-in rates.

With `--no-rotate` (`run_match(rotate=False)`), only one hanchan is played per wall. The seats are rotated in turn by wall number (wall w uses rotation w % 4, or w % 2 in 2-vs-2), and every hanchan uses a different wall, so they are independent of one another. Use it as a baseline for comparison, for example when measuring how much seat rotation reduces variance. Standard errors are then per hanchan.

With `--no-split` (`run_match(split_wall=False)`), split walls are not used and all seats draw in turn from one wall. Calls shift the draws, as in ordinary mahjong. To play on ordinary random walls, pass `--no-rotate --no-split --wall-base fresh` (see the [README](../README.en.md#arena-settings)).

## 3. Python

``` python
from yamawake import RulebotAgent, TsumogiriAgent, build_agent, run_match

r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()], walls=50)
print(r.table())          # standings (same as the command line's Standings table)
r.paired()                # 2v2 differences paired by wall
agent = build_agent("suzume-rla225")   # same spec as on the command line
```

`run_match` takes the same arguments as the command-line options (`walls`, `wall_base`, `single_kyoku`, `record_logs`, `save_scores` and so on). Statistics are collected only when `play_stats=True` and can be read with `r.play_stats()`. Timing is in `r.time_stats()`, and `yamawake.report.report(r)` gives the same tables as the command line. `run_match` itself prints nothing.

**Running a match in parts**: `match_plan(players, walls=..., wall_base=...)` (since 0.16.0) returns the hanchan `run_match` plays, without playing them. Each has its wall index, seat rotation, core seed, agent per seat, seat per slot and wall, in `run_match`'s order (by wall, then rotation). Split them into any parts and play each with `play_games(**plan.play_args(games))`: every hanchan gets the scores `run_match` gives it. `run_match` itself plays from this list.

``` python
from yamawake import RulebotAgent, TsumogiriAgent, lineup, match_plan, play_games

plan = match_plan(lineup(TsumogiriAgent(), RulebotAgent(), "1v3"), walls=100)
part = plan.games[:40]                          # e.g. the first 10 walls (4 rotations each)
scores, _ = play_games(**plan.play_args(part))  # scores by seat (hanchan x 4)
```

**Single-position decisions**: `decide(player spec, observation, kyoku_events)` returns one decision of rulebot or suzume.
suzume returns it with `a1` as `rule`; rulebot, with `debug=True`, attaches the inputs to its decision.

## 4. Your own player

A player is any object that has the following.

| Attribute | Contents |
|---|---|
| `lane = "observation"` | must be this value. It tells the arena to pass observations to `act_batch` and let it choose the moves (the built-in rulebot and suzume decide inside Rust and use a different value) |
| `act_batch(decisions)` | takes a list of decisions and returns one legal action for each, as a list in the same order. The decisions of every game waiting on that player's turn arrive in one call, so a model can batch its inference |
| `describe()` | the name shown in the results and game logs (`{"kind": ..., "name": ...}`) |
| `reducer_factory` (optional) | creates an object that folds the events of each kyoku. It is rebuilt at the start of each kyoku, and `.apply(event)` is called for each event. Readable as `Decision.reducer` |
| `annotate(decision)` (optional) | returns a dict for each decision. It goes into `Decision.extra` |

Each decision, a `Decision`, has the following.

| Attribute | Contents |
|---|---|
| `obs` | a `riichienv.Observation`, the same as the observation the RiichiEnv environment returns ([RiichiEnv documentation](https://github.com/smly/RiichiEnv#observation)). Legal actions `legal_actions()`, hand `hand`, drawn tile `drawn_tile`, discards `discards`, scores `scores`, and so on. Tiles use the 136-tile numbering (`// 4` gives the tile type, 0–33) |
| `seat` | one's own seat number (0–3; the first dealer is 0, the same numbering as mjai's `actor`) |
| `kyoku_events` | the mjai-format events from the start of the kyoku until now (other players' draws hidden). `obs.events` holds only the events since one's previous turn, so use this to follow the course of the kyoku |
| `drawable_count` | number of tiles left in the wall |
| `reacting_to` | when responding to another player's discard or added kan (call, ron, robbing the kan), that player's seat number. `None` on one's own turn |
| `dora_indicator_types` | tile types of the dora indicators (0–33) |

### Plugging in a trained model

Below is a rough example of plugging in your own model.
Replace `my_encode` (a function that makes features from an observation) and `model.predict` (a function that runs batched inference) with your own.

``` python
import numpy as np
from yamawake import RulebotAgent, lineup, run_match


class MyModelAgent:
    lane = "observation"

    def __init__(self, model):
        self.model = model                                   # your trained model

    def act_batch(self, decisions):
        x = np.stack([my_encode(d) for d in decisions])      # your features, one row per decision
        scores = self.model.predict(x)                       # one batched forward: (n, n_actions)
        actions = []
        for d, s in zip(decisions, scores):
            legal = d.obs.legal_actions()
            actions.append(max(legal, key=lambda a: s[a.encode()]))   # best legal action
        return actions

    def describe(self):
        return {"kind": "my-model", "name": "my-model"}


r = run_match(lineup(MyModelAgent(load_my_model()), RulebotAgent(), "2v2"))
print(r.table())
```

`a.encode()` is riichienv's action number (0–81, in the same order as the legal-action mask of `obs.mask()`). If your model's output uses a different order, replace this with your own mapping. If you use riichienv's features, you can also use `d.obs.encode()` or similar instead of `my_encode`.

### A minimal working example

Example: a player that wins when it can, discards a known safe tile against another player's riichi, and otherwise discards the tile it drew. It uses `reducer_factory` to record the seats that declared riichi and `annotate` for per-decision preprocessing.

``` python
import riichienv as rv
from yamawake import RulebotAgent, lineup, run_match

AT = rv.ActionType


class RiichiTracker:
    """Seats that declared riichi in this kyoku (rebuilt at every kyoku)."""

    def __init__(self):
        self.seats = set()

    def apply(self, event):
        if event["type"] == "reach":
            self.seats.add(event["actor"])


class SafeFirstAgent:
    """Win when possible; against riichi, discard a tile every riichi seat has
    discarded (genbutsu); otherwise discard the drawn tile. Never calls."""

    lane = "observation"
    reducer_factory = RiichiTracker          # d.reducer

    def annotate(self, d):                   # d.extra
        return {"riichi_seats": sorted(d.reducer.seats - {d.seat})}

    def act_batch(self, decisions):
        return [self.act(d) for d in decisions]

    def act(self, d):
        legal = d.obs.legal_actions()
        by_type = {}
        for a in legal:
            by_type.setdefault(a.action_type, []).append(a)
        for t in (AT.TSUMO, AT.RON):
            if t in by_type:
                return by_type[t][0]
        discards = by_type.get(AT.DISCARD)
        if not discards:                     # calls, kyushu, ...: pass
            return by_type[AT.PASS][0] if AT.PASS in by_type else legal[0]
        riichi = d.extra["riichi_seats"]
        if riichi:
            safe = set.intersection(*({t // 4 for t in d.obs.discards[s]} for s in riichi))
            for a in discards:
                if a.tile // 4 in safe:
                    return a
        for a in discards:
            if a.tile == d.obs.drawn_tile:
                return a
        return discards[-1]

    def describe(self):
        return {"kind": "safe-first", "name": "safe-first"}


r = run_match(lineup(SafeFirstAgent(), RulebotAgent(), "2v2"), walls=100)
print(r.table())
```

### Specifying a player by name

Registering a player with `register_agent(name, factory)` lets you use it with the same specs as `build_agent` and the command line. `factory(spec, **options)` receives the whole spec (for `safe-first:xxx`, including `:xxx`) and the `--opt` values.

``` python
from yamawake import register_agent
from yamawake.arena import main

register_agent("safe-first", lambda spec, **opts: SafeFirstAgent())
main(["--mode", "2v2", "--players", "safe-first", "rulebot"])
```

To use it from the `yamawake-arena` command, put a registering function in the `yamawake.agents` entry-point group of your own package (it is called once with no arguments).

``` toml
[project.entry-points."yamawake.agents"]
safe-first = "my_package.agents:register"   # def register(): register_agent("safe-first", ...)
```

## 5. The low-level loop

When you want to control every turn yourself (self-play connected to a model server, for example), use `SelfPlayLoop` directly.

``` python
import riichienv as rv
from yamawake import SelfPlayLoop, SeatTracker, decision_of, mjai_action_spec
from yamawake.rulebot import STANDARD
from yamawake.walls import blob_for_wall

loop = SelfPlayLoop(seeds, 2,                      # 2 = hanchan
                    bot_seats=[[(3, "rulebot", STANDARD)] for _ in seeds],
                    wall_tables=[blob_for_wall(s) for s in seeds], split_wall=True)
trackers = {}
while (batch := loop.collect()) is not None:   # all rows waiting on a Python turn
    specs = []
    for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
        obs = rv.Observation.deserialize_from_base64(b64)
        tr = trackers.setdefault((g, s), SeatTracker())
        tr.feed(obs.new_events())
        specs.append(mjai_action_spec(my_choice(decision_of(g, s, obs, tr))))
    loop.apply(specs)                           # one action per row, in order
loop.scores(), loop.mjai_log(0)
```

`collect()` returns `None` when all games have finished. Before the next `collect()`, answer every row with `apply`. Bots that play inside the loop are placed by seat number with `bot_seats` (`"rulebot"`, or for suzume `"a1"` with the setting `{"a1": "suzume"}`).

## 6. Running suzume in another game environment

To have suzume play one seat in a game run by someone else (a game server or another game environment), use `yamawake.external.SuzumePlayer`. Create one per seat and pass it that seat's riichienv `Observation` and the mjai events of the current kyoku (from `start_kyoku`; other players' draws may stay hidden).

``` python
from yamawake.external import SuzumePlayer

player = SuzumePlayer("suzume")                 # any suzume player name (suzume-rla225, etc.)
action = player.act(obs, kyoku_events)          # one of obs.legal_actions() (rv.Action)
reply = player.act_mjai(obs, kyoku_events)      # the same action in mjai form (with tsumogiri)
player.last_rule                                # name of the routine that chose the action (a1 for suzume)
r = player.choose(obs, kyoku_events)            # the action and a score per legal action, in one call (a riichi with its tile)
```

- **Same moves as in the arena**: the decisions use the same Rust code as suzume inside the arena. What it remembers during a kyoku (such as the riichi declaration tile) is also kept per seat, as in the arena, and forgotten at a new `start_kyoku`.
- **One decision with scores**: `choose` (an A1 set; since 0.16.0) returns the fields of `seat_options` (`legal`, `choice`, `choice_index`, `basis`, `scores`, and `pmf` for a set with P / M / F experts; their meaning is in section 5 of [`play_out_spec.md`](play_out_spec.md), in Japanese). A riichi is one move, up to the tile it discards (`{"type": "reach", "pai": ..., "tsumogiri": ...}`): ask once before the declaration, then play the declaration and that discard. Asked again with `act` right after the declaration, the player discards the same tile.
- **A riichi not declared**: when suzume chose riichi but another discard was played in its place, the remembered riichi tile is dropped at its next decision (since 0.16.0; before, the next discard could be that tile).
- **Tiles left in the wall**: the Python version of riichienv's `Observation` lacks the number of tiles left in the wall (`drawable_count`) and the seat that discarded last (`last_discarder`). These two are derived from the events passed in (70 − the number of draws, and the seat of the last `dahai`).
- **New hanchan**: when starting a new hanchan in the same seat, call `player.reset()` (even without it, the player forgets when the kyoku changes).

`SuzumeObservationAgent` places the same player in the arena as a Python player (one that receives observations). It plays through the same path as in an external environment, and a test confirms that it produces the same games as suzume inside the arena ([`tests/arena/test_external.py`](../tests/arena/test_external.py)).

### From Rust

Since 0.17.1 the same decision can be called from Rust (`yamawake::external`). The Python `SuzumePlayer` goes through it too, so the moves and scores are the same as in Python.

``` toml
# Cargo.toml: no features (no Python)
[dependencies]
yamawake = { git = "https://github.com/ShintaroMinami/yamawake-mahjong-arena", tag = "v0.17.1" }
```

``` rust
use yamawake::external::{Suzume, SuzumeSeat, observation_from_base64};

let bot = Suzume::from_spec("suzume")?;                       // written as a player name (suzume:aggression=0.3, etc.)
let obs = observation_from_base64(&obs_b64, &kyoku_events)?;  // from Python's obs.serialize_to_base64()
let c = bot.choose(&obs, &kyoku_events)?;                     // the decision of SuzumePlayer("suzume").choose(obs, events)
c.action                                                      // the move (the engine action act gives; a riichi: the declaration)
c.riichi_tile                                                 // for a riichi, the tile discarded after the declaration (136 id)
let a = bot.decide(&obs, &kyoku_events)?;                     // the move only, without scores
let mut seat = SuzumeSeat::new(bot);                          // to play one seat through a kyoku
```

- **Dependency**: added without features, it brings in no pyo3. yamawake re-exports its riichienv-core fork and serde_json (`yamawake::riichienv_core`, `yamawake::serde_json`); pass observations and events as their types (`Observation`, parsed `&[Value]`).
- **`Suzume` and `SuzumeSeat`**: `Suzume::choose` and `Suzume::decide` decide once on an empty memo (as a new `SuzumePlayer` does). `SuzumeSeat` keeps a memo per seat and forgets it at a new `start_kyoku`. After a riichi it discards the tile chosen with the declaration.
- **The result (`Choice`)**: the fields of Python's `choose`, plus the three below. `to_json()` is the dict Python's `choose` returns.
  - The engine action of each legal move (`legal[i].action`; for a riichi, the discard declared with it).
  - The move to play, `action` (the same as `decide`).
  - For a riichi, the tile discarded after the declaration, `riichi_tile`. It is the tile the seat discards after the declaration.
- **Threads**: `Suzume` is `Copy`, `Send` and `Sync`. Many threads (rayon, etc.) can call one `Suzume` at once.
- **Spec**: `<set>[:KEY=VALUE...]`. The set is `suzume`, `suzume-rla225` or an A1 model directory. The keys are `aggression`, `call_shift`, `riichi_shift`, `win`, `call` and `mlp` (`f64` or `f32`, [suzume.en.md](suzume.en.md)). The style targets (`call_rate`, etc.) are converted into options in Python, so they are not accepted here; pass the options they convert to.
- **Same as Python**: [`tests/external/choose_cases.json`](../tests/external/choose_cases.json) records 54 decisions of the Python player. It covers three specs, with discards, riichi, calls, kans, wins and passes. [`test_choose_cases.py`](../tests/external/test_choose_cases.py) checks that the Python player gives the recorded answers, and [`tests/external.rs`](../tests/external.rs) (`cargo test`) that the Rust entry does. After an intended change of play, re-record with `uv run python tests/external/choose_cases.py`.

## 7. Game logs

### handstats

Prints statistics for the player with the given name from the game logs written by `--record-logs` (records or mjai). The final scores give the riichi sticks left at the end to first place, as the engine does. With `--vs`, two runs played on the same walls are paired by kyoku and by hanchan and compared.

``` bash
python -m yamawake.handstats LOGDIR --name suzume
python -m yamawake.handstats LOGDIR --name suzume --vs OTHER --vs-name rulebot --json out.json
```

The statistics include the rank distribution, ready-hand rate and hand value, the rates and points of wins, deal-ins and exhaustive draws, riichi and call results, yaku frequencies, and more. See `--help` for the detailed definitions.

### tenhou6

Converts game logs to the format read by Tenhou's game log viewer (tenhou.net/6 JSON).

``` python
from yamawake.tenhou6 import convert_file, convert_records, dump
dump(convert_file("logs/arena-0-000000.json.gz"), "out/game.json")     # one mjai file
dump(convert_records("records/", 0), "out/game.json")                 # game 0 of a records directory
```
