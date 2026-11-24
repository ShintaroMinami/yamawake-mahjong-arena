# arena：固定分割牌山の対戦環境

**日本語** | [English](arena.en.md)

- コード：[`src/rust/arena/`](../src/rust/arena/)（対局ループ）、[`src/python/yamawake/`](../src/python/yamawake/)（Python のインターフェイス）
- 対局エンジン：riichienv-core 0.4.8 の fork（[`src/engine/README.md`](../src/engine/README.md)）

## 1. 仕組み

- **分割牌山**：各席は自分専用の牌山からツモる。副露があってもどの席のツモもずれない。
- **固定牌山**：局 × 本場（東 1〜北 4 × 12）ごとに 136 枚の牌山を固定します。連荘しても各局で使う牌山は変わらない。
- **席の回転**：1 つの牌山を、席を回して全プレイヤーが同じ牌山を1度ずつ打つ。1v3では席をずらして4回、2v2（`[A, B, A, B]`）は 2 回打つ（残りの 2 回は同じ対局になるため）。
- **標準牌山**：デフォルトでは`wall_base 0` から作り、その先頭 1,000 個を使う(設定で変更可能)。常に同じ規定の牌山を使用すれば、結果の違いは打ち手の違いだけを反映する。

対局は N 局を並列に進めます。rulebot と suzume は Rust の中で判断し、Python を呼ぶのは Python のプレイヤーの手番だけです。

Python のプレイヤーの判断には、判断そのものとは別に、1 回あたり約 20 µs の受け渡しの時間がかかります（Rust で作った観測を Python に渡して読み戻す変換と、判断ごとの記録の組み立て）。arena 自体の 1 判断あたりの時間（約 10 µs）の 2 倍程度です。suzume の 1 判断（約 0.16 ms）のような普通のモデルでは目立ちませんが、非常に高速なアルゴリズムを評価する場合は、この時間が全体の速さに影響する可能性があります。例えば、同じ手を選ぶツモ切りボットでも、Rust の中で判断する `tsumogiri` 4 人では約 3,070 半荘/分、Python で判断する `tsumogiri-python` 4 人では約 1,570 半荘/分です（Apple M4 Pro の 1 スレッド、20 牌山、席の回転なし）。Time の表の arena の行には、この受け渡しの時間が含まれます。

## 2. コマンドライン

``` bash
uv run yamawake-arena --mode 2v2 --players suzume rulebot
uv run yamawake-arena --players rulebot rulebot:kobalab rulebot rulebot \
    --walls 1000 --out results.jsonl --save-scores scores.npz --record-logs logs/
```

| オプション | 意味 |
|---|---|
| `--players` | 4 人のプレイヤー。`--mode` と併せて 2 人でもよい |
| `--mode 1v3\|2v2` | 2 人を `[A, B, B, B]` か `[A, B, A, B]` に並べる |
| `--walls N` | 牌山の数（既定は標準牌山の先頭 1,000） |
| `--wall-base fresh\|Integer` | `fresh` で新しい牌山を引く。値を渡すとその牌山を再現する |
| `--wall-seed S` / `--wall-start K` | シード S の牌山の K 番目から `--walls` 個を打つ（S、K は 0 以上 2³² 未満、既定 K = 0）。牌山を足すときは K に打った数を渡す。シードが違えば牌山は重ならない。`--wall-base` とは併用できない |
| `--opt KEY=VALUE` | プレイヤーに渡すオプション（rulebot の設定など） |
| `--single-kyoku` | 全対局を東 1 局だけにする（下記） |
| `--no-rotate` | 牌山を席で回さず、1 つの牌山で 1 半荘だけ打つ（下記） |
| `--no-split` | 分割牌山を使わず、普通の麻雀と同じく全席が 1 つの山からツモる（下記） |
| `--game-rule tenhou\|mjsoul` | 対局のルール。既定は天鳳（下記） |
| `--out FILE` | 結果の集計を jsonl に追記する |
| `--save-scores FILE` | 対局ごとの最終点と席の並びを npz に書く |
| `--record-logs DIR` | 全対局の牌譜を mjai 形式（gzip）で書き出す |
| `--record-format mjai\|records` | `--record-logs` の形式（0.17.0 から）。`records` は配列に詰めた 1 つのディレクトリ（[`records.md`](records.md)。新しいか空のディレクトリに書く） |
| `--no-stats` | 和了率などの統計を出さない（統計は牌譜から集計するので、約 3% 遅くなる） |
| `--flight N` | 同時に進める対局の数（既定 256） |

進み具合は標準エラーにバーで出し、結果は標準出力に次の表で出します。値の下のかっこは、牌山単位の標準誤差です。

| 表 | 中身 |
|---|---|
| Standings | 平均順位、ポイント、1〜4 位の率。2v2 では、牌山ごとに対にした差も出す |
| Play per kyoku | 1 局あたりの和了率、放銃率、副露率（チー・ポン・大明槓をした局）、立直率（宣言した局）と、平均和了点（和了した局の局収支の平均。本場と供託を含み、自分の立直棒を引く）、平均放銃点（放銃で払った点。本場を含み、自分の立直棒は含めない）、調整打点効率（和了率 × 平均和了点 − 放銃率 × 平均放銃点、1 局あたりの点） |
| Time | プレイヤーごとの判断の数、時間、1 判断あたりの時間と、arena 自体（対局エンジンとループ）の時間。内蔵のボットと arena はスレッドごとの時間の合計、Python のプレイヤーは `act_batch` にかかった時間 |

Play per kyoku の副露率・立直率は全局を分母とした値です。打ち方の座標の `call_rate`・`riichi_rate`（[`suzume.md`](suzume.md)）は、副露・立直が一度でも可能だった局を分母とするので、値が違います。

ポイントは 25,000 点持ち 30,000 点返し、順位点 +90 / +45 / 0 / −135、オカ 20 の精算です（`rules.DEFAULT_RULE`）。

`--game-rule`（`run_match(game_rule=...)`、0.17.0 から）は、役と対局の進行のルールを選びます。0.17.0 から、既定（`tenhou`）の進行は天鳳と同じになり、終局時に残った供託は 1 位が取ります（それまでは誰も取らなかった）。精算（上のポイント）はどちらでも同じです。

| ルール | 中身 |
|---|---|
| `tenhou`（既定） | 天鳳の段位戦 |
| `mjsoul` | 雀魂の段位戦（四人南） |

2 つのルールの違いは次のとおりです。

| 項目 | `tenhou` | `mjsoul` |
|---|---|---|
| 二倍役満（国士無双十三面、四暗刻単騎、純正九蓮宝燈、大四喜） | なし | あり |
| 三家和 | 途中流局 | トリプルロン（積み棒と供託は放銃者から見て最も近い和了者が取る） |
| 暗槓への国士無双の槍槓 | なし | あり |
| 複合役満のパオ（大三元・大四喜） | 全体を責任払い | パオの役満の分だけ |
| 西場で親が連荘したとき | 親がトップで 30,000 点以上のときだけ終える | 誰かが 30,000 点以上なら終える（親を含むダブロン・トリロンでの連荘は除く） |

次は 2 つとも同じです。

- オーラスに親が和了するか聴牌で流局したとき、親がトップで 30,000 点以上なら終える。
- オーラスの途中流局（九種九牌など）は、親がトップでも続ける。
- 流し満貫は親が聴牌なら連荘する。
- 南 4 局の終了時に 30,000 点以上がいなければ西入し、西 4 局で終える。
- 終局時に残った供託は 1 位（同点なら起家に近い席）が取る。

`tenhou` の進行は、天鳳の牌譜（鳳凰卓 東南喰赤、2016〜2020 年）から等間隔に選んだ 20,000 半荘を、牌譜から組み立てた牌山の上で牌譜どおりに打ち直して確かめました（全打牌・各局の開始状態・ドラ・最終点数がすべて一致）。`mjsoul` は、雀魂の公式ルール（<https://mahjongsoul.com/news/46>）に従っています。ただし、オーラスの途中流局については公式ルールに明記がありません。自動の和了止め・聴牌止めは和了と聴牌だけが対象なので、続けるものとしています。

`--single-kyoku` では、全員 25,000 点、本場 0 で、半荘の東 1 局と同じ牌山を打ち、その 1 局で終えます。点数状況の影響なしに手組みを比べるためのもので、局の収支と和了率・放銃率を出します。

`--no-rotate`（`run_match(rotate=False)`）では、牌山ごとに 1 半荘だけ打ちます。席は牌山の番号で順に回し（牌山 w では w % 4 回目の回転、2v2 では w % 2 回目）、どの半荘も別の牌山なので互いに独立です。席の回転による分散低減の効果を測るときなど、比較の基準に使います。標準誤差は半荘単位になります。

`--no-split`（`run_match(split_wall=False)`）では、分割牌山を使わず、全席が 1 つの山から順にツモります。副露するとツモがずれる、普通の麻雀と同じツモの流れです。一般的なランダム牌山で対局するには `--no-rotate --no-split --wall-base fresh` を指定します（[README](../README.md#アリーナの設定)）。

## 3. Python

``` python
from yamawake import RulebotAgent, TsumogiriAgent, build_agent, run_match

r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()], walls=50)
print(r.table())          # 順位表（コマンドラインの Standings の表と同じ）
r.paired()                # 2v2 の牌山ごとの対の差
agent = build_agent("suzume-rla225")   # コマンドラインと同じ指定
```

`run_match` の引数は、コマンドラインのオプションと同じです（`walls`、`wall_base`、`single_kyoku`、`record_logs`、`save_scores` など）。統計は `play_stats=True` のときだけ集計し、`r.play_stats()` で読めます。時間は `r.time_stats()`、コマンドラインと同じ表は `yamawake.report.report(r)` で得られます。`run_match` 自体は何も表示しません。

**対局を分けて回す**：`match_plan(players, walls=..., wall_base=...)`（0.16.0 から）は、`run_match` が打つ半荘の一覧を、打たずに返します。半荘ごとに、牌山の番号、席の回転、コアの seed、席ごとのエージェント、枠ごとの席、牌山があり、並びは `run_match` と同じです（牌山ごと、その中で回転の順）。好きな単位に分けて `play_games(**plan.play_args(games))` で打つと、それぞれの半荘で `run_match` と同じ得点になります。`run_match` 自体もこの一覧で打ちます。

``` python
from yamawake import RulebotAgent, TsumogiriAgent, lineup, match_plan, play_games

plan = match_plan(lineup(TsumogiriAgent(), RulebotAgent(), "1v3"), walls=100)
part = plan.games[:40]                          # 例えば最初の 10 牌山（各 4 回転）
scores, _ = play_games(**plan.play_args(part))  # 席ごとの得点（半荘 × 4）
```

**1 局面の判断**：`decide(プレイヤーの指定, 観測, kyoku_events)` は、rulebot か suzume の判断を 1 つ返します。
suzume は `rule` に `a1` を、rulebot は `debug=True` で判断の材料を付けて返します。

## 4. 自作のプレイヤー

プレイヤーは、次のものを持つ任意のオブジェクトです。

| 属性 | 中身 |
|---|---|
| `lane = "observation"` | 必ずこの値にする。arena が観測を `act_batch` に渡して手を選ばせる、という指定（内蔵の rulebot・suzume は Rust の中で判断するので別の値） |
| `act_batch(decisions)` | 判断のリストを受け、合法手を 1 つずつ、同じ順のリストで返す。そのプレイヤーの手番を待つ全ての対局の判断が 1 回で届くので、モデルはまとめて推論できる |
| `describe()` | 結果と牌譜に載せる名前（`{"kind": ..., "name": ...}`） |
| `reducer_factory`（任意） | 局ごとにイベントを畳み込むオブジェクトを作る。局の始めに作り直し、イベントごとに `.apply(event)` を呼ぶ。`Decision.reducer` で読める |
| `annotate(decision)`（任意） | 判断ごとに dict を返す。`Decision.extra` に入る |

各判断 `Decision` は次のものを持ちます。

| 属性 | 中身 |
|---|---|
| `obs` | RiichiEnv の環境が返す観測と同じ `riichienv.Observation`（[RiichiEnv の説明](https://github.com/smly/RiichiEnv#observation)）。合法手 `legal_actions()`、手牌 `hand`、ツモ牌 `drawn_tile`、河 `discards`、点数 `scores` など。牌は 136 の番号（`// 4` で牌種 0〜33） |
| `seat` | 自分の席番号（0〜3。起家が 0 で、mjai の `actor` と同じ番号） |
| `kyoku_events` | その局の始めから今までの mjai 形式のイベント（他家のツモは伏せる）。`obs.events` には自分の前の手番からの差分しか入っていないので、局の経過を読むにはこちらを使う |
| `drawable_count` | 山の残り枚数 |
| `reacting_to` | 他家の打牌・加槓への応答（鳴き・ロン・槍槓）のとき、その相手の席番号。自分の手番では `None` |
| `dora_indicator_types` | ドラ表示牌の牌種（0〜33） |

### 作ったモデルを組み込む例

以下は、手元のモデルを導入する際のおおまかな例です。
`my_encode`（観測から特徴量を作る関数）と `model.predict`（まとめて推論する関数）は、自分のものに置き換えます。

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

`a.encode()` は、riichienv の行動の番号（0〜81。`obs.mask()` の合法手の印と同じ並び）です。モデルの出力が別の並びなら、そこを自分の対応に置き換えます。riichienv の特徴量を使うなら、`my_encode` の代わりに `d.obs.encode()` なども使えます。

### 実際に動作する簡単な例

例：和了できれば和了し、他家の立直には現物を切り、それ以外はツモ切りするプレイヤー。立直した席番号の記録に `reducer_factory` を、判断ごとの前処理に `annotate` を使っています。

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

### 名前で指定する

`register_agent(名前, factory)` で登録すると、`build_agent` やコマンドラインと同じ指定で使えます。`factory(spec, **options)` は指定の全体（`safe-first:xxx` なら `:xxx` も含む）と `--opt` の値を受け取ります。

``` python
from yamawake import register_agent
from yamawake.arena import main

register_agent("safe-first", lambda spec, **opts: SafeFirstAgent())
main(["--mode", "2v2", "--players", "safe-first", "rulebot"])
```

`yamawake-arena` コマンドから使うには、自作のパッケージのエントリポイントのグループ `yamawake.agents` に、登録する関数を置きます（引数なしで 1 回呼ばれる）。

``` toml
[project.entry-points."yamawake.agents"]
safe-first = "my_package.agents:register"   # def register(): register_agent("safe-first", ...)
```

## 5. 低水準のループ

手番を全て自分で制御したいとき（モデルのサーバーとつなぐ自己対戦など）は、`SelfPlayLoop` を直接使います。

``` python
import riichienv as rv
from yamawake import SelfPlayLoop, SeatTracker, decision_of, mjai_action_spec
from yamawake.rulebot import STANDARD
from yamawake.walls import blob_for_wall

loop = SelfPlayLoop(seeds, 2,                      # 2 = 半荘
                    bot_seats=[[(3, "rulebot", STANDARD)] for _ in seeds],
                    wall_tables=[blob_for_wall(s) for s in seeds], split_wall=True)
trackers = {}
while (batch := loop.collect()) is not None:   # Python の手番を待つ全ての行
    specs = []
    for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
        obs = rv.Observation.deserialize_from_base64(b64)
        tr = trackers.setdefault((g, s), SeatTracker())
        tr.feed(obs.new_events())
        specs.append(mjai_action_spec(my_choice(decision_of(g, s, obs, tr))))
    loop.apply(specs)                           # 全ての行に、順番どおり 1 手ずつ
loop.scores(), loop.mjai_log(0)
```

`collect()` は全対局が終わると `None` を返します。次の `collect()` の前に、`apply` で全ての行に答えます。ループの中で打つボットは `bot_seats` で席番号ごとに置けます（`"rulebot"` か、suzume なら `"a1"` と設定 `{"a1": "suzume"}`）。

### 対局の分岐（fork）

`collect()` が返した判断で、対局をいくつかの手に分岐させ、それぞれを局の終わりまで打てます（0.19.0 から）。分岐した対局は、ほかの対局と同じ `collect()` / `apply()` で判断が回ってきます。

``` python
batch = loop.collect()
g, s = batch["games"][i], batch["seats"][i]           # 分岐させる判断（apply の前に）
forks = loop.fork(g, s, [spec_a, spec_b, spec_c])    # 手ごとに 1 対局。新しい対局の番号が返る
loop.apply(specs)                                     # 元の対局は自分の手で続く
...
results = loop.fork_results(forks, events=True)      # 局が終わった分岐の結果（終わっていなければ None）
```

- **何を複製するか**：エンジンの状態をすべて複製します（全員の手牌、山または席ごとのツモの列、王牌、ドラと裏ドラ）。各分岐は、指定した手を打ったあと、ふつうの対局として局の終わりまで打ちます。次の局は配りません。席ごとのレーンとループ内のボットは、元の対局と同じです。元の対局は変わりません。
- **同時の判断**：鳴きやロンで、同じ打牌に複数の席が判断する場面では、分岐で決まるのは指定した席の手だけです。ほかの席には、分岐の最初の `collect()` でもう一度同じ判断が回ってきます。ループ内のボットのその場面の判断は、そのまま引き継ぎます。
- **結果**：`fork_results` は `play_out` の結果と同じ項目（`end`、`delta`、`scores`、`kyotaku`、`honba`、`renchan`、`hora`、`ryukyoku`。[play_out_spec.md](play_out_spec.md) 3 節）に、`parent`、`seat`、`action` を加えた辞書を返します。`delta` は分岐の時点からの点数の変化です。`events=True` で、分岐の手からの局の出来事も付きます。
- **解放**：`fork_results` は、既定（`release=True`）で結果を返した分岐を解放します。解放した番号は、次の分岐で使い回します。局の途中でやめる分岐は `loop.release(games)` で解放します（判断を待っている分岐は解放できません）。
- **観測の続き**：分岐の観測の `new_events()` は、元の対局のその席の観測の続きです。席ごとに出来事を積み上げている場合（`SeatTracker` など）は、分岐の時点の元の対局のものを複製して使います。分岐の `mjai_log` は分岐の後の行しか持たないので、出来事は `events=True` で受け取ります。
- **共通の乱数と決定性**：`split_wall=True` では各席が自分のツモの列から引くので、ある分岐で鳴いても、ほかの席のツモはずれません（兄弟の分岐は、各席の次のツモを共有します。鳴きが多く、ある席の列が尽きたときだけ、ほかの列から借りるのでずれます。槓の嶺上牌と新しいドラは王牌から引くので、分岐ごとに変わりえます）。分岐の結果は、元の対局の状態、分岐の手、各席の手だけで決まります。
- **大きさ**：分岐 1 つは作った時点で約 5 KB で、局が進むと、その後の出来事のぶん増えます。

## 6. 外部の対局環境で suzume を動かす

ほかの人が動かす対局（対局サーバーや別の対戦環境）の 1 席として suzume を打たせるときは、`yamawake.external.SuzumePlayer` を使います。席ごとに 1 つ作り、その席の riichienv の `Observation` と、その局の mjai の出来事（`start_kyoku` から。他家のツモは伏せたままでよい）を渡します。

``` python
from yamawake.external import SuzumePlayer

player = SuzumePlayer("suzume")                 # suzume のプレイヤー名なら何でも（suzume-rla225 など）
action = player.act(obs, kyoku_events)          # obs.legal_actions() の 1 つ（rv.Action）
reply = player.act_mjai(obs, kyoku_events)      # 同じ手を mjai の形で（tsumogiri つき）
player.last_rule                                # 手を決めた処理の名前（suzume は a1）
r = player.choose(obs, kyoku_events)            # 手と、合法な手ごとの点数を 1 回で（立直は宣言牌まで 1 手）
```

- **arena と同じ手**：判断は arena の中の suzume と同じ Rust の処理です。局の間に覚えておくもの（立直の宣言牌など）も、arena と同じく席ごとに持ち、新しい `start_kyoku` で忘れます。
- **点数つきの 1 回の判断**：`choose`（A1 の組。0.16.0 から）は、`seat_options` と同じ欄（`legal`・`choice`・`choice_index`・`basis`・`scores`、P / M / F の組では `pmf`。意味は [`play_out_spec.md`](play_out_spec.md) の 5 節）を返します。立直は「立直して切る牌」までの 1 手（`{"type": "reach", "pai": ..., "tsumogiri": ...}`）なので、宣言の前に 1 回聞き、宣言とその打牌を打ちます。宣言の直後に `act` で聞き直しても、同じ牌を切ります。
- **宣言しなかった立直**：suzume が立直を選んだのに、外の手で別の打牌をした場合、覚えた立直の牌は次の判断で捨てます（0.16.0 から。それまでは次の打牌でその牌を切ることがあった）。
- **山の残り枚数**：riichienv の Python 版の `Observation` には、山の残り枚数（`drawable_count`）と最後に打牌した席（`last_discarder`）がありません。この 2 つは、渡された出来事から求めます（70 − ツモの数、最後の `dahai` の席）。
- **新しい半荘**：同じ席で新しい半荘を始めるときは `player.reset()` を呼びます（呼ばなくても、局が変わると忘れます）。

`SuzumeObservationAgent` は、同じプレイヤーを arena の Python のプレイヤー（観測を受け取る側）として置いたものです。外部の環境と同じ経路で打ち、arena の中の suzume と同じ対局になることをテストで確かめています（[`tests/arena/test_external.py`](../tests/arena/test_external.py)）。

### Rust から呼ぶ

0.17.1 から、同じ判断を Rust からも呼べます（`yamawake::external`）。Python の `SuzumePlayer` も中ではこれを通るので、手と点数は Python と同じです。

``` toml
# Cargo.toml：機能を付けずに入れる（Python は使わない）
[dependencies]
yamawake = { git = "https://github.com/ShintaroMinami/yamawake-mahjong-arena", tag = "v0.17.1" }
```

``` rust
use yamawake::external::{Suzume, SuzumeSeat, observation_from_base64};

let bot = Suzume::from_spec("suzume")?;                       // プレイヤー名と同じ書き方（suzume:aggression=0.3 など）
let obs = observation_from_base64(&obs_b64, &kyoku_events)?;  // Python の obs.serialize_to_base64() から
let c = bot.choose(&obs, &kyoku_events)?;                     // SuzumePlayer("suzume").choose(obs, events) と同じ判断
c.action                                                      // 打つ手（act と同じエンジンの手。立直は宣言）
c.riichi_tile                                                 // 立直のとき、宣言の後に切る牌（136 の id）
let a = bot.decide(&obs, &kyoku_events)?;                     // 点数なしで手だけ
let mut seat = SuzumeSeat::new(bot);                          // 1 局を通して 1 席を打つとき
```

- **依存の入れ方**：機能を付けずに入れると、pyo3 は入りません。yamawake は riichienv-core のフォーク版と serde_json を再公開しています（`yamawake::riichienv_core`、`yamawake::serde_json`）。観測と出来事は、これらの型（`Observation`、解析済みの `&[Value]`）で渡します。
- **`Suzume` と `SuzumeSeat`**：`Suzume::choose` と `Suzume::decide` は、空の memo で 1 回判断します（新しい `SuzumePlayer` と同じです）。`SuzumeSeat` は席ごとに memo を持ち、新しい `start_kyoku` で忘れます。立直の後の打牌は、宣言のときに選んだ牌です。
- **結果（`Choice`）**：Python の `choose` と同じ欄に加えて、次の 3 つを持ちます。`to_json()` は、Python の `choose` が返す dict と同じです。
  - 合法手ごとのエンジンの手（`legal[i].action`。立直は宣言で切る牌）。
  - 打つ手 `action`（`decide` と同じ）。
  - 立直のときに宣言の後で切る牌 `riichi_tile`。宣言の後にその席が打つ牌と一致します。
- **並列**：`Suzume` は `Copy` で、`Send` と `Sync` です。1 つの `Suzume` を、rayon などの多くのスレッドから同時に呼べます。`choose` と `decide` は、和了の判定の結果をスレッドごとに残して次の呼び出しでも使います（2026-11-25 から。1 スレッドあたり最大約 9 MB）。判定の鍵には場風・ドラ表示牌・副露まで含まれるので、どの席・局・対局の判断が使っても、結果は空の memo で判断したときと同じです。
- **spec**：`<組>[:KEY=VALUE...]` です。組は `suzume`、`suzume-rla225`、または A1 のモデルのディレクトリです。キーは `aggression`、`call_shift`、`riichi_shift`、`win`、`call`、`mlp`（`f64` か `f32`、[suzume.md](suzume.md)）です。スタイルの目標値（`call_rate` など）は Python で調整値に換算するものなので、受け取りません。換算した調整値を渡してください。
- **Python との一致**：[`tests/external/choose_cases.json`](../tests/external/choose_cases.json) に、Python のプレイヤーの判断を 54 局面記録しています。3 つの spec で、打牌・立直・鳴き・槓・和了・見送りを含みます。この記録と同じ答えになることを、Python 側は [`test_choose_cases.py`](../tests/external/test_choose_cases.py) で、Rust 側は [`tests/external.rs`](../tests/external.rs)（`cargo test`）で確かめています。打ち方を意図して変えたときは、`uv run python tests/external/choose_cases.py` で記録し直します。

## 7. 牌譜

### handstats

`--record-logs` の牌譜（records でも mjai でも）から、指定した名前のプレイヤーの統計を出します。最終の点数は、エンジンと同じく、残った供託を 1 位に渡して求めます。`--vs` を付けると、同じ牌山で打った 2 つの実行を局ごと・半荘ごとに対にして比べます。

``` bash
python -m yamawake.handstats LOGDIR --name suzume
python -m yamawake.handstats LOGDIR --name suzume --vs OTHER --vs-name rulebot --json out.json
```

出る統計は、順位の分布、聴牌率と打点、和了・放銃・流局の率と点数、立直と副露の成績、役の頻度などです。定義の詳細は `--help` にあります。

### tenhou6

牌譜を、天鳳の牌譜ビューアが読む形式（tenhou.net/6 の JSON）に変換します。

``` python
from yamawake.tenhou6 import convert_file, convert_records, dump
dump(convert_file("logs/arena-0-000000.json.gz"), "out/game.json")     # mjai の 1 ファイル
dump(convert_records("records/", 0), "out/game.json")                 # records の半荘 0
```
