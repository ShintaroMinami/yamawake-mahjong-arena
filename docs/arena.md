# arena：固定分割牌山の対戦環境

- コード：[`src/rust/arena/`](../src/rust/arena/)（対局ループ）、[`src/python/yamawake/`](../src/python/yamawake/)（Python のインターフェイス）
- 対局エンジン：riichienv-core 0.4.8 の fork（[`src/engine/README.md`](../src/engine/README.md)）

## 1. 仕組み

- **分割牌山**：各席は自分専用の牌山からツモる。副露があってもどの席のツモもずれない。
- **固定牌山**：局 × 本場（東 1〜北 4 × 12）ごとに 136 枚の牌山を固定します。連荘しても各局で使う牌山は変わらない。
- **席の回転**：1 つの牌山を、席を回して全プレイヤーが同じ牌山を1度ずつ打つ。1v3では席をずらして4回、2v2（`[A, B, A, B]`）は 2 回打つ（残りの 2 回は同じ対局になるため）。
- **標準牌山**：デフォルトでは`wall_base 0` から作り、その先頭 1,000 個を使う(設定で変更可能)。常に同じ規定の牌山を使用すれば、結果の違いは打ち手の違いだけを反映する。

対局は N 局を並列に進めます。rulebot と suzume は Rust の中で判断し、Python を呼ぶのは Python のプレイヤーの手番だけです。

Python のプレイヤーの判断には、判断そのものとは別に、1 回あたり約 20 µs の受け渡しの時間がかかります（Rust で作った観測を Python に渡して読み戻す変換と、判断ごとの記録の組み立て）。arena 自体の 1 判断あたりの時間（約 10 µs）の 2 倍程度です。suzume の 1 判断（約 0.23 ms）のような普通のモデルでは目立ちませんが、非常に高速なアルゴリズムを評価する場合は、この時間が全体の速さに影響する可能性があります。例えば、同じ手を選ぶツモ切りボットでも、Rust の中で判断する `tsumogiri` 4 人では約 3,070 半荘/分、Python で判断する `tsumogiri-python` 4 人では約 1,570 半荘/分です（Apple M4 Pro の 1 スレッド、20 牌山、席の回転なし。[`docs/suzume.md`](suzume.md) の 6 節と同じ条件）。Time の表の arena の行には、この受け渡しの時間が含まれます。

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
| `--out FILE` | 結果の集計を jsonl に追記する |
| `--save-scores FILE` | 対局ごとの最終点と席の並びを npz に書く |
| `--record-logs DIR` | 全対局の牌譜を mjai 形式（gzip）で書き出す |
| `--no-stats` | 和了率などの統計を出さない（統計は牌譜から集計するので、約 3% 遅くなる） |
| `--flight N` | 同時に進める対局の数（既定 256） |

進み具合は標準エラーにバーで出し、結果は標準出力に次の表で出します。値の下のかっこは、牌山単位の標準誤差です。

| 表 | 中身 |
|---|---|
| Standings | 平均順位、ポイント、1〜4 位の率。2v2 では、牌山ごとに対にした差も出す |
| Play per kyoku | 1 局あたりの和了率、放銃率、副露率（チー・ポン・大明槓をした局）、立直率（宣言した局）と、平均和了点（和了した局の局収支の平均。本場と供託を含み、自分の立直棒を引く）、平均放銃点（放銃で払った点。本場を含み、自分の立直棒は含めない）、調整打点効率（和了率 × 平均和了点 − 放銃率 × 平均放銃点、1 局あたりの点） |
| Time | プレイヤーごとの判断の数、時間、1 判断あたりの時間と、arena 自体（対局エンジンとループ）の時間。内蔵のボットと arena はスレッドごとの時間の合計、Python のプレイヤーは `act_batch` にかかった時間 |

ポイントは 25,000 点持ち 30,000 点返し、順位点 +90 / +45 / 0 / −135、オカ 20 の精算です（`rules.DEFAULT_RULE`）。

`--single-kyoku` では、全員 25,000 点、本場 0 で、半荘の東 1 局と同じ牌山を打ち、その 1 局で終えます。点数状況の影響なしに手組みを比べるためのもので、局の収支と和了率・放銃率を出します。

`--no-rotate`（`run_match(rotate=False)`）では、牌山ごとに 1 半荘だけ打ちます。席は牌山の番号で順に回し（牌山 w では w % 4 回目の回転、2v2 では w % 2 回目）、どの半荘も別の牌山なので互いに独立です。席の回転による分散低減の効果を測るときなど、比較の基準に使います。標準誤差は半荘単位になります。

## 3. Python

``` python
from yamawake import RulebotAgent, TsumogiriAgent, build_agent, run_match

r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()], walls=50)
print(r.table())          # 順位表（コマンドラインの Standings の表と同じ）
r.paired()                # 2v2 の牌山ごとの対の差
agent = build_agent("suzume-rla225")   # コマンドラインと同じ指定
```

`run_match` の引数は、コマンドラインのオプションと同じです（`walls`、`wall_base`、`single_kyoku`、`record_logs`、`save_scores` など）。統計は `play_stats=True` のときだけ集計し、`r.play_stats()` で読めます。時間は `r.time_stats()`、コマンドラインと同じ表は `yamawake.report.report(r)` で得られます。`run_match` 自体は何も表示しません。

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

## 6. 外部の対局環境で suzume を動かす

ほかの人が動かす対局（対局サーバーや別の対戦環境）の 1 席として suzume を打たせるときは、`yamawake.external.SuzumePlayer` を使います。席ごとに 1 つ作り、その席の riichienv の `Observation` と、その局の mjai の出来事（`start_kyoku` から。他家のツモは伏せたままでよい）を渡します。

``` python
from yamawake.external import SuzumePlayer

player = SuzumePlayer("suzume")                 # suzume のプレイヤー名なら何でも（suzume-rla225 など）
action = player.act(obs, kyoku_events)          # obs.legal_actions() の 1 つ（rv.Action）
reply = player.act_mjai(obs, kyoku_events)      # 同じ手を mjai の形で（tsumogiri つき）
player.last_rule                                # 手を決めた処理の名前（suzume は a1）
```

- **arena と同じ手**：判断は arena の中の suzume と同じ Rust の処理です。局の間に覚えておくもの（立直の宣言牌など）も、arena と同じく席ごとに持ち、新しい `start_kyoku` で忘れます。
- **山の残り枚数**：riichienv の Python 版の `Observation` には、山の残り枚数（`drawable_count`）と最後に打牌した席（`last_discarder`）がありません。この 2 つは、渡された出来事から求めます（70 − ツモの数、最後の `dahai` の席）。
- **新しい半荘**：同じ席で新しい半荘を始めるときは `player.reset()` を呼びます（呼ばなくても、局が変わると忘れます）。

`SuzumeObservationAgent` は、同じプレイヤーを arena の Python のプレイヤー（観測を受け取る側）として置いたものです。外部の環境と同じ経路で打ち、arena の中の suzume と同じ対局になることをテストで確かめています（[`tests/arena/test_external.py`](../tests/arena/test_external.py)）。

## 7. 牌譜

### handstats

`--record-logs` の牌譜から、指定した名前のプレイヤーの統計を出します。`--vs` を付けると、同じ牌山で打った 2 つの実行を局ごと・半荘ごとに対にして比べます。

``` bash
python -m yamawake.handstats LOGDIR --name suzume
python -m yamawake.handstats LOGDIR --name suzume --vs OTHER --vs-name rulebot --json out.json
```

出る統計は、順位の分布、聴牌率と打点、和了・放銃・流局の率と点数、立直と副露の成績、役の頻度などです。定義の詳細は `--help` にあります。

### tenhou6

mjai の牌譜を、天鳳の牌譜ビューアが読む形式（tenhou.net/6 の JSON）に変換します。

``` python
from yamawake.tenhou6 import convert_file, dump
dump(convert_file("logs/arena-0-000000.json.gz"), "out/game.json")
```

## 8. 制限

- 対局の最後に残った供託の扱いは、riichienv に従います。`handstats` の最終の点数は、残った供託を誰にも渡さずに計算します。
