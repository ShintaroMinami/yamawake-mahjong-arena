# arena：分割固定牌山の対戦環境

yamawake-mahjong-arena の対局場です。牌の運を打ち消して打ち手の差だけを測ることと、速いことを目的にしています。

- コード：`src/rust/arena/`（対局ループ）、`src/python/yamawake/`（`match.py`・`arena.py`・`driver.py`・`agents.py` など）
- 対局エンジン：`src/engine/riichienv-core/`（riichienv-core 0.4.8 の fork。`src/engine/README.md`）

## 1. 考え方

デュプリケートブリッジでは全卓が同じ配牌で打つので、結果は札ではなく打ち手の差を表します。arena は麻雀で同じことをします。

- **固定牌山**：*牌山の行列* が、対局の各座標（局 × 本場、東 1〜北 4 × 12）の 136 枚の牌山を固定します。連荘しても、その局がどの牌山で打たれるかはずれません。
- **split 牌山**：各席は、山のうち自分専用に決まった列からだけツモります。鳴きがあっても、他の席のこの先のツモはずれません。
- **席の回転**：1 つの牌山を、席を回して 4 回打ちます（2v2 の `[A, B, A, B]` は 2 回。残りの 2 回は同じ対局の繰り返しになるため）。どの打ち手も全ての席の牌を打つことになり、同じ決定的な打ち手 4 人なら平均順位はちょうど 2.5 になります。
- **標準牌山**：記録した 1 つの元の値（`wall_base 0`。2026-10-14 までは `6172671062617`）から作り直す 5,000 個の行列です（`walls.standard_walls()`、4 つの指紋で検査。`src/python/yamawake/data/eval_walls.json`）。測定がこれを共有するので、2 つの結果の違いは打ち手の違いだけから生まれます。

対局は N 局を足並みをそろえて並列（rayon）に進めます。rulebot と suzume は Rust の中で判断し、Python に入るのは 1 手番に 1 回、それが必要なプレイヤーのためだけです。速さの目安（ノート PC 1 台、200 牌山、4 人とも同じ bot）：rulebot 約 11,000 半荘/分、suzume 約 3,300、`suzume-mlp1` 約 4,900。

## 2. コマンドライン

    uv run yamawake-arena --mode 1v3 --players tsumogiri rulebot --walls 50
    uv run python -m yamawake.arena --mode 2v2 --players suzume rulebot
    uv run python -m yamawake.arena --players rulebot rulebot:latest rulebot rulebot \
        --walls 1000 --out results.jsonl --save-scores scores.npz --record-logs logs/

| オプション | 意味 |
|---|---|
| `--players` | 4 人のプレイヤー、または `--mode` と 2 人のプレイヤー |
| `--mode 1v3\|2v2` | 2 人のプレイヤーを標準の並びに展開する（1v3：`[A, B, B, B]`、2v2：`[A, B, A, B]`） |
| `--walls N` | 牌山の数（既定は標準牌山の先頭 1,000） |
| `--wall-base fresh\|値` | `fresh` で一度も使っていない牌山を引く。記録した元の値を渡せばその牌山を再現する |
| `--opt KEY=VALUE` | プレイヤーのファクトリに渡すオプション（suzume のオプションも届く） |
| `--out FILE` | 結果の集計を jsonl に追記する |
| `--save-scores FILE` | 対局ごとの最終点と席の並び（`scores`・`seat_of_slot`）を npz に書く |
| `--record-logs DIR` | 全対局を gzip した mjai（`arena-<wall_base>-NNNNNN.json.gz`）で書き出す。suzume が打つ対局は `<id>.suzume.json`（判断ごとに決めたルール）も書く |
| `--single-kyoku` | 全対局を東 1 局だけにする（下記） |
| `--flight N` | 同時に進める対局の数（既定 256） |

結果の表には、プレイヤーごとの平均順位とポイントと、牌山単位の標準誤差が出ます。ポイントは `rules.DEFAULT_RULE` の精算で、((最終点 − 30,000) / 1,000 + 順位の加点) / 100 です。順位の加点は +90 / +45 / 0 / −135 に、オカ 20 を 1 位に足したもの（1 位は +110）です（25,000 点持ち 30,000 点返し）。2v2 では、牌山ごとに対にした差も出ます。最後に、半荘数、かかった秒数、1 分あたりの半荘数を表示します。

**速さの測り方**：4 人とも同じ bot にして、たとえば `--players suzume suzume suzume suzume --walls 200` のように打たせます。4 人とも同じ bot のときの席の回転は同じ対局の繰り返しなので、このときの順位は使いません。

**東 1 局だけ（`--single-kyoku`、`run_match(..., single_kyoku=True)`）**：本場 0、供託なし、全員 25,000 点、席 0 が親で、半荘の東 1 局と同じ牌山と席で打ち、結果にかかわらずその局で終わります（riichienv の game mode 0）。点数と順位の影響なしに手組みを比べられます。表、jsonl の行（`kyoku_stats`）、2v2 の対の行には、局の収支の平均（最終点 − 25,000）と、席ごとの和了率・放銃率が出ます。`--save-scores` には最終点と、対局・席ごとの `win`・`deal_in` が入ります。

## 3. プレイヤー

| プレイヤー | 中身 |
|---|---|
| `tsumogiri` | ツモ切り（できなければ見送り、それもなければ最初の合法手） |
| `rulebot`、`rulebot:latest` | ルールベースのボット（`docs/rulebot.md`）。`rulebot` は標準の設定（`STANDARD`） |
| `suzume[-組][:KEY=VALUE...]` | 教師に合わせた MLP で打つボット（`docs/suzume.md`）。`suzume`（= `suzume-mlp3`）、`suzume-mlp3-flat`、`suzume-mlp3-avoid4`、`suzume-mlp1`、`suzume-mlp1-flat`。例 `suzume-mlp1:call=0` |

ほかのパッケージは、`yamawake.register_agent`・`register_resolver` か、エントリポイントのグループ `yamawake.agents` でプレイヤーを足せます。`--opt KEY=VALUE` はそのファクトリにオプションを渡します。

## 4. Python

    from yamawake import RulebotAgent, TsumogiriAgent, run_match

    r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()],
                  walls=50)
    print(r.table())          # プレイヤーごとの平均順位 / pt と、牌山単位の標準誤差
    r.paired()                # 2v2 の牌山ごとの対の差

`build_agent(プレイヤーの指定)` で、コマンドラインと同じ指定からプレイヤーを作れます。

**打ち手の作り方**：`lane = "observation"`、`act_batch(decisions) -> [riichienv.Action]`、`describe() -> dict` を持つ任意のオブジェクトです。`Decision` は `riichienv.Observation` と、エンコーダが要る文脈（その局のイベント `kyoku_events`、`drawable_count`、`reacting_to`、`dora_indicator_types`）を持ちます。打ち手は `reducer_factory`（局ごとに作り直すイベントの畳み込み。`Decision.reducer` で読める）と `annotate(decision) -> dict`（`Decision.extra` に入る）も定義できます。

**1 局面の判断**：`decide(プレイヤーの指定, 観測, その局の mjai の出来事)` は、rulebot か suzume の判断を 1 つ返します（`{"action": (action_type, tile, consume_tiles)}`。suzume は決めたルール `rule` 付き、rulebot は `debug=True` で判断の材料 `debug` 付き）。対局ループと同じ関数で判断するので、対局中と同じ答えになります（4 席を `decide("rulebot")` で打たせた半荘は、ループの中の rulebot の半荘と同じになる）。ただし suzume は、その局の前の判断の記憶（押し引きの状態など）を持たずに始めます。`Decision` の `kyoku_events` をそのまま渡せます（山の残り枚数は観測が持つ）。

### 低水準：足並みをそろえたループ

手番を全て自分で制御したいとき（例：モデルのサーバーが考えている間にエンコードするオンラインの自己対戦）：

    import riichienv as rv
    from yamawake import SelfPlayLoop, SeatTracker, decision_of, mjai_action_spec
    from yamawake.rulebot import STANDARD
    from yamawake.walls import blob_for_wall

    loop = SelfPlayLoop(seeds, 2,                          # 2 = 半荘
                        bot_seats=[[(3, "rulebot", STANDARD)] for _ in seeds],
                        wall_tables=[blob_for_wall(s) for s in seeds], split_wall=True)
    trackers = {}
    while (batch := loop.collect()) is not None:       # GIL を解放。rulebot は中で打つ
        specs = []
        for g, s, b64 in zip(batch["games"], batch["seats"], batch["obs"]):
            obs = rv.Observation.deserialize_from_base64(b64)
            tr = trackers.setdefault((g, s), SeatTracker())
            tr.feed(obs.new_events())                   # 各イベントは 1 度だけ届く
            specs.append(mjai_action_spec(my_choice(decision_of(g, s, obs, tr))))
        loop.apply(specs)                               # 行ごとに正確に 1 手、順番どおり
    loop.scores(), loop.mjai_log(0)

`collect()` は全対局が終わると `None`、それまでは `{"n", "games", "seats", "obs"}` を返します。次の `collect()` の前に、`apply` で全ての行に答える必要があります。ループの中で判断するプレイヤーは `bot_seats=[[(席, "rulebot" か "suzume", 設定), ...], ...]` で置けます。

ほかに `yamawake._core` には、`run_sp`（一人麻雀の向聴・打点の DP）と `suzume_rules`（suzume のルールの一覧）があります。`rules` モジュールには精算（`UmaRule`：ウマ・オカ、`rank_of`、`final_pt`）があります。

## 5. 牌譜を見る・調べる

### handstats

`--record-logs` で書いた牌譜から、指定した名前のプレイヤーの統計を出します。同じ牌山・同じ並びの 2 つの実行を、局（ファイル・局・本場・席、配牌が同じもの）と半荘（ファイル・席）で対にして比べられます。標準誤差は対局のファイルごとにまとめて計算します。

    python -m yamawake.handstats LOGDIR --name suzume
    python -m yamawake.handstats LOGDIR --name suzume --vs OTHER --vs-name rulebot --json out.json

| 区分 | 統計 |
|---|---|
| 半荘 | 平均順位、順位分布、最終の点数、トビ率 |
| 速さと打点 | 聴牌率、巡目ごとの聴牌率、初聴牌の巡目、聴牌時の打点の見込み（ロン・ツモ）、立直・門前・役なし・待ちなしの率、ダマの率 |
| 局の結果 | 和了・放銃・被ツモ・横移動・流局の率と点数、流局時の聴牌率、立直率、副露率、親の連荘率、局の収支 |
| 立直 | 先制の率、巡目、待ちの見えていない枚数、良形率、その後の和了・放銃・流局の率、和了点、一発・裏ドラの率 |
| 副露 | 副露数、その後の和了・放銃の率、和了点 |
| 和了 | 巡目、ツモの率、翻数、満貫以上・跳満以上などの率、役の頻度 |
| 放銃 | 相手の状態（立直・副露・ダマ）、自分の状態（立直中・副露中・門前） |

和了の役と打点は、牌譜に書かれていないので、riichienv の評価器で計算し直します（牌譜の点数と、ダブロンの 2 人目の本場を除いて一致する）。定義の詳細は `python -m yamawake.handstats --help` にあります。

### tenhou6

mjai の牌譜を、天鳳の牌譜ビューアが読む tenhou.net/6 の JSON に変換します（`yamawake.tenhou6`）。

    from yamawake.tenhou6 import convert_file, dump

## 6. 制限

- arena の rulebot は大明槓をしません（`docs/rulebot.md`）。
- 対局の最後に残った供託の扱いは、エンジン（riichienv）に従います。`handstats` の最終の点数は、残った供託を誰にも渡さずに計算します。
