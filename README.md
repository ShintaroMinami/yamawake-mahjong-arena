# yamawake-mahjong-arena

**日本語** | [English](README.en.md)

### 複製牌山での麻雀

手札複製ブリッジ(Duplicate Bridge)では、全卓が同じ手札構成でプレイします。これは、対戦結果が手札の運に左右されずプレイヤーの実力差をより正確に測ることを目指した工夫です。yamawake-mahjong-arena は麻雀の実力測定に同じ工夫を導入します。

　対局の各局の牌山を固定し、各席が自分専用の山からツモります（*固定分割牌山*）。このようにすることで、ゲーム中に発生する副露や連荘によって予定されたツモ牌がずれることなく最後まで維持されます。
この固定分割牌山を席ごとに割り当て、順に回して使用します。
全プレイヤーが全ての固定分割牌山を１度ずつ使用することで、結果の違いがプレイヤーの選択だけから生じるようにしています。

>　**重要 :** なお、ツモと局ごとの牌山を固定する同様の対局方式は、電脳麻将の作者である小林聡（kobalab）氏が 2020 年に「デュプリケート麻雀」として既に実装し、公開されていました（[デュプリケート麻雀の実装](https://blog.kobalab.net/entry/2020/12/19/075529)、[majiang-ai](https://github.com/kobalab/majiang-ai)）。実装してから気づきましたが、下調べは大事です。

### リポジトリの主な内容

- **arena**：固定分割牌山の対戦環境。Rust の対局ループで N 局を並列に進めます（[`docs/arena.md`](docs/arena.md)を参照）。
- **rulebot**：高速なルールベースのボット。ベースラインとして有用です（[`docs/rulebot.md`](docs/rulebot.md)を参照）。
- **suzume**：強化学習した麻雀AIの蒸溜モデル。判断ごとの小さな MLP を用いた軽量のボット（[`docs/suzume.md`](docs/suzume.md)を参照）。

1 CPU（AMD EPYC 7702、最大 2.18 GHz で計測）を使用した場合、
rulebot 4 人なら約 470 半荘/分、
suzume 4 人なら約 170 半荘/分 で動作します。
参考として、Mac（Apple M4 Pro）では 1 スレッドで rulebot 4 人が約 1,120 半荘/分、suzume 4 人が約 400 半荘/分、全てのスレッド（14 コア）を使うと rulebot 4 人が約 10,000 半荘/分、suzume 4 人が約 3,500 半荘/分です。
Python のプレイヤーは、判断 1 回ごとに観測の受け渡しで約 55 µs かかります（同じ EPYC の 1 CPU）。これは、arena対局の1判断あたりのシステム時間（約 23 µs）の2倍程度に相当します。非常に高速なアルゴリズムを評価する場合は、この時間が全体の速さに影響する可能性があります（[`docs/arena.md`](docs/arena.md)を参照）。

## suzume の性能

ここに対戦ベースで測定した suzume の強さを記載
- ツモ切り
- 電脳麻将
- mortal-298k
- etc.


## インストール

### リリースの wheel から（Rust の環境は不要）

各 [リリース](https://github.com/ShintaroMinami/yamawake-mahjong-arena/releases) に、Linux x86_64/aarch64（manylinux2014）と macOS arm64、CPython 3.11〜3.13 の wheel があります。
Rust の部分は wheel の中にビルド済みです。
`riichienv` と `numpy` は PyPI から入ります。
``` bash
    gh release download -R ShintaroMinami/yamawake-mahjong-arena -p '*.whl' -D wheels
    pip install --find-links wheels yamawake
```
### ソースから

Python 3.11 以上、Rust の環境、[uv](https://docs.astral.sh/uv/) が必要です。
``` bash
    git clone https://github.com/ShintaroMinami/yamawake-mahjong-arena.git
    cd yamawake-mahjong-arena
    uv sync                # 拡張を（maturin で）.venv にビルド
    uv run pytest          # テスト
```
uv を使わずに、Rust の環境があれば `pip install git+https://github.com/ShintaroMinami/yamawake-mahjong-arena.git` でも入ります。

パッケージの名前は `yamawake`、コマンドの名前は `yamawake-arena` です。

## 使い方
``` bash
    uv run yamawake-arena --mode 2v2 --players tsumogiri rulebot --walls 100
    uv run python -m yamawake.arena --mode 1v3 --players suzume rulebot
```    
結果は、順位とポイント、1 局あたりの和了率・放銃率・副露率・立直率、平均打点と調整打点効率、プレイヤーごとと arena 自体の計算時間を、表にまとめて標準出力に出します（[`docs/arena.md`](docs/arena.md#2-コマンドライン)）。

Python からの利用は以下のように行います。
``` python
    from yamawake import RulebotAgent, TsumogiriAgent, run_match
    r = run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(), RulebotAgent()], walls=100)
    print(r.table())
```

| プレイヤー | 説明 |
|---|---|
| `tsumogiri` | ツモ切りボット |
| `rulebot` | ルールベースの高速ボット |
| `suzume`（= `suzume-mlp3`）、`suzume-mlp3-flat`、`suzume-mlp3-avoid4` | 軽量かつある程度強いボット (3 層の MLP を使用) |
| `suzume-mlp1`、`suzume-mlp1-flat` | 強さを少し落とした、より軽量なボット (1 層の MLP を使用) |

> **Note :** flat は東 1 局の対局データのみで学習したモデル。avoid4はラス回避を重視したルールで学習したモデルです。

## 自作のプレイヤーとの対局

プレイヤーは、`lane = "observation"`、`act_batch(decisions)`、`describe()` を持つ任意のオブジェクトとし、以下のように実装します。
- `act_batch` では、そのプレイヤーの手番を待つ全ての対局の判断`decisions`をリストとして受ける。(リストとして渡されるのでモデルはまとめて推論できます)
- `decisions` の各判断 `d` は、観測 `d.obs`（`riichienv.Observation`。合法手・手牌・河・点数など含む）と、その局の mjai 形式のイベント `d.kyoku_events` などを持つ。
- 判断ごとに合法手から 1 つを選んで、`actions`としてリストを返す。

```python
    import riichienv as rv
    from yamawake import RulebotAgent, lineup, run_match

    class MyAgent:
        lane = "observation"

        def act_batch(self, decisions):
            actions = []
            for d in decisions:
                legal = d.obs.legal_actions()
                # 和了できれば和了、それ以外は最初の合法手
                win = [a for a in legal if a.action_type in (rv.ActionType.TSUMO, rv.ActionType.RON)]
                actions.append(win[0] if win else legal[0])
            return actions

        def describe(self):
            return {"kind": "mine", "name": "my-agent"}

    r = run_match(lineup(MyAgent(), RulebotAgent(), "2v2"), walls=100)   # MyAgent x 2 対 rulebot x 2
    print(r.table())
```

`register_agent("my-agent", lambda spec, **opts: MyAgent())` で名前を登録すると、同じプロセスの中で `yamawake.arena.main(["--players", "my-agent", "rulebot", ...])` のように名前で指定できます。`yamawake-arena` コマンドから使うには、自作のパッケージのエントリポイントのグループ `yamawake.agents` に登録の関数を置きます。判断に使える情報の一覧と、より詳しいコード例は [`docs/arena.md`](docs/arena.md#4-自作のプレイヤー) にあります。


その他、低水準のループ、1 局面の判断（`decide`）、牌譜の統計（`handstats`）と天鳳の牌譜への変換（`tenhou6`）など、詳細は [`docs/arena.md`](docs/arena.md) を参照してください。

## 謝辞

以下のリポジトリや文章を参考にしています。
素晴らしいソフトウェアや知見を提供してくださった著者/開発者の方々に心から感謝いたします。

- **[RiichiEnv](https://github.com/smly/RiichiEnv)**（smly、Apache-2.0）がシミュレータの土台となっています。yamawake-mahjong-arena の対局エンジンは riichienv-core 0.4.8 を取り込んだ fork で、席ごとのツモの列と、明示的な（局、本場）の牌山表を追加したバージョンです（[`src/engine/README.md`](src/engine/README.md)）。Python 側は `riichienv` パッケージを使います。
- **[電脳麻将](https://github.com/kobalab/Majiang)**（小林聡 / kobalab。[majiang-ai](https://github.com/kobalab/majiang-ai)、MIT、および電脳麻将のプロジェクト全体）：rulebot の設計はこちらのリポジトリを参考にしています。[`src/rust/rulebot/kobalab_ev.rs`](src/rust/rulebot/kobalab_ev.rs) は majiang-ai の `eval_shoupai` / `eval_fulou` の移植で、副露の判断は `select_fulou` を、カンの判断は `select_gang` を参考にしています。
- 手牌価値計算のためのアルゴリズム(動的計画法,DP)は、[tomohxx](https://github.com/tomohxx) の『麻雀アルゴリズム』（[6 章 和了確率](https://tomohxx.github.io/mahjong-algorithm-book/probability/)）の手法を、[nekobean](https://github.com/nekobean) の [mahjong-cpp](https://github.com/nekobean/mahjong-cpp) の解説に沿って実装しています。細かな実装の一部は libriichi（[Mortal](https://github.com/Equim-chan/Mortal)）を参考にしました。
- ほかに、riichienv-core を通して [Nyanten](https://github.com/Cryolite/nyanten)（MIT）の向聴表を使わせていただいています。

これらのライセンスの全文は [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) にあります。

## ライセンス

このリポジトリのライセンスは MIT（[`LICENSE`](LICENSE)）です。取り込んだ riichienv-core は Apache-2.0（[`src/engine/riichienv-core/LICENSE`](src/engine/riichienv-core/LICENSE)）です。

