# suzume：隠れマルコフモデルで推定した潜在モードにおける方策蒸留モデル

suzume は、強化学習した麻雀エージェントRLActor-225（教師）の方策を、判断/モードごとの小さな MLP に蒸留した軽量のエージェントです。
一般的な強化学習エージェントは arena で大量に対局させるには計算量が多いので、
"強化学習エージェントに近い強さで速く打てる対戦相手"を作ることを目的として開発されました。

- コード：[`src/rust/suzume/`](../src/rust/suzume/)（判断）、[`src/python/yamawake/suzume.py`](../src/python/yamawake/suzume.py)（Python 側：指定できるオプションと、判断の規則の一覧）
- モデルと固定のパラメータ：[`params/`](../params/)（ビルドのときに埋め込む）

## 1. 使い方

``` bash
uv run yamawake-arena --mode 2v2 --players suzume rulebot
uv run yamawake-arena --mode 1v3 --players suzume-mlp1:call=0 rulebot
```

``` python
from yamawake import SuzumeAgent
agent = SuzumeAgent({"riichi": 0})
```

### モデルの組

| モデル | 手組み | 状態予測 | 特徴 |
|---|---|---|---|
| `suzume`（= `suzume-mlp3`） | `flat_mlp3` | `normal_mlp3` | デフォルトの3層MLP。状態予測は局・順位・点差を考慮 |
| `suzume-mlp3-flat` | `flat_mlp3` | `flat_mlp3` | 状態予測も東 1 局で学習（点数状況を見ない） |
| `suzume-mlp3-avoid4` | `normal_mlp3` | `normal_mlp3` | 手組みも点数状況を見る。ラス回避型になる |
| `suzume-mlp1` | `flat_mlp1` | `normal_mlp1` | 1 層の軽量版（局・順位・点差を考慮） |
| `suzume-mlp1-flat` | `flat_mlp1` | `flat_mlp1` | 1 層、状態予測も東 1 局で学習 |

名前は学習データ（`flat` 東 1 局、`normal` 東南の半荘）と隠れ層の数（`mlp1` 1 層、`mlp3` 3 層）を表します。手組みのモデルは [`params/hand/`](../params/hand/) の下で、表の「手組み」の名前のフォルダに判断ごとに 1 ファイルずつ（`call_P.json` など 11 個）あります。状態予測のモデルは [`params/state/`](../params/state/) の下で、表の「状態予測」の名前の `.json` です。

### オプション

常に反映されるオプション：

| オプション | 意味 | デフォルト |
|---|---|---|
| `kan` | 0/1：カンする | 1 |
| `call` | 0/1：チー・ポンする（大明槓は `kan` に従う） | 1 |
| `riichi` | 0/1：立直する | 1 |
| `pf` | 0/1：押し引きの状態を判断して打ち分ける（3 節）。0 にすると、他家の立直への対応は下の `fold` に替わる | 1 |

`pf=0` のときだけ反映されるオプション（`pf=1` では無視されます）：

| オプション | 意味 | デフォルト |
|---|---|---|
| `fold` | 0/1：他家の立直にベタオリする | 1 |
| `fold_shanten` | 0〜8：ベタオリに入る、打牌後の最小向聴（`fold=1` のときだけ） | 1 |

これ以外の値（DP の定数など、モデルが学習の前提にした値）はオプションにせず、[`params/suzume.json`](../params/suzume.json) に固定しています。

## 2. 判断の仕組み

判断ごとに次のルールを上から試します（[`src/rust/suzume/registry.rs`](../src/rust/suzume/registry.rs)）。

1. 和了できれば和了する。九種九牌は流局にする。
2. 押し引き（`pf=1`）：自分の打牌で状態を予測し、M・F ならその状態のモデルで切る（3 節）。
3. カン：カンの MLP がカンを高く評価したときだけ実行。
4. 打牌・立直・副露：判断ごとの MLP で選ぶ。
5. 合法な打牌が 1 つしかなければそれを切る。

判断ごとの MLP は、打牌（向聴 1〜2、3 以上、立直できない聴牌）、立直、副露、カンの 6 種類です。判断の各選択肢を、全判断で共通の特徴量（76 列、[`src/rust/suzume/universal.rs`](../src/rust/suzume/universal.rs)）の 1 行で表し、MLP のスコアが最大の選択肢を選びます。特徴量の中心は、手牌価値計算の DP（[`src/rust/eval/dp.rs`](../src/rust/eval/dp.rs)）により計算された打牌後の手の価値（和了率・聴牌率・期待値）です。これに向聴と受け入れ、待ち、手役、打牌の性質、僅差の判断の中の比較を加えます。

## 3. 押し引き

自分の打牌ごとに、押す（P）・回す（M）・オリる（F）の状態を判断し、状態ごとのモデルで打ちます（[`src/rust/suzume/pf.rs`](../src/rust/suzume/pf.rs)）。

| 判断 | P | M | F |
|---|---|---|---|
| 打牌 | 手組みの MLP | 回し打ち（`discard_M`） | ベタオリ（`discard_F`） |
| 立直 | `riichi_P` | `riichi_M` | `riichi_M` |
| チー・ポン | `call_P` | `call_M` | `call_F` |
| カン | `kan` | `kan` | しない |

- **状態予測**：危険度の特徴量を他家 3 人 × 合法な牌種の行ごとに MLP に通し、平均と最大でまとめて P・M・F の確率を出します。確率が最大の状態を使います。`normal_*` は、局、本場、順位、点差などの点数状況（21 個）も読みます。副露の判断は、直前の自分の打牌の状態を使います。
- **ベタオリ・回し打ち**：牌種ごとに、他家 3 人それぞれへの安全度を MLP で採点し、最も危険な相手を重く見る形（−τ log Σ exp(−s/τ)）で 1 つにまとめます。この値が最も高い牌種を切ります。特徴量（38 個）は、現物か、放銃パターン数（両面・カンチャン・ペンチャン・シャンポン・単騎）、河の読み（早い外側・宣言牌のスジなど）、他家の副露と見え方、自分の手の損、残る安全牌です。

## 4. 学習

- 教師（RLActor-225、鳴きあり）×4 の対局で、各判断の教師の方策に合わせて学習しました（クロスエントロピー最小化）。
- P・M・F の状態は、教師の打牌に当てた HMM の事後確率で決めます。各状態のモデルは、その状態の事後確率で重みを付けた行で学習しました。状態予測は、その事後確率を目標に学習しています。
- データは、`flat_*` が東 1 局 10 万局、`normal_*` が半荘です。どちらもランダム牌山（split なし）で、対局ごとに 7 : 2 : 1（train・valid・test）に分割。

## 5. ベンチマーク

suzume と現在公開されているモデルとの性能比較は以下の通りです。

<img src="images/suzume_performance.svg" alt="suzume との 2 vs 2 での平均順位と、モデルごとの 1 判断の時間、半荘/分">

- suzumeはmortal-v4の第三者学習モデルグループより少し弱い程度の性能です。
- suzumeはRLActor-225(未公開)を教師として学習されたモデルですが、この教師モデルはmortal再学習モデルより少し強いです。
- 速度はTanyaoDojoの10倍程度、mortal-v4の25倍程度と高速に動作します。

比較したモデルの出典：

| モデル | 出典 |
|---|---|
| mortal-298k | [VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k)（[mortal-v4](https://github.com/Equim-chan/Mortal)、298k ステップ） |
| mortal-582500 | [Yuchen1457/mortal-582500](https://huggingface.co/Yuchen1457/mortal-582500)（mortal-v4、582,500 ステップ） |
| mortal-S42 | [haor/Mortal-S42](https://huggingface.co/haor/Mortal-S42)（mortal-v4、S42.8k） |
| TanyaoDojo | [Twu31/TanyaoDojo](https://huggingface.co/Twu31/TanyaoDojo) の `bc_v2_g186.pkl`（コード：[twu3202/TanyaoDojo](https://github.com/twu3202/TanyaoDojo)） |
| 電脳麻将 | [kobalab/majiang-ai](https://github.com/kobalab/majiang-ai) 1.3.0 を [kobalab/mjai-bot](https://github.com/kobalab/mjai-bot) 2.0.0 で実行 |
| mjai-manue | [Apricot-S/mjai-manue-go](https://github.com/Apricot-S/mjai-manue-go)（[gimite/mjai-manue](https://github.com/gimite/mjai-manue) の Go 移植） |
| rulebot | このリポジトリ（[`rulebot.md`](rulebot.md)） |
| RLActor-225 | 未公開 |

平均順位は、各モデル 2 体と suzume（`suzume-mlp3`）2 体の 2 vs 2 で、標準牌山の先頭 1,000 牌山を席の回転 2 回ずつ（2,000 半荘）打った結果です。
図のバイオリンとボックスプロットは、牌山を選び直して推定し直した 400 回（ブートストラップ）の分布で、箱は四分位、ひげは 95% の範囲、点は推定値です。
表の時間は 4 人とも同じモデルで、CPU 1 個だけを使って推論したときの値です（電脳麻将、mjai-manue は 20 牌山、そのほかは 256 牌山。下の「6. 計算速度」は 20 牌山の値のため、少し異なります）。

## 6. 計算速度

1 CPU（1 コアに固定し、1 スレッドで実行）、4 人とも同じボット、256 牌山、席の回転なし（20 半荘）、1 回の計測。
CPU は AMD EPYC 7702（Zen 2、定格 2.0 GHz・最大 3.35 GHz。計測したマシンの周波数の上限は 2.18 GHz）：

| ボット | 半荘/分 | 1 判断あたりの時間 |
|---|---|---|
| `suzume-mlp3(-flat)` | 約 162.9 | 約 0.53 ms |
| `suzume-mlp3-avoid4` | 約 159.7 | 約 0.54 ms |
| `suzume-mlp1(-flat)` | 約 192.2 | 約 0.43 ms |
| rulebot（参考） | 約 515.1 | 約 0.16 ms |
| ツモ切り（参考、`tsumogiri`） | 約 1,420 | 約 0.017 ms |

1 判断あたりの時間は、ボット自身の判断にかかった時間です（対局エンジンとループの時間を含まない。出力の Time の表の ms/decision と同じ）。

参考として、Mac（Apple M4 Pro、P コア最大 4.5 GHz）での値です。条件は上と同じですが、macOS ではコアに固定できないため、1 スレッドで実行しています：

| ボット | 半荘/分 | 1 判断あたりの時間 |
|---|---|---|
| `suzume-mlp3` | 約 400 | 約 0.23 ms |
| `suzume-mlp3-flat` | 約 390 | 約 0.22 ms |
| `suzume-mlp3-avoid4` | 約 380 | 約 0.23 ms |
| `suzume-mlp1` | 約 450 | 約 0.19 ms |
| `suzume-mlp1-flat` | 約 460 | 約 0.19 ms |
| rulebot（参考） | 約 1,120 | 約 0.07 ms |
| ツモ切り（参考、`tsumogiri`） | 約 3,100 | 約 0.008 ms |

全てのスレッド（14 コア：P コア 10・E コア 4）を使うと、500 牌山で `suzume-mlp3` 4 人が約 3,500 半荘/分、rulebot 4 人が約 10,000 半荘/分です。
