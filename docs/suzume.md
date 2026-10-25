# suzume：判断ごとの小さな MLP による方策蒸留モデル

suzume は、強化学習した麻雀エージェント [Mortal-S42](https://huggingface.co/haor/Mortal-S42)（教師）の方策を、判断の種類ごとの小さな MLP に蒸留した軽量のエージェントです。
一般的な強化学習エージェントは arena で大量に対局させるには計算量が多いので、
"強化学習エージェントに近い強さで速く打てる対戦相手"を作ることを目的として開発されました。

- コード：[`src/rust/suzume/a1.rs`](../src/rust/suzume/a1.rs)（判断）、[`src/rust/suzume/universal.rs`](../src/rust/suzume/universal.rs)（選択肢の特徴量）、[`src/rust/suzume/fold_model.rs`](../src/rust/suzume/fold_model.rs)（危険度の特徴量）
- モデル：[`params/suzume/`](../params/suzume/)（ビルドのときに埋め込む）

## 1. 使い方

``` bash
uv run yamawake-arena --mode 2v2 --players suzume rulebot
uv run yamawake-arena --mode 1v3 --players suzume-rla225 rulebot
```

``` python
from yamawake import SuzumeAgent
agent = SuzumeAgent()                          # suzume
agent = SuzumeAgent(models="suzume-rla225")    # 別の教師で学習した組
```

### モデルの組

| モデル | 教師 | フォルダ |
|---|---|---|
| `suzume` | Mortal-S42 | [`params/suzume/suzume/`](../params/suzume/suzume/) |
| `suzume-rla225` | RLActor-225（未公開） | [`params/suzume/rla225/`](../params/suzume/rla225/) |

2 つの組は、教師と学習データだけが違い、モデルの形と学習の手順は同じです（4 節）。

### 打ち方の調節

`suzume` には、打ち方を調節する 3 つのパラメータがあります（2026-10 に追加）。
既定ではどれも 0（調節なし）で、0 のときはこれまでの `suzume` と同じに打ちます。
目的は打ち方の違う対戦相手を作ることで、強くすることではありません。

| パラメータ | 調節するもの | 値 | 仕組み |
|---|---|---|---|
| `aggression` | 押し引き | -1 〜 1 | 脅威のある局面で、押しのモデルと降りのモデルの方策を補間する |
| `call_shift` | 副露の度合い | -1 〜 1 | 副露（チー・ポン・大明槓）の選択肢のスコアを動かす |
| `riichi_shift` | 立直の度合い | -1 〜 1 | 立直の選択肢のスコアを動かす |

``` bash
uv run yamawake-arena --mode 2v2 --players suzume:aggression=-0.5:riichi_shift=0.5 suzume
```
``` python
agent = SuzumeAgent({"aggression": 0.5, "call_shift": 0.5})
```

- `aggression`：判断の種類（打牌 4 種、副露、槓）ごとに、押しのモデルと降りのモデルの 2 つの小さな MLP を学習させてあります（[`params/suzume/suzume/`](../params/suzume/suzume/) の `*_push.json` / `*_fold.json`）。教師の方策のうち、他家の立直や副露などの脅威がある局面の判断だけを、教師が押したときの方策（押しのモデル）、または最も強い脅威に降りたときの方策（降りのモデル）に置き換えて学習したものです。正の値では `suzume` の対数方策に押しのモデルの対数方策を、負の値では降りのモデルの対数方策を、|値| の割合で混ぜます（`(1 - |値|) log π + |値| log π_端点`）。1 で押しのモデル、-1 で降りのモデルの打ち方になります。立直するかダマにするかは `suzume` 自身の選択のままで、`aggression` はその中で打牌を選びます。
- `call_shift` / `riichi_shift`：その選択肢のスコアに、値に倍率を掛けたものを足します。倍率は正負で分けてあり、`call_shift` は負で 2.5・正で 3.4、`riichi_shift` は負で 4.5・正で 3.5 です。鳴ける（立直できる）判断で鳴く（立直する）オッズが、±1 で `call_shift` はおよそ e^1.5 倍（1/e^1.5 倍）、`riichi_shift` はおよそ e³ 倍（1/e³ 倍）になるように、既定の `suzume` との対局（標準牌山 0〜1999）で決めました。
- 3 つは組み合わせて使えます。持つのは `suzume` だけです（`suzume-rla225` で指定するとエラーになります）。

## 2. 判断の仕組み

判断ごとに次の順に決めます（[`src/rust/suzume/a1.rs`](../src/rust/suzume/a1.rs) の `decide`）。

1. 和了できれば和了する。九種九牌は流局にする。
2. 立直を宣言した直後は、立直の選択肢で選んだ牌を切る。
3. 暗槓・加槓ができるときは、カンの MLP が、カンしない場合（下の 4. で選ぶ手）と各カンを比べる。
4. 打牌・立直・副露：判断の種類ごとの MLP で、スコアが最大の選択肢を選ぶ。
5. どれでも決まらないときは、合法な打牌が 1 種類ならそれを、なければパス、ツモ切り、最初の合法手の順に選ぶ。

判断の種類と MLP：

| MLP | 判断 | 大きさ |
|---|---|---|
| `discard0` | 聴牌からの打牌（立直できるかどうかによらない。立直の選択肢を含む） | 3 層 × 64 |
| `discard1` | 打牌後の最小の向聴が 1 の打牌 | 3 層 × 144 |
| `discard2` | 打牌後の最小の向聴が 2 の打牌 | 3 層 × 144 |
| `discard3` | 打牌後の最小の向聴が 3 以上の打牌 | 3 層 × 128 |
| `call` | チー・ポン・大明槓とパス | 3 層 × 64 |
| `kan` | 暗槓・加槓とカンしない場合 | 3 層 × 16 |

判断の各選択肢を 1 行の特徴量で表し、MLP のスコアが最大の選択肢を選びます。行の特徴量は次の通りです。

| 特徴量 | 数 | 入る選択肢 | 内容 |
|---|---|---|---|
| 共通の特徴量（[`universal.rs`](../src/rust/suzume/universal.rs)） | 最大 76 列 | 全て | 手牌価値計算の DP（[`dp.rs`](../src/rust/eval/dp.rs)）による打牌後の手の価値（和了率・聴牌率・期待値）、向聴と受け入れ、待ち、手役、打牌の性質、僅差の選択肢どうしの比較 |
| 危険度（[`fold_model.rs`](../src/rust/suzume/fold_model.rs)） | 他家 3 人 × 37 個 | 打牌・立直 | 切る牌の、各他家（下家・対面・上家の順）への危険度：現物か、放銃のパターン数（両面・カンチャン・ペンチャン・シャンポン・単騎）、河の読み（早い外側・宣言牌のスジなど）、他家の副露と見え方、残る安全牌など。下の席の特徴量 9 個を含む |
| 席の特徴量 | 他家 3 人 × 9 個 | 副露 | 他家ごとの立直、副露の数と副露のドラ、親か、捨て牌の数、役牌・染め手・タンヤオ・チャンタの副露 |
| 点差 | 他家 3 人 × 1 個 | 打牌・立直・副露 | 自分とその他家との点差 |
| 局面の状況 | 21 個 | 全て（判断ごとに共通） | 局（東 1〜南 4）、本場、供託、順位、点数と上下との点差 |
| 状態の one-hot | 1 組 | 全て（判断ごとに共通） | 手の型（一般・七対子・染め手・役牌）、序盤か中盤以降か、他家の立直があるかどうか |

モデルごとに使う列は少しずつ違います（各モデルのファイルの `inputs`）。

## 3. 特徴量の計算に使うモデル

カンの判断では、カンしない場合の行を作るために、打牌の手を 1 つ仮に選びます。
この仮の手と、危険度の特徴量の表（待ちになっている割合など）は、[`params/suzume/features/`](../params/suzume/features/) のモデルと表から作ります。
suzume の手そのものは、この仮の手ではなく 2 節の MLP が決めます。

## 4. 学習

- 教師 ×4 の半荘 1 万局を記録し、各判断の教師の方策に合わせて学習しました（クロスエントロピー最小化）。教師の方策は、教師の Q 値の softmax（温度 1）です。
- 牌山はランダム（split なし）です。牌山の seed の md5 の値で、7 : 2 : 1（train・valid・test）に分けました。
- 最適化は schedule-free の RAdam（学習率 0.01、weight decay なし）です。200 エポック（カンは 100 エポック）学習し、valid で最も良かったエポックのモデルを使います。
- MLP の幅は、打牌のモデルごとに幅を変えて学習し、2 vs 2 の対局の強さで選びました。

## 5. ベンチマーク

suzume と現在公開されているモデルとの性能比較は以下の通りです。

<img src="images/suzume_performance.svg" alt="suzume との 2 vs 2 での平均順位と、モデルごとの 1 判断の時間、半荘/分">

- suzumeは、mortal-v4の第三者学習モデルのうちmortal-298kとほぼ同じ強さで、mortal-582500やTanyaoDojoより少し弱い程度の性能です。
- suzumeの教師のmortal-S42との差は、平均順位で0.05程度です。
- suzume-rla225（教師はRLActor-225）はsuzumeとほぼ同じ強さです（平均順位で0.015程度弱い）。
- 速度はTanyaoDojoの10倍程度、mortal-v4の30倍程度と高速に動作します。

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
| windshadow-sl | [windshadow233/Mahjong-AI](https://github.com/windshadow233/Mahjong-AI)（Suphx の教師あり学習の部分の再現） |
| RLActor-225 | 未公開 |

平均順位は、各モデル 2 体と suzume 2 体の 2 vs 2 で、標準牌山の先頭 1,000 牌山を席の回転 2 回ずつ（2,000 半荘）打った結果です。
図のバイオリンとボックスプロットは、牌山を選び直して推定し直した 400 回（ブートストラップ）の分布で、箱は四分位、ひげは 95% の範囲、点は推定値です。
表の時間は 4 人とも同じモデルで、CPU 1 個だけを使って対局したときの値です（電脳麻将、mjai-manue、RLActor-225 は 20 牌山、そのほかは 256 牌山）。

## 6. 計算速度

1 CPU（1 コアに固定し、1 スレッドで実行）、4 人とも同じボット、256 牌山、席の回転なし（256 半荘）、1 回の計測。
CPU は AMD EPYC 7702（Zen 2、定格 2.0 GHz・最大 3.35 GHz。計測したマシンの周波数の上限は 2.0 GHz）：

| ボット | 半荘/分 | 1 判断あたりの時間 |
|---|---|---|
| `suzume` | 約 173 | 約 0.52 ms |
| `suzume-rla225` | 約 167 | 約 0.51 ms |
| rulebot（参考） | 約 515 | 約 0.16 ms |

1 判断あたりの時間は、ボット自身の判断にかかった時間です（対局エンジンとループの時間を含まない。出力の Time の表の ms/decision と同じ）。

参考として、Mac（Apple M4 Pro、P コア最大 4.5 GHz）での値です。条件は上と同じですが、macOS ではコアに固定できないため、1 スレッドで実行しています：

| ボット | 半荘/分 | 1 判断あたりの時間 |
|---|---|---|
| `suzume` | 約 400 | 約 0.23 ms |
| `suzume-rla225` | 約 380 | 約 0.22 ms |
| rulebot（参考） | 約 1,200 | 約 0.07 ms |

全てのスレッド（14 コア：P コア 10・E コア 4）を使うと、500 牌山で `suzume` 4 人が約 3,700 半荘/分、rulebot 4 人が約 10,000 半荘/分です。
