# rulebot：高速なルールベースのボット

rulebot は、作者自身のルールベースのボットを riichienv に移植したものです。arena の物差し（強さを測る相手）として、速く、決まった打ち方をすることを目的にしています。対局ループの中で Rust のまま判断します。

- コード：[`src/rust/rulebot/`](../src/rust/rulebot/)（判断）、[`src/rust/eval/`](../src/rust/eval/)（一人麻雀の DP、点数、向聴。suzume と共通）、[`src/python/yamawake/rulebot.py`](../src/python/yamawake/rulebot.py)（設定）
- 移植元と謝辞：[`docs/THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

## 1. 判断の流れ

1 回の判断（`rulebot::react`）は、次の順に決めます。

1. 和了できれば和了する。
2. 暗槓・加槓（設定 `kan`）：majiang-ai の `select_gang` の移植（`kan.rs`）。他家が立直していて自分が聴牌でなければしない。聴牌から 2 向聴までは、カン後の手の評価（`kobalab_ev`）がカン前より下がらないものの最大。3 向聴以上は、役を考えた向聴数が変わらない最初のもの。大明槓はしない。
3. 立直できるときは、立直するかを決める（`should_riichi`：ダマのロンの打点が最も低い待ちでも 6400 点以上ならダマ）。
4. 打牌（`discard.rs`）。
5. 鳴き（`call.rs`）。

## 2. 打牌

脅威がなければ、手を進める打牌を選びます。2 向聴以内は一人麻雀の DP（`sp_adapter.rs`、`eval::dp`）の期待値が最大の牌。それより遠いか DP の値がないときは、役を考えた向聴数（`yaku_shanten.rs`、majiang-ai の `Player#xiangting`）、受け入れ、ドラの保持、孤立牌の順で比べます。

脅威は、他家の立直と、テンパイらしい副露（巡目に対して副露が多く、見えている価値が 3 以上）です。脅威があるときは、2 段で決めます（`threatened_discard`）。

- 2 向聴以上：最も安全な牌（同じ安全さなら手の損が小さい牌）を切る。
- 1 向聴と聴牌：手の価値と危険度の 1 本の軸で、押すかオリるかを決める（`push_or_fold`。各打牌の DP の期待値から危険度 × 係数を引き、最も安全な牌を価値 0 として候補に加えて最大を選ぶ。係数は 1 向聴が 50、聴牌が 5）。DP の値がないときは、1 向聴は向聴数を保つ牌のうち最も安全な牌、聴牌は手を進める打牌。

危険度は、待ちの形の数え上げで見積もります（`danger.rs`。majiang-ai の基礎の危険度と同じく見えていない枚数で重み付けし、ドラと親を重くする）。複数の脅威には最大の危険度を使い、脅威 1 つ増えるごとに 1.6 倍します。

## 3. 鳴き

majiang-ai の `select_fulou` に従います。3 向聴以上では、役を考えた向聴数が下がる鳴きをします（脅威があれば鳴かない）。2 向聴以内では、鳴いたときと見送ったときの手の価値を比べ、上回る鳴きのうち最大のものをします。脅威があるときは、鳴いた後の価値が 750 点（聴牌なら 250 点）未満の鳴きはしません。手の価値は、標準の設定では一人麻雀の DP（`call_ev_sp`。見送りの価値は立直を仮定しない `call_sp_flat_pass`）、`LATEST` では `kobalab_ev`（majiang-ai の `eval_shoupai` / `eval_fulou` の移植）で測ります。

## 4. 設定

設定は `RulebotConfig`（[`src/rust/rulebot/config.rs`](../src/rust/rulebot/config.rs)）の欄で、2 つの設定を区別するものだけです（`call_ev_sp`、`call_sp_flat_pass`、`kan`。既定はすべて off で、`LATEST` と同じ）。両方に共通の値（上の 50、5、1.6、6400 など）は、同じファイルの定数です。2 つの設定は Python 側に置いています（[`src/python/yamawake/rulebot.py`](../src/python/yamawake/rulebot.py)）。

| 設定 | 中身 |
|---|---|
| `STANDARD`（プレイヤー名 `rulebot`） | 標準の設定。`LATEST` の鳴きの評価を一人麻雀の DP に替え、暗槓・加槓をする（2026-10-13 から） |
| `LATEST`（プレイヤー名 `rulebot:latest`） | 設定の探索の挑戦者（鳴きの評価は `kobalab_ev`） |

Python からは `RulebotAgent(設定)` で座らせます。指定しなかった欄は Rust の既定のままで、知らない欄はエラーになります。

## 5. 1 局面の判断

`decide("rulebot", 観測, その局の mjai の出来事, debug=True)` で、対局中と同じ判断と、判断の材料（向聴、脅威、手の価値など）を得られます（[`docs/arena.md`](arena.md) 3 節）。

## 6. 検査

- golden（[`tests/golden/`](../tests/golden/)）：標準の rulebot の対局（rulebot ×4、ツモ切り 1 対 rulebot 3）が、記録した点数と牌譜のとおりになることを確かめます。
- カン（[`tests/rulebot/`](../tests/rulebot/)）：標準の設定は暗槓・加槓をして大明槓はしないこと、`kan` を切ればカンしないことを確かめます。
- 一人麻雀の DP（[`tests/eval/test_sp.py`](../tests/eval/test_sp.py)）：reljong v0.1 と同じ答えを出すことを確かめます。

## 7. 制限

- 大明槓はしません。大明槓は、他家の牌で新しいドラと短い山を卓に与える賭けで、このボットはその値踏みをしません。一人麻雀の DP は、カンの後の手も評価します（ドラ表示牌 1 枚につき裏ドラ表示牌 1 枚）。
