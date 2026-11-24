# 牌譜の形式（yamawake records）

半荘の牌譜を numpy の配列（`.npy`）に詰めて、1 つのディレクトリに置く形式です。
yamawake が正式な定義を持ち、他のリポジトリ（tide、mjrelnet など）はこれに合わせます。

- mjai の行との往復は完全に可逆です（yamawake のエンジンが出す行と、文字列として一致）。
- イベントは 1 件 2 バイトの固定長です。和了・流局・副露の中身は、型の付いた別の表に持ちます。そのため、コーパス全体を Python のループなしに numpy で集計できます（例：`op = events[:, 0] >> 4`）。
- 圧縮しないので、`np.load(mmap_mode="r")` で任意の半荘・局へ直接読みに行けます。
- 他の値から決まるものは持ちません。オフセットも持たず、件数から読み込み時に累積和で求めます。そのため、2 つのディレクトリは配列をつなぐだけで連結できます（`players.json` の添字だけ付け替える）。

読み書きは `yamawake.records` です。

``` python
from yamawake.records import Records, RecordWriter

w = RecordWriter()
w.add_game(events, scores=final_scores)      # events: mjai の dict（start_game から end_game まで）
w.save("out/")

r = Records("out/")
r.events(0)                                  # 半荘 0 の mjai の dict
r.lines(0)                                   # エンジンが出すのと同じ mjai の行
r["events"]                                  # (E, 2) uint8、全局のイベント
```

`run_match(record_logs=DIR, record_format="records")`（コマンドラインでは `--record-format records`）で、対局をこの形式で書き出します。mjai の gzip（`.json.gz`）のディレクトリは `python -m yamawake.records convert SRC OUT` で変換できます。

## 1. ファイル

| ファイル | 中身 |
|---|---|
| `meta.json` | `format`（`"yamawake-records"`）、`version`（今は 1）、半荘・局・イベントの数。ほかに出どころ（`source`）など自由な欄 |
| `players.json` | プレイヤー名の一覧（`game_player` の添字の先） |
| `*.npy` | 下の配列。リトルエンディアン |

## 2. 半荘（G 件）

| 配列 | 型 | 中身 |
|---|---|---|
| `game_n_kyoku` | (G,) uint16 | 局の数 |
| `game_player` | (G, 4) int32 | 各席のプレイヤー（`players.json` の添字） |
| `game_scores` | (G, 4) int16 | 最終点数（100 点単位）。終局時に残った供託を 1 位に渡した後の点数 |

出どころに固有の配列（例：`game_seed`）は `game_` で始まる名前で G 行持てます。読み手は知らない配列を無視します。

## 3. 局（K 件、半荘の順、半荘の中は局の順）

| 配列 | 型 | 中身 |
|---|---|---|
| `kyoku_info` | (K, 4) uint8 | 局の番号（場風 × 4 + 局 − 1。東 1 局が 0、北 4 局が 15）、本場、供託、ドラ表示牌（牌の 1 バイト） |
| `kyoku_scores` | (K, 4) int16 | 開始時の点数（100 点単位） |
| `kyoku_haipai` | (K, 4, 13) uint8 | 配牌（牌譜に書かれた順のまま） |
| `kyoku_n_events` | (K,) uint16 | イベントの数 |
| `kyoku_n_meld_tiles` | (K,) uint8 | `meld_tiles` の牌の数 |
| `kyoku_n_wins` | (K,) uint8 | 和了の数（1〜3）。0 は流局（`draw_deltas` に 1 行） |

親は持ちません（局 − 1）。

## 4. 牌の 1 バイト

- 下位 6 ビット：牌種 0〜33（萬 0〜8、筒 9〜17、索 18〜26、東南西北白發中 27〜33）。
- ビット 6：赤 5。
- ビット 7：打牌のツモ切り（それ以外は 0）。

## 5. イベント（`events`、(E, 2) uint8）

1 バイト目は `(op << 4) | (aux << 2) | actor`、2 バイト目は引数です。aux は相手の相対席（`(target − actor) mod 4`）で、和了では 0 がツモです。

| op | 種類 | aux | 2 バイト目 | 別の表 |
|---|---|---|---|---|
| 0 | tsumo | 0 | 牌 | — |
| 1 | dahai | 0 | 牌（ビット 7 にツモ切り） | — |
| 2 | chi | 相手 | 鳴いた牌 | `meld_tiles` に 2 枚 |
| 3 | pon | 相手 | 鳴いた牌 | `meld_tiles` に 2 枚 |
| 4 | daiminkan | 相手 | 鳴いた牌 | `meld_tiles` に 3 枚 |
| 5 | ankan | 0 | `pai`（槓の牌） | `meld_tiles` に 4 枚 |
| 6 | kakan | 0 | 加えた牌 | `meld_tiles` に 3 枚 |
| 7 | reach | 0 | 0 | — |
| 8 | reach_accepted | 0 | 0 | — |
| 9 | dora | 0（actor 0） | ドラ表示牌 | — |
| 10 | hora | 放銃者（0 はツモ） | 0 | 和了の表に 1 行 |
| 11 | ryukyoku | 0（actor 0） | 流局の種類 | `draw_deltas` に 1 行 |

- `start_game`、`start_kyoku`、`end_kyoku`、`end_game` は入れません。半荘と局の表から復元します。
- 晒した牌（`consumed`）は牌譜に書かれた順のまま、`meld_tiles`（uint8、1 次元）にイベントの順で並べます。

流局の種類：

| 値 | `reason` |
|---|---|
| 0 | なし（不明。天鳳由来の牌譜など） |
| 1 | `exhaustive_draw` |
| 2 | `kyushu_kyuhai` |
| 3 | `sufuurenta` |
| 4 | `suucha_riichi` |
| 5 | `suukansansen` |
| 6 | `sanchaho` |
| 7 | `nagashimangan` |

## 6. 和了（H 件）と流局（R 件）

どちらもイベントの順です。

| 配列 | 型 | 中身 |
|---|---|---|
| `win_deltas` | (H, 4) int16 | 点数の移動（100 点単位。本場と供託を含む） |
| `win_han` | (H,) uint8 | 翻（0 は不明） |
| `win_fu` | (H,) uint8 | 符（役満は 0） |
| `win_yaku` | (H,) uint64 | 役 id のビット（ビット i が役 id i。riichienv の id） |
| `win_ura` | (H, 5) uint8 | 裏ドラ表示牌（使わない欄は 0xFF） |
| `draw_deltas` | (R, 4) int16 | 点数の移動（100 点単位。立直棒は含まない） |

- 牌譜に翻・符・役が 1 件もない（天鳳由来など）ときは、`win_han`・`win_fu`・`win_yaku` を置きません。読み手は、ないときは翻 0（不明）として扱います。
- 役満かどうかは持ちません（役 id 35 以上のビットがあれば役満）。
- ツモかどうかは aux（0 がツモ）で決まります。

## 7. mjai との対応

`Records.events(g)` は、yamawake のエンジンが出す mjai と同じ dict を返します。`Records.lines(g)` は、それをキーの昇順・区切り `(",", ":")` で書いた行で、エンジンの行と文字列として一致します。

- `start_game` は `{"names": [...], "type": "start_game"}` です。
- `hora` は `han`・`fu`・`yaku`（昇順）・`yakuman`・`ura_markers` を持ち、ツモでは `"tsumo": true` も持ちます。`win_han` が 0（不明）なら、`han`・`fu`・`yaku`・`yakuman` を出しません。
- `ryukyoku` は、種類が 0 でなければ `reason` を持ちます。
- `ankan` は `pai` を持ちます。`pai` のない牌譜（天鳳由来など）から作るときは、`consumed` の 1 枚目を入れます。

## 8. 範囲

- 点数と点数の移動は 100 の倍数で、int16（100 点単位）に収まること。
- 1 局の和了は 3 回まで、裏ドラ表示牌は 5 枚まで、役 id は 63 まで。
- 書き込み（`RecordWriter.add_game`）は、これに合わない牌譜を `ValueError` で断ります。
