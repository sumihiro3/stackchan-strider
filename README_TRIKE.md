# Stackchan Trike(逆三輪車)

M5Stack 製品版スタックチャン(K151 / CoreS3)を載せて走る、3D プリントの逆三輪車。前の 2 輪で走り、後ろの 1 輪でかじを取る。

- スタックチャンが首を振る(ヨー)と、ターンテーブルと歯車を通して前の駆動輪が回る。
- うなずくと、頭の右側に差したレバーが赤い引き棒を前後に押し引きし、後輪のフォークが左右に切れる(うなずき 0〜30° で ±30°、15° で直進)。
- ネジと接着剤は使わない。C クリップとツメで留める。

**状態: 設計と印刷データまで。実物の組み立てと走行はまだ確かめていない(2026-11-23)。**

## 組立ガイド

[`docs_trike/guide.html`](docs_trike/guide.html) をブラウザで開く(1 つのファイルで完結する。three.js は CDN から読み込む)。スクロールに合わせて、3D で部品が 1 つずつ付いていく。手順は 12。

作り直すとき:

```bash
.venv/bin/python car/export_guide_trike.py
```

## 印刷

| ファイル | 中身 |
|---|---|
| `print_trike/stackchan_trike.3mf` | 色ごとのプレート(グレー 2 枚、赤、黒)。Bambu Lab P2S 用 |
| `print_trike/stackchan_trike_1color.3mf` | 1 色で刷るとき。2 枚目は car500 と同じ形の駆動部 5 部品 |
| `print_trike/stl/` | 部品ごとの STL(造形時の向き) |
| `print_trike/parts.json` | 部品表 |

どの部品もサポートは要らない。ほかに 10 円玉 6 枚(後ろのカップに入れるおもり)を用意する。

## 設計と確認

- 設計: `car/trike_design.py`(駆動部などは `car/car_design.py` の部品を使う)
- 干渉と寸法・重心の確認:

  ```bash
  cd car && ../.venv/bin/python verify_car.py --model trike --one-color
  ```

- 印刷データの書き出し:

  ```bash
  .venv/bin/python car/build_car.py --model trike --one-color
  ```

## ファームウェアとコントローラー

[`firmware/README.md`](firmware/README.md) を見る。載せたあと、操作ページの「設定 → 車の種類」で「トライク」を選ぶ(直進 15°、スティックの左右が入れ替わる)。手に持つコントローラーとして M5Stack StopWatch が使える。

## まだ確かめていないこと

- レバーのピンが頭の穴にどのくらいきつく入るか
- 引き棒の上の端をたわませて、J のピンをレバーに差せるか(組立の手順 12)
- 縦に刷るシャフトの根元の強さ、左の駆動輪が圧入だけで抜けないか
- 実際の重心と、後輪にかかる重さ(計算では全体の 13%)
