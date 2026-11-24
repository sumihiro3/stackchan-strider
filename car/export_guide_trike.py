"""トライクの組立ガイド(スクロールに合わせて部品が付いていく 3D のページ)を書き出す。

使い方: python car/export_guide_trike.py
出力: docs_trike/guide.html(ダブルクリックで開ける単体ファイル。three.js は CDN から読み込む)

car500 のガイド(export_guide_car500.py、underslung ブランチ)の作りをそのまま使う。
手順と部品の動き(どの手順で、どこから来て付くか)は MOVES に書く。10 円玉は印刷しないので、ここで円柱を足して見せる。
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
from manifold3d import Manifold

sys.argv = [sys.argv[0], "--model", "trike"]
import model_select  # noqa: E402

c = model_select.load()
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("guide_trike_template.html")
OUT = ROOT / "docs_trike" / "guide.html"
FRAMES = 60  # 走っている姿のコマ数(車軸 1 回転)
STEER_SWING_DEG = 11.0  # 走っている姿で、うなずきを直進の角度から前後に振る幅

# 手順の番号(ページの section の順。0 = 完成、1 = 部品)
(S_TABLE, S_YOKE, S_FRAME_LEFT, S_FRAME_RIGHT, S_DRIVE_WHEELS, S_AUX, S_BRIDGE, S_FORK, S_COINS, S_LEVER, S_STACKCHAN,
 S_LINK) = range(2, 14)
SC_HOVER = 150.0  # レバーを差す間、スタックチャンを浮かせておく高さ [mm]
FORK_HOVER = -55.0  # 後輪を付ける間、フォークを橋の下に置いておく高さ [mm]

# 部品 → 動き [(手順, 開始, 長さ, 動く前のずれ, 動いたあとのずれ)]。ずれは組み上がった位置からの量 [mm](機体座標: X 右、Y 前、Z 上)。
# 最初の動きで現れる。開始・長さは、その手順のスクロールの中の割合(0〜1)
_UP = (0.0, 0.0, 0.0)


def _one(step: int, order: int, count: int, origin) -> list:
    return [(step, 0.08 + 0.55 * order / count, 0.26, origin, _UP)]


def _hover(step: int, start: float, origin, at) -> tuple:
    return (step, start, 0.24, origin, at)


_SC = (0.0, 0.0, SC_HOVER)
_FORK = (0.0, 0.0, FORK_HOVER)
MOVES: dict[str, list] = {
    # 駆動部: デッキに歯車・ソケット枠・ヨークを重ねてから、左の側板でまとめて留める。
    # デッキだけを先に側板へ留めると、ヨークの左のツメを側板の角穴へ入れられない(実物、2026-11-24)。
    # ピニオンは歯車とかみ合うので横からは入らない。デッキの下から入れる(計算で、下からなら当たらない)
    "deck": _one(S_TABLE, 0, 3, (0, 0, 60)),
    "gear_table": _one(S_TABLE, 1, 3, (0, 0, 85)),
    "socket_frame": _one(S_TABLE, 2, 3, (0, 0, 85)),
    "yoke": _one(S_YOKE, 0, 1, (0, 0, 60)),
    "pinion": _one(S_FRAME_LEFT, 0, 2, (0, 0, -60)),
    "rail_L": _one(S_FRAME_LEFT, 1, 2, (-75, 0, 0)),
    "sleeve": _one(S_FRAME_RIGHT, 0, 2, (120, 0, 0)),
    "rail_R": _one(S_FRAME_RIGHT, 1, 2, (75, 0, 0)),
    # 駆動輪: 右はシャフトと一体で、右から通す。左は六角に押し込む
    "wheel_drive_shaft": _one(S_DRIVE_WHEELS, 0, 2, (150, 0, 0)),
    "wheel_drive": _one(S_DRIVE_WHEELS, 1, 3, (-60, 0, 0)),
    "clip_drive": _one(S_DRIVE_WHEELS, 2, 3, (-25, 0, 0)),
    "wheel_aux_R": _one(S_AUX, 0, 2, (35, 0, 0)),
    "wheel_aux_L": _one(S_AUX, 0, 2, (-35, 0, 0)),
    "clip_aux_R": _one(S_AUX, 1, 2, (25, 0, 0)),
    "clip_aux_L": _one(S_AUX, 1, 2, (-25, 0, 0)),
    # 後ろ: 橋を腕の下から当て、後輪を付けたフォークを下から通す
    "bridge": _one(S_BRIDGE, 0, 1, (0, 0, -45)),
    "fork": [_hover(S_FORK, 0.06, (0, 0, FORK_HOVER - 40), _FORK), (S_FORK, 0.6, 0.2, _FORK, _UP)],
    "wheel_trike": [_hover(S_FORK, 0.2, (40, 0, FORK_HOVER), _FORK), (S_FORK, 0.6, 0.2, _FORK, _UP)],
    "clip_trike": [_hover(S_FORK, 0.36, (25, 0, FORK_HOVER), _FORK), (S_FORK, 0.6, 0.2, _FORK, _UP)],
    "clip_king": [(S_FORK, 0.82, 0.14, (0, 0, 25), _UP)],
    "coins": _one(S_COINS, 0, 2, (0, 0, 70)),
    "cup_lid": _one(S_COINS, 1, 2, (0, 0, 40)),
    # スタックチャン: 浮かせたままレバーを差し、まとめて下ろす
    **{n: [_hover(S_LEVER, 0.04, (0, 0, SC_HOVER + 90), _SC), (S_STACKCHAN, 0.12, 0.45, _SC, _UP)]
       for n in ("sc_base", "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes")},
    "lever": [_hover(S_LEVER, 0.2, (40, 0, SC_HOVER), _SC), (S_STACKCHAN, 0.12, 0.45, _SC, _UP)],
    "lever_peg": [_hover(S_LEVER, 0.4, (40, 0, SC_HOVER), _SC), (S_STACKCHAN, 0.12, 0.45, _SC, _UP)],
    "lego_pin": [_hover(S_LEVER, 0.6, (40, 0, SC_HOVER), _SC), (S_STACKCHAN, 0.12, 0.45, _SC, _UP)],
    # 引き棒: 横板をフォークの腕のピンへ下ろし、上の端をたわませて J のピンをレバーへ差す
    "link": [(S_LINK, 0.06, 0.22, (14, 0, 40), (6, 0, 0)), (S_LINK, 0.32, 0.16, (6, 0, 0), _UP)],
    "clip_arm": [(S_LINK, 0.52, 0.18, (0, 0, 25), _UP)],
    "clip_j": [(S_LINK, 0.7, 0.18, (0, 0, 25), _UP)],
}

GHOST: dict[str, list] = {}  # 部品 → 透かして見せる手順(中に入る部品を見せるため)

COIN_PART = "coins"


def explode(name: str) -> list[float]:
    """「部品」の場面で、部品をばらして見せるずれ [mm]。"""
    if name.startswith("sc_") or name in ("lever", "lever_peg", "lego_pin", "link", "clip_j"):
        return [0.0, 0.0, 230.0]
    if name in ("wheel_drive_shaft", "wheel_aux_R", "clip_aux_R"):
        return [45.0, 0.0, 0.0]
    if name in ("wheel_drive", "clip_drive", "wheel_aux_L", "clip_aux_L"):
        return [-45.0, 0.0, 0.0]
    if name in ("bridge", "cup_lid", "fork", "wheel_trike", "clip_trike", "clip_king", "clip_arm", COIN_PART):
        return [0.0, -50.0, 0.0]
    return [0.0, 0.0, 0.0]


def _coin_stack() -> Manifold:
    """カップに入れる 10 円玉の重なり(設計の座標)。"""
    _, _, (x, y, z) = c.EXTRA_MASSES[0]
    h = c.COINS * c.COIN_T
    return Manifold.cylinder(h, c.COIN_R, c.COIN_R, 64).translate((x, y, z - h / 2))


def _b64(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def collect() -> dict:
    """部品の形、置き方(走っている姿のコマ)、組立の動きを、ページに埋め込む形にまとめる。"""
    defs = c.part_defs()
    neutral = c.steering_geometry()[0]
    frames = [c.instances(2 * math.pi * k / FRAMES, neutral + math.radians(STEER_SWING_DEG) * math.sin(2 * math.pi * k / FRAMES))
              for k in range(FRAMES)]
    rest = c.instances(0.0, neutral)
    missing = [i.name for i in rest if i.name not in MOVES]
    if missing:
        raise SystemExit(f"組立の動きが決まっていない部品: {missing}")
    used = sorted({i.part for i in rest})
    parts = {}
    for key in used:
        mesh = c.d.to_trimesh(c.part_shape(key))
        part = defs[key]
        parts[key] = dict(label=part.label, color=part.color, printable=part.printable, v=_b64(mesh.vertices, np.float32), i=_b64(mesh.faces, np.uint32))
    instances = []
    for index, inst in enumerate(rest):
        entry = dict(name=inst.name, part=inst.part, m=[round(float(v), 5) for v in inst.matrix[:3, :].reshape(-1)], ex=explode(inst.name),
                     moves=[dict(step=s, start=a, span=b, origin=list(o), to=list(t)) for s, a, b, o, t in MOVES[inst.name]])
        stack = np.array([frame[index].matrix[:3, :].reshape(-1) for frame in frames])
        assert all(frame[index].name == inst.name for frame in frames)
        if np.abs(stack - stack[0]).max() > 1e-9:
            entry["f"] = _b64(stack, np.float32)
        instances.append(entry)
    coin = c.d.to_trimesh(_coin_stack())
    parts[COIN_PART] = dict(label="10 円玉", color="#b87333", printable=False, v=_b64(coin.vertices, np.float32), i=_b64(coin.faces, np.uint32))
    instances.append(dict(name=COIN_PART, part=COIN_PART, m=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], ex=explode(COIN_PART),
                          moves=[dict(step=s, start=a, span=b, origin=list(o), to=list(t)) for s, a, b, o, t in MOVES[COIN_PART]]))
    unused = ("lever",) if c.LEVER_LEGO else ("lever_lego", "lever_peg")  # レバーは 2 種類のうち片方だけ刷る
    printed = sum(p.count for k, p in defs.items() if p.printable and k not in unused)
    return dict(frames=FRAMES, parts=parts, instances=instances, ground=round(-c.GROUND_Z, 2), printed=printed, ghost=GHOST)


def main() -> int:
    data = collect()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload)
    page = ('<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"guide: {OUT} ({len(page) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
