"""ピボット(pivot)の組立ガイド(スクロールに合わせて部品が付いていく 3D のページ)を書き出す。

使い方: python car/export_guide_pivot.py
出力: docs_pivot/guide.html(ダブルクリックで開ける単体ファイル。three.js は CDN から読み込む)

トライクのガイド(export_guide_trike.py)の作りをそのまま使う。
手順と部品の動き(どの手順で、どこから来て付くか)は MOVES に書く。
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], "--model", "pivot"]
import model_select  # noqa: E402

c = model_select.load()
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("guide_pivot_template.html")
OUT = ROOT / "docs_pivot" / "guide.html"
FRAMES = 60  # 走っている姿のコマ数(車軸 1 回転)
# 走っている姿のうなずき: 前半は頭が水平(直進)、後半は頭を上げきる(旋回)。上げ下げにかけるコマ数
SPIN_FROM, SPIN_TO, NOD_RAMP = 26, 54, 6

# 手順の番号(ページの section の順。0 = 完成、1 = 部品)
(S_FOOT, S_TABLE, S_YOKE, S_FRAME_LEFT, S_FRAME_RIGHT, S_DRIVE_WHEELS, S_AUX, S_STACKCHAN, S_LEVER, S_LINK) = range(2, 12)
# 足の腕は、右の側板を車体に付ける前に差す(あとからはピニオンに当たって入らない)。
# 手順 1 で側板と足の腕を車体の右に組んでおき、右の側板を付ける手順でまとめて寄せる
PARK = (110.0, 0.0, 0.0)
SC_HOVER = 160.0  # スタックチャンを下ろし始める高さ [mm]

# 部品 → 動き [(手順, 開始, 長さ, 動く前のずれ, 動いたあとのずれ)]。ずれは組み上がった位置からの量 [mm](機体座標: X 右、Y 前、Z 上)。
# 最初の動きで現れる。開始・長さは、その手順のスクロールの中の割合(0〜1)
_UP = (0.0, 0.0, 0.0)


def _one(step: int, order: int, count: int, origin) -> list:
    return [(step, 0.08 + 0.55 * order / count, 0.26, origin, _UP)]


def _parked(start: float, dx: float) -> list:
    """手順 1 で車体の右(PARK)に組み、右の側板を付ける手順で車体へ寄せる。dx は PARK へ来る前の左右のずれ [mm]。"""
    return [(S_FOOT, start, 0.24, (PARK[0] + dx, PARK[1], PARK[2]), PARK), (S_FRAME_RIGHT, 0.42, 0.26, PARK, _UP)]


MOVES: dict[str, list] = {
    "rail_R": _parked(0.06, 50.0),
    "foot": _parked(0.3, -50.0),  # 側板の内側から、軸を穴へ通す
    "clip_pivot": _parked(0.6, 25.0),
    # 駆動部: デッキに歯車・ソケット枠・ヨークを重ねてから、左の側板でまとめて留める。
    # デッキだけを先に側板へ留めると、ヨークの左のツメを側板の角穴へ入れられない(実物、2026-11-24)。
    # ピニオンは歯車とかみ合うので横からは入らない。デッキの下から入れる(計算で、下からなら当たらない)
    "deck": _one(S_TABLE, 0, 3, (0, 0, 60)),
    "gear_table": _one(S_TABLE, 1, 3, (0, 0, 85)),
    "socket_frame": _one(S_TABLE, 2, 3, (0, 0, 85)),
    "yoke": _one(S_YOKE, 0, 1, (0, 0, 60)),
    "pinion": _one(S_FRAME_LEFT, 0, 2, (0, 0, -60)),
    "rail_L": _one(S_FRAME_LEFT, 1, 2, (-75, 0, 0)),
    # スリーブは、置いてある右の側板の穴へ内側から通し、側板と一緒に寄せる
    "sleeve": [(S_FRAME_RIGHT, 0.06, 0.24, (PARK[0] - 45, PARK[1], PARK[2]), PARK), (S_FRAME_RIGHT, 0.42, 0.26, PARK, _UP)],
    # 駆動輪: 右はシャフトと一体で、右から通す。左は六角に押し込む
    "wheel_drive_shaft": _one(S_DRIVE_WHEELS, 0, 2, (150, 0, 0)),
    "wheel_drive": _one(S_DRIVE_WHEELS, 1, 2, (-60, 0, 0)),
    "wheel_aux_R": _one(S_AUX, 0, 2, (35, 0, 0)),
    "wheel_aux_L": _one(S_AUX, 0, 2, (-35, 0, 0)),
    "clip_aux_R": _one(S_AUX, 1, 2, (25, 0, 0)),
    "clip_aux_L": _one(S_AUX, 1, 2, (-25, 0, 0)),
    # スタックチャンを載せてから、レバーを横から付ける(先に付けると、輪がヨークの耳に乗り上げて載らない)
    **{n: [(S_STACKCHAN, 0.12, 0.45, (0, 0, SC_HOVER), _UP)] for n in ("sc_base", "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes")},
    "lever_R": [(S_LEVER, 0.1, 0.28, (55, 0, 0), _UP)],
    "lego_pins_R": [(S_LEVER, 0.48, 0.26, (45, 0, 0), _UP)],
    # 押し板: 2 本のピンへ外から同時に差し、C クリップで留める
    "link": [(S_LINK, 0.3, 0.26, (45, 0, 0), _UP)],
    "clip_pin": [(S_LINK, 0.62, 0.16, (0, -25, 12), _UP)],
    "clip_in": [(S_LINK, 0.74, 0.16, (0, -25, 0), _UP)],
}

GHOST: dict[str, list] = {}  # 部品 → 透かして見せる手順(中に入る部品を見せるため)


def explode(name: str) -> list[float]:
    """「部品」の場面で、部品をばらして見せるずれ [mm]。"""
    if name.startswith("sc_") or name in ("lever_R", "lego_pins_R", "clip_pin"):
        return [0.0, 0.0, 200.0]
    if name in ("link", "clip_in"):
        return [70.0, -40.0, 60.0]
    if name in ("foot", "clip_pivot"):
        return [0.0, -90.0, 0.0]
    if name in ("wheel_drive_shaft", "wheel_aux_R", "clip_aux_R"):
        return [45.0, 0.0, 0.0]
    if name in ("wheel_drive", "wheel_aux_L", "clip_aux_L"):
        return [-45.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0]


def nod(frame: int) -> float:
    """走っている姿のコマ → うなずき [rad]。前半は水平、後半は上げきる。"""
    lo, hi = (math.radians(v) for v in c.NOD_RANGE)

    def ramp(v: float) -> float:
        t = min(1.0, max(0.0, v))
        return t * t * (3 - 2 * t)

    return lo + (hi - lo) * (ramp((frame - SPIN_FROM) / NOD_RAMP) - ramp((frame - SPIN_TO) / NOD_RAMP))


def _b64(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def collect() -> dict:
    """部品の形、置き方(走っている姿のコマ)、組立の動きを、ページに埋め込む形にまとめる。"""
    defs = c.part_defs()
    frames = [c.instances(2 * math.pi * k / FRAMES, nod(k)) for k in range(FRAMES)]
    rest = c.instances(0.0, c.NEUTRAL)
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
    printed = sum(p.count for p in defs.values() if p.printable)
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
    print(f"guide: {OUT} ({len(page) / 1e6:.2f} MB、印刷部品 {data['printed']} 個)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
