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
(S_TABLE, S_YOKE, S_AXLE, S_FRAME_LEFT, S_FRAME_RIGHT, S_AUX, S_FOOT, S_STACKCHAN, S_LEVER, S_LINK) = range(2, 12)
AXLE_PARK = (0.0, 0.0, -75.0)  # 車軸は車体の下で組み、まとめて持ち上げる
FOOT_PARK = (0.0, 0.0, -60.0)  # 足の腕は車体の下から持ち上げる
SC_HOVER = 160.0  # スタックチャンを下ろし始める高さ [mm]

# 部品 → 動き [(手順, 開始, 長さ, 動く前のずれ, 動いたあとのずれ)]。ずれは組み上がった位置からの量 [mm](機体座標: X 右、Y 前、Z 上)。
# 最初の動きで現れる。開始・長さは、その手順のスクロールの中の割合(0〜1)
_UP = (0.0, 0.0, 0.0)


def _one(step: int, order: int, count: int, origin) -> list:
    return [(step, 0.08 + 0.55 * order / count, 0.26, origin, _UP)]


def _under(step: int, park, start: float, dx: float, lift_at: float) -> list:
    """車体の下(park)で横から dx [mm] ずれた所から寄せて組み、lift_at でまとめて持ち上げる。"""
    origin = (park[0] + dx, park[1], park[2])
    return [(step, start, 0.16, origin, park), (step, lift_at, 0.24, park, _UP)]


MOVES: dict[str, list] = {
    # 駆動部: デッキに歯車・ソケット枠・ヨークを重ね、下から車軸をかみ合わせてから、左右の側板でまとめて留める。
    # デッキだけを先に側板へ留めると、ヨークの左のツメを側板の角穴へ入れられない(実物、2026-11-24)
    "deck": _one(S_TABLE, 0, 3, (0, 0, 60)),
    "gear_table": _one(S_TABLE, 1, 3, (0, 0, 85)),
    "socket_frame": _one(S_TABLE, 2, 3, (0, 0, 85)),
    "yoke": _one(S_YOKE, 0, 1, (0, 0, 60)),
    # 車軸: シャフトにピニオン、駆動輪、スリーブを通し、デッキの下から持ち上げて歯車にかみ合わせる
    "shaft": _under(S_AXLE, AXLE_PARK, 0.04, 0.0, 0.7),
    "pinion": _under(S_AXLE, AXLE_PARK, 0.16, 90.0, 0.7),
    "wheel_drive_R": _under(S_AXLE, AXLE_PARK, 0.3, 80.0, 0.7),
    "wheel_drive_L": _under(S_AXLE, AXLE_PARK, 0.3, -80.0, 0.7),
    "sleeve_R": _under(S_AXLE, AXLE_PARK, 0.48, 60.0, 0.7),
    "sleeve_L": _under(S_AXLE, AXLE_PARK, 0.48, -60.0, 0.7),
    "rail_L": _one(S_FRAME_LEFT, 0, 1, (-75, 0, 0)),
    "rail_R": _one(S_FRAME_RIGHT, 0, 1, (75, 0, 0)),
    "wheel_aux_R": _one(S_AUX, 0, 2, (35, 0, 0)),
    "wheel_aux_L": _one(S_AUX, 0, 2, (-35, 0, 0)),
    "clip_aux_R": _one(S_AUX, 1, 2, (25, 0, 0)),
    "clip_aux_L": _one(S_AUX, 1, 2, (-25, 0, 0)),
    # 足: 足の腕を車体の下から入れて、軸の棒を左から通す
    "foot": _under(S_FOOT, FOOT_PARK, 0.04, 0.0, 0.32),
    "hinge_rod": [(S_FOOT, 0.6, 0.2, (-120, 0, 0), _UP)],
    "clip_hinge_L": [(S_FOOT, 0.82, 0.14, (-25, 0, 0), _UP)],
    "clip_hinge_R": [(S_FOOT, 0.82, 0.14, (25, 0, 0), _UP)],
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
    if name in ("foot", "hinge_rod", "clip_hinge_R", "clip_hinge_L"):
        return [0.0, 0.0, -70.0]
    if name in ("wheel_drive_R", "sleeve_R", "wheel_aux_R", "clip_aux_R"):
        return [45.0, 0.0, 0.0]
    if name in ("wheel_drive_L", "sleeve_L", "wheel_aux_L", "clip_aux_L"):
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
