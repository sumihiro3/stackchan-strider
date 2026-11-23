"""car500 の組立ガイド(スクロールに合わせて部品が付いていく 3D のページ)を書き出す。

使い方: python car/export_guide_car500.py
出力: docs_car500/guide.html(ダブルクリックで開ける単体ファイル。three.js は CDN から読み込む)

walker の組立ガイド(walker/guide_template.html)の見た目と操作をまねた、車用の別のページ。
手順と部品の動き(どの手順で、どこから来て付くか)は MOVES に書く。
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], "--model", "car500"]
import model_select  # noqa: E402

c = model_select.load()
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("guide_car500_template.html")
OUT = ROOT / "docs_car500" / "guide.html"
FRAMES = 60  # 走っている姿のコマ数(車軸 1 回転)
STEER_SWING_DEG = 11.0  # 走っている姿で、うなずきを直進の角度から左右に振る幅

# 手順の番号(ページの section の順。0 = 完成、1 = 部品)
(S_FRAME_LEFT, S_TABLE, S_YOKE, S_FRAME_RIGHT, S_REAR, S_BEAM, S_FRONT, S_FENDER, S_BALLAST, S_SHELL, S_BUMPER, S_STACKCHAN,
 S_LINK, S_DECOR) = range(2, 16)
SHELL_HOVER = 115.0  # おもりを入れる間、殻を浮かせておく高さ [mm]

# 部品 → 動き [(手順, 開始, 長さ, 動く前のずれ, 動いたあとのずれ)]。ずれは組み上がった位置からの量 [mm](機体座標: X 右、Y 前、Z 上)。
# 最初の動きで現れる。開始・長さは、その手順のスクロールの中の割合(0〜1)
_UP = (0.0, 0.0, 0.0)


def _one(step: int, order: int, count: int, origin) -> list:
    return [(step, 0.08 + 0.55 * order / count, 0.26, origin, _UP)]


MOVES: dict[str, list] = {
    # 駆動部(walker の吊り下げ版と同じ順。ヨークが横向きのツメで側板に留まるので、右の側板より先に付ける)
    "pinion": _one(S_FRAME_LEFT, 0, 3, (70, 0, 0)),
    "rail_L": _one(S_FRAME_LEFT, 1, 3, (-75, 0, 0)),
    "deck": _one(S_FRAME_LEFT, 2, 3, (45, 0, 0)),
    "gear_table": _one(S_TABLE, 0, 2, (0, 0, 85)),
    "socket_frame": _one(S_TABLE, 1, 2, (0, 0, 85)),
    "yoke": _one(S_YOKE, 0, 1, (45, 0, 60)),
    "sleeve": _one(S_FRAME_RIGHT, 0, 3, (120, 0, 0)),
    "rail_R": _one(S_FRAME_RIGHT, 1, 3, (75, 0, 0)),
    "shaft": _one(S_FRAME_RIGHT, 2, 3, (150, 0, 0)),
    # 後輪
    "wheel_rear_R": _one(S_REAR, 0, 4, (55, 0, 0)),
    "wheel_rear_L": _one(S_REAR, 1, 4, (-55, 0, 0)),
    "hubcap_rear_R": _one(S_REAR, 2, 4, (35, 0, 0)),
    "hubcap_rear_L": _one(S_REAR, 3, 4, (-35, 0, 0)),
    # 前の横棒とナックル
    "beam": _one(S_BEAM, 0, 4, (0, 0, 60)),
    "knuckle_R": _one(S_BEAM, 1, 4, (0, 0, 55)),
    "knuckle_L": _one(S_BEAM, 1, 4, (0, 0, 55)),
    "clip_R": _one(S_BEAM, 3, 4, (0, 0, -30)),
    "clip_L": _one(S_BEAM, 3, 4, (0, 0, -30)),
    # タイロッドと前輪
    "tie_rod": _one(S_FRONT, 0, 6, (0, 50, 0)),
    "clip_tie_R": _one(S_FRONT, 1, 6, (0, 0, 25)),
    "clip_tie_L": _one(S_FRONT, 1, 6, (0, 0, 25)),
    "wheel_front_R": _one(S_FRONT, 2, 6, (55, 0, 0)),
    "wheel_front_L": _one(S_FRONT, 2, 6, (-55, 0, 0)),
    "clip_wheel_R": _one(S_FRONT, 3, 6, (35, 0, 0)),
    "clip_wheel_L": _one(S_FRONT, 3, 6, (-35, 0, 0)),
    "hubcap_front_R": _one(S_FRONT, 5, 6, (35, 0, 0)),
    "hubcap_front_L": _one(S_FRONT, 5, 6, (-35, 0, 0)),
    "fender_front_R": _one(S_FENDER, 0, 1, (0, 0, 45)),
    "fender_front_L": _one(S_FENDER, 0, 1, (0, 0, 45)),
    # 殻: 浮かせた殻の鼻の筒に、下から 10 円玉とふたを入れる。そのあと、まとめて真上から下ろす
    "body500": [(S_BALLAST, 0.06, 0.22, (0, 0, SHELL_HOVER + 90), (0, 0, SHELL_HOVER)),
                (S_SHELL, 0.12, 0.45, (0, 0, SHELL_HOVER), _UP)],
    "ballast": [(S_BALLAST, 0.34, 0.24, (0, 0, SHELL_HOVER - 55), (0, 0, SHELL_HOVER)),
                (S_SHELL, 0.12, 0.45, (0, 0, SHELL_HOVER), _UP)],
    "ballast_lid": [(S_BALLAST, 0.6, 0.24, (0, 0, SHELL_HOVER - 45), (0, 0, SHELL_HOVER)),
                    (S_SHELL, 0.12, 0.45, (0, 0, SHELL_HOVER), _UP)],
    "bumper_front500": _one(S_BUMPER, 0, 2, (0, 50, 0)),
    "bumper_rear500": _one(S_BUMPER, 1, 2, (0, -50, 0)),
    # スタックチャン(ハットをかぶせたまま)
    **{n: _one(S_STACKCHAN, 0, 1, (0, 0, 150)) for n in ("sc_base", "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes", "hat", "hat_back")},
    "link": _one(S_LINK, 0, 3, (0, 30, 40)),
    "clip_j": _one(S_LINK, 2, 3, (25, 0, 0)),
    "clip_drag": _one(S_LINK, 2, 3, (0, 0, 25)),
    # 飾り
    "headlight_R": _one(S_DECOR, 0, 5, (0, 45, 0)),
    "headlight_L": _one(S_DECOR, 0, 5, (0, 45, 0)),
    "badge": _one(S_DECOR, 1, 5, (0, 40, 0)),
    "windshield500": _one(S_DECOR, 2, 5, (0, 0, 45)),
    "mirror500_R": _one(S_DECOR, 3, 5, (0, 0, 35)),
    "mirror500_L": _one(S_DECOR, 3, 5, (0, 0, 35)),
    "top500": _one(S_DECOR, 4, 5, (0, 0, 45)),
}


# 部品 → 透かして見せる手順(中に入る部品を見せるため)
GHOST = {"body500": [S_BALLAST]}


def explode(name: str) -> list[float]:
    """「部品」の場面で、部品をばらして見せるずれ [mm]。"""
    if name.startswith("sc_") or name in ("hat", "hat_back", "clip_j", "link", "clip_drag"):
        return [0.0, 0.0, 230.0]
    if name in ("body500", "ballast", "ballast_lid", "badge", "headlight_R", "headlight_L", "windshield500", "top500", "mirror500_R", "mirror500_L"):
        return [0.0, 0.0, 120.0]
    if name.startswith(("wheel_", "hubcap_", "clip_wheel_")):
        return [45.0 if name.endswith("_R") else -45.0, 0.0, 0.0]
    if name == "bumper_front500":
        return [0.0, 45.0, 0.0]
    if name == "bumper_rear500":
        return [0.0, -45.0, 0.0]
    return [0.0, 0.0, 0.0]


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
    printed = sum(p.count for p in defs.values() if p.printable)
    return dict(frames=FRAMES, parts=parts, instances=instances, ground=round(-c.GROUND_Z, 2), printed=printed, ghost=GHOST)


def main() -> int:
    data = collect()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload)
    page = ('<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    OUT.write_text(page, encoding="utf-8")
    print(f"guide: {OUT} ({len(page) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
