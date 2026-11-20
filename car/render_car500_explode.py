"""car500 の分解図と、鼻のおもりの断面図を描く(組立説明用)。

使い方: python car/render_car500_explode.py
出力: docs_car500/{explode,explode_below,ballast_section}.png

render_car.py の描き方をそのまま使い、部品の置き方(c.instances / c.placed)だけ差し替える。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--model", "car500"]  # render_car が import 時に --model を読む
import render_car as r  # noqa: E402
from manifold3d import Manifold  # noqa: E402

c = r.c
DOCS = Path(__file__).resolve().parent.parent / "docs_car500"
# 分解図で持ち上げる高さ [mm]。組む順の逆に、下から ふた → 10 円玉 → 殻と飾り と並べる
SHELL_PARTS = {"body500", "badge", "headlight", "windshield500", "top500", "mirror500_R", "mirror500_L"}
LIFT = {**{k: 110.0 for k in SHELL_PARTS}, "ballast": 55.0, "ballast_lid": 25.0}
# 断面図に出す部品(鼻のまわりだけ)
SECTION_PARTS = {"body500", "ballast", "ballast_lid", "beam", "tie_rod", "knuckle_R", "knuckle_L", "wheel_front", "tire",
                 "headlight", "badge", "bumper_front500", "rail_R", "rail_L"}
EXPLODE_VIEWS = {"explode": ((300, 360, 190), (0, 40, 50)), "explode_below": ((230, 330, -10), (0, 60, 60))}
SECTION_VIEWS = {"ballast_section": ((120, 70, 25), (0, 88, 14))}

_instances, _placed = c.instances, c.placed
OUT = c.OUT


def _exploded_instances(drive=0.0, pitch=None, with_stackchan=True):
    """スタックチャンとハットを除いた部品(殻の中が見えるように)。"""
    return [i for i in _instances(drive, pitch, False) if i.part not in {"hat", "hat_back"}]


def _exploded(instance) -> Manifold:
    return _placed(instance).translate((0, 0, LIFT.get(instance.part, 0.0)))


def _cut(instance) -> Manifold:
    """x < 0 の半分だけを残す(車の中心で縦に切り、切り口を右から見る)。"""
    return _placed(instance) ^ Manifold.cube((200, 400, 400)).translate((-200, -200, -200))


def _section_instances(drive=0.0, pitch=None, with_stackchan=True):
    return [i for i in _instances(drive, pitch, False) if i.part in SECTION_PARTS and not _cut(i).is_empty()]


def _render(name: str, instances, placed, views) -> list[Path]:
    # MuJoCo は同じパスのメッシュを使い回すので、描き方ごとに出力先を分ける
    c.instances, c.placed, c.OUT, r.VIEWS = instances, placed, OUT / name, views
    out = []
    for path in r.render(0.0, None):
        dst = DOCS / Path(path).name
        shutil.copy(path, dst)
        out.append(dst)
    return out


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    paths = _render("explode", _exploded_instances, _exploded, EXPLODE_VIEWS)
    paths += _render("section", _section_instances, _cut, SECTION_VIEWS)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
