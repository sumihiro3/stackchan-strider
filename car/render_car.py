"""組み上がった車を MuJoCo のオフスクリーン描画で画像にする(見た目の確認用)。

使い方: python car/render_car.py [--model car|car500] [--pitch 13] [--drive 0](前輪の切れ角はうなずきから決まる。--pitch を省くと直進)
出力: out/car/views/{front34,side,top,front,rear34,ear}.png
"""
from __future__ import annotations

import argparse
import math
import sys

import mujoco
import numpy as np
from PIL import Image

import model_select

c = model_select.load()

MM = 0.001
VIEWS = getattr(c, "VIEWS", None) or {  # 名前: (カメラ位置[mm]、注視点[mm])。地面は z = -30
    "front34": ((260, 330, 150), (0, 40, 10)),
    "rear34": ((-260, -300, 170), (0, 30, 10)),
    "side": ((520, 40, 20), (0, 40, 20)),
    "top": ((0, 41, 560), (0, 40, 0)),
    "front": ((0, 560, 30), (0, 40, 30)),
    "ear": ((190, 90, 110), (25, 5, 62)),  # 右の耳当てとヨークの寄り
}


def _rgba(color: str) -> str:
    return " ".join(f"{int(color[i:i + 2], 16) / 255:.3f}" for i in (1, 3, 5)) + " 1"


def _axes(pos, target) -> str:
    look = np.asarray(target, float) - np.asarray(pos, float)
    look /= np.linalg.norm(look)
    up_hint = np.array([0, 1.0, 0]) if abs(look[2]) > 0.99 else np.array([0, 0, 1.0])
    right = np.cross(look, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    return " ".join(f"{v:.5f}" for v in list(right) + list(up))


def build_xml(drive: float, pitch: float | None) -> str:
    mesh_dir = c.OUT / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    defs = c.part_defs()
    assets, geoms = [], []
    for inst in c.instances(drive, pitch):
        mesh = d_to_trimesh(c.placed(inst))
        path = mesh_dir / f"{inst.name}.stl"
        mesh.export(path)
        assets.append(f'<mesh name="{inst.name}" file="{path}" scale="{MM} {MM} {MM}"/>')
        geoms.append(f'<geom type="mesh" mesh="{inst.name}" rgba="{_rgba(defs[inst.part].color)}" contype="0" conaffinity="0"/>')
    cams = "\n".join(f'<camera name="{n}" pos="{" ".join(str(v * MM) for v in p)}" xyaxes="{_axes(p, t)}" fovy="32"/>' for n, (p, t) in VIEWS.items())
    ground = c.GROUND_Z * MM
    return f'''<mujoco>
  <visual><global offwidth="1200" offheight="900"/><quality shadowsize="4096"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/></visual>
  <asset>{"".join(assets)}
    <texture name="floor" type="2d" builtin="checker" rgb1="0.93 0.93 0.94" rgb2="0.88 0.89 0.90" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="30 30"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.88 0.90 0.92" width="64" height="64"/></asset>
  <worldbody>
    <light pos="0.5 0.3 0.6" dir="-0.5 -0.3 -0.6" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom type="plane" size="2 2 0.01" pos="0 0 {ground}" material="floor"/>
    {cams}
    {"".join(geoms)}
  </worldbody>
</mujoco>'''


def d_to_trimesh(shape):
    return c.d.to_trimesh(shape)


def render(drive: float = 0.0, pitch: float | None = None, suffix: str = "") -> list[str]:
    model = mujoco.MjModel.from_xml_string(build_xml(drive, pitch))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out_dir = c.OUT / "views"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with mujoco.Renderer(model, 900, 1200) as renderer:
        for name in VIEWS:
            renderer.update_scene(data, camera=name)
            path = out_dir / f"{name}{suffix}.png"
            Image.fromarray(renderer.render()).save(path)
            paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(model_select.MODELS), default="car")
    parser.add_argument("--pitch", type=float, default=None, help="うなずき [deg](省くと直進の角度)")
    parser.add_argument("--drive", type=float, default=0.0, help="車軸の回転角 [deg]")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    pitch = None if args.pitch is None else math.radians(args.pitch)
    for p in render(math.radians(args.drive), pitch, args.suffix):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
