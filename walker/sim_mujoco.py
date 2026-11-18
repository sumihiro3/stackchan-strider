"""MuJoCo で Stackchan Strider を物理シミュレーションし、実際に歩けるかを確かめる。

モデル化:
  - 車体(側板・デッキ・ヨーク + スタックチャンの頭部)を浮動ベースにする。頭はイヤーパッドで車体に固定とみなす
  - クランク軸(ピニオン・クランク込み)は X 軸ヒンジ、ターンテーブル(歯車・枠・土台)は Z 軸ヒンジ
  - 傘歯車 1:1 は関節どうしの等式拘束(ターンテーブル角 = −クランク角)。バックラッシュは入れない
  - 4 脚はそれぞれ 6 リンク。木構造 + 3 か所のピン結合(connect 拘束)で閉リンクにする
  - ヨーサーボは、頭(車体)と土台(ターンテーブル)の間に働くトルク。SCS0009 のトルク-速度特性を直線で近似
  - 床と接触するのは足先(円柱)と、転倒時のための車体・頭の箱
使い方: python walker/sim_mujoco.py [--video] [--seconds 12]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

import design as d
import gait

OUT = d.ROOT / "out" / "mujoco"
MM = 0.001
THETA0 = 0.0
PRINT_DENSITY = gait.PLA_DENSITY * gait.FILL_FACTOR * 1e6  # g/mm^3 → kg/m^3
SC_BASE_MASS = 0.022  # 土台側(土台・底蓋・基板)の質量の仮定 [kg]
SC_HEAD_MASS = d.SC_MASS_G / 1000 - SC_BASE_MASS
# Feetech SCS0009 @4.8V: ストール 1.89 kg·cm、無負荷 80 rpm
SERVO_STALL = 1.89 * 9.81e-2
SERVO_NOLOAD = 80 * 2 * math.pi / 60
# 関節の摩擦トルクの仮定 [N·m]
FRICTION_PIN = 5e-4  # 印刷ピンの回転ジョイント
# クランク軸(スリーブ 2 か所)。吊り下げ版は短軸 2 か所・平歯車の噛み合い 2 か所・つば 2 か所が加わるので 3 倍に見積もる
FRICTION_SHAFT = 3e-3 if d.UNDERSLUNG else 1e-3
FRICTION_TABLE = 7e-3  # ターンテーブルのスラスト面(摩擦係数 0.3 × 荷重 2.5 N × 半径 9 mm)
# 感度確認用の上書き(環境変数): 関節摩擦の倍率、頭部重心の前後・上下ずれ [mm]
FRICTION_SCALE = float(os.environ.get("STRIDER_JOINT_FRICTION_SCALE", "1"))
HEAD_CG_SHIFT_Y = float(os.environ.get("STRIDER_HEAD_CG_DY", "0"))
HEAD_CG_SHIFT_Z = float(os.environ.get("STRIDER_HEAD_CG_DZ", "0"))
FRICTION_PIN *= FRICTION_SCALE
FRICTION_SHAFT *= FRICTION_SCALE
FRICTION_TABLE *= FRICTION_SCALE


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _rgba(color: str, alpha: float = 1.0) -> str:
    return _fmt([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [alpha])


CAMERA_POS = (0.26, 0.36, 0.10)  # 車体から見たカメラ位置 [m](右前方の少し上)
CAMERA_TARGET = (0.0, 0.0, 0.02)


def _camera_axes(position, target) -> str:
    """カメラの右方向と上方向(MuJoCo の xyaxes)。"""
    look = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    look /= np.linalg.norm(look)
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    return _fmt(list(right) + list(up))


# 吊り下げ版: 駆動軸まわり(ピニオン・シャフト・スリーブ・平歯車)と歯車カバーは、質量だけを車体に載せる(回転はクランク軸の関節で代表させる)
LOW_SHAFT_NAMES = ["pinion", "shaft", "sleeve", "spur_R", "spur_L", "cover_R", "cover_L"] if d.UNDERSLUNG else []


def build_xml(friction: float) -> str:
    mesh_dir = OUT / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    defs = d.part_defs()
    instances = {inst.name: inst for inst in d.instances(THETA0)}
    assets, geoms = [], {}
    for name, inst in instances.items():
        mesh = d.to_trimesh(d.placed(inst))
        mesh.export(mesh_dir / f"{name}.stl")
        assets.append(f'<mesh name="{name}" file="meshes/{name}.stl" scale="{MM} {MM} {MM}" inertia="exact"/>')
        printable = defs[inst.part].printable
        density = PRINT_DENSITY if printable else 0
        geoms[name] = (
            f'<geom type="mesh" mesh="{name}" rgba="{_rgba(defs[inst.part].color)}" density="{density:.1f}" '
            f'contype="0" conaffinity="0" group="1"/>'
        )

    def geom_lines(names) -> str:
        return "\n".join(geoms[name] for name in names)

    legs_xml = {"chassis": [], "shaft": []}
    equalities = []
    foot_geoms = {}
    for leg, sx, sy in d.LEGS:
        joints = d.solve_leg(d.LEG, d.leg_canonical_angle(THETA0, sx, sy))
        first = d.leg_first_layer(sx, sy)

        def point(joint: str, layer: float) -> str:
            x = sx * (d.L1 + (first + layer) * d.LAYER_PITCH + d.LAYER_T / 2)
            return _fmt([x * MM, sy * joints[joint][0] * MM, (joints[joint][1] + d.LEG_Z) * MM])

        def hinge(name: str, joint: str, layer: float) -> str:
            return f'<joint name="{leg}_{name}" type="hinge" axis="1 0 0" pos="{point(joint, layer)}" frictionloss="{FRICTION_PIN}" damping="1e-5"/>'

        foot_x0, foot_x1 = d.foot_x_range(sx, sy)
        hy, hz = sy * joints["H"][0], joints["H"][1] + d.LEG_Z
        foot = (
            f'<geom name="{leg}_foot" type="cylinder" fromto="{_fmt([foot_x0 * MM, hy * MM, hz * MM, foot_x1 * MM, hy * MM, hz * MM])}" '
            f'size="{d.FOOT_R * MM}" friction="{friction} 0.005 0.0001" mass="0" rgba="0 0 0 0" group="3"/>'
        )
        legs_xml["chassis"].append(
            f'''<body name="{leg}_tri_upper">
  {hinge("tri_upper", "B", 1)}
  {geoms[leg + "_tri_upper"]}
  <body name="{leg}_f">
    {hinge("f", "E", 0.5)}
    {geoms[leg + "_f"]}
  </body>
</body>
<body name="{leg}_c">
  {hinge("c", "B", 0)}
  {geoms[leg + "_c"]}
  <body name="{leg}_tri_lower">
    {hinge("tri_lower", "F", 1)}
    {geoms[leg + "_tri_lower"]}
    {foot}
  </body>
</body>'''
        )
        legs_xml["shaft"].append(
            f'''<body name="{leg}_j">
  {hinge("j", "C", 0)}
  {geoms[leg + "_j"]}
</body>
<body name="{leg}_k">
  {hinge("k", "C", 2)}
  {geoms[leg + "_k"]}
</body>'''
        )
        for body1, body2, joint, layer in ((f"{leg}_j", f"{leg}_tri_upper", "D", 0.5), (f"{leg}_k", f"{leg}_c", "F", 1), (f"{leg}_f", f"{leg}_tri_lower", "G", 0.5)):
            equalities.append(f'<connect body1="{body1}" body2="{body2}" anchor="{point(joint, layer)}" solref="0.002 1"/>')

    head_center = [0, ((d.SC_FRONT - 29.2) / 2 + HEAD_CG_SHIFT_Y) * MM, (d.TABLE_Z1 + (d.SC_HEAD_BOTTOM + d.SC_HEAD_TOP) / 2 + HEAD_CG_SHIFT_Z) * MM]
    start_z = -d.GROUND_Z * MM + 0.002
    chassis_names = ["side_plate_R", "side_plate_L", "deck", "yoke", "cowl_F", "cowl_R", *LOW_SHAFT_NAMES, "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes", "sc_camera", "sc_leds"]
    shaft_names = ["crank_R", "crank_L", "clip_R", "clip_L"] if d.UNDERSLUNG else ["pinion", "shaft", "sleeve", "crank_R", "crank_L", "clip_R", "clip_L"]
    table_names = ["gear_table", "socket_frame", "sc_base"]
    return f'''<mujoco model="stackchan_strider">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <size memory="200M"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <rgba haze="0.93 0.94 0.95 1"/>
  </visual>
  <asset>
    {chr(10).join(assets)}
    <texture name="floor" type="2d" builtin="checker" rgb1="0.90 0.91 0.92" rgb2="0.84 0.86 0.88" width="256" height="256" mark="edge" markrgb="0.72 0.75 0.78"/>
    <material name="floor" texture="floor" texrepeat="80 80" reflectance="0"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.97 0.98" rgb2="0.86 0.88 0.90" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="floor" friction="{friction} 0.005 0.0001"/>
    <body name="chassis" pos="0 0 {start_z:.5f}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="{_fmt(CAMERA_POS)}" xyaxes="{_camera_axes(CAMERA_POS, CAMERA_TARGET)}"/>
      <camera name="side" mode="trackcom" pos="0.42 0 0.02" xyaxes="0 1 0 0 0 1"/>
      {geom_lines(chassis_names)}
      <geom name="head_mass" type="box" pos="{_fmt(head_center)}" size="0.025 0.025 0.025" mass="{SC_HEAD_MASS:.4f}" rgba="0 0 0 0" group="3" friction="{friction} 0.005 0.0001"/>
      <geom name="belly" type="box" pos="0 0 -0.004" size="0.027 0.042 0.010" mass="0" rgba="0 0 0 0" group="3" friction="{friction} 0.005 0.0001"/>
      {chr(10).join(legs_xml["chassis"])}
      <body name="shaft">
        <joint name="shaft" type="hinge" axis="1 0 0" pos="0 0 {d.LEG_Z * MM:.5f}" frictionloss="{FRICTION_SHAFT}" damping="1e-5"/>
        {geom_lines(shaft_names)}
        {chr(10).join(legs_xml["shaft"])}
      </body>
      <body name="table">
        <joint name="table" type="hinge" axis="0 0 1" pos="0 0 0" frictionloss="{FRICTION_TABLE}" damping="1e-5"/>
        {geom_lines(table_names)}
        <geom name="base_mass" type="box" pos="0 0 {(d.TABLE_Z1 + 4) * MM:.5f}" size="0.022 0.026 0.004" mass="{SC_BASE_MASS}" rgba="0 0 0 0" contype="0" conaffinity="0" group="3"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint name="bevel" joint1="table" joint2="shaft" polycoef="0 {1 if d.UNDERSLUNG else -1} 0 0 0" solref="0.002 1"/>
    {chr(10).join(equalities)}
  </equality>
  <actuator>
    <general name="yaw_servo" joint="table" ctrlrange="-1 1" gaintype="fixed" gainprm="{SERVO_STALL}" biastype="affine" biasprm="0 0 {-SERVO_STALL / SERVO_NOLOAD:.6f}" forcerange="{-SERVO_STALL} {SERVO_STALL}"/>
  </actuator>
</mujoco>
'''


def load(friction: float):
    OUT.mkdir(parents=True, exist_ok=True)
    xml = build_xml(friction)
    path = OUT / "stackchan_strider.xml"
    path.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def _tilt(quat: np.ndarray) -> tuple[float, float]:
    """車体の前後(正 = 前のめり)と左右(正 = 右へ傾く)の傾き [deg]。"""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    up_in_body = rot.reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.asin(np.clip(up_in_body[1], -1, 1))), math.degrees(math.asin(np.clip(up_in_body[0], -1, 1)))


def run(friction: float, command: float, seconds: float, settle: float = 1.0, video: str | None = None) -> dict:
    model, data = load(friction)
    shaft_adr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft")]
    shaft_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft")]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot") for leg, _, _ in d.LEGS]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    belly_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("belly", "head_mass")}
    renderer, frames = None, []
    if video:
        renderer = mujoco.Renderer(model, 720, 960)
    steps = int(round(seconds / model.opt.timestep))
    record_every = int(round(0.005 / model.opt.timestep))
    frame_every = int(round((1 / 30) / model.opt.timestep))
    log = []
    fallen = False
    for step in range(steps):
        time = step * model.opt.timestep
        data.ctrl[0] = command * min(1.0, max(0.0, (time - settle) / 0.5)) if time >= settle else 0.0
        mujoco.mj_step(model, data)
        if step % record_every == 0:
            pitch, roll = _tilt(data.qpos[3:7])
            on_floor = set()
            body_hit = False
            for contact in data.contact[: data.ncon]:
                pair = {contact.geom1, contact.geom2}
                if floor_id in pair:
                    other = (pair - {floor_id}).pop()
                    if other in foot_ids:
                        on_floor.add(foot_ids.index(other))
                    if other in belly_ids:
                        body_hit = True
            fallen = fallen or body_hit or abs(pitch) > 45 or abs(roll) > 45
            log.append((time, data.qpos[0], data.qpos[1], data.qpos[2], pitch, roll, data.qpos[shaft_qadr], data.qvel[shaft_adr], data.actuator_force[0], len(on_floor)))
        if renderer is not None and step % frame_every == 0:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())
    log = np.array(log)
    if renderer is not None:
        import imageio

        imageio.mimsave(video, frames, fps=30, quality=8, macro_block_size=None)
        renderer.close()
    walking = log[log[:, 0] >= settle + 1.0]
    duration = walking[-1, 0] - walking[0, 0]
    revolutions = abs(walking[-1, 6] - walking[0, 6]) / (2 * math.pi)
    forward = walking[-1, 2] - walking[0, 2]
    sideways = walking[-1, 1] - walking[0, 1]
    return dict(
        friction=friction,
        command=command,
        seconds=seconds,
        fallen=bool(fallen),
        crank_rpm=float(revolutions / duration * 60),
        forward_mm=float(forward * 1000),
        sideways_mm=float(sideways * 1000),
        speed_mm_s=float(forward * 1000 / duration),
        advance_per_rev_mm=float(forward * 1000 / revolutions) if revolutions > 0.2 else 0.0,
        pitch_deg=[float(walking[:, 4].min()), float(walking[:, 4].max())],
        roll_deg=[float(walking[:, 5].min()), float(walking[:, 5].max())],
        servo_torque_mNm_mean=float(np.abs(walking[:, 8]).mean() * 1000),
        servo_torque_mNm_peak=float(np.abs(walking[:, 8]).max() * 1000),
        feet_on_floor_mean=float(walking[:, 9].mean()),
        feet_on_floor_min=int(walking[:, 9].min()),
        total_mass_g=float(mujoco.mj_getTotalmass(model) * 1000),
        log=log,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    results = []
    print("摩擦  指令   転倒  クランクrpm  前進mm  横ずれmm  速度mm/s  mm/回転  前後の傾き       左右の傾き      サーボ平均/最大 mN·m  接地足(平均/最少)")
    for friction in (0.3, 0.5, 0.8):
        for command in (1.0, -1.0, 0.5):
            if command == 0.5 and friction != 0.5:
                continue
            result = run(friction, command, args.seconds)
            result.pop("log")
            results.append(result)
            print(
                f"{friction:4.1f} {command:+5.1f}  {'あり' if result['fallen'] else 'なし'}  {result['crank_rpm']:9.1f}  {result['forward_mm']:7.1f}  {result['sideways_mm']:7.1f}  "
                f"{result['speed_mm_s']:7.1f}  {result['advance_per_rev_mm']:7.1f}  {result['pitch_deg'][0]:+5.1f}〜{result['pitch_deg'][1]:+5.1f}°  "
                f"{result['roll_deg'][0]:+5.1f}〜{result['roll_deg'][1]:+5.1f}°  {result['servo_torque_mNm_mean']:6.1f} / {result['servo_torque_mNm_peak']:6.1f}      "
                f"{result['feet_on_floor_mean']:.2f} / {result['feet_on_floor_min']}"
            )
    with open(OUT / "results.json", "w", encoding="utf-8") as handle:
        json.dump(dict(results=results, assumptions=dict(
            servo_stall_Nm=SERVO_STALL, servo_noload_rad_s=SERVO_NOLOAD, friction_pin_Nm=FRICTION_PIN, friction_shaft_Nm=FRICTION_SHAFT,
            friction_table_Nm=FRICTION_TABLE, print_density_kg_m3=PRINT_DENSITY, sc_head_mass_kg=SC_HEAD_MASS, sc_base_mass_kg=SC_BASE_MASS)),
            handle, ensure_ascii=False, indent=1)
    if args.video:
        forward_sign = max((r for r in results if r["friction"] == 0.5 and abs(r["command"]) == 1.0), key=lambda r: r["forward_mm"])["command"]
        run(0.5, forward_sign, args.seconds, video=str(OUT / "walk.mp4"))
        print("動画:", OUT / "walk.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
