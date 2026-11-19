"""構想比較: スタックチャンのサーボ 2 個で走る「自動車型」を、単純な形状の MuJoCo モデルで試す。

Strider の脚機構は使わない。部品の形状は箱と円柱だけで、寸法は仮の値。
  - 首振りサーボ(連続回転)→ 傘歯車 1:1 → 後輪の車軸。トルク-速度特性は sim_mujoco と同じ SCS0009 の直線近似
  - うなずきサーボ(角度指令)→ 押し棒 → 前輪のかじ取り。うなずき 0〜60° を、かじ取り ±30° に 1:1 で割り当てる仮定
  - 前輪は左右を同じ角度に切る(平行かじ取り。アッカーマンではない)
  - 車輪は床とだけ接触させる(車台との干渉は形状の設計で避ける前提)
  - 後輪は「左右を 1 本の軸で回す」と「右だけ駆動、左は空転」の 2 通り
使い方: python walker/sim_concept_car.py [--json out/mujoco/concept_car.json] [--video]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import mujoco
import numpy as np

import design as d
import sim_mujoco as sm

OUT = sm.OUT
WHEEL_R = 20.0  # 車輪の半径 [mm]
WHEEL_W = 8.0
WHEELBASE = 100.0  # 前後の車軸の間隔 [mm]
TRACK = 100.0  # 左右の車輪の間隔 [mm]
DECK_Z = 30.0  # 車台の上面(スタックチャンの土台を載せる面)の高さ [mm]
STRUCT_MASS = 0.10  # 車台・車輪・歯車など印刷部品の質量の仮定 [kg]
SC_MASS = d.SC_MASS_G / 1000
SC_CG_H = 40.0  # 土台の底面からスタックチャンの重心までの高さの仮定 [mm]
STEER_MAX = math.radians(30)
STEER_KP = 2.0  # かじ取りの位置制御ゲイン [N·m/rad](トルクはストールで頭打ち)
FRICTION_DRIVE = sm.FRICTION_TABLE + sm.FRICTION_SHAFT
MM = 0.001


def build_xml(friction: float, axle: str) -> str:
    wr, ww, wb, tr = WHEEL_R * MM, WHEEL_W * MM, WHEELBASE * MM, TRACK * MM
    fr = f'friction="{friction} 0.005 0.0001"'
    wheel = lambda x: f'<geom type="cylinder" fromto="{x - ww / 2} 0 0 {x + ww / 2} 0 0" size="{wr}" mass="0.008" rgba="0.2 0.2 0.22 1" contype="2" conaffinity="0" {fr}/>'
    drive = f'frictionloss="{FRICTION_DRIVE}" damping="1e-5"'
    if axle == "rigid":
        rear = f'''<body name="rear_axle" pos="0 {-wb / 2} {wr}">
        <joint name="drive" type="hinge" axis="1 0 0" {drive}/>
        {wheel(tr / 2)}{wheel(-tr / 2)}
      </body>'''
    else:
        rear = f'''<body name="rear_R" pos="{tr / 2} {-wb / 2} {wr}">
        <joint name="drive" type="hinge" axis="1 0 0" {drive}/>
        {wheel(0)}
      </body>
      <body name="rear_L" pos="{-tr / 2} {-wb / 2} {wr}">
        <joint name="idle_L" type="hinge" axis="1 0 0" frictionloss="{sm.FRICTION_PIN}" damping="1e-5"/>
        {wheel(0)}
      </body>'''
    front = "\n".join(
        f'''<body name="knuckle_{s}" pos="{sx * tr / 2} {wb / 2} {wr}">
        <joint name="steer_{s}" type="hinge" axis="0 0 1" range="-0.6 0.6" frictionloss="{sm.FRICTION_PIN}" damping="1e-4"/>
        <geom type="box" size="0.004 0.004 0.004" mass="0.003" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
        <body name="front_{s}">
          <joint name="roll_{s}" type="hinge" axis="1 0 0" frictionloss="{sm.FRICTION_PIN}" damping="1e-5"/>
          {wheel(0)}
        </body>
      </body>'''
        for s, sx in (("R", 1), ("L", -1))
    )
    deck_z = DECK_Z * MM
    sc_z = deck_z + SC_CG_H * MM
    return f'''<mujoco model="concept_car">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="960" offheight="720"/><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.90 0.91 0.92" rgb2="0.84 0.86 0.88" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="40 40"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" conaffinity="3" {fr}/>
    <body name="chassis" pos="0 0 0.0005">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0.35 -0.35 0.30" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
      <geom name="deck" type="box" pos="0 0 {(deck_z + wr) / 2}" size="{tr / 2 - ww} {wb / 2 + 0.01} {(deck_z - wr) / 2 + 0.003}" mass="{STRUCT_MASS - 0.044}" rgba="0.85 0.3 0.25 1" {fr}/>
      <geom name="stackchan" type="box" pos="0 0 {sc_z}" size="0.024 0.027 {SC_CG_H * MM}" mass="{SC_MASS}" rgba="0.95 0.95 0.95 1" {fr}/>
      {rear}
      {front}
    </body>
  </worldbody>
  <equality>
    <joint joint1="steer_L" joint2="steer_R" polycoef="0 1 0 0 0" solref="0.002 1"/>
  </equality>
  <actuator>
    <general name="drive" joint="drive" ctrlrange="-1 1" gainprm="{sm.SERVO_STALL}" biastype="affine" biasprm="0 0 {-sm.SERVO_STALL / sm.SERVO_NOLOAD:.6f}" forcerange="{-sm.SERVO_STALL} {sm.SERVO_STALL}"/>
    <general name="steer" joint="steer_R" ctrlrange="-0.6 0.6" gainprm="{STEER_KP}" biastype="affine" biasprm="0 {-STEER_KP} -0.05" forcerange="{-sm.SERVO_STALL} {sm.SERVO_STALL}"/>
  </actuator>
</mujoco>'''


def heading(quat) -> float:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    fwd = rot.reshape(3, 3)[:, 1]
    return math.atan2(fwd[1], fwd[0])


def run(friction: float, axle: str, command: float, steer_deg: float, seconds: float = 8.0, video: str | None = None) -> dict:
    """前進は command > 0(車輪の回転の向きを前進にそろえてある)。steer_deg > 0 で左へ切る。"""
    model = mujoco.MjModel.from_xml_string(build_xml(friction, axle))
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    body_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("deck", "stackchan")}
    drive_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drive")]
    renderer, frames = (mujoco.Renderer(model, 720, 960), []) if video else (None, None)
    rec = int(0.005 / model.opt.timestep)
    log, fallen = [], False
    for step in range(int(seconds / model.opt.timestep)):
        t = step * model.opt.timestep
        ramp = min(1.0, max(0.0, (t - 0.5) / 0.5))
        # 車輪の関節は +x 軸まわり。+ 回転で車体は -y(後ろ)へ進むので、前進の指令は符号を反転する
        data.ctrl[0] = -command * ramp
        data.ctrl[1] = math.radians(steer_deg) * ramp
        mujoco.mj_step(model, data)
        if step % rec == 0:
            pitch, roll = sm._tilt(data.qpos[3:7])
            hit = any((c.geom1 == floor and c.geom2 in body_geoms) or (c.geom2 == floor and c.geom1 in body_geoms) for c in data.contact[: data.ncon])
            fallen = fallen or hit or abs(roll) > 45 or abs(pitch) > 45
            log.append((t, data.qpos[0], data.qpos[1], heading(data.qpos[3:7]), pitch, roll, data.qvel[drive_dof], data.actuator_force[0], data.actuator_force[1]))
        if renderer is not None and step % int((1 / 30) / model.opt.timestep) == 0:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())
    if renderer is not None:
        import imageio

        imageio.mimsave(video, frames, fps=30, quality=8, macro_block_size=None)
        renderer.close()
    log = np.array(log)
    w = log[log[:, 0] >= 2.0]
    dt = w[-1, 0] - w[0, 0]
    hd = np.unwrap(w[:, 3])
    yaw = math.degrees(hd[-1] - hd[0])
    path = float(np.hypot(np.diff(w[:, 1]), np.diff(w[:, 2])).sum())
    # 車体の向きに沿った前進速度(符号つき)
    fwd = np.array([[math.cos(h), math.sin(h)] for h in hd[:-1]])
    vel = np.diff(w[:, 1:3], axis=0) / np.diff(w[:, 0])[:, None]
    speed = float((fwd * vel).sum(axis=1).mean() * 1000)
    wheel_speed = float(np.abs(w[:, 6]).mean() * WHEEL_R)
    return dict(
        friction=friction, axle=axle, command=command, steer_deg=steer_deg, fallen=bool(fallen),
        speed_mm_s=speed, wheel_surface_mm_s=wheel_speed, slip=1 - abs(speed) / wheel_speed if wheel_speed > 1 else 0.0,
        yaw_rate_deg_s=yaw / dt,
        radius_mm=path * 1000 / abs(math.radians(yaw)) if abs(yaw) > 2 else float("inf"),
        radius_geom_mm=WHEELBASE / math.tan(math.radians(abs(steer_deg))) if steer_deg else float("inf"),
        roll_deg=float(np.abs(w[:, 5]).max()), pitch_deg=float(np.abs(w[:, 4]).max()),
        drive_mNm_mean=float(np.abs(w[:, 7]).mean() * 1000), drive_mNm_peak=float(np.abs(w[:, 7]).max() * 1000),
        steer_mNm_mean=float(np.abs(w[:, 8]).mean() * 1000), steer_mNm_peak=float(np.abs(w[:, 8]).max() * 1000),
        mass_g=float(mujoco.mj_getTotalmass(model) * 1000),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cases = []
    for axle in ("rigid", "single"):
        for friction in (0.3, 0.5, 0.8, 1.0):
            for command, steer in ((0.6, 0), (-0.6, 0), (0.6, 30), (0.6, -30), (1.0, 30)):
                cases.append((friction, axle, command, steer))
    rows = []
    print("後輪    摩擦 指令  かじ角  転倒  速度mm/s 車輪周速 すべり  旋回°/s 半径mm(幾何)  傾き最大(左右/前後)  駆動平均/最大 mN·m  かじ取り平均/最大")
    for c in cases:
        r = run(*c)
        rows.append(r)
        print(f"{r['axle']:6s} {r['friction']:4.1f} {r['command']:+4.1f}  {r['steer_deg']:+4.0f}   {'あり' if r['fallen'] else 'なし'}  {r['speed_mm_s']:+7.1f}  {r['wheel_surface_mm_s']:7.1f}  {r['slip']:+5.2f}  "
              f"{r['yaw_rate_deg_s']:+6.1f}  {r['radius_mm']:6.0f}({r['radius_geom_mm']:4.0f})    {r['roll_deg']:4.1f} / {r['pitch_deg']:4.1f}°        "
              f"{r['drive_mNm_mean']:5.1f} / {r['drive_mNm_peak']:5.1f}       {r['steer_mNm_mean']:5.1f} / {r['steer_mNm_peak']:5.1f}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(rows=rows), f, ensure_ascii=False, indent=1)
    if args.video:
        run(0.5, "single", 0.6, 30, video=str(OUT / "concept_car.mp4"))
        print("動画:", OUT / "concept_car.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
