"""構想比較: スタックチャンのサーボ 2 個で歩く「2 サーボ歩行型」(昆虫型)を、単純な形状の MuJoCo モデルで試す。

Strider の脚機構は使わない。形状は箱と円柱と球だけで、寸法は仮の値。
  - 前脚と後ろ脚: それぞれ左右の足先を 1 本の横棒でつなぎ、車体の縦軸(z)まわりに振る。前後の棒は同じ角度で動く(連結棒の仮定)。
    首振りサーボを角度指令で使い、これを振る(1:1)
  - 中脚: 左右の足先を 1 本の棒でつなぎ、車体の前後軸(y)まわりに傾ける。片側の中脚が床を押し、その側の前後の足を浮かせる。
    うなずきサーボで傾ける(1:1)。うなずきの軸は左右(x)向きなので、実機では向きを変えるリンクが要る(モデルには入れていない)
  - サーボは位置制御で、トルクは SCS0009 のストール(sim_mujoco.SERVO_STALL)で頭打ち。指令の角速度は無負荷速度より十分遅くする
  - 歩き方: 振り θ = Θ·sin(ωt + β 項)、傾き φ = Φ·sat(sin(ωt + ψ)) + τ
使い方: python walker/sim_concept_walker.py [--json out/mujoco/concept_walker.json] [--video]
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
MM = 0.001
HIP_Z = 40.0  # 脚の付け根(振り・傾きの軸)の高さ [mm]
FOOT_R = 4.0
CORNER_X = 45.0  # 前後の足先の左右位置 [mm]
CORNER_Y = 50.0  # 前後の脚の付け根の前後位置 [mm]
MID_X = 35.0  # 中脚の足先の左右位置 [mm]
MID_EXTRA = 0.0  # 中脚を前後の脚より長くする量 [mm]
STRUCT_MASS = 0.10  # 印刷部品の質量の仮定 [kg]
SC_MASS = d.SC_MASS_G / 1000
SC_CG_H = 40.0  # スタックチャンの土台底面から重心まで [mm]
KP = 3.0  # サーボの位置ゲイン [N·m/rad]
KV = 0.03  # 速度ゲイン [N·m·s/rad]

CORNERS = ("FR", "FL", "RR", "RL")
MIDS = ("MR", "ML")


def build_xml(friction: float, mid_x: float = MID_X, mid_extra: float = MID_EXTRA) -> str:
    fr = f'friction="{friction} 0.005 0.0001"'
    hz = HIP_Z * MM
    foot_z = -(HIP_Z - FOOT_R) * MM

    def leg(name: str, x: float, z: float) -> str:
        return (f'<geom type="capsule" fromto="0 0 0 {x} 0 {z}" size="0.003" mass="0.006" rgba="0.85 0.3 0.25 1" contype="0" conaffinity="0"/>'
                f'<geom name="{name}" type="sphere" pos="{x} 0 {z}" size="{FOOT_R * MM}" mass="0.002" rgba="0.2 0.2 0.22 1" {fr}/>')

    bars = []
    for tag, y in (("F", CORNER_Y), ("R", -CORNER_Y)):
        bars.append(f'''<body name="bar_{tag}" pos="0 {y * MM} {hz}">
        <joint name="swing_{tag}" type="hinge" axis="0 0 1" range="-0.8 0.8" frictionloss="{sm.FRICTION_PIN}" damping="1e-4"/>
        {leg(tag + "R", CORNER_X * MM, foot_z)}{leg(tag + "L", -CORNER_X * MM, foot_z)}
      </body>''')
    mz = foot_z - mid_extra * MM
    bars.append(f'''<body name="bar_M" pos="0 0 {hz}">
        <joint name="tilt" type="hinge" axis="0 1 0" range="-0.8 0.8" frictionloss="{sm.FRICTION_PIN}" damping="1e-4"/>
        {leg("MR", mid_x * MM, mz)}{leg("ML", -mid_x * MM, mz)}
      </body>''')
    body_mass = STRUCT_MASS - 7 * 0.008
    sc_z = hz + 0.01 + SC_CG_H * MM
    start = 0.0007 + max(0.0, mid_extra * MM)
    return f'''<mujoco model="concept_walker">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="960" offheight="720"/><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.90 0.91 0.92" rgb2="0.84 0.86 0.88" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="40 40"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" {fr}/>
    <body name="body" pos="0 0 {start}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0.30 -0.30 0.28" xyaxes="0.707 0.707 0 -0.42 0.42 0.80"/>
      <geom name="deck" type="box" pos="0 0 {hz + 0.004}" size="0.028 {(CORNER_Y + 12) * MM} 0.006" mass="{body_mass}" rgba="0.95 0.6 0.3 1" {fr}/>
      <geom name="stackchan" type="box" pos="0 0 {sc_z}" size="0.024 0.027 {SC_CG_H * MM}" mass="{SC_MASS}" rgba="0.95 0.95 0.95 1" {fr}/>
      {chr(10).join(bars)}
    </body>
  </worldbody>
  <equality>
    <joint joint1="swing_R" joint2="swing_F" polycoef="0 1 0 0 0" solref="0.002 1"/>
  </equality>
  <actuator>
    <general name="swing" joint="swing_F" ctrlrange="-0.8 0.8" gainprm="{KP}" biastype="affine" biasprm="0 {-KP} {-KV}" forcerange="{-sm.SERVO_STALL} {sm.SERVO_STALL}"/>
    <general name="tilt" joint="tilt" ctrlrange="-0.8 0.8" gainprm="{KP}" biastype="affine" biasprm="0 {-KP} {-KV}" forcerange="{-sm.SERVO_STALL} {sm.SERVO_STALL}"/>
  </actuator>
</mujoco>'''


def _sat(x: float, k: float = 3.0) -> float:
    """sin を台形に近づける(傾けたまま振る時間を長くする)。"""
    return math.tanh(k * x) / math.tanh(k)


def heading(quat) -> float:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    fwd = rot.reshape(3, 3)[:, 1]
    return math.atan2(fwd[1], fwd[0])


def make(model):
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in (*CORNERS, *MIDS, "floor", "deck", "stackchan")}
    return ids


def contacts(model, data, ids) -> set[str]:
    on = set()
    for c in data.contact[: data.ncon]:
        pair = {c.geom1, c.geom2}
        if ids["floor"] in pair:
            other = (pair - {ids["floor"]}).pop()
            for n, g in ids.items():
                if g == other:
                    on.add(n)
    return on


def static_tilt(friction: float, phi_deg: float, mid_x: float = MID_X, mid_extra: float = MID_EXTRA) -> dict:
    """傾けたまま止めて、傾きサーボのトルクと浮いた足を見る。"""
    model = mujoco.MjModel.from_xml_string(build_xml(friction, mid_x, mid_extra))
    data = mujoco.MjData(model)
    ids = make(model)
    tilt_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")]
    for step in range(int(3.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        data.ctrl[1] = math.radians(phi_deg) * min(1.0, max(0.0, (t - 0.5) / 1.0))
        mujoco.mj_step(model, data)
    on = contacts(model, data, ids)
    pitch, roll = sm._tilt(data.qpos[3:7])
    return dict(phi_cmd=phi_deg, phi_actual=math.degrees(data.qpos[tilt_q]), tilt_mNm=float(abs(data.actuator_force[1]) * 1000),
                feet_down=sorted(on), roll_deg=roll, saturated=bool(abs(data.actuator_force[1]) >= sm.SERVO_STALL * 0.99))


def run(friction: float, swing_deg: float = 20, tilt_deg: float = 15, period: float = 1.0, psi_deg: float = 90, tilt_bias_deg: float = 0,
        swing_bias_deg: float = 0, tilt_mult: int = 1, cycles: int = 8, mid_x: float = MID_X, mid_extra: float = MID_EXTRA, video: str | None = None) -> dict:
    model = mujoco.MjModel.from_xml_string(build_xml(friction, mid_x, mid_extra))
    data = mujoco.MjData(model)
    ids = make(model)
    renderer, frames = (mujoco.Renderer(model, 720, 960), []) if video else (None, None)
    omega = 2 * math.pi / period
    start = 1.5
    total = start + cycles * period
    rec = int(0.005 / model.opt.timestep)
    log, fallen = [], False
    for step in range(int(total / model.opt.timestep)):
        t = step * model.opt.timestep
        ramp = min(1.0, max(0.0, (t - 0.5) / 1.0))
        phase = omega * max(0.0, t - start)
        if t >= start:
            swing = math.radians(swing_bias_deg) + math.radians(swing_deg) * math.sin(phase)
            tilt = math.radians(tilt_bias_deg) + math.radians(tilt_deg) * _sat(math.sin(tilt_mult * phase + math.radians(psi_deg)))
        else:
            swing = math.radians(swing_bias_deg) * ramp
            tilt = (math.radians(tilt_bias_deg) + math.radians(tilt_deg) * _sat(math.sin(math.radians(psi_deg)))) * ramp
        data.ctrl[0], data.ctrl[1] = swing, tilt
        mujoco.mj_step(model, data)
        if step % rec == 0:
            pitch, roll = sm._tilt(data.qpos[3:7])
            on = contacts(model, data, ids)
            fallen = fallen or "deck" in on or "stackchan" in on or abs(roll) > 45 or abs(pitch) > 45
            log.append((t, data.qpos[0], data.qpos[1], heading(data.qpos[3:7]), pitch, roll, data.actuator_force[0], data.actuator_force[1], len(on & set(CORNERS))))
        if renderer is not None and step % int((1 / 30) / model.opt.timestep) == 0:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())
    if renderer is not None:
        import imageio

        imageio.mimsave(video, frames, fps=30, quality=8, macro_block_size=None)
        renderer.close()
    log = np.array(log)
    w = log[log[:, 0] >= start + period]  # 最初の 1 周期は過渡として捨てる
    n = cycles - 1
    hd = np.unwrap(w[:, 3])
    h0 = hd[0]
    dxy = w[-1, 1:3] - w[0, 1:3]
    fwd = np.array([math.cos(h0), math.sin(h0)])
    left = np.array([-math.sin(h0), math.cos(h0)])
    yaw = math.degrees(hd[-1] - hd[0])
    path = float(np.hypot(np.diff(w[:, 1]), np.diff(w[:, 2])).sum())
    return dict(
        friction=friction, swing_deg=swing_deg, tilt_deg=tilt_deg, period=period, psi_deg=psi_deg, tilt_bias_deg=tilt_bias_deg,
        swing_bias_deg=swing_bias_deg, tilt_mult=tilt_mult, fallen=bool(fallen),
        fwd_mm_per_cycle=float(dxy @ fwd * 1000 / n), side_mm_per_cycle=float(dxy @ left * 1000 / n),
        speed_mm_s=float(dxy @ fwd * 1000 / (n * period)), yaw_deg_per_cycle=yaw / n, yaw_rate_deg_s=yaw / (n * period),
        radius_mm=path * 1000 / abs(math.radians(yaw)) if abs(yaw) > 5 else float("inf"),
        roll_deg=float(np.abs(w[:, 5]).max()), pitch_deg=float(np.abs(w[:, 4]).max()),
        swing_mNm_mean=float(np.abs(w[:, 6]).mean() * 1000), swing_mNm_peak=float(np.abs(w[:, 6]).max() * 1000),
        tilt_mNm_mean=float(np.abs(w[:, 7]).mean() * 1000), tilt_mNm_peak=float(np.abs(w[:, 7]).max() * 1000),
        corners_down_min=int(w[:, 8].min()), mass_g=float(mujoco.mj_getTotalmass(model) * 1000),
    )


def fmt(r: dict) -> str:
    return (f"{'転倒' if r['fallen'] else '    '} 前{r['fwd_mm_per_cycle']:+6.1f} 横{r['side_mm_per_cycle']:+5.1f} mm/周期  速度{r['speed_mm_s']:+6.1f} mm/s  "
            f"旋回{r['yaw_deg_per_cycle']:+6.1f}°/周期 {r['yaw_rate_deg_s']:+6.1f}°/s 半径{r['radius_mm']:6.0f}  傾き{r['roll_deg']:4.1f}/{r['pitch_deg']:4.1f}°  "
            f"振り{r['swing_mNm_mean']:5.1f}/{r['swing_mNm_peak']:5.1f} 傾け{r['tilt_mNm_mean']:5.1f}/{r['tilt_mNm_peak']:5.1f} mN·m")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    out = dict(static=[], gaits=[])
    print("== 傾けたまま止めたときのトルク(摩擦 0.5)")
    for phi in (5, 10, 15, 20):
        s = static_tilt(0.5, phi)
        out["static"].append(s)
        print(f"指令 {phi:2d}° 実際 {s['phi_actual']:5.1f}°  トルク {s['tilt_mNm']:6.1f} mN·m{' (頭打ち)' if s['saturated'] else ''}  接地 {','.join(s['feet_down'])}  車体の左右傾き {s['roll_deg']:+5.1f}°")
    print("== 歩き方")
    gaits = [("前進/後退 ψ=+90", dict(psi_deg=90)), ("前進/後退 ψ=-90", dict(psi_deg=-90))]
    gaits += [(f"ψ={p}", dict(psi_deg=p)) for p in (0, 45, 135, 180)]
    gaits += [(f"傾きを片寄せ τ={tb}", dict(psi_deg=90, tilt_bias_deg=tb)) for tb in (5, 10)]
    gaits += [(f"振りを片寄せ β={sb}", dict(psi_deg=90, swing_bias_deg=sb)) for sb in (10, 20)]
    gaits += [("振り 30° 片寄せ 15°", dict(psi_deg=90, swing_deg=30, swing_bias_deg=15))]
    gaits += [("傾き 2 倍周波 振り 30°", dict(psi_deg=90, tilt_mult=2, swing_deg=30))]
    gaits += [(f"傾き 2 倍周波 ψ={p}", dict(psi_deg=p, tilt_mult=2)) for p in (0, 90, 180, 270)]
    for friction in (0.3, 0.5, 0.8, 1.0):
        for label, kw in gaits:
            if friction != 0.5 and not label.startswith("前進"):
                continue
            r = run(friction, **kw)
            r["label"] = label
            out["gaits"].append(r)
            print(f"摩擦 {friction:.1f} {label:18s} {fmt(r)}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    if args.video:
        # 前進 4 周期のあと、振りを片寄せて曲がる様子は別の動画にする
        run(0.5, psi_deg=90, video=str(OUT / "concept_walker.mp4"))
        run(0.5, psi_deg=90, swing_bias_deg=20, video=str(OUT / "concept_walker_turn.mp4"))
        print("動画:", OUT / "concept_walker.mp4", OUT / "concept_walker_turn.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
