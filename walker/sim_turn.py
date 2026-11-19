"""MuJoCo で、片側のクランクを駆動から切り離したときに旋回できるかを確かめる。

旋回クラッチの案(片側の平歯車とシャフトの間のドグクラッチ)を、機構を作る前に評価する。
左右のクランクを別々の関節にしたモデル(sim_mujoco.build_xml(split=True))で、まず直進し、
切り離す側のクランクが指定の角度に来た瞬間にクラッチを切る。切った側は次のどちらかにする。
  lock: その角度で止める(デテントやブレーキで固定した場合)
  free: 摩擦だけで空転させる(何も止めるものが無い場合)
使い方: python walker/sim_turn.py [--json out/mujoco/turn.json] [--video]
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

SETTLE = 1.0  # 着地を待つ時間 [s]
STRAIGHT = 2.0  # クラッチを切るまで直進する時間の下限 [s]
TURN = 10.0  # クラッチを切ってから測る時間 [s]
SKIP = 0.5  # 切った直後の過渡を測定から外す時間 [s]


def _heading(quat: np.ndarray) -> float:
    """車体の前(+y)の向きの水平面内の角度 [rad]。反時計回り(上から見て左回り)が正。"""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    fwd = rot.reshape(3, 3)[:, 1]
    return math.atan2(fwd[1], fwd[0])


def run_turn(friction: float, command: float, side: str, mode: str, phase_deg: float, video: str | None = None) -> dict:
    """side 側のクランクを、その関節角が phase_deg(初期姿勢からの角度、0〜360)を過ぎた瞬間に切り離して旋回させる。"""
    model, data = sm.load(friction, split=True)

    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def eid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    q_adr = model.jnt_qposadr[jid("shaft_" + side)]
    other = "L" if side == "R" else "R"
    q_other = model.jnt_qposadr[jid("shaft_" + other)]
    bevel, lock = eid("bevel_" + side), eid("lock_" + side)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot") for leg, _, _ in d.LEGS]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    body_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("belly", "head_mass")}
    renderer, frames = (mujoco.Renderer(model, 720, 960), []) if video else (None, None)
    frame_every = int(round((1 / 30) / model.opt.timestep))
    record_every = int(round(0.005 / model.opt.timestep))

    target = math.radians(phase_deg)
    t_cut = None
    prev = 0.0
    log = []
    fallen = False
    step = 0
    max_steps = int((SETTLE + STRAIGHT + 4 + TURN) / model.opt.timestep)
    while step < max_steps:
        time = step * model.opt.timestep
        data.ctrl[0] = command * min(1.0, max(0.0, (time - SETTLE) / 0.5)) if time >= SETTLE else 0.0
        q = data.qpos[q_adr]
        if t_cut is None and time >= SETTLE + STRAIGHT:
            # 関節角が target + 2πn を跨いだ瞬間に切る(回る向きはどちらでもよい)
            if math.floor((q - target) / (2 * math.pi)) != math.floor((prev - target) / (2 * math.pi)):
                t_cut = time
                data.eq_active[bevel] = 0
                if mode == "lock":
                    model.eq_data[lock][0] = q
                    data.eq_active[lock] = 1
        prev = q
        mujoco.mj_step(model, data)
        if step % record_every == 0:
            pitch, roll = sm._tilt(data.qpos[3:7])
            on_floor, body_hit = set(), False
            for contact in data.contact[: data.ncon]:
                pair = {contact.geom1, contact.geom2}
                if floor_id in pair:
                    o = (pair - {floor_id}).pop()
                    if o in foot_ids:
                        on_floor.add(foot_ids.index(o))
                    if o in body_ids:
                        body_hit = True
            fallen = fallen or body_hit or abs(pitch) > 45 or abs(roll) > 45
            log.append((time, data.qpos[0], data.qpos[1], _heading(data.qpos[3:7]), pitch, roll,
                        data.qpos[q_adr], data.qpos[q_other], data.actuator_force[0], len(on_floor)))
        if renderer is not None and step % frame_every == 0:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())
        step += 1
        if t_cut is not None and time >= t_cut + TURN:
            break
    if renderer is not None:
        import imageio

        imageio.mimsave(video, frames, fps=30, quality=8, macro_block_size=None)
        renderer.close()
    log = np.array(log)
    if t_cut is None:
        raise RuntimeError("クラッチを切る角度に達しなかった")
    straight = log[(log[:, 0] >= SETTLE + 1.0) & (log[:, 0] < t_cut)]
    turn = log[log[:, 0] >= t_cut + SKIP]
    heading = np.unwrap(turn[:, 3])
    dt = turn[-1, 0] - turn[0, 0]
    yaw_deg = math.degrees(heading[-1] - heading[0])
    path = np.hypot(np.diff(turn[:, 1]), np.diff(turn[:, 2])).sum()
    chord = math.hypot(turn[-1, 1] - turn[0, 1], turn[-1, 2] - turn[0, 2])
    rev_driven = abs(turn[-1, 7] - turn[0, 7]) / (2 * math.pi)
    rev_cut = abs(turn[-1, 6] - turn[0, 6]) / (2 * math.pi)
    yaw_rate = yaw_deg / dt
    # 進んだ道のり ÷ 回った角度 = 車体中心の旋回半径(その場旋回なら 0 に近い)
    radius = path / abs(math.radians(yaw_deg)) * 1000 if abs(yaw_deg) > 1 else float("inf")
    return dict(
        friction=friction, command=command, side=side, mode=mode, phase_deg=phase_deg,
        fallen=bool(fallen),
        yaw_deg=yaw_deg, yaw_rate_deg_s=yaw_rate,
        deg_per_driven_rev=yaw_deg / rev_driven if rev_driven > 0.1 else 0.0,
        driven_rpm=rev_driven / dt * 60, cut_side_rev=rev_cut,
        path_mm=path * 1000, chord_mm=chord * 1000, radius_mm=radius,
        pitch_deg=[float(turn[:, 4].min()), float(turn[:, 4].max())],
        roll_deg=[float(turn[:, 5].min()), float(turn[:, 5].max())],
        feet_on_floor_min=int(turn[:, 9].min()), feet_on_floor_mean=float(turn[:, 9].mean()),
        servo_torque_mNm_mean=float(np.abs(turn[:, 8]).mean() * 1000),
        straight_heading_drift_deg=float(math.degrees(np.unwrap(straight[:, 3])[-1] - np.unwrap(straight[:, 3])[0])) if len(straight) > 2 else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cases = []
    # 1) 止める角度の影響(摩擦 0.5、指令 +0.6、左を止める)。角度は左クランク関節の初期姿勢からの角度
    for phase in range(0, 360, 45):
        cases.append((0.5, 0.6, "L", "lock", phase))
    # 以下は、止めた側の 2 本の足が前後に大きく開いて両方接地する角度(左 90° = 右 270°)で調べる
    # 2) 空転させた場合
    cases.append((0.5, 0.6, "L", "free", 90))
    # 3) 左右・前後の対称性
    for side, command, phase in (("L", -0.6, 90), ("R", 0.6, 270), ("R", -0.6, 270)):
        cases.append((0.5, command, side, "lock", phase))
    # 4) 床の摩擦と速さ
    for friction in (0.3, 0.8, 1.0):
        cases.append((friction, 0.6, "L", "lock", 90))
    cases.append((0.5, 0.3, "L", "lock", 90))
    cases.append((0.5, 1.0, "L", "lock", 90))
    rows = []
    print("摩擦 指令  止める側 方式  角度  転倒  旋回°   °/s    °/駆動1回転  半径mm  駆動rpm  切った側の回転  前後傾き       左右傾き       接地(最少/平均)")
    for friction, command, side, mode, phase in cases:
        r = run_turn(friction, command, side, mode, phase)
        rows.append(r)
        print(
            f"{friction:3.1f} {command:+4.1f}  {side}      {mode:4s}  {phase:3d}  {'あり' if r['fallen'] else 'なし'}  {r['yaw_deg']:+6.1f}  {r['yaw_rate_deg_s']:+5.1f}  {r['deg_per_driven_rev']:+7.1f}     "
            f"{r['radius_mm']:6.0f}  {r['driven_rpm']:6.1f}  {r['cut_side_rev']:6.2f}        {r['pitch_deg'][0]:+5.1f}〜{r['pitch_deg'][1]:+5.1f}°  "
            f"{r['roll_deg'][0]:+5.1f}〜{r['roll_deg'][1]:+5.1f}°  {r['feet_on_floor_min']} / {r['feet_on_floor_mean']:.2f}",
            flush=True,
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(dict(turn_seconds=TURN, rows=rows), handle, ensure_ascii=False, indent=1)
    if args.video:
        out = sm.OUT / "turn.mp4"
        run_turn(0.5, -0.6, "L", "lock", 90, video=str(out))
        print("動画:", out)
    return 0 if not any(r["fallen"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
