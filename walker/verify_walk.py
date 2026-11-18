"""歩行モーション1周を通した検証。

  A. 干渉検出の自己診断(わざと部品をずらして検出できることを確かめる)
  B. クランク1回転の全姿勢で部品どうしの干渉体積が 0 であること(スタックチャン本体も含む)
  C. 連結していない部品どうしの最小すきま
  D. 抜け止め(全突起の C クリップ)の掛かりと、軸方向の遊びの合計
  E. 頭の回り止め(イヤーパッド)と、うなずき動作の干渉
  F. ターンテーブルの浮き上がり止め
  G. 接地姿勢(傾き・上下動・前進量)と必要トルクの見積り
使い方: python walker/verify_walk.py [--step 2] [--json out/verify_walk.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys

import numpy as np

import collide
import design as d
import gait

GAP_REQUIRED = 0.25  # 連結していない部品どうしに求める最小すきま [mm]
GAP_SEARCH = 1.5
SERVO_RATED_NM = 0.65 * 9.81e-2  # SCS0009 定格トルク 0.65 kg·cm @4.8V
SERVO_STALL_NM = 1.89 * 9.81e-2
TILT_LIMIT_DEG = 8.0

# 設計上つながっている(すきまが小さくて当然の)組み合わせ。部品名の接頭辞で判定する
CHASSIS = ("side_plate", "deck", "yoke", "cowl", "cover")
SHAFT = ("pinion", "shaft", "sleeve", "crank", "clip", "spur")
TABLE = ("gear_table", "socket_frame", "sc_base")
HEAD = ("sc_",)  # sc_base は TABLE が先に拾う


def _family(name: str) -> str:
    for label, prefixes in (("chassis", CHASSIS), ("shaft", SHAFT), ("table", TABLE), ("head", HEAD)):
        if name.startswith(prefixes):
            return label
    return name[:2]  # 脚は RR / RF / LR / LF


def _jointed(a: str, b: str) -> bool:
    """設計上、軸・穴・当たり面でつながっている組み合わせか。"""
    fa, fb = _family(a), _family(b)
    if fa == fb:
        return True  # 同じ剛体グループ、または同じ脚の中(ジョイントで連結)
    pair = {fa, fb}
    if not pair & {"chassis", "shaft", "table", "head"}:
        # 別の脚どうし: 同じ側の 2 本は、リンク1・2 を 1 本のクランクピンに重ねて通す(層のすきまで隣り合う)
        return fa[0] == fb[0] and a.endswith(("_j", "_k")) and b.endswith(("_j", "_k"))
    if pair <= {"chassis", "shaft", "table", "head"}:
        return True  # 軸受・歯車・カラー・イヤーパッドなど(B の干渉判定と個別の検証で見る)
    leg = (pair - {"chassis", "shaft", "table", "head"}).pop()
    other = (pair - {leg}).pop()
    leg_part = a if _family(a) == leg else b
    other_part = b if leg_part == a else a
    if other == "shaft":  # クランクピンに掛かるのは j と k、クリップはその外側
        return leg_part.endswith(("_j", "_k")) and other_part.startswith(("crank", "clip"))
    if other == "chassis":  # 固定ピボット B に掛かるのは c と上三角
        return leg_part.endswith(("_c", "_tri_upper", "_clip_B")) and other_part.startswith(("side_plate", "cover"))
    return False


def self_test() -> dict:
    """リンクを層方向へ 2 mm、足を前後へずらして、干渉として検出されることを確かめる。"""
    shifted = collide.find_collisions(0.3, nudge={"RR_k": d.translation(-2.0, 0, 0)})
    lifted = collide.find_collisions(0.3, nudge={"gear_table": d.translation(0, 0, -1.0)})
    ok = any("RR_k" in (a, b) for a, b, _ in shifted) and any("gear_table" in (a, b) for a, b, _ in lifted)
    return dict(ok=ok, shifted_hits=len(shifted), lowered_gear_hits=len(lifted))


def sweep_collisions(step_deg: float) -> dict:
    worst: dict[tuple[str, str], tuple[float, float]] = {}
    count = int(round(360 / step_deg))
    for n in range(count):
        theta = -2 * math.pi * n / count
        for a, b, volume in collide.find_collisions(theta):
            key = (a, b)
            if key not in worst or volume > worst[key][0]:
                worst[key] = (volume, math.degrees(theta) % 360)
    hits = [dict(a=a, b=b, volume_mm3=v, crank_deg=deg) for (a, b), (v, deg) in sorted(worst.items(), key=lambda kv: -kv[1][0])]
    return dict(ok=not hits, poses=count, hits=hits)


def sweep_gaps(step_deg: float) -> dict:
    smallest: dict[tuple[str, str], tuple[float, float]] = {}
    count = int(round(360 / step_deg))
    for n in range(count):
        theta = -2 * math.pi * n / count
        placed = []
        for inst in d.instances(theta):
            shape = d.placed(inst)
            placed.append((inst.name, shape, shape.bounding_box()))
        for (na, sa, ba), (nb, sb, bb) in itertools.combinations(placed, 2):
            if _jointed(na, nb):
                continue
            if any(ba[i] > bb[i + 3] + GAP_SEARCH or bb[i] > ba[i + 3] + GAP_SEARCH for i in range(3)):
                continue
            gap = sa.min_gap(sb, GAP_SEARCH)
            key = (na, nb)
            if gap < GAP_SEARCH and (key not in smallest or gap < smallest[key][0]):
                smallest[key] = (gap, math.degrees(theta) % 360)
    rows = [dict(a=a, b=b, gap_mm=g, crank_deg=deg) for (a, b), (g, deg) in sorted(smallest.items(), key=lambda kv: kv[1][0])]
    minimum = rows[0]["gap_mm"] if rows else GAP_SEARCH
    return dict(ok=minimum >= GAP_REQUIRED, min_gap_mm=minimum, required_mm=GAP_REQUIRED, closest=rows[:12])


def retention() -> dict:
    """抜け止め: すべての突起とクランクのピンを C クリップで留める。姿勢によらず効くので、寸法だけ確かめる。

    クリップが穴より大きいこと(掛かり)、溝に入ること、重ねた部品が軸方向に動ける量(遊びの合計)を出す。
    """
    overlap = d.CLIP_R_OUT - (d.STUD_R + d.FIT)
    groove_fit = (d.STUD_R - d.GROOVE_DEPTH + d.FIT) < d.STUD_R and d.GROOVE_W >= d.CLIP_T
    rows = []
    stacks = [("クランクのピン", 4)] + [(f"{joint} の突起", layers) for joint, (_o, _a, layers) in d.STUD_JOINTS.items()]
    for label, layers in stacks:
        # 根元のすきま + 層の間 + 端 + 溝の中でクリップが動ける量
        play = d.GAP + (layers - 1) * d.GAP + d.END_GAP + (d.GROOVE_W - d.CLIP_T)
        rows.append(dict(joint=label, layers=layers, axial_play_mm=play))
    return dict(ok=overlap >= 1.0 and groove_fit, clip_overlap_mm=overlap, radial_fit_mm=d.FIT, joints=rows,
                max_axial_play_mm=max(row["axial_play_mm"] for row in rows))


def _head_hits(**kwargs) -> bool:
    hits = collide.find_collisions(0.0, **kwargs)
    head_parts = ("sc_head", "sc_core", "sc_face")
    return any(a.startswith(head_parts) or b.startswith(head_parts) for a, b, _ in hits)


def head_lock() -> dict:
    """頭がヨー方向に何度まで遊ぶか(イヤーパッドに当たるまで)と、うなずきで当たらない範囲。"""
    plays = []
    for direction in (1, -1):
        low, high = 0.0, math.radians(10)
        if not _head_hits(head_yaw=direction * high):
            plays.append(math.degrees(high))
            continue
        for _ in range(12):
            mid = (low + high) / 2
            if _head_hits(head_yaw=direction * mid):
                high = mid
            else:
                low = mid
        plays.append(math.degrees((low + high) / 2))
    nod_free = [deg for deg in range(-45, 46, 5) if not _head_hits(head_pitch=math.radians(deg))]
    torque_arm = d.SC_HEAD_HALF_W / 1000
    return dict(
        ok=max(plays) < 3.0 and 0 in nod_free,
        yaw_play_deg=plays,
        nod_free_range_deg=[min(nod_free), max(nod_free)] if nod_free else None,
        pad_force_at_stall_N=SERVO_STALL_NM / (2 * torque_arm),
        note="ピッチ軸の位置(机面から 43 mm、ヨー軸の真上)は図面からの推定。うなずき範囲はこの仮定での値",
    )


def table_retention() -> dict:
    def lifted(amount: float) -> bool:
        nudge = {name: d.translation(0, 0, amount) for name in ("gear_table", "socket_frame", "sc_base")}
        hits = collide.find_collisions(0.0, nudge=nudge, with_stackchan=False)
        return any("yoke" in (a, b) for a, b, _ in hits)

    free = not lifted(0.3)
    blocked = lifted(1.0)
    return dict(ok=free and blocked, free_at_0_3mm=free, blocked_at_1_0mm=blocked)


def gait_report(frames: int) -> dict:
    mass = gait.mass_model()
    thetas, poses, forward = gait.walk_cycle(frames)
    missing = sum(pose is None for pose in poses)
    valid = [pose for pose in poses if pose is not None]
    pitch = np.array([pose.pitch_deg for pose in valid])
    roll = np.array([pose.roll_deg for pose in valid])
    height = np.array([pose.height for pose in valid])
    tilt = np.array([pose.tilt_deg for pose in valid])
    # 必要トルク(準静的): 重心を持ち上げる区間の dh/dθ × 重量。姿勢が切り替わる瞬間(落下側)は除く
    dtheta = 2 * math.pi / frames
    rise = np.diff(height) / dtheta
    continuous = np.abs(np.diff(pitch)) < 1.5
    lift = rise[continuous & (rise > 0)]
    lift_peak = float(lift.max()) if len(lift) else 0.0
    weight = mass.total_g / 1000 * gait.GRAVITY
    torque = weight * lift_peak / 1000
    tip_angle = math.degrees(math.atan2(min(abs(c[0][0]) for c in d.foot_contacts(0.0)), mass.cg[2] - d.GROUND_Z))
    stride = forward[-1]
    ok = missing == 0 and tilt.max() <= TILT_LIMIT_DEG and torque <= SERVO_RATED_NM
    return dict(
        ok=bool(ok),
        mass_total_g=mass.total_g,
        mass_walker_g=mass.walker_g,
        cg_body_mm=mass.cg.tolist(),
        cg_height_mm=[float(height.min()), float(height.max())],
        pitch_deg=[float(pitch.min()), float(pitch.max())],
        roll_deg=[float(roll.min()), float(roll.max())],
        tilt_max_deg=float(tilt.max()),
        tilt_limit_deg=TILT_LIMIT_DEG,
        static_tip_angle_deg=tip_angle,
        poses_without_rest=missing,
        min_support_feet=min(len(pose.support) for pose in valid),
        advance_per_rev_mm=float(stride),
        speed_mm_s_at_40rpm=float(stride * 40 / 60),
        torque_peak_Nm=float(torque),
        servo_rated_Nm=SERVO_RATED_NM,
        servo_stall_Nm=SERVO_STALL_NM,
        foot_lift_mm=float(max(pose.feet_clear.max() for pose in valid)),
        assumptions=[
            f"スタックチャンの重心はヨー軸から前へ {gait.SC_CG_ASSUMED[1]} mm、土台底面から {gait.SC_CG_ASSUMED[2]} mm と仮定",
            f"印刷部品の質量は体積 × PLA 密度 × 充填率 {gait.FILL_FACTOR} の概算",
            "接地姿勢は静的なつり合い。足は床で滑らないと仮定(前進量は接地足の移動量から算出)",
            "トルクは重心を持ち上げる仕事だけの準静的な値で、ジョイントと歯車の摩擦は含まない",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=2.0, help="干渉を調べるクランク角の刻み [deg]")
    parser.add_argument("--gap-step", type=float, default=6.0)
    parser.add_argument("--json")
    args = parser.parse_args()

    report = {}
    report["self_test"] = self_test()
    print(f"A. 干渉検出の自己診断: {'OK' if report['self_test']['ok'] else 'NG'} {report['self_test']}")
    report["collisions"] = sweep_collisions(args.step)
    col = report["collisions"]
    print(f"B. 歩行1周の干渉({col['poses']} 姿勢): {'OK 干渉なし' if col['ok'] else 'NG'}")
    for hit in col["hits"][:20]:
        print(f"     {hit['a']} × {hit['b']}: {hit['volume_mm3']:.3f} mm^3 (クランク {hit['crank_deg']:.0f}°)")
    report["gaps"] = sweep_gaps(args.gap_step)
    gaps = report["gaps"]
    print(f"C. 連結していない部品の最小すきま: {gaps['min_gap_mm']:.2f} mm (要求 {GAP_REQUIRED} mm) {'OK' if gaps['ok'] else 'NG'}")
    for row in gaps["closest"][:8]:
        print(f"     {row['a']} - {row['b']}: {row['gap_mm']:.2f} mm (クランク {row['crank_deg']:.0f}°)")
    report["keys"] = retention()
    keys = report["keys"]
    print(f"D. 抜け止め(全突起を C クリップで留める): {'OK' if keys['ok'] else 'NG'}  クリップの掛かり {keys['clip_overlap_mm']:.2f} mm、穴と軸のすきま 片側 {keys['radial_fit_mm']:.2f} mm")
    for row in keys["joints"]:
        print(f"     {row['joint']}: 部品 {row['layers']} 枚  軸方向の遊びの合計 {row['axial_play_mm']:.2f} mm")
    report["head_lock"] = head_lock()
    head = report["head_lock"]
    print(f"E. 頭の回り止め: ヨーの遊び +{head['yaw_play_deg'][0]:.2f}° / -{head['yaw_play_deg'][1]:.2f}°, "
          f"うなずき可能範囲 {head['nod_free_range_deg']}°, ストール時のパッド荷重 {head['pad_force_at_stall_N']:.1f} N  {'OK' if head['ok'] else 'NG'}")
    report["table_retention"] = table_retention()
    print(f"F. ターンテーブルの浮き止め: {'OK' if report['table_retention']['ok'] else 'NG'} {report['table_retention']}")
    report["gait"] = gait_report(180)
    g = report["gait"]
    print(f"G. 接地姿勢: {'OK' if g['ok'] else 'NG'}")
    print(f"     総質量 {g['mass_total_g']:.0f} g(印刷部品 {g['mass_walker_g']:.0f} g)  重心高さ {g['cg_height_mm'][0]:.1f}〜{g['cg_height_mm'][1]:.1f} mm")
    print(f"     前後の傾き {g['pitch_deg'][0]:+.1f}〜{g['pitch_deg'][1]:+.1f}°  左右 {g['roll_deg'][0]:+.1f}〜{g['roll_deg'][1]:+.1f}°  最大 {g['tilt_max_deg']:.1f}° (上限 {TILT_LIMIT_DEG}°, 静的転倒角 {g['static_tip_angle_deg']:.0f}°)")
    print(f"     支持脚の最少本数 {g['min_support_feet']}  足上げ最大 {g['foot_lift_mm']:.1f} mm  前進 {g['advance_per_rev_mm']:.1f} mm/回転  40rpm で {g['speed_mm_s_at_40rpm']:.0f} mm/s")
    print(f"     必要トルク(持ち上げ分のピーク) {g['torque_peak_Nm'] * 1000:.1f} mN·m / サーボ定格 {g['servo_rated_Nm'] * 1000:.0f} mN·m, ストール {g['servo_stall_Nm'] * 1000:.0f} mN·m")
    ok = all(section["ok"] for section in report.values())
    report["ok"] = ok
    print("総合判定:", "合格" if ok else "不合格")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
