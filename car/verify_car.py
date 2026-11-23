"""車の部品どうしの干渉と、寸法・質量・重心を確かめる。

調べる姿勢: 車軸(=ターンテーブル)の回転 0〜315°(45° 刻み)× うなずき 0〜30°(2° 刻み)。前輪の切れ角はうなずきから決まる。
使い方: python car/verify_car.py [--model car|trike|pivot] [--json out/car/verify.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys

import numpy as np

import model_select

c = model_select.load()

VOLUME_TOL = 0.02  # これ未満の重なり体積 [mm^3] は面が接しているだけの数値誤差とみなす
PLA_DENSITY = 1.24e-3  # g/mm^3
FILL = 0.55  # 壁と充填を合わせた見かけの充填率(walker の gait.FILL_FACTOR 相当の仮定)


def _boxes_overlap(a, b) -> bool:
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


def collisions(drive: float, pitch: float):
    placed = [(i, c.placed(i)) for i in c.instances(drive, pitch)]
    placed = [(i, s, s.bounding_box()) for i, s in placed]
    hits = []
    for (ia, sa, ba), (ib, sb, bb) in itertools.combinations(placed, 2):
        if ia.name.startswith("sc_") and ib.name.startswith("sc_"):
            continue
        if not _boxes_overlap(ba, bb):
            continue
        v = (sa ^ sb).volume()
        if v > VOLUME_TOL:
            hits.append((ia.name, ib.name, round(v, 2)))
    return hits


def _touches(moving: str, fixed: str, shift) -> bool:
    """組み上がった位置から moving を shift [mm] だけずらすと fixed に食い込むか(=そこで止まるか)。"""
    shapes = {i.name: c.placed(i) for i in c.instances() if i.name in (moving, fixed)}
    return (shapes[moving].translate(shift) ^ shapes[fixed]).volume() > VOLUME_TOL


def fit_checks() -> list[tuple[str, bool]]:
    """はめ込みの止まり方。組み上がった位置より奥へ押し込めないこと、がたつかないことを確かめる。

    設計モジュールに FIT_CHECKS [(説明, 動かす部品, 止める部品, ずらす量)] があればそれを使う。
    """
    if hasattr(c, "FIT_CHECKS"):
        return [(label, _touches(moving, fixed, shift)) for label, moving, fixed, shift in c.FIT_CHECKS]
    bumper = "bumper_front500" if any(i.name == "bumper_front500" for i in c.instances()) else "bumper_front"
    checks = [(f"{bumper} を 1 mm 奥へ押すと横棒に当たる(腕の段で止まる)", _touches(bumper, "beam", (0, -1.0, 0)))]
    body = "body500" if bumper == "bumper_front500" else "body_rear"
    for tag in ("R", "L"):
        checks.append((f"{body} を 0.3 mm 持ち上げると出っぱりが rail_{tag} の台に掛かる", _touches(body, f"rail_{tag}", (0, 0, 0.3))))
        checks.append((f"fender_front_{tag} を 0.3 mm 下げると台に当たる(座で止まる)", _touches(f"fender_front_{tag}", f"knuckle_{tag}", (0, 0, -0.3))))
    return checks


def summary() -> dict:
    defs = c.part_defs()
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    mass, moment = 0.0, np.zeros(3)
    for inst in c.instances():
        shape = c.placed(inst)
        b = np.array(shape.bounding_box())
        lo, hi = np.minimum(lo, b[:3]), np.maximum(hi, b[3:])
        if defs[inst.part].printable:
            mesh = c.d.to_trimesh(shape)
            m = mesh.volume * PLA_DENSITY * FILL
            mass += m
            moment += m * mesh.center_mass
    sc = c.d.SC_MASS_G
    sc_center = np.array([0.0, 4.0, c.d.TABLE_Z1 + 40.0])  # スタックチャンの重心の仮定(土台の底から 40 mm 上、やや前)
    extra = sum(g for _, g, _ in c.EXTRA_MASSES)  # 鼻先のおもりなど、印刷しない重り
    for _, g, at in c.EXTRA_MASSES:
        moment = moment + g * np.asarray(at, float)
    total = mass + sc + extra
    center = (moment + sc * sc_center) / total
    low_parts = sorted(((float(np.array(c.placed(i).bounding_box())[2]) - c.GROUND_Z, i.name) for i in c.instances()
                        if not i.part.startswith(("wheel", "hubcap"))))[:3]
    return dict(
        length=float(hi[1] - lo[1]), width=float(hi[0] - lo[0]), height=float(hi[2] - c.GROUND_Z),
        printed_g=round(mass, 1), total_g=round(total, 1), center_mm=[round(float(v), 1) for v in center],
        center_height_mm=round(float(center[2] - c.GROUND_Z), 1),
        rear_axle_load=round(1 - float(center[1]) / c.WHEELBASE, 3), extra_g=round(extra, 1),
        lowest=[(round(z, 1), n) for z, n in low_parts],
    )


MIN_WHEEL_LIFT = 1.5  # 足を下ろした側の駆動輪が、これ以上浮くこと [mm]
MIN_TILT_CLEAR = 0.5  # 足を下ろして車体が傾いたとき、ほかの支えや低い所が地面からこれ以上離れていること [mm]


def statics(total_g: float, center) -> list[dict]:
    """足を下ろして片側の駆動輪を浮かせる車(設計モジュールに STATICS があるもの)の、支えごとの重さと駆動輪の浮き。

    車は「足・地面を押す駆動輪・前後の支えのどれか 1 つ」の 3 点で立つ。3 つ目の支えは、重さが負にならないもののうち、
    車体が傾いたとき最初に地面に着くもの(浮き ÷ 足と駆動輪を結ぶ線からの距離 が最小)。重心が 3 点の外なら ok = False。
    支えは (x, y, 直進のときの地面からの浮き)。
    """
    spec = c.STATICS
    weight = total_g * 9.81e-3  # [N]
    below = spec["foot_below_ground"]()
    rows = []
    for case in spec["cases"]:
        foot, wheel = np.array(case["foot"], float), np.array(case["wheel"], float)
        line = (wheel - foot) / np.linalg.norm(wheel - foot)
        best = None
        for sx, sy, lift in case["supports"]:
            support = (sx, sy)
            pts = np.array([foot, wheel, support], float)
            loads = np.linalg.solve(np.vstack([np.ones(3), pts[:, 0], pts[:, 1]]), weight * np.array([1.0, center[0], center[1]]))
            off = np.array(support, float) - foot
            dist = abs(line[0] * off[1] - line[1] * off[0])
            if loads.min() >= -1e-9 and dist > 1e-6 and (best is None or lift / dist < best[0]):
                best = (lift / dist, support, loads, pts, lift)
        if best is None:
            rows.append(dict(name=case["name"], ok=False))
            continue
        _, support, loads, pts, lift = best
        # 車体の座標で見た地面: 駆動輪の下は 0、足の下は -below、支えの下は +lift を通る平面
        plane = np.linalg.solve(np.column_stack([pts, np.ones(3)]), np.array([-below, 0.0, lift]))
        wheel_lift = -float(plane @ np.array([*case["lifted"], 1.0]))
        # 傾いた姿勢で、3 点のほかに地面に着く所がないか(着くとこすって旋回のじゃまになる)
        others = [(f"支え ({x:.0f}, {y:.0f})", x, y, up) for x, y, up in case["supports"] if (x, y) != tuple(support)] + list(case.get("low_points", []))
        clear = min(((up - float(plane @ np.array([x, y, 1.0])), name) for name, x, y, up in others), default=(float("inf"), "-"))
        rows.append(dict(name=case["name"], ok=wheel_lift >= MIN_WHEEL_LIFT and clear[0] >= MIN_TILT_CLEAR, tilt_clear=clear,
                         support=tuple(float(v) for v in support),
                         foot_n=float(loads[0]), wheel_n=float(loads[1]), support_n=float(loads[2]), share=[float(v / weight) for v in loads],
                         wheel_lift=wheel_lift, torque_nmm=float(loads[0]) * spec["torque_per_newton"]()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(model_select.MODELS), default="car")
    parser.add_argument("--json")
    args = parser.parse_args()
    s = summary()
    print(f"全長 {s['length']:.1f} 全幅 {s['width']:.1f} 全高 {s['height']:.1f} mm  印刷部品 {s['printed_g']} g  おもり {s['extra_g']} g  合計 {s['total_g']} g")
    drive_label, other_label = getattr(c, "AXLE_LABELS", ("後輪", "前輪"))
    print(f"重心 {s['center_mm']}(地面から {s['center_height_mm']} mm)  {drive_label}にかかる重さの割合 {s['rear_axle_load']:.0%}({other_label} {1 - s['rear_axle_load']:.0%})  地面に近い部品 {s['lowest']}")
    lifts = statics(s["total_g"], s["center_mm"]) if hasattr(c, "STATICS") else []
    if lifts:
        print(f"足を下ろしたとき(足は地面の線より {c.STATICS['foot_below_ground']():.2f} mm 下がる):")
        for r in lifts:
            if "support" not in r:
                print(f"  [NG] {r['name']}: 重心が 3 つの支えの外にある")
                continue
            print(f"  [{'OK' if r['ok'] else 'NG'}] {r['name']}: 足 {r['foot_n']:.2f} N({r['share'][0]:.0%})、押す駆動輪 {r['wheel_n']:.2f} N({r['share'][1]:.0%})、"
                  f"支え {r['support']} {r['support_n']:.2f} N({r['share'][2]:.0%})、浮く駆動輪 {r['wheel_lift']:.2f} mm、首のサーボ {r['torque_nmm']:.1f} N·mm、"
                  f"ほかでいちばん地面に近い所 {r['tilt_clear'][1]} {r['tilt_clear'][0]:.2f} mm")
    fits = fit_checks()
    print("はめ込み:")
    for label, ok in fits:
        print(f"  [{'OK' if ok else 'NG'}] {label}")
    found = {}
    lo, hi = (int(v) for v in c.NOD_RANGE)
    for drive in range(0, 360, 45):
        for pitch in range(lo, hi + 1, 2):
            steer = round(math.degrees(c.steer_from_nod(math.radians(pitch))), 1)
            for a, b, v in collisions(math.radians(drive), math.radians(pitch)):
                key = (a, b)
                if key not in found or v > found[key][0]:
                    found[key] = (v, drive, steer, pitch)
    if found:
        print("干渉:")
        for (a, b), (v, dr, st, pi) in sorted(found.items(), key=lambda kv: -kv[1][0]):
            print(f"  {a} × {b}: {v:.2f} mm^3(車軸 {dr}°、うなずき {pi}°、切れ角 {st}°)")
    else:
        print("干渉なし")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(summary=s, collisions=[dict(a=a, b=b, volume=v, drive=dr, steer=st, pitch=pi) for (a, b), (v, dr, st, pi) in found.items()]), f, ensure_ascii=False, indent=1)
    return 0 if not found and all(ok for _, ok in fits) and all(r["ok"] for r in lifts) else 1


if __name__ == "__main__":
    sys.exit(main())
