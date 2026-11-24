"""組立手順の検証: 手順どおりに部品を 1 つずつ動かし、すでに付いている部品にぶつからずに入るかを確かめる。

歩行中の干渉(verify_walk.py)とは別に、「その順番で、実際に重ねられるか・クリップをはめられるか」を見る。
  - まっすぐ重ねる部品: 外側から最終位置まで、層の方向(X)に少しずつ動かして重なりを調べる
  - C クリップ: 口の向きに沿って横からはめる(はめる突起とは、広がりながら通るので、その組み合わせだけ除く)
使い方: python walker/verify_assembly.py [--json out/verify_assembly.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

import assembly_plan as plan
import collide
import design as d

TRAVEL = 24.0  # 差し込みを調べる距離 [mm]
STEPS = 49
BODY_PARTS = ("side_plate_R", "side_plate_L", "deck", "yoke", "cowl_F", "cowl_R", "gear_table", "socket_frame", "pinion", "shaft", "sleeve", "crank_R", "crank_L") + (("spur_R", "spur_L", "cover_R", "cover_L") if d.UNDERSLUNG else ())


def _shapes(theta: float) -> dict:
    return {inst.name: d.placed(inst) for inst in d.instances(theta, with_stackchan=False)}


def _overlap(moving, statics: dict, skip=()) -> tuple[float, str]:
    worst, name = 0.0, ""
    box = moving.bounding_box()
    for other, shape in statics.items():
        if other in skip:
            continue
        ob = shape.bounding_box()
        if any(box[i] > ob[i + 3] or ob[i] > box[i + 3] for i in range(3)):
            continue
        volume = (moving ^ shape).volume()
        if volume > worst:
            worst, name = volume, other
    return worst, name


def _union(shapes: dict, names) -> object:
    out = shapes[names[0]]
    for name in names[1:]:
        out = out + shapes[name]
    return out


def clip_approach(theta: float, name: str) -> np.ndarray:
    """クリップ name の口の向き(機体座標の単位ベクトル)。クリップはこの向きへ滑らせてはめる。"""
    inst = next(i for i in d.instances(theta, with_stackchan=False) if i.name == name)
    return inst.matrix[:3, :3] @ np.array([0.0, 1.0, 0.0])


def check_op(op: plan.Op, placed: list[str]) -> dict:
    shapes = _shapes(op.crank)
    statics = {name: shapes[name] for name in placed}
    sx = plan.LEG_SIGNS[op.leg][0]
    moving = _union(shapes, op.parts)
    worst, where, at = 0.0, "", ""
    if op.kind == "radial":  # C クリップ: 口の反対側から、口の向きへ滑らせる
        mouth = clip_approach(op.crank, op.parts[0])
        for s in np.linspace(12.0, 0.0, STEPS):
            volume, other = _overlap(moving.translate(tuple(-mouth * s)), statics, skip=(op.owner,))
            if volume > worst:
                worst, where, at = volume, other, f"手前 {s:.1f} mm"
    else:
        for s in np.linspace(TRAVEL, 0.0, STEPS):
            volume, other = _overlap(moving.translate((sx * s, 0, 0)), statics)
            if volume > worst:
                worst, where, at = volume, other, f"外側 {s:.1f} mm"
    return dict(op=op.key, leg=op.leg, parts=list(op.parts), crank_deg=round(math.degrees(op.crank) % 360, 1), kind=op.kind, line=op.line,
                overlap_mm3=float(worst), with_part=where, where=at, ok=worst <= collide.VOLUME_TOL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args()
    results = []
    for side in ("R", "L"):
        placed = list(BODY_PARTS)
        for op in plan.side_ops(side):
            result = check_op(op, placed)
            result["side"] = side
            results.append(result)
            placed += list(op.parts)
            print(f"[{'OK' if result['ok'] else 'NG'}] {side} {op.leg} {'+'.join(p.split('_',1)[1] for p in op.parts):28s} クランク {result['crank_deg']:5.1f}°  "
                  f"重なり {result['overlap_mm3']:.3f} mm^3" + (f"  相手 {result['with_part']}({result['where']})" if not result["ok"] else ""))
    ok = all(r["ok"] for r in results)
    print("判定:", "合格" if ok else "不合格")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(dict(ok=ok, ops=results), handle, ensure_ascii=False, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
