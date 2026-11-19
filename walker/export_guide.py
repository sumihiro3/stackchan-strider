"""組立説明サイト(スクロールに合わせて組立アニメーションが進むページ)を書き出す。

out/guide/index.html … ダブルクリックで開ける単体ファイル(three.js は CDN から読み込む)
out/guide/stackchan-strider-guide.html … 同じ内容の本文だけ(ほかのページへ埋め込む用)
事前に build.py と verify_*.py --json を実行しておくこと(形状データは export_viewer.collect() を使う)。
"""
from __future__ import annotations

import json
import sys

import numpy as np

import assembly_plan as plan
import design as d
import export_viewer

OUT = d.ROOT / "out"

# 手順の番号(ページ側の section の順番と対応する)
STEP_FRAME_LEFT, STEP_TABLE, STEP_YOKE, STEP_FRAME_RIGHT, STEP_SHAFT, STEP_CRANK = 2, 3, 4, 5, 6, 7
STEP_SIDE = {"R": 8, "L": 9}  # 右側の脚 2 本、左側の脚 2 本
STEP_COWL, STEP_STACKCHAN = 10, 11

# 脚の手順の時間割(手順の中での進み具合 0〜1)。はじめにクランクを組立位置へ回し、そのあと操作を等間隔に並べる
LEG_TURN = (0.03, 0.08)
LEG_FIRST, LEG_LAST, LEG_SPAN = 0.10, 0.90, 0.042


def leg_timing(count: int) -> list[tuple[float, float]]:
    return [(LEG_FIRST + (LEG_LAST - LEG_FIRST) * index / (count - 1), LEG_SPAN) for index in range(count)]


def leg_assembly() -> dict[str, dict]:
    """脚の部品ごとの組立情報。assembly_plan の手順(verify_assembly.py で検証済み)をそのまま使う。"""
    out = {}
    for side, step in STEP_SIDE.items():
        ops = plan.side_ops(side)
        timing = leg_timing(len(ops))
        for index, op in enumerate(ops):
            begin, span = timing[index]
            sx = plan.LEG_SIGNS[op.leg][0]
            for part in op.parts:
                entry = dict(step=step, start=begin, span=span, op=index, origin=[sx * 55.0, 0.0, 0.0])
                if op.kind == "radial":  # C クリップは口の反対側から横へ滑らせる
                    inst = next(i for i in d.instances(op.crank, with_stackchan=False) if i.name == part)
                    mouth = inst.matrix[:3, :3] @ np.array([0.0, 1.0, 0.0])
                    entry["origin"] = [float(v) for v in -mouth * 14.0]
                out[part] = entry
    return out


def assembly(name: str, legs: dict[str, dict]) -> dict:
    """部品ごとの「どの手順で、いつ、どこから来て付くか」。origin は取り付け前の位置のずれ [mm](機体座標)。"""
    # 吊り下げ版はヨークが横向きのツメで側板に留まるので、ターンテーブルとヨークを右の側板より先に付ける
    fixed = {
        "pinion": (STEP_FRAME_LEFT, 0, 3, (70, 0, 0)),  # 筒を左の側板の下の穴に内側から通す
        "side_plate_L": (STEP_FRAME_LEFT, 1, 3, (-75, 0, 0)),
        "deck": (STEP_FRAME_LEFT, 2, 3, (45, 0, 0)),  # 左のツメを側板の角穴へ横から押し込む
        "gear_table": (STEP_TABLE, 0, 2, (0, 0, 85)),
        "socket_frame": (STEP_TABLE, 1, 2, (0, 0, 85)),
        "yoke": (STEP_YOKE, 0, 1, (45, 0, 60)),
        "sleeve": (STEP_FRAME_RIGHT, 0, 2, (120, 0, 0)),
        "side_plate_R": (STEP_FRAME_RIGHT, 1, 2, (75, 0, 0)),
        "shaft": (STEP_SHAFT, 0, 3, (135, 0, 0)),
        "spur_R": (STEP_SHAFT, 1, 3, (45, 0, 0)),
        "spur_L": (STEP_SHAFT, 2, 3, (-45, 0, 0)),
        "crank_R": (STEP_CRANK, 0, 4, (45, 0, 0)),
        "crank_L": (STEP_CRANK, 1, 4, (-45, 0, 0)),
        "cover_R": (STEP_CRANK, 2, 4, (45, 0, 0)),
        "cover_L": (STEP_CRANK, 3, 4, (-45, 0, 0)),
        "cowl_F": (STEP_COWL, 0, 2, (0, 70, 0)),
        "cowl_R": (STEP_COWL, 1, 2, (0, -70, 0)),
    }
    if name in fixed:
        step, order, count, origin = fixed[name]
        return dict(step=step, start=0.1 + 0.5 * order / count, span=0.26, origin=list(origin))
    if name.startswith("sc_"):
        return dict(step=STEP_STACKCHAN, start=0.1, span=0.26, origin=[0, 0, 120])
    return legs[name]


def main() -> int:
    data = export_viewer.collect()
    legs = leg_assembly()
    for inst in data["instances"]:
        inst["asm"] = assembly(inst["name"], legs)
    ops = plan.side_ops("R")
    # list: 操作 → 説明の箇条書きの何行目か(0 行目はクランクを回す)
    data["legPlan"] = dict(steps=sorted(STEP_SIDE.values()), turn=LEG_TURN, ops=leg_timing(len(ops)), list=[op.line + 1 for op in ops],
                           flipped=[name for name, sx, sy in d.LEGS if sx * sy < 0])  # flipped: 三角板を裏返して付ける脚
    # 刻印: 部品ごとに、溝を埋める形(光らせる用)と中心位置、文字、種類(リンク番号 / ジョイント記号)
    for key, link in (("link_j", "j"), ("link_k", "k"), ("link_c", "c"), ("link_f", "f"), ("tri_upper", "tri_upper"), ("tri_lower", "tri_lower")):
        marks = []
        for mark in d.engraving_marks(link):
            mesh = d.to_trimesh(d.mark_inlay(mark))
            marks.append(dict(char=mark["char"], kind=mark["kind"], at=[round(mark["x"], 3), round(mark["y"], 3), d.LAYER_T],
                              v=export_viewer._b64(mesh.vertices, np.float32), i=export_viewer._b64(mesh.faces, np.uint32)))
        data["parts"][key]["marks"] = marks
    slim = {key: data[key] for key in ("roles", "frames", "parts", "instances", "body", "level", "legs", "legPlan")}
    slim["ground"] = round(-d.GROUND_Z, 2)
    slim["bom"] = dict(total=data["bom"]["total_printed_parts"], kinds=data["bom"]["unique_stl"], envelope=data["bom"]["envelope_mm"])
    sim = data["sim"]
    typical = next((row for row in sim["rows"] if row["friction"] == 0.5 and row["command"] == 0.6), None) if sim else None
    slim["walk"] = dict(speed=round(typical["speed_mm_s"]), rpm=round(typical["crank_rpm"])) if typical else None
    template = (d.ROOT / "walker" / "guide_template.html").read_text(encoding="utf-8")
    payload = json.dumps(slim, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = template.replace("/*__DATA__*/", payload)
    out_dir = OUT / "guide"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stackchan-strider-guide.html").write_text(body, encoding="utf-8")
    page = (
        '<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
        + body + "\n</body>\n</html>\n"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"guide: {out_dir / 'index.html'} ({len(page) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
