"""ブラウザ用 3D ビューアーを書き出す。

out/viewer/index.html … ダブルクリックで開ける単体ファイル(three.js は CDN から読み込む)
out/viewer/stackchan-strider.html … 同じ内容の本文だけ(ほかのページへ埋め込む用)
事前に build.py / verify_gears.py --json / verify_walk.py --json を実行しておくこと。
"""
from __future__ import annotations

import base64
import json
import math
import sys

import numpy as np

import build
import design as d
import gait

OUT = d.ROOT / "out"
FRAMES = 120


def _b64(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _explode(name: str) -> list[float]:
    """分解表示で部品を動かす方向と量 [mm]。"""
    side = 1 if "_R" in name or name.startswith(("RR", "RF")) or name == "sleeve" else -1
    if name.startswith(("RR", "RF", "LR", "LF")):
        kind = name.split("_", 1)[1]
        layer = 4 if kind.startswith("clip_") else {"j": 1, "c": 1, "f": 1, "tri_upper": 2, "tri_lower": 2, "k": 3}[kind]
        fore = 1 if name[1] == "F" else -1
        return [side * (42 + 16 * layer), fore * 14.0, 0.0]
    table = {
        "side_plate_R": [26, 0, 0], "side_plate_L": [-26, 0, 0], "deck": [0, 0, -34],
        "yoke": [0, 0, 78], "gear_table": [0, 0, 26], "socket_frame": [0, 0, 50],
        "pinion": [0, 0, -14], "shaft": [0, 0, -60], "sleeve": [14, 0, -60],
        "crank_R": [48, 0, 0], "crank_L": [-48, 0, 0], "clip_R": [118, 0, 0], "clip_L": [-118, 0, 0],
        "sc_base": [0, 0, 104],
        # 改造で追加した部品
        "cowl_F": [0, 30, -20], "cowl_R": [0, -30, -20],
        "spur_R": [40, 0, -30], "spur_L": [-40, 0, -30], "cover_R": [64, 0, 0], "cover_L": [-64, 0, 0],
    }
    if name.startswith("sc_") and name not in table:
        return [0.0, 0.0, 122.0]
    return [float(v) for v in table[name]]


BODY_SETTLE = 0.08  # 胴体が次の姿勢へ倒れ込むのにかける長さ(歩行 1 周に対する割合)


def smooth_body(body: np.ndarray) -> np.ndarray:
    """胴体の姿勢(コマ × 12)を、支える足が切り替わる瞬間に一気に動かず、少し遅れて倒れ込むようにならす。

    接地姿勢は静的なつり合いで求めているので、支える 3 本が入れ替わるコマで姿勢が飛ぶ。
    表示では、前のコマの姿勢から一定の割合で近づける(1 次遅れ)。周期の継ぎ目が出ないよう 3 周まわして最後の 1 周を使う。
    """
    frames = len(body)
    rate = 1.0 - math.exp(-1.0 / (BODY_SETTLE * frames))
    out = body.copy()
    state = body[-1].copy()
    for _lap in range(3):
        for index in range(frames):
            state = state + rate * (body[index] - state)
            out[index] = state
    # ならした回転部分を、直交行列に戻す
    for index in range(frames):
        matrix = out[index].reshape(3, 4)
        u, _s, vt = np.linalg.svd(matrix[:, :3])
        matrix[:, :3] = u @ vt
    return out


def collect() -> dict:
    defs = d.part_defs()
    parts = {}
    role_of = {d.COLORS[name]: name for name in ("chassis", "accent", "link")}
    for key, part in defs.items():
        mesh = d.to_trimesh(d.part_shape(key))
        parts[key] = dict(
            label=part.label, color=part.color, printable=part.printable, finish=part.finish,
            role=role_of.get(part.color) if part.printable else None,
            up=[float(v) for v in build.PRINT_POSE[key][0][2, :3]] if key in build.PRINT_POSE else None,
            v=_b64(mesh.vertices, np.float32), i=_b64(mesh.faces, np.uint32), tris=int(len(mesh.faces)),
        )
    thetas, poses, forward = gait.walk_cycle(FRAMES)
    per_frame = [d.instances(theta) for theta in thetas[:FRAMES]]
    instances = []
    for index, inst in enumerate(per_frame[0]):
        stack = np.array([frame[index].matrix[:3, :].reshape(-1) for frame in per_frame])
        entry = dict(name=inst.name, part=inst.part, group=inst.group, ex=_explode(inst.name))
        if np.abs(stack - stack[0]).max() < 1e-9:
            entry["m"] = [round(float(v), 5) for v in stack[0]]
        else:
            entry["f"] = _b64(stack, np.float32)
        instances.append(entry)
    body = smooth_body(np.array([gait.body_transform(pose, 0.0)[:3, :].reshape(-1) for pose in poses[:FRAMES]]))
    level = d.translation(0, 0, -d.GROUND_Z)[:3, :].reshape(-1)
    feet = np.array([d.foot_contacts(theta)[:, :, :].mean(axis=1) for theta in thetas[:FRAMES]])  # (F, 4, 3)
    with open(OUT / "verify_gears.json", encoding="utf-8") as handle:
        gears = json.load(handle)
    with open(OUT / "verify_walk.json", encoding="utf-8") as handle:
        walk = json.load(handle)
    with open(OUT / "parts.json", encoding="utf-8") as handle:
        bom = json.load(handle)
    assembly = None
    if (OUT / "verify_assembly.json").exists():
        with open(OUT / "verify_assembly.json", encoding="utf-8") as handle:
            assembly = json.load(handle)
    sim = None
    sweep_path = OUT / "mujoco" / "sweep.json"
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as handle:
            sim = json.load(handle)
        sim["sensitivity"] = []
        labels = {
            "HEAD_CG_DY=8": "スタックチャンの重心が前へ 8 mm", "HEAD_CG_DY=-8": "重心が後ろへ 8 mm",
            "HEAD_CG_DY=12": "重心が前へ 12 mm", "HEAD_CG_DY=-12": "重心が後ろへ 12 mm",
            "HEAD_CG_DZ=12": "重心が上へ 12 mm", "JOINT_FRICTION_SCALE=3": "関節の摩擦が 3 倍",
        }
        for key, label in labels.items():
            path = OUT / "mujoco" / f"sweep_{key}.json"
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    other = json.load(handle)
                if other["leg_scale"] == sim["leg_scale"] and other["crank_ratio"] == sim["crank_ratio"]:
                    sim["sensitivity"].append(dict(label=label, falls=other["falls"], cases=other["cases"],
                                                   torque=other["servo_torque_mNm_max_mean"], pitch=other["worst_pitch_deg"]))
    return dict(
        roles={name: d.COLORS[name] for name in ("chassis", "accent", "link")},
        sim=sim,
        assembly=assembly,
        frames=FRAMES,
        parts=parts,
        instances=instances,
        body=_b64(body, np.float32),
        level=[float(v) for v in level],
        forward=[round(float(v), 3) for v in forward[: FRAMES + 1]],
        pitch=[round(p.pitch_deg, 2) for p in poses[:FRAMES]],
        roll=[round(p.roll_deg, 2) for p in poses[:FRAMES]],
        support=[list(p.support) for p in poses[:FRAMES]],
        feet=_b64(feet, np.float32),
        legs=[name for name, _, _ in d.LEGS],
        gears=gears,
        walk=walk,
        bom=bom,
        dims=dict(
            leg_scale=d.LEG_SCALE, crank_radius=round(d.LEG.m, 2), module=d.GEAR.module, teeth=d.GEAR.teeth,
            table_r=d.TABLE_R, ground=round(-d.GROUND_Z, 1), pocket="48 × 56 × 8",
            links={k: round(v * d.LEG_SCALE, 2) for k, v in d.HOLY.items() if k != "m"},
        ),
    )


def main() -> int:
    data = collect()
    template = (d.ROOT / "walker" / "viewer_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = template.replace("/*__DATA__*/", payload)
    out_dir = OUT / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stackchan-strider.html").write_text(body, encoding="utf-8")
    page = (
        '<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
        + body + "\n</body>\n</html>\n"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"viewer: {out_dir / 'index.html'} ({len(page) / 1e6:.2f} MB), 三角形 {sum(p['tris'] for p in data['parts'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
