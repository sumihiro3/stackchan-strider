"""印刷する部品を造形時の向きで STL に書き出し、色ごとのプレートに並べた 3MF と部品表を作る。

使い方: python car/build_car.py
出力: out/car/stl/<部品>_x<個数>.stl(ベッド面 = z 0 の造形時の向き)、out/car/3mf/stackchan_car.3mf(グレー・赤・黒のプレート)、out/car/parts.json
公開用に同じものを print_car/ へ写す(out/ は git の対象外)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

import car_design as c

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "walker"))
import build as wb  # noqa: E402  walker の向きの行列・オーバーハングの数え方・輪郭での詰め方・Bambu Studio の 3MF をそのまま使う

d = c.d
BED = wb.BED
Y_UP = d.mat([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]])  # ローカル +Y を上へ(X 軸まわり 90°)
Z_UP, Z_DOWN, X_UP, X_DOWN, Y_DOWN = np.eye(4), wb.FLIP, wb.X_UP, wb.X_DOWN, wb.Y_DOWN

# 部品 → (造形時の向きへの行列, 向きの説明, サポートの要否)。6 方向のオーバーハング面積とベッドに接する面積を比べて選んだ
PRINT_POSE = {
    "rail_R": (X_DOWN, "内面を下", False),
    "rail_L": (X_UP, "内面を下", False),
    "deck": (Z_UP, "そのまま(ボスを上)", False),
    "yoke": (Z_UP, "カラーを下、パッドを上", False),
    "gear_table": (Z_DOWN, "円板を下、歯と軸を上", False),
    "socket_frame": (Z_DOWN, "上面を下、ツメを上", False),
    "pinion": (X_DOWN, "背面を下、歯と筒を上", False),
    "sleeve": (X_UP, "立てる", False),
    "shaft": (Z_UP, "六角の平らな面を下にして寝かせる", False),
    "beam": (Z_UP, "寝かせる", False),
    "knuckle_R": (Y_UP, "後ろの面を下(ピンはすべて寝かせる)", True),
    "knuckle_L": (Y_UP, "後ろの面を下(ピンはすべて寝かせる)", True),
    "hat": (Y_UP, "後ろの口を下、前(ひさし)を上", True),
    "hat_back": (Y_UP, "寝かせる", False),
    "link": (Z_UP, "水平の板を下、縦の板を立てる", True),
    "tie_rod": (Z_UP, "寝かせる", False),
    "clip": (X_UP, "寝かせる", False),
    "wheel_rear": (X_DOWN, "外の面を下(皿の下面はリムとハブの間 11 mm の橋)", False),
    "wheel_front": (X_UP, "内の面を下(皿の下面は 11 mm の橋)", False),
    "hubcap": (X_UP, "差し込みの筒を下", True),
    "tire": (X_UP, "寝かせる", False),
    "body_rear": (Y_UP, "後ろの面を下", True),
    "body_front": (Y_UP, "後ろ(風防側)の切り口を下", True),
    "fender_rear_R": (X_UP, "内側の縁を下(外の面を上)", False),
    "fender_rear_L": (X_DOWN, "内側の縁を下(外の面を上)", False),
    "fender_front_R": (X_UP, "内側の縁を下(外の面を上)", False),
    "fender_front_L": (X_DOWN, "内側の縁を下(外の面を上)", False),
    "molding_body_R": (Z_DOWN, "寝かせる", False),
    "molding_body_L": (Z_DOWN, "寝かせる", False),
    "molding_hood_R": (X_DOWN, "表の面を下", True),
    "molding_hood_L": (X_UP, "表の面を下", True),
    "mirror_R": (Z_UP, "差し込みの棒を下", True),
    "mirror_L": (Z_UP, "差し込みの棒を下", True),
    "windshield": (Y_UP, "寝かせる", False),
    "top": (Z_UP, "組んだときの向き", True),
    "bumper": (Z_DOWN, "上面を下", True),
    "grille": (Y_UP, "後ろの面を下、盾を上", False),
    "headlight": (Y_DOWN, "レンズを下", True),
}
# 3MF の中(Bambu Studio の画面)で使う英語名
ENGLISH_NAMES = {
    "rail_R": "Rail R", "rail_L": "Rail L", "deck": "Deck", "yoke": "Yoke (car)", "gear_table": "Turntable gear",
    "socket_frame": "Socket frame", "pinion": "Pinion", "sleeve": "Sleeve", "shaft": "Rear axle", "beam": "Front beam",
    "knuckle_R": "Knuckle R", "knuckle_L": "Knuckle L", "hat": "Hat", "hat_back": "Hat back plate", "link": "Drag link",
    "tie_rod": "Tie rod", "clip": "C-clip", "wheel_rear": "Rear wheel", "wheel_front": "Front wheel", "hubcap": "Hubcap",
    "tire": "Tire", "body_rear": "Body rear", "body_front": "Body front", "fender_rear_R": "Fender rear R",
    "fender_rear_L": "Fender rear L", "fender_front_R": "Fender front R", "fender_front_L": "Fender front L",
    "molding_body_R": "Molding body R", "molding_body_L": "Molding body L", "molding_hood_R": "Molding hood R",
    "molding_hood_L": "Molding hood L", "mirror_R": "Mirror R", "mirror_L": "Mirror L", "windshield": "Windshield",
    "top": "Folded top", "bumper": "Bumper", "grille": "Grille", "headlight": "Headlight",
}
_WALKER_SETTINGS = wb.project_settings_text
PLATE_ORDER = (("gray", "Gray"), ("red", "Red"), ("black", "Black"))  # フィラメント 1〜3 の順


def print_shape(key: str):
    """造形時の向き(ベッド面 = z 0、外形の左下手前が原点)に置いた部品。"""
    shape = d.apply(c.part_shape(key), PRINT_POSE[key][0])
    low = shape.bounding_box()[:3]
    return shape.translate((-low[0], -low[1], -low[2]))


def bed_contact(mesh) -> float:
    """ベッドに接する下向きの面の面積 [mm^2]。"""
    n, ctr = mesh.face_normals, mesh.triangles_center
    return float(mesh.area_faces[(n[:, 2] < -0.99) & (ctr[:, 2] < 0.05)].sum())


def _project_settings_text() -> str:
    settings = json.loads(_WALKER_SETTINGS())
    colours = [c.COLORS[role].upper() for role, _ in PLATE_ORDER]
    settings["filament_colour"] = settings["filament_multi_colour"] = colours
    settings["filament_colour_type"] = ["1"] * len(colours)
    return json.dumps(settings, ensure_ascii=False, indent=4)


def write_plates(defs, shapes, meshes) -> list[dict]:
    """色ごとにベッドへ輪郭で詰め、全プレートを 1 つの 3MF に書く(walker/build.py の詰め方と書き方を使う)。"""
    out_dir = c.OUT / "3mf"
    out_dir.mkdir(parents=True, exist_ok=True)
    plates, result = [], []
    for slot, (role, plate_name) in enumerate(PLATE_ORDER, start=1):
        items = {}
        for key, part in defs.items():
            if key in shapes and part.color == c.COLORS[role]:
                mask = wb._footprint(shapes[key])
                for _ in range(part.count):
                    items[len(items)] = (key, mask)
        packed = wb.nest(items)
        for number, plate in enumerate(packed, start=1):
            name = f"{plate_name}" + (f" {number}" if len(packed) > 1 else "") + f" {c.COLORS[role]}"
            plates.append((name, slot, [(ENGLISH_NAMES[key], meshes[key], k, x, y, shapes[key]) for key, k, x, y in plate]))
            counts = {}
            for key, *_ in plate:
                counts[key] = counts.get(key, 0) + 1
            result.append(dict(plate=len(plates), name=name, color=role, parts=len(plate), contents=counts))
    wb.project_settings_text = _project_settings_text  # フィラメントの色だけ車の色にする(他の設定は walker と同じ A1 mini 用)
    path = out_dir / "stackchan_car.3mf"
    path.unlink(missing_ok=True)
    wb.write_bambu_project(path, plates)
    return result


def main() -> int:
    out = c.OUT / "stl"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.stl"):  # このスクリプトの出力だけが入るフォルダ。名前の変わった部品の古いファイルを残さない
        old.unlink()
    defs = c.part_defs()
    missing = [k for k, p in defs.items() if p.printable and k not in PRINT_POSE]
    if missing:
        raise SystemExit(f"印刷の向きが決まっていない部品: {missing}")
    rows, shapes, meshes = [], {}, {}
    color_name = {v: k for k, v in c.COLORS.items()}
    for key, part in defs.items():
        if not part.printable:
            continue
        shapes[key] = print_shape(key)
        meshes[key] = mesh = d.to_trimesh(shapes[key])
        mesh.export(out / f"{key}_x{part.count}.stl")
        size = mesh.extents
        _, how, support = PRINT_POSE[key]
        rows.append(dict(key=key, label=part.label, color=color_name.get(part.color, part.color), count=part.count,
                         size_mm=[round(float(v), 1) for v in size], volume_cm3=round(float(mesh.volume) / 1000, 2),
                         overhang_mm2=round(wb.overhang_area(mesh), 1), bed_mm2=round(bed_contact(mesh), 1),
                         orientation=how, support=support,
                         watertight=bool(mesh.is_watertight), fits_bed=bool(size[0] <= BED and size[1] <= BED and size[2] <= BED)))
    plates = write_plates(defs, shapes, meshes)
    (c.OUT / "parts.json").write_text(json.dumps(dict(parts=rows, plates=plates), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'部品':22s} 色     個数  幅×奥×高 mm            体積 cm3  張り出し mm2  接地 mm2  サポート  向き")
    for r in rows:
        flag = "" if r["watertight"] and r["fits_bed"] else "  ← 水密でない/ベッドに収まらない"
        print(f"{r['label']:22s} {r['color']:6s} {r['count']:3d}  {str(r['size_mm']):22s} {r['volume_cm3']:7.2f}  {r['overhang_mm2']:11.1f}  {r['bed_mm2']:8.1f}  "
              f"{'要' if r['support'] else '-':4s}  {r['orientation']}{flag}")
    by_color = {}
    for r in rows:
        by_color[r["color"]] = by_color.get(r["color"], 0) + r["volume_cm3"] * r["count"]
    print("色ごとの体積(cm3): " + "、".join(f"{k} {v:.1f}" for k, v in by_color.items()))
    publish = c.ROOT / "print_car"
    shutil.rmtree(publish, ignore_errors=True)
    shutil.copytree(out, publish / "stl")
    shutil.copy2(c.OUT / "3mf" / "stackchan_car.3mf", publish)
    shutil.copy2(c.OUT / "parts.json", publish)
    for p in plates:
        print(f"プレート {p['plate']} {p['name']}: {p['parts']} 個 " + "、".join(f"{k}×{n}" for k, n in p["contents"].items()))
    return 0 if all(r["watertight"] and r["fits_bed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
