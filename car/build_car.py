"""印刷する部品を造形時の向きで STL に書き出し、色ごとのプレートに並べた 3MF と部品表を作る。

使い方: python car/build_car.py [--model car|trike] [--one-color] [--reference <Bambu Studio で P2S を選んで保存した .3mf>]
  一部の部品だけの 3MF: python car/build_car.py --model trike --only wheel_rear,wheel_front
  --reference を渡すと、その 3MF のプロジェクト設定を car/bambu_p2s_settings.json に写して正本にする
出力: out/car/stl/<部品>_x<個数>.stl(ベッド面 = z 0 の造形時の向き)、out/car/3mf/stackchan_car.3mf(グレー・赤・黒のプレート)、out/car/parts.json
公開用に同じものを print_car/ へ写す(out/ は git の対象外)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import model_select

c = model_select.load()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "walker"))
import build as wb  # noqa: E402  walker の向きの行列・オーバーハングの数え方・輪郭での詰め方・Bambu Studio の 3MF をそのまま使う

d = c.d
# 印刷は Bambu Lab P2S(ベッド 256 mm 角、高さ 256 mm)。walker の詰め方はモジュールの BED を見るので、ここで差し替える
BED = 256.0
wb.BED = BED
wb.PLATE_STRIDE = BED * 1.2  # Bambu Studio のプレート格子の間隔(ベッド幅 × 1.2)
P2S_SETTINGS = Path(__file__).with_name("bambu_p2s_settings.json")
Y_UP = d.mat([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]])  # ローカル +Y を上へ(X 軸まわり 90°)
Z_UP, Z_DOWN, X_UP, X_DOWN, Y_DOWN = np.eye(4), wb.FLIP, wb.X_UP, wb.X_DOWN, wb.Y_DOWN

# 部品 → (造形時の向きへの行列, 向きの説明, サポートの要否)。6 方向のオーバーハング面積とベッドに接する面積を比べて選んだ
PRINT_POSE = {
    "rail_R": (X_DOWN, "内面を下", True),  # Bambu Studio でスライスするとサポートが要った
    "rail_L": (X_UP, "内面を下", True),
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
    "wheel_rear": (X_DOWN, "外の面を下(タイヤと一体。皿の下面はリムとハブの間 11 mm の橋)", False),
    "wheel_front": (X_UP, "内の面を下(タイヤと一体。皿の下面は 11 mm の橋)", False),
    "hubcap": (X_UP, "差し込みの筒を下", True),
    "body_rear": (Y_UP, "後ろの面を下", True),
    "body_front": (Y_UP, "後ろ(風防側)の切り口を下", True),
    "fender_rear_R": (X_DOWN, "外側の縁を下、ピンを上", False),
    "fender_rear_L": (X_UP, "外側の縁を下、ピンを上", False),
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
    "bumper_front": (Z_UP, "腕と棒の下面を下(ツメの爪は層の中で開く)", False),
    "bumper_rear": (Z_UP, "腕と棒の下面を下(ツメの爪は層の中で開く)", False),
    "grille": (Y_UP, "後ろの面を下、盾を上", True),  # Bambu Studio でスライスするとサポートが要った
    "headlight": (Y_DOWN, "レンズを下", True),
    # car500(FIAT 500 風)
    "body500": (Y_UP, "後ろの面を下", True),
    "badge": (Y_DOWN, "表を下(ピンを上)", True),
    "bumper_front500": (Z_UP, "腕と棒の下面を下(ツメの爪は層の中で開く)", False),
    "bumper_rear500": (Z_UP, "腕と棒の下面を下(ツメの爪は層の中で開く)", False),
    "windshield500": (Y_UP, "寝かせる", False),
    "top500": (Z_UP, "組んだときの向き", True),
    "mirror500_R": (Z_UP, "差し込みの棒を下", True),
    "mirror500_L": (Z_UP, "差し込みの棒を下", True),
    "ballast_lid": (Z_UP, "つばを下", False),
}
# 3MF の中(Bambu Studio の画面)で使う英語名
ENGLISH_NAMES = {
    "rail_R": "Rail R", "rail_L": "Rail L", "deck": "Deck", "yoke": "Yoke (car)", "gear_table": "Turntable gear",
    "socket_frame": "Socket frame", "pinion": "Pinion", "sleeve": "Sleeve", "shaft": "Rear axle", "beam": "Front beam",
    "knuckle_R": "Knuckle R", "knuckle_L": "Knuckle L", "hat": "Hat", "hat_back": "Hat back plate", "link": "Drag link",
    "tie_rod": "Tie rod", "clip": "C-clip", "wheel_rear": "Rear wheel", "wheel_front": "Front wheel", "hubcap": "Hubcap",
    "body_rear": "Body rear", "body_front": "Body front", "fender_rear_R": "Fender rear R",
    "fender_rear_L": "Fender rear L", "fender_front_R": "Fender front R", "fender_front_L": "Fender front L",
    "molding_body_R": "Molding body R", "molding_body_L": "Molding body L", "molding_hood_R": "Molding hood R",
    "molding_hood_L": "Molding hood L", "mirror_R": "Mirror R", "mirror_L": "Mirror L", "windshield": "Windshield",
    "top": "Folded top", "bumper_front": "Bumper front", "bumper_rear": "Bumper rear", "grille": "Grille", "headlight": "Headlight",
    "body500": "Body (500)", "badge": "Nose badge", "bumper_front500": "Bumper front (500)", "bumper_rear500": "Bumper rear (500)",
    "windshield500": "Windshield (500)", "top500": "Folded canvas top", "mirror500_R": "Door mirror R", "mirror500_L": "Door mirror L",
    "ballast_lid": "Ballast lid",
}
# 車ごとの印刷の向きと英語名。設計モジュールの PRINT_POSE_OVERRIDE は {部品: (行列の名前, 向きの説明, サポートの要否)}
_POSES = {"Z_UP": Z_UP, "Z_DOWN": Z_DOWN, "X_UP": X_UP, "X_DOWN": X_DOWN, "Y_UP": Y_UP, "Y_DOWN": Y_DOWN}
PRINT_POSE.update({k: (_POSES[m], how, sup) for k, (m, how, sup) in getattr(c, "PRINT_POSE_OVERRIDE", {}).items()})
ENGLISH_NAMES.update(getattr(c, "ENGLISH_NAMES", {}))
PLATE_ORDER = c.PLATE_ROLES  # フィラメント 1〜3 の順
# 部品どうしの間隔 [mm]。自動ブリム(最大 brim_width)が両隣の部品から伸びても重ならない幅にする。
# 5 mm ではブリムどうしが重なり、Bambu Studio で「パスの競合」になった。
# サポートは部品の輪郭より外へ広がるので、サポートの要る部品のプレートはさらに広く空ける
_BRIM = float(json.loads(P2S_SETTINGS.read_text(encoding="utf-8")).get("brim_width", 5))
PLATE_GAP = {False: 2 * _BRIM + 2.0, True: max(20.0, 2 * _BRIM + 2.0)}
# サポートの要否と違うプレートに載せる部品(部品 1〜2 個だけのプレートを作らないため)。True = support のプレート。
# 車ごとに設計モジュールの PLATE_GROUP で決める。既定の hat はユーザーが Bambu Studio で並べ替えた配置に合わせた
# (サポートは全体設定の「重要な領域のみ」で付く)
PLATE_GROUP = getattr(c, "PLATE_GROUP", {"hat": False})


def print_shape(key: str):
    """造形時の向き(ベッド面 = z 0、外形の左下手前が原点)に置いた部品。"""
    shape = d.apply(c.part_shape(key), PRINT_POSE[key][0])
    low = shape.bounding_box()[:3]
    return shape.translate((-low[0], -low[1], -low[2]))


def bed_contact(mesh) -> float:
    """ベッドに接する下向きの面の面積 [mm^2]。"""
    n, ctr = mesh.face_normals, mesh.triangles_center
    return float(mesh.area_faces[(n[:, 2] < -0.99) & (ctr[:, 2] < 0.05)].sum())


def import_reference_settings(reference: Path) -> None:
    """Bambu Studio が保存した P2S のプロジェクトから、プロジェクト設定をそのまま P2S_SETTINGS に写す。

    純正プリセットを継承で解決しただけの設定は、Studio が保存時に足す項目(nozzle_volume_type、filament_map など
    約 190 個)が欠けていて、Studio 02.09 で開けなかった。そのため Studio 自身が書いた設定を正本にする。
    """
    import zipfile

    with zipfile.ZipFile(reference) as archive:
        settings = json.loads(archive.read("Metadata/project_settings.config"))
    if settings.get("printer_model") != "Bambu Lab P2S" or settings.get("printable_area") != ["0x0", f"{BED:g}x0", f"{BED:g}x{BED:g}", f"0x{BED:g}"]:
        raise SystemExit(f"P2S の設定ではない: {settings.get('printer_model')} {settings.get('printable_area')}")
    P2S_SETTINGS.write_text(json.dumps(settings, ensure_ascii=False, indent=4), encoding="utf-8")


def _project_settings_text() -> str:
    settings = json.loads(P2S_SETTINGS.read_text(encoding="utf-8"))
    colours = [c.COLORS[role].upper() for role, _ in PLATE_ORDER]
    settings["filament_colour"] = settings["filament_multi_colour"] = colours
    settings["filament_colour_type"] = ["1"] * len(colours)
    return json.dumps(settings, ensure_ascii=False, indent=4)


def write_plates(defs, shapes, meshes, project: str | None = None) -> list[dict]:
    """色とサポートの要否ごとにベッドへ輪郭で詰め、全プレートを 1 つの 3MF に書く(walker/build.py の詰め方と書き方を使う)。

    サポートの要る部品は別のプレートにまとめ、間隔を PLATE_GAP[True] に広げる(PLATE_GROUP で載せ替える部品を除く)。
    サポートは全体設定(car/bambu_p2s_settings.json)でオン、「重要な領域のみ」。
    """
    out_dir = c.OUT / "3mf"
    out_dir.mkdir(parents=True, exist_ok=True)
    plates, result = [], []
    for slot, (role, plate_name) in enumerate(PLATE_ORDER, start=1):
        for support in (False, True):
            items = {}
            for key, part in defs.items():
                if key in shapes and part.color == c.COLORS[role] and PLATE_GROUP.get(key, PRINT_POSE[key][2]) == support:
                    mask = wb._footprint(shapes[key])
                    for _ in range(part.count):
                        items[len(items)] = (key, mask)
            if not items:
                continue
            wb.PLATE_GAP = PLATE_GAP[support]
            packed = wb.nest(items)
            for number, plate in enumerate(packed, start=1):
                name = f"{plate_name}" + (" support" if support else "") + (f" {number}" if len(packed) > 1 else "") + f" {c.COLORS[role]}"
                plates.append((name, slot, [(ENGLISH_NAMES[key], meshes[key], k, x, y, shapes[key]) for key, k, x, y in plate]))
                counts = {}
                for key, *_ in plate:
                    counts[key] = counts.get(key, 0) + 1
                result.append(dict(plate=len(plates), name=name, color=role, support=support, parts=len(plate), contents=counts))
    wb.project_settings_text = _project_settings_text  # Studio が保存した P2S の設定のまま、フィラメントの色だけ車の色にする
    wb.BS_VERSION = json.loads(P2S_SETTINGS.read_text(encoding="utf-8"))["version"]  # 3MF の Application の版を、設定を保存した Studio にそろえる
    path = out_dir / f"{project or c.PROJECT_NAME}.3mf"
    path.unlink(missing_ok=True)
    wb.write_bambu_project(path, plates)
    return result


def nest_hard(items, tries: int = 400) -> list:
    """wb.nest で 2 枚以上になるとき、大きい順を少しずつ崩した並べ方を tries 通りまで試し、1 枚に収まった時点で返す。

    walker/build.py の nest は 14 通りしか試さないので、1 枚に余裕を持って並べたいプレートで使う(walker/ は書き換えない)。
    """
    best = wb.nest(items)
    if len(best) == 1:
        return best
    numbers = list(items)
    area = {n: int(items[n][1].sum()) for n in numbers}
    base = sorted(numbers, key=lambda n: -area[n])
    rng = np.random.default_rng(7)
    for _ in range(tries):
        order = list(base)
        for _ in range(4):
            i, j = rng.integers(0, len(order), 2)
            order[i], order[j] = order[j], order[i]
        plates = wb._nest(items, order)
        if (len(plates), len(plates[-1])) < (len(best), len(best[-1])):
            best = plates
        if len(best) == 1:
            break
    return best


def write_one_color(defs, shapes, meshes) -> list[dict]:
    """色分けせず、全部品をフィラメント 1 のプレートに載せた <PROJECT_NAME>_1color.3mf を書く(1 色での試し刷り用)。

    設計モジュールの ONE_COLOR_PLATES [(プレート名, 部品の一覧 | None)] で、プレートを分ける(None = 残り全部)。
    ONE_COLOR_MARGIN があれば、ベッドの端からの余白をそれにする(端は糸を引きやすいので広く取る)。
    """
    groups = getattr(c, "ONE_COLOR_PLATES", [("All", None)])
    listed = {k for _, keys in groups if keys for k in keys}
    margin = wb.PLATE_MARGIN
    wb.PLATE_MARGIN = getattr(c, "ONE_COLOR_MARGIN", margin)
    plates, result = [], []
    try:
        for group, keys in groups:
            members = [k for k in shapes if (k in keys if keys else k not in listed)]
            for support in (False, True):
                items = {}
                for key in members:
                    if PLATE_GROUP.get(key, PRINT_POSE[key][2]) != support:
                        continue
                    mask = wb._footprint(shapes[key])
                    for _ in range(defs[key].count):
                        items[len(items)] = (key, mask)
                if not items:
                    continue
                wb.PLATE_GAP = PLATE_GAP[support]
                packed = nest_hard(items)
                for number, plate in enumerate(packed, start=1):
                    name = group + (" support" if support else "") + (f" {number}" if len(packed) > 1 else "")
                    plates.append((name, 1, [(ENGLISH_NAMES[key], meshes[key], k, x, y, shapes[key]) for key, k, x, y in plate]))
                    counts = {}
                    for key, *_ in plate:
                        counts[key] = counts.get(key, 0) + 1
                    result.append(dict(plate=len(plates), name=name, support=support, parts=len(plate), contents=counts))
    finally:
        wb.PLATE_MARGIN = margin
    wb.project_settings_text = _project_settings_text
    wb.BS_VERSION = json.loads(P2S_SETTINGS.read_text(encoding="utf-8"))["version"]
    path = c.OUT / "3mf" / f"{c.PROJECT_NAME}_1color.3mf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    wb.write_bambu_project(path, plates)
    return result


def write_only(keys: list[str]) -> int:
    """指定した部品だけを並べた 3MF を <PROJECT_NAME>_<部品>.3mf として作り、公開先にも写す(試し刷り用)。"""
    defs = c.part_defs()
    unknown = [k for k in keys if k not in defs or not defs[k].printable]
    if unknown:
        raise SystemExit(f"印刷する部品ではない: {unknown}")
    defs = {k: defs[k] for k in keys}
    shapes = {k: print_shape(k) for k in keys}
    meshes = {k: d.to_trimesh(shapes[k]) for k in keys}
    name = f"{c.PROJECT_NAME}_" + "_".join(keys)
    plates = write_plates(defs, shapes, meshes, name)
    shutil.copy2(c.OUT / "3mf" / f"{name}.3mf", c.PUBLISH)
    for p in plates:
        print(f"プレート {p['plate']} {p['name']}: {p['parts']} 個 " + "、".join(f"{k}×{n}" for k, n in p["contents"].items()))
    print(c.PUBLISH / f"{name}.3mf")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(model_select.MODELS), default="car", help="作る車(car = レトロカー、trike = 逆三輪車)")
    parser.add_argument("--reference", type=Path, help="Bambu Studio で P2S を選んで保存した 3MF(設定の正本を取り替えるとき)")
    parser.add_argument("--only", help="この部品だけの 3MF を別に作る(カンマ区切り。例 wheel_rear,wheel_front)。全体の出力は変えない")
    parser.add_argument("--one-color", action="store_true", help="色分けせず全部品を 1 色のプレートに並べた <PROJECT_NAME>_1color.3mf も作る")
    args = parser.parse_args()
    if args.only:
        return write_only(args.only.split(","))
    if args.reference:
        import_reference_settings(args.reference)
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
    if args.one_color:
        one = write_one_color(defs, shapes, meshes)
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
    publish = c.PUBLISH
    shutil.rmtree(publish, ignore_errors=True)
    shutil.copytree(out, publish / "stl")
    shutil.copy2(c.OUT / "3mf" / f"{c.PROJECT_NAME}.3mf", publish)
    shutil.copy2(c.OUT / "parts.json", publish)
    if args.one_color:
        shutil.copy2(c.OUT / "3mf" / f"{c.PROJECT_NAME}_1color.3mf", publish)
        for p in one:
            print(f"1 色 プレート {p['plate']} {p['name']}: {p['parts']} 個 " + "、".join(f"{k}×{n}" for k, n in p["contents"].items()))
    for p in plates:
        print(f"プレート {p['plate']} {p['name']}: {p['parts']} 個 " + "、".join(f"{k}×{n}" for k, n in p["contents"].items()))
    return 0 if all(r["watertight"] and r["fits_bed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
