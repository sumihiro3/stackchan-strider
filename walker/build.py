"""印刷用 STL の書き出しと印刷性チェック。

各部品を「造形時の向き」(ベッド面 = z 0)に置き直して out/stl/ に書き出し、部品表を out/parts.json に保存する。
使い方: python walker/build.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import design as d

OUT = d.ROOT / "out"
BED = 180.0  # Bambu Lab A1 mini のベッド一辺 [mm]
OVERHANG_LIMIT_DEG = 50.0  # 鉛直からこの角度を超えて倒れた下向き面をオーバーハングとして数える

X_UP = d.mat([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])  # ローカル +X を上へ
X_DOWN = d.mat([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])  # ローカル +X を下へ(Y 軸まわり 90°)
FLIP = d.mat([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])  # 上下反転(X 軸まわり 180°)
MIRROR_U = d.mat([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
Y_DOWN = d.mat([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]])  # ローカル +Y を下へ(X 軸まわり -90°)

# 部品 → (造形時の向きへの行列, 向きの説明)
PRINT_POSE = {
    "side_plate": (X_UP, "内面を下、スタッドを上"),
    "deck": (np.eye(4), "そのまま(ボスを上)"),
    "yoke": (np.eye(4), "カラーを下、イヤーパッドを上"),
    "cowl": (Y_DOWN, "鼻先を下、舌を上"),
    "gear_table": (FLIP, "円板を下、歯と軸を上"),
    "socket_frame": (FLIP, "上面を下、ツメを上"),
    "pinion": (X_DOWN, "背面を下、歯と筒を上"),
    "shaft": (np.eye(4), "六角の平らな面を下にして寝かせる"),
    "sleeve": (X_UP, "立てる"),
    "crank": (X_UP, "円板を下、ピンを上"),
    "spur": (X_UP, "内面を下、合わせ印を上"),
    "cover": (X_UP, "内面を下、座を上"),
    "clip": (X_UP, "寝かせる"),
    "link_j": (np.eye(4), "スタッドを上"),
    "link_k": (np.eye(4), "寝かせる"),
    "link_c": (np.eye(4), "スタッドを上"),
    "link_f": (np.eye(4), "スタッドを上"),
    "tri_upper": (np.eye(4), "寝かせる"),
    "tri_lower": (np.eye(4), "足先の幅広部を上"),
}


def print_shape(key: str, mirrored: bool = False):
    shape = d.part_shape(key)
    if mirrored:
        shape = d.apply(shape, MIRROR_U)
    shape = d.apply(shape, PRINT_POSE[key][0])
    low = shape.bounding_box()[:3]
    return shape.translate((-low[0], -low[1], -low[2]))


def overhang_area(mesh) -> float:
    """ベッドに接していない下向き面のうち、OVERHANG_LIMIT_DEG を超える面積 [mm^2]。"""
    normals = mesh.face_normals
    centers = mesh.triangles_center
    limit = -math.sin(math.radians(OVERHANG_LIMIT_DEG))
    mask = (normals[:, 2] < limit) & (centers[:, 2] > 0.15)
    return float(mesh.area_faces[mask].sum())


def main() -> int:
    # --reprint 部品名,部品名 … 指定した部品だけを並べた 3MF(刷り直し用)を書いて終わる
    if "--reprint" in sys.argv:
        keys = set(sys.argv[sys.argv.index("--reprint") + 1].split(","))
        unknown = keys - set(d.part_defs())
        if unknown:
            raise SystemExit(f"知らない部品名: {sorted(unknown)}")
        for plate in write_plates(only=keys, file_name="stackchan_strider_reprint.3mf"):
            names = "、".join(f"{label}×{count}" for label, count in plate["contents"])
            print(f"3MF {plate['file']} プレート{plate['plate']}「{plate['name']}」: {plate['parts']} 個  {names}")
        return 0
    stl_dir = OUT / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)
    for old in stl_dir.glob("*.stl"):
        old.unlink()
    rows = []
    defs = d.part_defs()
    jobs = []
    for key, part in defs.items():
        if not part.printable:
            continue
        jobs.append((key, key, part.label, part.count, False))
    for key, name, label, count, mirrored in jobs:
        shape = print_shape(key, mirrored)
        mesh = d.to_trimesh(shape)
        if not mesh.is_watertight or mesh.volume <= 0:
            raise RuntimeError(f"{name}: 水密でない、または裏返っている")
        size = shape.bounding_box()[3:]
        fits = size[0] <= BED and size[1] <= BED and size[2] <= BED
        bodies = len(shape.decompose())
        file_name = f"{name}_x{count}.stl"
        mesh.export(stl_dir / file_name)
        rows.append(
            dict(
                key=name,
                label=label,
                count=count,
                file=file_name,
                size_mm=[round(float(v), 2) for v in size],
                volume_cm3=round(mesh.volume / 1000, 2),
                orientation=PRINT_POSE[key][1],
                overhang_mm2=round(overhang_area(mesh), 1),
                bodies=bodies,
                fits_bed=bool(fits),
                triangles=int(len(mesh.faces)),
            )
        )
    total_parts = sum(row["count"] for row in rows)
    total_volume = sum(row["volume_cm3"] * row["count"] for row in rows)
    summary = dict(
        parts=rows,
        total_printed_parts=total_parts,
        unique_stl=len(rows),
        total_volume_cm3=round(total_volume, 1),
        purchased_parts=0,
        envelope_mm=envelope(),
        ok=all(row["fits_bed"] and row["bodies"] == 1 for row in rows),
    )
    with open(OUT / "parts.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print(f"{'部品':24s} 個数  寸法(造形時) mm          体積cm3  オーバーハングmm2  向き")
    for row in rows:
        size = " x ".join(f"{v:5.1f}" for v in row["size_mm"])
        flag = "" if row["fits_bed"] and row["bodies"] == 1 else "  ← NG"
        print(f"{row['key']:24s} {row['count']:3d}  {size}  {row['volume_cm3']:6.2f}  {row['overhang_mm2']:8.1f}  {row['orientation']}{flag}")
    print(f"印刷部品 {total_parts} 個(STL {len(rows)} 種)、体積合計 {total_volume:.1f} cm3、購入部品 0 個")
    print("全体寸法(スタックチャン込み):", summary["envelope_mm"])
    plates = write_plates()
    summary["plates"] = plates
    with open(OUT / "parts.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    for plate in plates:
        names = "、".join(f"{label}×{count}" for label, count in plate["contents"])
        print(f"3MF {plate['file']} プレート{plate['plate']}「{plate['name']}」: {plate['parts']} 個  {names}")
    return 0 if summary["ok"] else 1


# ---------------------------------------------------------------- 3MF(色ごとの造形プレート)
PLATE_MARGIN = 5.0  # ベッド端からの余白 [mm]
PLATE_GAP = 3.0  # 部品どうしの間隔 [mm]
NEST_CELL = 0.5  # 輪郭をマス目にするときの 1 マス [mm]
NEST_STEP = 2  # 置き場所を探す刻み [マス]
COLOR_ROLES = (("chassis", "1_骨格"), ("accent", "2_主役"), ("link", "3_つなぎ"))
# 3MF の中(Bambu Studio の画面)で使う英語名
PLATE_NAMES = {"chassis": "Frame", "accent": "Accent", "link": "Links"}
ENGLISH_NAMES = {
    "side_plate": "Side plate", "deck": "Deck", "yoke": "Yoke (collar + ear pads)", "gear_table": "Turntable gear",
    "socket_frame": "Socket frame", "cowl": "Cowl (front / rear)", "pinion": "Pinion", "shaft": "Hex shaft", "sleeve": "Sleeve", "crank": "Crank gear", "spur": "Shaft spur gear", "cover": "Gear cover",
    "clip": "C-clip", "link_j": "Link 1 (j, C-D)", "link_k": "Link 2 (k, C-F)", "link_c": "Link 3 (c, B-F)",
    "link_f": "Link 4 (f, E-G)", "tri_upper": "Upper triangle", "tri_lower": "Lower triangle (foot)",
}


def _footprint(shape) -> np.ndarray:
    """造形時の向きの部品を真上から見た輪郭を、マス目(True = 部品がある)にする。穴の中は空きとして残す。"""
    from PIL import Image, ImageDraw

    polygons = [np.asarray(points) for points in shape.project().to_polygons()]
    size = shape.bounding_box()[3:5]
    image = Image.new("1", (int(math.ceil(size[0] / NEST_CELL)) + 1, int(math.ceil(size[1] / NEST_CELL)) + 1), 0)
    draw = ImageDraw.Draw(image)

    def signed_area(points: np.ndarray) -> float:
        return 0.5 * float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]))

    for points in sorted(polygons, key=lambda pts: -abs(signed_area(pts))):  # 外周 → 穴 の順に塗る
        draw.polygon([(float(x) / NEST_CELL, float(y) / NEST_CELL) for x, y in points], fill=1 if signed_area(points) > 0 else 0)
    return np.array(image, dtype=bool).T  # [x, y]


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """マス目を cells だけ太らせる(部品どうしの間隔を確保する)。"""
    padded = np.pad(mask, cells)
    out = np.zeros_like(padded)
    for dx in range(-cells, cells + 1):
        for dy in range(-cells, cells + 1):
            if dx * dx + dy * dy <= cells * cells:
                out |= np.roll(np.roll(padded, dx, axis=0), dy, axis=1)
    return out


def _nest(items, order):
    """輪郭どうしが重ならないように、手前左から順に詰める。items = {番号: (名前, 輪郭マス)}。

    戻り値はプレートごとの [(名前, 回転 k(×90°), 回した輪郭の左下を置く位置 x, y [mm])]。
    """
    side = int(round((BED - 2 * PLATE_MARGIN) / NEST_CELL))
    pad = int(round(PLATE_GAP / NEST_CELL))
    plates, grids = [], []
    for number in order:
        name, mask = items[number]
        rotations = []
        for k in range(4):
            turned = np.rot90(mask, k)
            if turned.shape[0] <= side and turned.shape[1] <= side:
                rotations.append((k, turned, _dilate(turned, pad)))
        if not rotations:
            raise RuntimeError(f"{name} がベッドに収まらない")
        placed = False
        for plate, grid in zip(plates, grids):
            best = None
            for k, turned, fat in rotations:
                w, h = turned.shape
                for y in range(0, side - h + 1, NEST_STEP):
                    if best is not None and y + h >= best[0]:
                        break
                    for x in range(0, side - w + 1, NEST_STEP):
                        # 太らせた輪郭(周囲に pad マスの余白つき)が、置き済みの部品と重ならないか
                        window = grid[x:x + w + 2 * pad, y:y + h + 2 * pad]
                        if not (window & fat).any():
                            if best is None or (y + h, x) < best[:2]:
                                best = (y + h, x, y, k, turned)
                            break
            if best is not None:
                _, x, y, k, turned = best
                grid[x + pad:x + pad + turned.shape[0], y + pad:y + pad + turned.shape[1]] |= turned
                plate.append((name, k, x * NEST_CELL, y * NEST_CELL))
                placed = True
                break
        if not placed:
            grid = np.zeros((side + 2 * pad, side + 2 * pad), dtype=bool)
            k, turned, _ = rotations[0]
            grid[pad:pad + turned.shape[0], pad:pad + turned.shape[1]] |= turned
            grids.append(grid)
            plates.append([(name, k, 0.0, 0.0)])
    return plates


def nest(items) -> list:
    """並べる順を何通りか試し、プレート数が最も少ない(同数なら最後のプレートが空いている)詰め方を返す。"""
    numbers = list(items)
    area = {n: int(items[n][1].sum()) for n in numbers}
    longest = {n: max(items[n][1].shape) for n in numbers}
    orders = [sorted(numbers, key=lambda n: -area[n]), sorted(numbers, key=lambda n: -longest[n])]
    rng = np.random.default_rng(1)
    for _ in range(12):
        orders.append([numbers[i] for i in rng.permutation(len(numbers))])
    best = None
    for order in orders:
        plates = _nest(items, order)
        score = (len(plates), len(plates[-1]))
        if best is None or score < best[0]:
            best = (score, plates)
    return best[1]


def placement_transform(shape, k: int, x: float, y: float, origin=(0.0, 0.0)) -> str:
    """部品を Z まわりに k×90° 回し、回した後の外形の左下を (x, y) に置く 3MF の変換(行ベクトル × 行列)。"""
    low_x, low_y, _, high_x, high_y, _ = shape.bounding_box()
    width, depth = high_x - low_x, high_y - low_y
    # np.rot90 は配列 [x, y] を反時計回りに回す。対応する点の写像と、回した後に左下が原点へ来るずらし量
    rows, shift = {
        0: ("1 0 0 0 1 0", (0.0, 0.0)),
        1: ("0 1 0 -1 0 0", (depth, 0.0)),
        2: ("-1 0 0 0 -1 0", (width, depth)),
        3: ("0 -1 0 1 0 0", (0.0, width)),
    }[k % 4]
    return f"{rows} 0 0 1 {origin[0] + PLATE_MARGIN + x + shift[0]:.3f} {origin[1] + PLATE_MARGIN + y + shift[1]:.3f} 0"


# 自動生成の出力先。out/3mf/stackchan_strider.3mf は Bambu Studio で手を入れて保存した正本なので、ここからは上書きしない
PROJECT_FILE = "stackchan_strider_generated.3mf"
PLATE_STRIDE = BED * 1.2  # Bambu Studio はプレートを「ベッド幅 × 1.2」間隔の格子に並べる(180 mm なら 216 mm)


def plate_origin(index: int, count: int) -> tuple[float, float]:
    """Bambu Studio のプレート格子での index 番目(0 始まり)の原点。列数は ceil(sqrt(count))、行は -Y 方向へ増える。"""
    columns = math.ceil(math.sqrt(count))
    return (index % columns) * PLATE_STRIDE, -(index // columns) * PLATE_STRIDE


# Bambu Studio 同梱の純正プリセット。プロジェクト設定はここから継承を解決して作る(値は一切変えない)
BS_PROFILES = Path(r"C:\Program Files\Bambu Studio\resources\profiles\BBL")
BS_VERSION = "02.06.01.55"
BS_MACHINE = "Bambu Lab A1 mini 0.4 nozzle"
BS_PROCESS = "0.20mm Standard @BBL A1M"
BS_FILAMENT = "Bambu PLA Basic @BBL A1M"
# プレートの色ごとのフィラメント番号と、画面上の色(手持ちのフィラメントの色)
FILAMENT_SLOT = {"chassis": 1, "accent": 2, "link": 3}
FILAMENT_COLOURS = ["#CCCCCC", "#2A86EA", "#FCFCFC"]
PRESET_META_KEYS = {"type", "name", "inherits", "from", "setting_id", "instantiation", "description", "include", "renamed_from"}


def _preset_index(kind: str) -> dict:
    folder = BS_PROFILES / kind
    if not folder.is_dir():
        raise RuntimeError(f"Bambu Studio のプリセットが見つからない: {folder}")
    index = {}
    for file in folder.glob("*.json"):
        with open(file, encoding="utf-8") as handle:
            data = json.load(handle)
        index[data.get("name", file.stem)] = data
    return index


def resolve_preset(kind: str, name: str, index: dict | None = None) -> dict:
    """純正プリセットを、親(inherits)と取り込み(include)をたどって 1 つの辞書にまとめる。"""
    index = _preset_index(kind) if index is None else index
    if name not in index:
        raise RuntimeError(f"プリセット {name!r} が {kind} に無い")
    data = index[name]
    merged = resolve_preset(kind, data["inherits"], index) if data.get("inherits") else {}
    for included in data.get("include", []):
        merged.update(resolve_preset(kind, included, index))
    merged.update({key: value for key, value in data.items() if key not in PRESET_META_KEYS})
    return merged


def project_settings() -> dict:
    """A1 mini 0.4 mm / 0.20mm Standard / PLA Basic ×3 色のプロジェクト設定。純正プリセットから変更なし。"""
    machine = resolve_preset("machine", BS_MACHINE)
    process = resolve_preset("process", BS_PROCESS)
    filament = resolve_preset("filament", BS_FILAMENT)
    slots = len(FILAMENT_COLOURS)
    config = {}
    config.update(process)
    config.update(machine)
    for key, value in filament.items():  # フィラメントの値はスロット数ぶん並べる
        config[key] = value * slots if isinstance(value, list) and key != "compatible_printers" else value
    config.update(
        {
            "from": "project",
            "name": "project_settings",
            "version": BS_VERSION,
            "printer_settings_id": BS_MACHINE,
            "print_settings_id": BS_PROCESS,
            "filament_settings_id": [BS_FILAMENT] * slots,
            "filament_colour": FILAMENT_COLOURS,
            "filament_ids": [filament.get("filament_id", "")] * slots,
            "filament_self_index": [str(n + 1) for n in range(slots)],  # 各フィラメントの設定列がどのスロットのものか
            "different_settings_to_system": [""] * (slots + 2),
            "inherits_group": [""] * (slots + 2),
            "curr_bed_type": "Textured PEI Plate",
        }
    )
    return config


REFERENCE_PROJECT = d.ROOT / "print" / "stackchan_strider.3mf"


def project_settings_text() -> str:
    """プロジェクト設定の JSON。Bambu Studio が入っていない環境では、元の配布 3MF(A1 mini 用)の設定をそのまま借りる。"""
    if (BS_PROFILES / "machine").is_dir():
        return json.dumps(project_settings(), ensure_ascii=False, indent=4)
    import zipfile

    with zipfile.ZipFile(REFERENCE_PROJECT) as archive:
        return archive.read("Metadata/project_settings.config").decode("utf-8")


def write_bambu_project(path, plates) -> None:
    """Bambu Studio のプロジェクト形式で、複数プレートを 1 つの 3MF に書く。

    plates = [(プレート名, フィラメント番号, [(表示名, trimesh, 回転 k, x, y, 造形時の形状)])]。x, y はプレート内の位置。
    プロジェクト設定(project_settings.config)も入れる。これが無いと Bambu Studio は形状だけを読み、プレート分けを捨てる。
    """
    import zipfile

    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">',
             f' <metadata name="Application">BambuStudio-{BS_VERSION}</metadata>',
             ' <metadata name="BambuStudio:3mfVersion">1</metadata>',
             ' <metadata name="Title">Stackchan Strider</metadata>',
             " <resources>"]
    build, config = [], ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    plate_config, assemble = [], []
    next_id = 1
    for plate_index, (plate_name, slot, items) in enumerate(plates):
        origin_x, origin_y = plate_origin(plate_index, len(plates))
        instances = []
        for label, mesh, k, x, y, shape in items:
            mesh_id, object_id = next_id, next_id + 1
            next_id += 2
            model.append(f'  <object id="{mesh_id}" type="model"><mesh><vertices>')
            model.append("".join(f'<vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>' for v in mesh.vertices))
            model.append("</vertices><triangles>")
            model.append("".join(f'<triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in mesh.faces))
            model.append("</triangles></mesh></object>")
            model.append(f'  <object id="{object_id}" type="model"><components>'
                         f'<component objectid="{mesh_id}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object>')
            transform = placement_transform(shape, k, x, y, (origin_x, origin_y))
            build.append(f'  <item objectid="{object_id}" transform="{transform}" printable="1"/>')
            config.append(f'  <object id="{object_id}">\n    <metadata key="name" value="{label}"/>\n    <metadata key="extruder" value="{slot}"/>\n'
                          f'    <part id="{mesh_id}" subtype="normal_part">\n      <metadata key="name" value="{label}"/>\n'
                          f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n    </part>\n  </object>')
            instances.append(f'    <model_instance>\n      <metadata key="object_id" value="{object_id}"/>\n'
                             f'      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="{100 + object_id}"/>\n    </model_instance>')
            assemble.append(f'   <assemble_item object_id="{object_id}" instance_id="0" transform="{transform}" offset="0 0 0" />')
        plate_config.append(f'  <plate>\n    <metadata key="plater_id" value="{plate_index + 1}"/>\n    <metadata key="plater_name" value="{plate_name}"/>\n'
                            f'    <metadata key="locked" value="false"/>\n' + "\n".join(instances) + "\n  </plate>")
    model += [" </resources>", " <build>"] + build + [" </build>", "</model>"]
    config += plate_config + ["  <assemble>"] + assemble + ["  </assemble>", "</config>"]
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                     ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                     ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", "\n".join(model))
        archive.writestr("Metadata/model_settings.config", "\n".join(config))
        archive.writestr("Metadata/project_settings.config", project_settings_text())


def write_plates(only: set | None = None, file_name: str = PROJECT_FILE) -> list[dict]:
    """色ごとにベッド(180 mm 角)へ輪郭で詰め、全プレートを 1 つの 3MF(out/3mf/stackchan_strider_generated.3mf)に書く。"""
    out_dir = OUT / "3mf"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / file_name).unlink(missing_ok=True)
    defs = d.part_defs()
    plates, result = [], []
    for role, _folder in COLOR_ROLES:
        shapes, meshes, items = {}, {}, {}
        for key, part in defs.items():
            if not part.printable or part.color != d.COLORS[role] or (only is not None and key not in only):
                continue
            shapes[key] = print_shape(key)
            meshes[key] = d.to_trimesh(shapes[key])
            mask = _footprint(shapes[key])
            for _copy in range(part.count):
                items[len(items)] = (key, mask)
        if not items:
            continue
        packed = nest(items)
        for number, plate in enumerate(packed, start=1):
            name = PLATE_NAMES[role] + (f" {number}" if len(packed) > 1 else "") + f" {d.COLORS[role]}"
            plates.append((name, FILAMENT_SLOT[role], [(ENGLISH_NAMES[key], meshes[key], k, x, y, shapes[key]) for key, k, x, y in plate]))
            counts = {}
            for key, *_ in plate:
                counts[key] = counts.get(key, 0) + 1
            result.append(dict(file=file_name, plate=len(plates), name=name, color=role, hex=d.COLORS[role], parts=len(plate),
                               contents=[(ENGLISH_NAMES[key], count) for key, count in counts.items()]))
    write_bambu_project(out_dir / file_name, plates)
    return result


def envelope() -> dict:
    low, high = np.full(3, 1e9), np.full(3, -1e9)
    for step in range(0, 360, 10):
        for inst in d.instances(math.radians(step)):
            box = d.placed(inst).bounding_box()
            low = np.minimum(low, box[:3])
            high = np.maximum(high, box[3:])
    return dict(width=round(float(high[0] - low[0]), 1), length=round(float(high[1] - low[1]), 1), height=round(float(high[2] - d.GROUND_Z), 1))


if __name__ == "__main__":
    sys.exit(main())
