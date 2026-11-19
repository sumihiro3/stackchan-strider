"""スタックチャン・レトロカー: 首振りサーボで後輪を回し、うなずきサーボで前輪を切る車。赤いボディ、黒いフェンダー・タイヤ、グレーの金物。

座標系(単位 mm)は Strider と同じ。X = 右、Y = 前、Z = 上。
原点 = 後輪の車軸(X 軸)とヨー軸(Z 軸)の交点 = 傘歯車のピッチ円錐の頂点。地面は z = -WHEEL_R。
駆動部(デッキ、ターンテーブル歯車、ソケット枠、ピニオン、スリーブ、ヨーク)は walker/design.py の部品をそのまま使う。
部品は「ローカル形状」と「姿勢行列」に分けて持ち、instances() が姿勢つきの全部品を返す。
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, Manifold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "walker"))
import design as d  # noqa: E402
from design import (  # noqa: E402
    AXIS_Z_TO_X, HOLE_FIT, SEG, apply, box, circle, cyl, cyl_x, hexagon, hull2, plate_yz, polygon, rect, rot_x, rot_z, translation,
)

# ピッチ軸の高さは実機の写真(定規を立て、うなずきを変えて 3 枚)で約 45 mm と測った(walker の推定は 43)。
# walker/ は書き換えず、車ではここで上書きする。PITCH_Z は walker の読み込み時に計算済みなので一緒に直す
d.SC_PITCH_AXIS_H = 45.0
d.PITCH_Z = d.TABLE_Z1 + d.SC_PITCH_AXIS_H

ROOT = d.ROOT
OUT = ROOT / "out" / "car"

# ---------------------------------------------------------------- 寸法
WHEEL_R = 30.0  # タイヤの外半径。ピニオンの歯先(半径 21.4)と地面のすきまが 8.6 mm 残る
RIM_R = 22.0  # ホイールの外半径 = タイヤの内半径
TIRE_W = 10.0
WHEEL_X0 = 58.0  # ホイールの内面 |x|(デッキとヨークのツメの先 53.4 と、ボディの外面 56 の外)
WHEEL_X1 = WHEEL_X0 + TIRE_W
WHEELBASE = 105.0
GROUND_Z = -WHEEL_R

# 後ろの車軸: walker の六角シャフトと同じ断面で、ホイールのハブまで伸ばす
SHAFT_HALF = WHEEL_X1 - 3.5  # 先はキャップの穴の手前で止める
HUB_X0 = d.PLATE_OUT + 0.4  # 後輪のハブの内面(側板の外面に当たって、車軸の左右の位置を決める)
HUB_R = 7.0
# 六角シャフトとホイールのはめあい。元の Strider はクランクの六角穴とシャフトの差が 0.5 mm(1 個あたり約 ±7° の空回り)で、
# 摩擦だけで留まっていたため抜けた。差を 0.1 mm にすると軽い圧入で止まる(GOROman 氏の H3C での修正値)。
# 圧入はホイールの外寄り TIGHT_LEN だけにして、残りは元のすきまで通す(全長を圧入にすると押し込めない)
SHAFT_ACTUAL_AF = d.SHAFT_AF - d.SHAFT_UNDERSIZE  # 7.8
WHEEL_SOCKET_AF = SHAFT_ACTUAL_AF + 0.1
WHEEL_SOCKET_LEAD = 0.3  # 圧入部の入口の面取り(二面幅で片側)
TIGHT_X = (0.5, 6.5)  # 圧入する範囲(ホイールの内面から。外端はシャフトの先 SHAFT_HALF の手前)
CAP_POCKET_R = 5.8  # キャップの穴。前輪では中に C クリップが入る
FRONT_BORE_X1 = 5.0  # 前輪の軸受け穴の奥(この先の壁を溝つきピンが貫き、外で C クリップが留める)
FRONT_WALL_X1 = 6.5

# 側板(後ろは walker の側板と同じ位置・同じ角穴。前へ伸びて、前輪の間で内側へ寄る)
RAIL_FRONT_X = (25.0, 30.0)  # 前の部分の内面・外面 |x|
RAIL_TAPER_Y = (48.0, 72.0)  # この区間で外から内へ寄せる
RAIL_FRONT_Y = WHEELBASE + 13.0
RAIL_BEAM_Z = (-8.0, 2.0)

# 前の横棒とキングピン(前輪の向きを変える縦の軸)
KINGPIN_X = 50.0
BOSS_R = 5.0
BEAM_Z = (-6.0, 2.0)
KNUCKLE_GAP = 0.3  # キングピンのボスとナックルの腕のすきま(上下)
KNUCKLE_Z = (BEAM_Z[1] + KNUCKLE_GAP, BEAM_Z[1] + KNUCKLE_GAP + 5.0)
UPRIGHT_X = (WHEEL_X0 - 2.3, WHEEL_X0 - 0.3)  # ナックルの縦板(ホイールの内面より内側)
STUB_R = 4.0  # 前輪の軸
STEER_MAX = math.radians(30)
STEER_ARM_T = 4.0
TIE_ARM = 15.0  # キングピンからタイロッドのピンまで(真前)
TIE_Z = (KNUCKLE_Z[0] + STEER_ARM_T + 0.3, KNUCKLE_Z[0] + STEER_ARM_T + 0.3 + 2.4)  # タイロッドの板の上下
TIE_PIN_CLEAR = TIE_Z[1] + 0.2 - (KNUCKLE_Z[0] + STEER_ARM_T)
DRAG_ARM = 7.0  # キングピンから引き棒のピンまで(内向き)。fit_drag_arm() が NOD_RANGE の両端で ±STEER_MAX になる長さに置き換える
LINK_TAB_Z = (KNUCKLE_Z[1] + 0.3, KNUCKLE_Z[1] + 0.3 + 2.4)  # 引き棒の前端の板(ドラッグアームの上)
DRAG_PIN_CLEAR = LINK_TAB_Z[1] + 0.2 - KNUCKLE_Z[1]

# ボディ(赤)。外形は球の凸包で丸く作り、肉厚 SKIN で殻にする
SKIN = 2.0
BODY_HALF_W = 56.0  # 後ろのボディの外面 |x|
BODY_REAR_Y = -62.0
BODY_SPLIT_Y = 47.0  # 後ろと前のボディの分け目(風防の根元)
BODY_BOTTOM_Z = -6.0
DECK_TOP_Z = 50.0  # 後ろのボディ上面(頭の下端 43.5 を隠す)
COCKPIT = dict(half_w=42.0, y0=-27.0, radius=12.0)  # 横はハットの腕とかじ取りの引き棒を通す幅
HOOD_HALF_W = 33.0  # 前輪の間のボンネットの外面 |x|(前輪を 30° 切ったときの内側の逃げが 41.8)
HOOD_NOSE_Y = 146.0
HOOD_BOTTOM_Z = BEAM_Z[1] + 3.0
COWL_Y1 = 60.0  # 幅の広い風防下の部分の前端
DOOR_Y = (6.0, 43.0)  # ドアの溝の前後(前ボディとの分け目 47 の手前)
DOOR_TOP_Z = 38.0
DOOR_HANDLE_Z = 32.0
LOUVER_Y = [77.0 + 4.0 * k for k in range(5)]  # ボンネット横の空気取り(縦の溝)
LOUVER_Z = (19.0, 30.0)
MIRROR_XY = (17.0, 124.0)  # フェンダーミラーの根元(右)

# フェンダー(赤)。後ろはボディの横に固定、前はナックルと一緒に向きを変える
FENDER_R = 35.5  # 内面の半径(タイヤの外面との間 5.5)
FENDER_T = 2.0
FENDER_X = (WHEEL_X0 - 1.5, WHEEL_X1 + 2.0)
REAR_FENDER_DEG = (-10.0, 200.0)  # 前の水平を 0°、上を 90° として覆う範囲
FRONT_FENDER_DEG = (25.0, 150.0)

COLORS = {
    "red": "#c8372d",
    "gray": "#9a9a96",
    "black": "#26262a",
    "dark": "#5f5e5a",
}


# ---------------------------------------------------------------- 形の道具
def _ball(x: float, y: float, z: float, r: float) -> Manifold:
    return Manifold.sphere(r, 48).translate((x, y, z))


def blob(points, radius: float, shrink: float = 0.0) -> Manifold:
    """球の凸包。shrink だけ小さい球で作ると、外形から shrink だけ内側へずれた形(殻の内面)になる。"""
    return Manifold.batch_hull([_ball(x, y, z, radius - shrink) for x, y, z in points])


def mirror_x(points):
    out = []
    for x, y, z in points:
        out += [(x, y, z), (-x, y, z)]
    return out


def shell(points, radius: float) -> Manifold:
    return blob(points, radius) - blob(points, radius, SKIN)


ENGRAVE = 0.9  # 溝の深さ(肉厚 SKIN の半分弱)


def engrave(body: Manifold, cutter: Manifold, outer: Manifold, inner: Manifold) -> Manifold:
    """外面から ENGRAVE だけの層(outer - inner)と cutter の重なりを削る。cutter は溝を外面に投影する柱。"""
    return body - (cutter ^ (outer - inner))


def _crowned(x0: float, x1: float, r_in: float, t: float, crown: float, steps: int = 12) -> list[tuple[float, float]]:
    """フェンダーの断面((x, r) 平面)。幅の中央が crown だけ外へふくらむ。"""
    mid, half = (x0 + x1) / 2, (x1 - x0) / 2
    outer = [(x0 + (x1 - x0) * i / steps, r_in + t + crown * (1 - ((x0 + (x1 - x0) * i / steps - mid) / half) ** 2)) for i in range(steps + 1)]
    inner = [(x, r - t) for x, r in reversed(outer)]
    return outer + inner


def ring_sector_x(section: list[tuple[float, float]], deg0: float, deg1: float) -> Manifold:
    """(x, r) の断面を X 軸まわりに deg0..deg1 だけ回す(0° = +Y、90° = +Z)。"""
    # revolve は (u, v) の u を半径として v 軸まわりに回し、v が Z になる。断面を (r, x) にして回してから Z → X へ向ける
    pts = [(r, x) for x, r in section]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    solid = Manifold.revolve(polygon(pts if area > 0 else pts[::-1]), 96, deg1 - deg0)
    # revolve の 0° は +X 方向。+X → +Y(前)、回転の軸 Z → X になるように並べ替え、deg0 だけ回す
    to_x = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    return solid.transform(to_x).rotate((deg0, 0, 0))


# ---------------------------------------------------------------- 駆動まわり(グレー)
@lru_cache(maxsize=None)
def shaft_part() -> Manifold:
    return apply(hexagon(d.SHAFT_AF - d.SHAFT_UNDERSIZE).extrude(2 * SHAFT_HALF), AXIS_Z_TO_X).translate((-SHAFT_HALF, 0, 0))


@lru_cache(maxsize=None)
def rail_part() -> Manifold:
    """右の側板。後ろは walker の側板と同じ軸受け穴と角穴、前は細い梁になって前輪の間へ内側に寄る。"""
    deck_mid = (d.DECK_Z0 + d.DECK_Z1) / 2
    lug_mid = d.COLLAR_Z0 + d.COLLAR_LUG_T / 2
    nodes = [circle(13.0, 0, 0)]
    for sign in (1, -1):
        nodes += [circle(7.5, sign * d.DECK_TENON_Y, deck_mid), circle(7.5, sign * d.YOKE_TENON_Y, lug_mid)]
    section = hull2(*nodes)
    section += rect(0, RAIL_TAPER_Y[0] + 2.0, RAIL_BEAM_Z[0], RAIL_BEAM_Z[1])
    section -= circle(d.SLEEVE_R + HOLE_FIT)
    for sign in (1, -1):
        section -= rect(sign * d.DECK_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.DECK_TENON_Y + d.TENON_W / 2 + 0.2, d.DECK_Z0 - 0.15, d.DECK_Z1 + 0.15)
        section -= rect(sign * d.YOKE_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.YOKE_TENON_Y + d.TENON_W / 2 + 0.2, d.COLLAR_Z0 - 0.15, d.COLLAR_Z0 + d.COLLAR_LUG_T + 0.15)
    rear = plate_yz(section, d.PLATE_IN, d.PLATE_OUT)
    # 前の梁: 平面図で外から内へ寄る帯を、梁の高さで押し出す
    (y0, y1), (fx0, fx1) = RAIL_TAPER_Y, RAIL_FRONT_X
    band = polygon([(d.PLATE_IN, RAIL_TAPER_Y[0] - 2.0), (d.PLATE_OUT, RAIL_TAPER_Y[0] - 2.0), (d.PLATE_OUT, y0), (fx1, y1), (fx1, RAIL_FRONT_Y),
                    (fx0, RAIL_FRONT_Y), (fx0, y1), (d.PLATE_IN, y0)])
    beam = band.extrude(RAIL_BEAM_Z[1] - RAIL_BEAM_Z[0]).translate((0, 0, RAIL_BEAM_Z[0]))
    # 前の横棒を受ける台
    seat = box(RAIL_FRONT_X[0] - 3.0, RAIL_FRONT_X[1], WHEELBASE - 7.0, WHEELBASE + 7.0, RAIL_BEAM_Z[0], RAIL_BEAM_Z[1])
    # 横棒が上から落ちる溝
    notch = box(RAIL_FRONT_X[0] - 4.0, RAIL_FRONT_X[1] + 1.0, WHEELBASE - BOSS_R - 0.15, WHEELBASE + BOSS_R + 0.15, BEAM_Z[0] - 0.15, BEAM_Z[1] + 1.0)
    return rear + beam + seat - notch


@lru_cache(maxsize=None)
def beam_part() -> Manifold:
    """前の横棒。両端にキングピンのボス(縦の穴)。"""
    bar = hull2(circle(BOSS_R, -KINGPIN_X, 0), circle(BOSS_R, KINGPIN_X, 0)).extrude(BEAM_Z[1] - BEAM_Z[0]).translate((0, WHEELBASE, BEAM_Z[0]))
    for sx in (1, -1):
        bar -= cyl(d.STUD_R + HOLE_FIT, BEAM_Z[0] - 1, BEAM_Z[1] + 1).translate((sx * KINGPIN_X, WHEELBASE, 0))
    return bar


@lru_cache(maxsize=None)
def knuckle_part() -> Manifold:
    """右前のナックル(ローカル: キングピンの軸 = Z、前輪の軸の中心 = (WHEEL 内面側, 0, 0))。

    ボスの上に乗る腕から、下向きのキングピン(C クリップの溝つき)がボスを貫く。外の縦板から前輪の軸が出て、
    上へ伸びた支柱がサイクルフェンダーを支える。
    """
    kx = KINGPIN_X
    arm = hull2(circle(BOSS_R, 0, 0), circle(3.0, UPRIGHT_X[1] - kx - 3.0, 0)).extrude(KNUCKLE_Z[1] - KNUCKLE_Z[0]).translate((0, 0, KNUCKLE_Z[0]))
    upright = box(UPRIGHT_X[0] - kx, UPRIGHT_X[1] - kx, -5.0, 5.0, -8.0, KNUCKLE_Z[1])
    strut = box(UPRIGHT_X[0] - kx, UPRIGHT_X[1] - kx, -4.0, 4.0, KNUCKLE_Z[1] - 1.0, FENDER_R - 0.1)
    stub = cyl_x(STUB_R, UPRIGHT_X[1] - kx - 0.01, WHEEL_X0 - kx + FRONT_BORE_X1 - 0.2)
    # 先は細い溝つきピン(walker と同じ太さ)。ホイールの壁を貫き、外で C クリップが抜けを止める
    stub += apply(d.grooved_pin(FRONT_BORE_X1 - 0.2, FRONT_CLIP_CLEAR), AXIS_Z_TO_X).translate((WHEEL_X0 - kx, 0, 0))
    # キングピン: ボスの下面の先に C クリップの溝
    pin = d.grooved_pin(0.0, (KNUCKLE_Z[0] - BEAM_Z[0])).mirror((0, 0, 1)).translate((0, 0, KNUCKLE_Z[0]))
    pin = pin.trim_by_plane((0, 0, -1), -KNUCKLE_Z[0] - 0.01)
    # タイロッドの腕: 真前へ伸ばし、先に上向きの溝つきピン。左右の腕とタイロッドで平行四辺形になる(左右の切れ角は同じ)
    steer_arm = hull2(circle(3.5, 0, 0), circle(3.5, 0, TIE_ARM)).extrude(STEER_ARM_T).translate((0, 0, KNUCKLE_Z[0]))
    steer_arm += d.grooved_pin(KNUCKLE_Z[0] + STEER_ARM_T, TIE_PIN_CLEAR).translate((0, TIE_ARM, 0))
    return arm + upright + strut + stub + pin + steer_arm


@lru_cache(maxsize=None)
def knuckle_right_part() -> Manifold:
    """右前のナックル。左と同じ形に、引き棒をつなぐ内向きの腕(ドラッグアーム)を足す。"""
    drag = hull2(circle(3.5, 0, 0), circle(3.5, -DRAG_ARM, 0)).extrude(KNUCKLE_Z[1] - KNUCKLE_Z[0]).translate((0, 0, KNUCKLE_Z[0]))
    drag += d.grooved_pin(KNUCKLE_Z[1], DRAG_PIN_CLEAR).translate((-DRAG_ARM, 0, 0))
    return knuckle_part() + drag


FRONT_CLIP_CLEAR = FRONT_WALL_X1 + 0.1 - (FRONT_BORE_X1 - 0.2)  # ピンの根元から溝まで


def _hex_bore_x(af: float, x0: float, x1: float) -> Manifold:
    return apply(hexagon(af).extrude(x1 - x0), AXIS_Z_TO_X).translate((x0, 0, 0))


def _wheel(bore: CrossSection | None, inner_hub: bool) -> Manifold:
    """ホイール(ローカル: 軸 = X、内面 x = 0、外へ +X)。皿形で、中心に赤いキャップの穴。"""
    w = TIRE_W
    rim = cyl_x(RIM_R, 0, w) - cyl_x(RIM_R - 2.0, 1.0, w + 1)
    ring = cyl_x(RIM_R + 1.5, -1, 1) - cyl_x(RIM_R - 1.0, -2, 2)
    flange = ring.translate((0.0, 0, 0)).trim_by_plane((1, 0, 0), 0).trim_by_plane((-1, 0, 0), -1.2) + \
        ring.translate((w - 0.6, 0, 0)).trim_by_plane((1, 0, 0), w - 1.2).trim_by_plane((-1, 0, 0), -w)  # タイヤの横ずれ止め(輪)
    # 皿は外の面から 1.5 mm 下げる。ハブは外の面まで通す(外の面を下にして印刷するとき、皿の下面がリムとハブの間の橋になる)
    dish = cyl_x(RIM_R - 1.0, w - 3.5, w - 1.5) + cyl_x(HUB_R + 2.0, 0, w)
    for k in range(8):  # 飾りの穴(スポーク風)
        a = 2 * math.pi * k / 8
        dish -= cyl_x(2.4, w - 4, w, 15.0 * math.cos(a), 15.0 * math.sin(a))
    wheel = rim + flange + dish
    if inner_hub:
        wheel += cyl_x(HUB_R, HUB_X0 - WHEEL_X0, 0.01)
    if bore is not None:
        wheel -= apply(bore.extrude(80), AXIS_Z_TO_X).translate((-40, 0, 0))
    wheel -= cyl_x(CAP_POCKET_R, FRONT_WALL_X1 if bore is None else w - 3.0, w + 1)  # キャップの穴
    return wheel


@lru_cache(maxsize=None)
def wheel_rear_part() -> Manifold:
    """後輪。六角穴は大半を元のすきま(8.3)で通し、外寄り TIGHT_X だけ 7.9 で圧入する。圧入部の入口は面取り。"""
    wheel = _wheel(None, True)
    loose = d.SHAFT_AF + d.SHAFT_BORE_CLEARANCE
    lead = Manifold.batch_hull([_hex_bore_x(WHEEL_SOCKET_AF + 2 * WHEEL_SOCKET_LEAD, TIGHT_X[0] - 0.01, TIGHT_X[0]),
                                _hex_bore_x(WHEEL_SOCKET_AF, TIGHT_X[0] + WHEEL_SOCKET_LEAD, TIGHT_X[0] + WHEEL_SOCKET_LEAD + 0.01)])
    return wheel - _hex_bore_x(loose, -20.0, TIGHT_X[0]) - lead - _hex_bore_x(WHEEL_SOCKET_AF, TIGHT_X[0], TIGHT_X[1] + 5.0)


@lru_cache(maxsize=None)
def wheel_front_part() -> Manifold:
    """前輪。軸受け穴(半径 STUB_R + HOLE_FIT)の奥に壁があり、溝つきピンの穴だけ通す。"""
    wheel = _wheel(None, False)
    return wheel - cyl_x(STUB_R + HOLE_FIT, -1.0, FRONT_BORE_X1) - cyl_x(d.STUD_R + HOLE_FIT, 0.0, FRONT_WALL_X1 + 1.0)


@lru_cache(maxsize=None)
def tire_part() -> Manifold:
    """タイヤ(黒)。外周は丸く、中央に O リング(太さ 2 mm)の溝。ホイールのつばの間にはまる。"""
    w = TIRE_W - 2.6
    pts = [(RIM_R + 0.1, 0.0), (WHEEL_R - 1.5, 0.0), (WHEEL_R, 1.5), (WHEEL_R, w - 1.5), (WHEEL_R - 1.5, w), (RIM_R + 0.1, w)]
    ring = polygon([(r, x) for r, x in pts]).revolve(96)
    ring = ring.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64))
    groove = cyl_x(WHEEL_R + 1, w / 2 - 1.1, w / 2 + 1.1) - cyl_x(WHEEL_R - 1.4, w / 2 - 2, w / 2 + 2)
    return (ring - groove).translate((1.3, 0, 0))


@lru_cache(maxsize=None)
def hubcap_part() -> Manifold:
    """ホイールの中心の赤いキャップ(差し込み)。"""
    plug = cyl_x(CAP_POCKET_R - 0.2, -3.0, 0.01) - cyl_x(d.CLIP_R_OUT + 0.3, -4.0, -0.6)  # 筒の中に C クリップとピンの頭が入る
    dome = Manifold.sphere(7.0, 48).scale((0.45, 1, 1)).trim_by_plane((1, 0, 0), 0)
    return plug + dome


# ---------------------------------------------------------------- 外装(赤)
def _rear_points(shrink_top: float = 0.0):
    r = 14.0
    pts = []
    for x in (BODY_HALF_W - r,):
        for y in (BODY_REAR_Y + r + 6.0, BODY_SPLIT_Y + 12.0):
            pts += [(x, y, BODY_BOTTOM_Z + r), (x, y, DECK_TOP_Z - r)]
    pts += [(BODY_HALF_W - r - 6.0, BODY_REAR_Y + r, DECK_TOP_Z - r - 6.0), (BODY_HALF_W - r - 6.0, BODY_REAR_Y + r, BODY_BOTTOM_Z + r + 4.0)]
    return mirror_x(pts), r


@lru_cache(maxsize=None)
def body_rear_part() -> Manifold:
    """後ろのボディ。駆動部を包む殻で、上に運転席の開口、横に後輪の軸の逃げ。"""
    pts, r = _rear_points()
    body = shell(pts, r).trim_by_plane((0, -1, 0), -BODY_SPLIT_Y)
    body = body.trim_by_plane((0, 0, 1), BODY_BOTTOM_Z + SKIN + 0.2)  # 底は開ける
    c = COCKPIT
    opening = hull2(circle(c["radius"], -c["half_w"] + c["radius"], c["y0"] + c["radius"]), circle(c["radius"], c["half_w"] - c["radius"], c["y0"] + c["radius"]),
                    rect(-c["half_w"], c["half_w"], c["y0"] + c["radius"], BODY_SPLIT_Y + 5))
    body -= opening.extrude(40).translate((0, 0, 30.0))
    # 後輪の上の切り欠き(フェンダーが上を覆う)。軸とハブを通す
    for sx in (1, -1):
        body -= cyl_x(FENDER_R - 1.0, sx * 40.0 if sx > 0 else -BODY_HALF_W - 5, BODY_HALF_W + 5 if sx > 0 else -40.0)
    # ドアの溝(前ヒンジでなく後ろヒンジ。下は後輪のフェンダーに隠れる)と、前寄りの取っ手
    y0, y1, z0, z1 = DOOR_Y[0], DOOR_Y[1], 2.0, DOOR_TOP_Z
    outline = CrossSection.square((y1 - y0 - 8, z1 - z0 - 8)).translate((y0 + 4, z0 + 4)).offset(4.0, d.JoinType.Round)
    ring = outline.offset(0.7) - outline.offset(-0.7)
    cutter = plate_yz(ring, 40.0, 70.0)
    cutter += cutter.mirror((1, 0, 0))
    body = engrave(body, cutter, blob(pts, r), blob(pts, r, ENGRAVE))
    for sx in (1, -1):
        body += Manifold.batch_hull([_ball(sx * (BODY_HALF_W - 0.5), y, DOOR_HANDLE_Z, 1.3) for y in (y1 - 9.0, y1 - 3.0)])
    return body - rails_clearance()


def _front_points():
    r_c, r_h = 14.0, 11.0
    cowl = mirror_x([(BODY_HALF_W - r_c, y, z) for y in (BODY_SPLIT_Y - 8.0, COWL_Y1 - r_c + 6.0) for z in (BODY_BOTTOM_Z + r_c, DECK_TOP_Z - r_c)])
    hood = mirror_x([(HOOD_HALF_W - r_h, BODY_SPLIT_Y, HOOD_BOTTOM_Z + r_h), (HOOD_HALF_W - r_h, BODY_SPLIT_Y, DECK_TOP_Z - 2.0 - r_h),
                     (HOOD_HALF_W - r_h - 2.0, HOOD_NOSE_Y - r_h, HOOD_BOTTOM_Z + r_h + 2.0), (HOOD_HALF_W - r_h - 4.0, HOOD_NOSE_Y - r_h - 2.0, DECK_TOP_Z - 10.0 - r_h)])
    return (cowl, r_c), (hood, r_h)


@lru_cache(maxsize=None)
def body_front_part() -> Manifold:
    """前のボディ: 風防の下の幅広い部分(カウル)と、前輪の間の細いボンネット。"""
    (cowl, r_c), (hood, r_h) = _front_points()
    outer = blob(cowl, r_c) + blob(hood, r_h)
    inner = blob(cowl, r_c, SKIN) + blob(hood, r_h, SKIN)
    body = (outer - inner).trim_by_plane((0, 1, 0), BODY_SPLIT_Y)
    body = body.trim_by_plane((0, 0, 1), BODY_BOTTOM_Z + SKIN + 0.2)  # 底は開ける
    # ボンネットの下は横棒とナックルの腕を逃がす
    body -= box(-HOOD_HALF_W - 1, HOOD_HALF_W + 1, COWL_Y1 + 2.0, HOOD_NOSE_Y + 5, BODY_BOTTOM_Z - 1, HOOD_BOTTOM_Z)
    # 引き棒がカウルの前の壁を抜けるすき間(前輪の陰)と、タイロッドがボンネットの下を通る切り欠き
    body -= box(LINK_X[0] - 1.0, LINK_X[1] + 1.5, 50.0, 78.0, 10.0, 40.0)
    body -= box(-HOOD_HALF_W - 1, HOOD_HALF_W + 1, WHEELBASE + TIE_ARM - 8.0, WHEELBASE + TIE_ARM + 8.0, BODY_BOTTOM_Z - 1, TIE_Z[1] + 1.5)
    # 運転席側は開ける(後ろのボディの開口とつなぐ)
    c = COCKPIT
    body -= box(-c["half_w"], c["half_w"], BODY_SPLIT_Y - 20, BODY_SPLIT_Y + 1.0, 30.0, 80.0)
    # グリルとヘッドライトは、穴にはめる(穴の形 = 部品の形)
    body -= apply(grille_outline(), GRILLE_POSE)
    for pose in HEADLIGHT_POSES:
        body -= apply(headlight_part(), pose)
    # 空気取り(ボンネットの横の縦溝)と、ボンネットの中央の合わせ目
    cutter = Manifold()
    for y in LOUVER_Y:
        slot = hull2(circle(0.8, y, LOUVER_Z[0] + 0.8), circle(0.8, y, LOUVER_Z[1] - 0.8))
        cutter += plate_yz(slot, 12.0, 40.0)
    cutter += cutter.mirror((1, 0, 0))
    cutter += box(-0.5, 0.5, COWL_Y1 + 8.0, HOOD_NOSE_Y - 10.0, 25.0, 60.0)
    inner = blob(cowl, r_c, ENGRAVE) + blob(hood, r_h, ENGRAVE)
    body = engrave(body, cutter, outer, inner)
    for sx in (1, -1):
        body -= cyl(MIRROR_PEG_R + HOLE_FIT, 20.0, 60.0).translate((sx * MIRROR_XY[0], MIRROR_XY[1], 0))
    return body - rails_clearance()


@lru_cache(maxsize=None)
def rails_clearance() -> Manifold:
    """左右の側板を 0.3 mm 太らせた形。ボディの底の丸みが側板に当たるところを、これで逃がす。"""
    grow = Manifold.cube((0.6, 0.6, 0.6), True)
    rail = rail_part().minkowski_sum(grow)
    return rail + rail.mirror((1, 0, 0))


def body_rear_outer() -> Manifold:
    pts, r = _rear_points()
    return blob(pts, r)


def body_front_outer() -> Manifold:
    (cowl, r_c), (hood, r_h) = _front_points()
    return blob(cowl, r_c) + blob(hood, r_h)


@lru_cache(maxsize=None)
def fender_rear_part() -> Manifold:
    """右後ろのフェンダー(ローカル: 後輪の軸 = X 軸、原点 = 軸心)。"""
    section = _crowned(*FENDER_X, FENDER_R, FENDER_T, 2.0)
    return ring_sector_x(section, *REAR_FENDER_DEG)


@lru_cache(maxsize=None)
def fender_front_part() -> Manifold:
    """右前のサイクルフェンダー(ローカル: ナックルと同じ。キングピンの軸 = Z)。"""
    section = _crowned(*FENDER_X, FENDER_R, FENDER_T, 2.0)
    return ring_sector_x(section, *FRONT_FENDER_DEG).translate((-KINGPIN_X, 0, 0))


# ---------------------------------------------------------------- 小物(グレー)
@lru_cache(maxsize=None)
def windshield_part() -> Manifold:
    """風防の枠(ガラスは入れない)。カウルの上に立てる。"""
    half, z0, z1 = 40.0, DECK_TOP_Z - 3.0, DECK_TOP_Z + 14.0
    frame = CrossSection.square((2 * half, z1 - z0)).translate((-half, z0)).offset(-0.01)
    frame = frame.offset(3.0, d.JoinType.Round) - CrossSection.square((2 * half - 6, z1 - z0 - 4)).translate((-half + 3, z0 + 3)).offset(2.0, d.JoinType.Round)
    plate = frame.extrude(2.4)
    plate = plate.transform(np.array([[1, 0, 0, 0], [0, 0, 1, BODY_SPLIT_Y + 1.0], [0, 1, 0, 0]], dtype=np.float64))
    return plate - body_front_outer()  # 根元をカウルの曲面に合わせる


@lru_cache(maxsize=None)
def top_part() -> Manifold:
    """たたんだ幌。後ろの甲板に載せる。"""
    y, z = BODY_REAR_Y + 14.0, DECK_TOP_Z + 4.0
    roll = Manifold.batch_hull([_ball(sx * 40.0, y, z, 8.0) for sx in (1, -1)] + [_ball(sx * 36.0, y - 6.0, z - 2.0, 6.5) for sx in (1, -1)])
    for x in (-24.0, 0.0, 24.0):  # たたんだ骨
        roll -= box(x - 0.6, x + 0.6, y - 20, y + 20, z + 5.5, z + 20)
    return roll.trim_by_plane((0, 0, 1), DECK_TOP_Z - 3.0) - body_rear_outer()  # 下面を甲板の曲面に合わせる


@lru_cache(maxsize=None)
def bumper_part() -> Manifold:
    """前のバンパー(後ろは同じ部品を Z まわり 180°)。両端が少し後ろへ回り込む。"""
    pts = [(sx * 42.0, 0.0, 0.0) for sx in (1, -1)]
    bar = Manifold.batch_hull([_ball(x, y, z, 3.2) for x, y, z in pts])
    for sx in (1, -1):
        bar += Manifold.batch_hull([_ball(sx * 42.0, 0, 0, 3.2), _ball(sx * 45.0, -6.0, 0, 3.2)])
    return bar


def _shield(grow: float = 0.0) -> CrossSection:
    """グリルの外形((x, z) 平面、中心が原点)。上が広く、下へすぼまる縦長の盾。"""
    shape = hull2(circle(4.5, -8.0, 11.5), circle(4.5, 8.0, 11.5), circle(4.0, -6.5, -9.0), circle(4.0, 6.5, -9.0), circle(2.5, 0.0, -14.5))
    return shape.offset(grow, d.JoinType.Round) if grow else shape


_XZ_TO_XY = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)  # (x, y, z) -> (x, z, y): 押し出しの向き Z を前(+Y)へ


def _shield_prism(y0: float, y1: float, grow: float = 0.0) -> Manifold:
    return _shield(grow).extrude(y1 - y0).translate((0, 0, y0)).transform(_XZ_TO_XY)


def grille_outline() -> Manifold:
    """グリルの穴(ボディから抜く形)。"""
    return _shield_prism(-12.0, 4.0)


@lru_cache(maxsize=None)
def grille_part() -> Manifold:
    """鼻先のラジエーターシェル。後ろはボンネットに差し込む柱、前は太い枠と縦の桟、上に飾りの紋章と給水口のキャップ。"""
    plug = _shield_prism(-11.0, 0.0)
    frame = _shield_prism(0.0, 3.0) - _shield_prism(1.0, 4.0, -2.2)
    bead = _shield_prism(3.0, 3.6, -0.3) - _shield_prism(2.9, 3.7, -1.3)  # 枠の前の細い縁取り
    slats = Manifold()
    for k in range(-4, 5):
        slats += box(2.4 * k - 0.45, 2.4 * k + 0.45, 0.0, 2.4, -20.0, 20.0)
    slats ^= _shield_prism(0.0, 2.4, -2.0)
    crest = Manifold.cylinder(3.6, 3.2, 3.2, 48).transform(_XZ_TO_XY).translate((0, 0.4, 12.0))  # 紋章(上の枠に重ねた円板)
    crest -= Manifold.cylinder(1.0, 2.2, 2.2, 48).transform(_XZ_TO_XY).translate((0, 3.2, 12.0))
    cap = cyl(2.0, 14.0, 18.0).translate((0, -3.0, 0)) + Manifold.sphere(2.4, 32).scale((1, 1, 0.5)).translate((0, -3.0, 18.0))
    cap -= apply(body_front_outer(), np.linalg.inv(GRILLE_POSE))  # 給水口の根元をボンネットに沿わせる
    return plug + frame + bead + slats + crest + cap


@lru_cache(maxsize=None)
def headlight_part() -> Manifold:
    """丸いヘッドライト(前を向いた殻と、レンズのくぼみ)。"""
    to_y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)
    pod = Manifold.sphere(7.5, 48).scale((1, 1.1, 1))
    pod -= Manifold.cylinder(4, 5.6, 5.6, 48).transform(to_y).translate((0, 6.2, 0))
    stalk = Manifold.cylinder(12.0, 2.2, 2.2, 32).transform(to_y).translate((0, -12.0, 0))  # 後ろへ伸びてボンネットの穴に差す
    rim = Manifold.revolve(circle(0.9, 6.3, 0.0), 64).transform(to_y).translate((0, 6.0, 0))  # レンズの縁
    return pod + stalk + rim


MIRROR_PEG_R = 1.5


def _surface_z(x: float, y: float) -> float:
    """前のボディの外面の、(x, y) での上面の高さ。"""
    probe = body_front_outer() ^ cyl(0.05, 0.0, 80.0, 8).translate((x, y, 0))
    return probe.bounding_box()[5]


@lru_cache(maxsize=None)
def mirror_part() -> Manifold:
    """右のフェンダーミラー(設計の座標のまま)。細い柱が外へ傾いて立ち、丸い鏡が後ろを向く。根元の足はボンネットに沿い、下の棒を穴に差す。"""
    x, y = MIRROR_XY
    z = _surface_z(x, y)
    foot = cyl(3.0, z - 3.0, z + 1.2).translate((x, y, 0)) - body_front_outer()
    peg = cyl(MIRROR_PEG_R, z - 5.0, z).translate((x, y, 0))
    stalk = Manifold.batch_hull([_ball(x, y, z + 0.5, 1.5), _ball(x + 3.0, y, z + 7.0, 1.2)])
    head = Manifold.sphere(5.0, 48).scale((1.0, 0.6, 0.85)).translate((x + 4.5, y, z + 10.5))
    head -= Manifold.sphere(4.0, 48).scale((1.0, 0.3, 0.8)).translate((x + 4.5, y - 2.6, z + 10.5))  # 後ろ向きの鏡面のくぼみ
    return foot + peg + stalk + head


MOLDING_T = 0.8  # モールの厚み
MOLDING_W = 2.0
MOLDING_Z = 40.5  # 後ろのボディと風防下の横のモールの下端(後輪のフェンダーの外面 39.5 の上)
HOOD_MOLDING_Y = (68.0, 132.0)
HOOD_MOLDING_Z = (40.0, 31.0)  # ボンネットの横では、前へ下がる


@lru_cache(maxsize=None)
def molding_body_part() -> Manifold:
    """右の横のモール(グレー、設計の座標のまま)。後ろのボディから風防の下まで、外面に沿って貼る。"""
    rear_pts, r = _rear_points()
    (cowl, r_c), _ = _front_points()
    grown = blob(rear_pts, r, -MOLDING_T) + blob(cowl, r_c, -MOLDING_T)
    band = grown - body_rear_outer() - body_front_outer()
    return band ^ box(30.0, 70.0, BODY_REAR_Y + 10.0, COWL_Y1, MOLDING_Z, MOLDING_Z + MOLDING_W)


@lru_cache(maxsize=None)
def molding_hood_part() -> Manifold:
    """右のボンネットの横のモール(グレー、設計の座標のまま)。前へ向かって下がる。"""
    _, (hood, r_h) = _front_points()
    band = blob(hood, r_h, -MOLDING_T) - body_front_outer()
    (y0, y1), (z0, z1) = HOOD_MOLDING_Y, HOOD_MOLDING_Z
    strip = plate_yz(polygon([(y0, z0), (y1, z1), (y1, z1 + MOLDING_W), (y0, z0 + MOLDING_W)]), 15.0, 60.0)
    return band ^ strip


# ---------------------------------------------------------------- ハットとかじ取り
# ハット(黒、次元大介の帽子のイメージ)は頭(スタックチャンの胴体シェル + CoreS3)に前から差し込み、後ろの止め板で抜け止めする。
# 左右の耳当てはヨークのパッドと同じ中心の円板で、頭とパッドの間に挟まる(うなずいても位置が変わらない)。CoreS3 の横と下は覆わない。
# 頭と一緒にうなずくので、右の腕のピン J は前後に動く。J から引き棒で右前輪のドラッグアームを押し引きし、タイロッドで左前輪も切る。
# 形はすべて頭のローカル座標(机面 z = 0、ピッチ軸は (y, z) = (0, SC_PITCH_AXIS_H))。
HEAD_X = d.SC_HEAD_HALF_W
HEAD_Y = (-29.2, d.SC_FRONT)  # 胴体シェルの背面〜CoreS3 の前面
HEAD_Z = (d.SC_HEAD_BOTTOM, d.SC_HEAD_TOP)
HAT_T = 1.6
HAT_SIDE_FIT, HAT_TOP_FIT = 0.15, 0.2  # 横は詰める(ハットの首振りが J の前後のガタになる)。導光バーは窓と逃げ溝でよける
HEAD_BACK_BAND = -27.2  # 帯の高さ(z 63〜66)での頭の背面(公式 STL の断面から)。下のほうは -29.2 まで出ている
STEER_FIT = 0.15  # かじ取りのピンと穴の半径すきま(HOLE_FIT 0.2 より詰める。2 か所のガタが切れ角の遊びになる)
BRIM_W = 12.0
CROWN_Z0 = 62.0  # 帽子のふち(画面の表示域の上端 60.7 より上)
LED_WINDOW = dict(y=(-17.5, 12.5), z=(66.3, HEAD_Z[1] + HAT_TOP_FIT))  # 導光バー(前から 6〜34 mm、上面から 1.2〜3.2 mm)を見せる窓
EAR_DISC_R, EAR_DISC_T = 12.5, 1.2  # 耳当て: ヨークのパッド(半径 13)と同じ中心の円板。頭とパッドの間に挟まる
EAR_DOME = (2.0, 0.6)  # 耳当ての外の中心のぽっち(底の半径, 高さ)。ヨークのパッドの穴に落ちてハットの浮きを止める
EAR_SLOT_DY = 2.0  # パッドの穴を前後に ±2 mm 長くする(頭の前後の位置ずれを逃がす)
LOWER_STRAP_Y = (10.0, 16.0)  # 耳当てから腕へ下るひも。ヨークの腕(|y| ≤ 8)と CoreS3(y ≥ 17.5)の間
UPPER_STRAP_Y = (-5.0, 5.0)  # 耳当てから帯へ上がるひも
PORT_WINDOW = dict(y=(18.5, 32.0), z=(CROWN_Z0 - 2.0, 68.5))  # CoreS3 の横(電源ボタンなど)を帯の壁から出す窓
ARM_X = (31.3, 33.8)  # 腕の内面はヨークの腕の外面(30.6)の外
ARM_Z = (25.5, 28.5)  # ヨークのパッド(ピッチ軸から半径 13)より下、ヨークの腕の斜面より上
J_LOCAL = (0.0, 27.0)  # 右の腕のピンの (y, z)。ピッチ軸の真下 16 mm なので、うなずくと主に前後へ動く
LINK_X = (ARM_X[1] + 1.6, ARM_X[1] + 1.6 + 2.5)  # 引き棒の板の左右(ヨークの腕の斜面から離す)
J_PIN_CLEAR = LINK_X[1] + 0.2 - ARM_X[1]
LINK_KNEE = (30.0, 48.5)  # 引き棒の折れ目 (y, z)。カラー(半径 44.7、上面 36.4)の外
NOD_RANGE = (0.0, 30.0)  # 使ううなずきの範囲 [deg](ファームの上限は 60。車ではこの範囲に制限する)


def _rounded(x: float, y0: float, y1: float, r: float) -> CrossSection:
    return rect(-x + r, x - r, y0 + r, y1 - r).offset(r, d.JoinType.Round, 2.0, 48)


def _layer(section: CrossSection, z: float) -> Manifold:
    return section.extrude(0.01).translate((0, 0, z))


@lru_cache(maxsize=None)
def hat_part() -> Manifold:
    """ハット(頭のローカル座標)。クラウン・つば・帯・LED と CoreS3 の窓・耳当て・左右の腕と、右の腕のピン J。"""
    xo, yb, yf = HEAD_X + HAT_SIDE_FIT + HAT_T, HEAD_Y[0] - 0.8 - HAT_T, HEAD_Y[1] + HAT_TOP_FIT + HAT_T
    top_z = HEAD_Z[1] + HAT_TOP_FIT
    base = _rounded(xo, yb, yf, 2.0)
    mid = _rounded(xo - 1.5, yb + 2.0, yf - 2.0, 9.0)
    crown_top = _rounded(xo - 7.0, yb + 6.0, yf - 6.0, 14.0)
    outer = Manifold.batch_hull([_layer(base, CROWN_Z0), _layer(base, top_z + 0.5), _layer(mid, top_z + 5.5), _layer(crown_top, top_z + 10.0)])
    # 中折れ(てっぺんの前後のへこみ)と、前のつまみ
    outer -= Manifold.sphere(1.0, 64).scale((5.5, 26.0, 3.0)).translate((0, 3.0, top_z + 12.0))
    for sx in (1, -1):
        outer -= Manifold.sphere(1.0, 48).scale((5.0, 7.0, 4.0)).translate((sx * 16.0, yf - 5.0, top_z + 10.5))
    hollow = Manifold.batch_hull([_layer(base.offset(-HAT_T), top_z + HAT_T), _layer(mid.offset(-HAT_T), top_z + 4.0),
                                  _layer(crown_top.offset(-HAT_T), top_z + 6.0)])
    head = box(-HEAD_X - HAT_SIDE_FIT, HEAD_X + HAT_SIDE_FIT, yb - 5.0, HEAD_Y[1] + HAT_TOP_FIT, CROWN_Z0 - 1.0, top_z)  # 後ろは開けて前から差し込む
    crown = outer - hollow - head
    # 帯
    crown += (base.offset(0.6, d.JoinType.Round) - base).extrude(3.5).translate((0, 0, CROWN_Z0))
    # LED の窓
    for sx in (1, -1):
        crown -= box(sx * (HEAD_X - 1.0), sx * (xo + 2.0), *LED_WINDOW["y"], *LED_WINDOW["z"]) if sx > 0 else box(-xo - 2.0, -HEAD_X + 1.0, *LED_WINDOW["y"], *LED_WINDOW["z"])
    # 後ろの止め板(hat_back_part)が下から差さる溝。板が頭の背面を押さえ、ハットが前へ抜けない
    crown -= box(-xo - 1.0, xo + 1.0, *BACK_SLOT_Y, CROWN_Z0 - 2.0, BACK_PLATE_Z[1])
    # 前から差し込むとき導光バー(|x| 27.25)をこすらないよう、窓より前の壁の内面に逃げ溝
    for sx in (1, -1):
        relief = box(HEAD_X - 0.5, HEAD_X + 0.45, LED_WINDOW["y"][1] - 1.0, yf - HAT_T + 0.01, 67.0, 69.6)
        crown -= relief if sx > 0 else relief.mirror((1, 0, 0))
    # つば: 前は少し下げ(目深)、横は反り上げる
    brim = ((base.offset(BRIM_W, d.JoinType.Round) - base.offset(-0.01)).extrude(1.6).translate((0, 0, CROWN_Z0 - 1.6))).refine(8)

    def _bend(v):
        x, y, z = v
        z += 0.022 * max(0.0, abs(x) - xo) ** 2 + 0.035 * max(0.0, yb - y) ** 2  # 横と後ろは反り上げる
        z -= 0.28 * max(0.0, y - (yf - 4.0))  # 前は目深に下げる
        return (x, y, z)

    brim = brim.warp(_bend)
    # CoreS3 の横を帯の壁から出す窓
    crown -= box(HEAD_X - 1.0, xo + 2.0, *PORT_WINDOW["y"], *PORT_WINDOW["z"])
    crown -= box(-xo - 2.0, -HEAD_X + 1.0, *PORT_WINDOW["y"], *PORT_WINDOW["z"])
    # 耳当て・ひも・腕(右。左は鏡写し)
    ax_z = d.SC_PITCH_AXIS_H
    ex0, ex1 = HEAD_X + HAT_SIDE_FIT, HEAD_X + HAT_SIDE_FIT + EAR_DISC_T
    base_r, dome_h = EAR_DOME
    dome_rs = (base_r ** 2 + dome_h ** 2) / (2 * dome_h)
    dome = Manifold.sphere(dome_rs, 48).translate((ex1 + dome_h - dome_rs, 0, ax_z)) ^ box(ex1 - 0.01, ex1 + dome_h, -base_r, base_r, ax_z - base_r, ax_z + base_r)
    ear = cyl_x(EAR_DISC_R, ex0, ex1, 0, ax_z, 96) + dome
    ear += box(ex0, ex1, *UPPER_STRAP_Y, ax_z, CROWN_Z0 + 0.5)
    ear += box(ex0, ex1, *LOWER_STRAP_Y, ARM_Z[0], ax_z)
    ear += box(ex0, ARM_X[1], *LOWER_STRAP_Y, *ARM_Z) + box(ARM_X[0], ARM_X[1], J_LOCAL[0] - 4.0, LOWER_STRAP_Y[1], *ARM_Z)
    ear += cyl_x(1.5, ARM_X[0], ARM_X[1], J_LOCAL[0] - 4.0, (ARM_Z[0] + ARM_Z[1]) / 2)  # 後ろ端を丸める
    hat = crown + brim + ear + ear.mirror((1, 0, 0))
    pin = apply(d.grooved_pin(0.0, J_PIN_CLEAR), AXIS_Z_TO_X).translate((ARM_X[1], J_LOCAL[0], J_LOCAL[1]))
    return hat + pin


@lru_cache(maxsize=None)
def yoke_car_part() -> Manifold:
    """車用のヨーク(グレー、設計の座標)。walker のヨークのパッドの内側を削って耳当ての入る場所を作り、ぽっちの穴をあける。"""
    yoke = d.yoke_part()
    x_in = HEAD_X + HAT_SIDE_FIT + EAR_DISC_T + 0.2  # 耳当てとのすきま 0.2
    z = d.PITCH_Z
    slot_r = EAR_DOME[0] + 0.2
    for sx in (1, -1):
        cut = cyl_x(EAR_DISC_R + 0.8, HEAD_X - 1.0, x_in, 0, z, 96)
        cut += Manifold.batch_hull([cyl_x(slot_r, x_in - 1.0, x_in + 6.0, dy, z, 48) for dy in (-EAR_SLOT_DY, EAR_SLOT_DY)])
        yoke -= cut if sx > 0 else cut.mirror((1, 0, 0))
    return yoke


BACK_PLATE_Y = (HEAD_BACK_BAND - 0.2 - 1.2, HEAD_BACK_BAND - 0.2)
BACK_SLOT_Y = BACK_PLATE_Y  # 溝と板は同じ厚み(軽い圧入で落ちない)
BACK_PLATE_Z = (CROWN_Z0 + 0.2, 68.8)  # 上端は頭の背面の上の張り出し(z 69 から)の下


@lru_cache(maxsize=None)
def hat_back_part() -> Manifold:
    """ハットの後ろの止め板(黒、頭のローカル座標)。頭を差し込んだあと、横の壁の溝へ下から差す。"""
    x = HEAD_X + HAT_SIDE_FIT + HAT_T - 0.2
    return box(-x, x, *BACK_PLATE_Y, *BACK_PLATE_Z)


def _head_pose(theta: float) -> np.ndarray:
    nod = translation(0, 0, d.SC_PITCH_AXIS_H) @ rot_x(theta) @ translation(0, 0, -d.SC_PITCH_AXIS_H)
    return translation(0, 0, d.TABLE_Z1) @ nod


def j_world(theta: float) -> np.ndarray:
    """引き棒の後ろの目(右の腕のピン J、引き棒の板の厚みの中央)の位置。"""
    return (_head_pose(theta) @ np.array([(LINK_X[0] + LINK_X[1]) / 2, J_LOCAL[0], J_LOCAL[1], 1.0]))[:3]


def k_world(delta: float) -> np.ndarray:
    """引き棒の前の目(右前のドラッグアームのピン)の位置。delta: 前輪の切れ角(正で左)。"""
    return np.array([KINGPIN_X - DRAG_ARM * math.cos(delta), WHEELBASE - DRAG_ARM * math.sin(delta), sum(LINK_TAB_Z) / 2])


def _solve_delta(theta: float, length: float) -> float:
    lo, hi = math.radians(-70), math.radians(70)  # 距離は delta について単調(K が前へ出るほど J から遠い)
    j = j_world(theta)
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.linalg.norm(k_world(mid) - j) > length:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=None)
def steering_geometry() -> tuple[float, float]:
    """(直進のうなずき角 [rad]、引き棒の目の間隔 [mm])。うなずきの両端で左右の切れ角が対称になる直進角を選ぶ。"""
    t0, t1 = (math.radians(v) for v in NOD_RANGE)
    lo, hi = t0, t1
    for _ in range(50):
        mid = (lo + hi) / 2
        length = float(np.linalg.norm(k_world(0.0) - j_world(mid)))
        if _solve_delta(t0, length) + _solve_delta(t1, length) > 0:  # 直進の角度を上げると、うなずき 0 のときの切れ角が増える
            hi = mid
        else:
            lo = mid
    neutral = (lo + hi) / 2
    return neutral, float(np.linalg.norm(k_world(0.0) - j_world(neutral)))


def fit_drag_arm() -> float:
    """NOD_RANGE の両端で切れ角がちょうど ±STEER_MAX になるドラッグアームの長さを求め、DRAG_ARM に入れる。"""
    global DRAG_ARM
    lo, hi = 4.0, 20.0
    for _ in range(40):
        DRAG_ARM = (lo + hi) / 2
        steering_geometry.cache_clear()
        if _solve_delta(math.radians(NOD_RANGE[0]), steering_geometry()[1]) > STEER_MAX:
            lo = DRAG_ARM
        else:
            hi = DRAG_ARM
    DRAG_ARM = round(DRAG_ARM, 1)
    steering_geometry.cache_clear()
    return DRAG_ARM


def steer_from_nod(theta: float) -> float:
    """うなずき角 [rad] から前輪の切れ角 [rad](正で左)。"""
    return _solve_delta(theta, steering_geometry()[1])


@lru_cache(maxsize=None)
def link_part() -> Manifold:
    """引き棒(直進の姿勢で、設計の座標のまま)。後ろは J の横穴、YZ 面の板で前へ下り、ドラッグアームの上で水平の板になって縦穴。"""
    neutral, _ = steering_geometry()
    j, k = j_world(neutral), k_world(0.0)
    m_y = k[1] - 8.0
    # 根元はヨークの腕の斜面とカラーの上を高いまま越え、カラーの先(LINK_KNEE)から前へ下りる
    knee = (LINK_KNEE[0], LINK_KNEE[1])
    bar = plate_yz(hull2(circle(3.4, j[1], j[2]), circle(3.4, *knee)) + hull2(circle(3.4, *knee), circle(3.5, m_y, k[2])), *LINK_X)
    tab = hull2(circle(3.5, j[0], m_y), circle(4.0, k[0], k[1])).extrude(LINK_TAB_Z[1] - LINK_TAB_Z[0]).translate((0, 0, LINK_TAB_Z[0]))
    link = (bar + tab).trim_by_plane((0, 0, 1), LINK_TAB_Z[0])  # 水平の板の下面より下を落とし、板を下にして印刷できるようにする
    link -= cyl_x(d.STUD_R + STEER_FIT, LINK_X[0] - 1, LINK_X[1] + 1, j[1], j[2])
    link -= cyl(d.STUD_R + STEER_FIT, LINK_TAB_Z[0] - 1, LINK_TAB_Z[1] + 1).translate((k[0], k[1], 0))
    return link


@lru_cache(maxsize=None)
def tie_rod_part() -> Manifold:
    """タイロッド(直進の姿勢で、設計の座標のまま)。左右のナックルの前向きの腕のピンをつなぐ。"""
    y = WHEELBASE + TIE_ARM
    bar = hull2(circle(4.0, -KINGPIN_X, y), circle(4.0, KINGPIN_X, y)) - circle(d.STUD_R + STEER_FIT, -KINGPIN_X, y) - circle(d.STUD_R + STEER_FIT, KINGPIN_X, y)
    return bar.extrude(TIE_Z[1] - TIE_Z[0]).translate((0, 0, TIE_Z[0]))


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, cos = np.cross(a, b), float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + cos)


def link_pose(theta: float, delta: float) -> np.ndarray:
    neutral, _ = steering_geometry()
    j0, k0, j, k = j_world(neutral), k_world(0.0), j_world(theta), k_world(delta)
    m = np.eye(4)
    m[:3, :3] = _rotation_between(k0 - j0, k - j)
    m[:3, 3] = j - m[:3, :3] @ j0
    return m


fit_drag_arm()

CLIP_UP = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])  # クリップ(軸 X、厚み +X)を、軸 Z、厚み +Z に


GRILLE_POSE = translation(0, HOOD_NOSE_Y - 1.5, 22.0)
HEADLIGHT_POSES = [translation(sx * 23.0, HOOD_NOSE_Y - 5.0, 31.0) for sx in (1, -1)]


# ---------------------------------------------------------------- 部品表と姿勢
@dataclass(frozen=True)
class PartDef:
    key: str
    label: str
    build: object
    color: str
    count: int
    printable: bool = True


def part_defs() -> dict[str, PartDef]:
    c, sc = COLORS, d.SC_COLORS
    defs = [
        PartDef("rail_R", "右の側板", rail_part, c["gray"], 1),
        PartDef("rail_L", "左の側板", lambda: rail_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("deck", "デッキ(walker と同じ)", d.deck_part, c["gray"], 1),
        PartDef("yoke", "ヨーク(車用。耳当ての逃げとぽっちの穴)", yoke_car_part, c["gray"], 1),
        PartDef("gear_table", "ターンテーブル歯車(walker と同じ)", d.gear_table_part, c["gray"], 1),
        PartDef("socket_frame", "ソケット枠(walker と同じ)", d.socket_frame_part, c["gray"], 1),
        PartDef("pinion", "ピニオン(walker と同じ)", d.pinion_part, c["gray"], 1),
        PartDef("sleeve", "スリーブ(walker と同じ)", d.sleeve_part, c["gray"], 1),
        PartDef("shaft", "後ろの車軸(六角)", shaft_part, c["gray"], 1),
        PartDef("beam", "前の横棒", beam_part, c["gray"], 1),
        PartDef("knuckle_R", "右前のナックル", knuckle_right_part, c["gray"], 1),
        PartDef("knuckle_L", "左前のナックル", lambda: knuckle_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("hat", "ハット", hat_part, c["black"], 1),
        PartDef("hat_back", "ハットの後ろの止め板", hat_back_part, c["black"], 1),
        PartDef("link", "かじ取りの引き棒", link_part, c["gray"], 1),
        PartDef("tie_rod", "タイロッド", tie_rod_part, c["gray"], 1),
        PartDef("clip", "C クリップ(walker と同じ)", d.clip_part, c["gray"], 8),
        PartDef("wheel_rear", "後輪のホイール", wheel_rear_part, c["gray"], 2),
        PartDef("wheel_front", "前輪のホイール", wheel_front_part, c["gray"], 2),
        PartDef("hubcap", "ホイールキャップ", hubcap_part, c["red"], 4),
        PartDef("tire", "タイヤ", tire_part, c["black"], 4),
        PartDef("body_rear", "後ろのボディ", body_rear_part, c["red"], 1),
        PartDef("body_front", "前のボディ", body_front_part, c["red"], 1),
        PartDef("fender_rear_R", "右後ろのフェンダー", fender_rear_part, c["black"], 1),
        PartDef("fender_rear_L", "左後ろのフェンダー", lambda: fender_rear_part().mirror((1, 0, 0)), c["black"], 1),
        PartDef("fender_front_R", "右前のフェンダー", fender_front_part, c["black"], 1),
        PartDef("fender_front_L", "左前のフェンダー", lambda: fender_front_part().mirror((1, 0, 0)), c["black"], 1),
        PartDef("molding_body_R", "右の横のモール", molding_body_part, c["gray"], 1),
        PartDef("molding_body_L", "左の横のモール", lambda: molding_body_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("molding_hood_R", "右のボンネットのモール", molding_hood_part, c["gray"], 1),
        PartDef("molding_hood_L", "左のボンネットのモール", lambda: molding_hood_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("mirror_R", "右のフェンダーミラー", mirror_part, c["gray"], 1),
        PartDef("mirror_L", "左のフェンダーミラー", lambda: mirror_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("windshield", "風防の枠", windshield_part, c["gray"], 1),
        PartDef("top", "たたんだ幌", top_part, c["gray"], 1),
        PartDef("bumper", "バンパー", bumper_part, c["gray"], 2),
        PartDef("grille", "グリル", grille_part, c["gray"], 1),
        PartDef("headlight", "ヘッドライト", headlight_part, c["gray"], 2),
        PartDef("sc_base", "スタックチャン土台", d.stackchan_base, sc["base"], 1, printable=False),
        PartDef("sc_servo", "スタックチャン サーボ部", d.stackchan_servo_body, sc["base"], 1, printable=False),
        PartDef("sc_head", "スタックチャン胴体シェル", d.stackchan_head, sc["shell"], 1, printable=False),
        PartDef("sc_core", "CoreS3 本体", d.stackchan_core, sc["shell"], 1, printable=False),
        PartDef("sc_face", "CoreS3 前面ガラス", d.stackchan_face, sc["glass"], 1, printable=False),
        PartDef("sc_eyes", "スタックチャンの顔", d.stackchan_eyes, sc["face"], 1, printable=False),
    ]
    return {p.key: p for p in defs}


@dataclass(frozen=True)
class Instance:
    name: str
    part: str
    matrix: np.ndarray
    group: str


def instances(drive: float = 0.0, head_pitch: float | None = None, with_stackchan: bool = True) -> list[Instance]:
    """drive: 後輪の車軸の回転角[rad](ターンテーブルは 1:1 で回る)、head_pitch: うなずき[rad](省略で直進の角度)。

    前輪の切れ角はうなずきから決まる(ハット → 引き棒 → 右のナックル → タイロッド → 左のナックル)。
    """
    eye = np.eye(4)
    if head_pitch is None:
        head_pitch = steering_geometry()[0]
    steer = steer_from_nod(head_pitch)
    head = _head_pose(head_pitch)
    out = [Instance(n, n, eye, "chassis") for n in ("rail_R", "rail_L", "deck", "yoke", "beam", "body_rear", "body_front", "windshield", "top",
                                                    "molding_body_R", "molding_body_L", "molding_hood_R", "molding_hood_L", "mirror_R", "mirror_L")]
    out += [Instance("fender_rear_R", "fender_rear_R", eye, "chassis"), Instance("fender_rear_L", "fender_rear_L", eye, "chassis")]
    out += [
        Instance("bumper_F", "bumper", translation(0, HOOD_NOSE_Y + 5.0, 4.0), "chassis"),
        Instance("bumper_R", "bumper", translation(0, BODY_REAR_Y - 5.0, 4.0) @ d.ROT180_Z, "chassis"),
        Instance("grille", "grille", GRILLE_POSE, "chassis"),
    ]
    for pose, tag in zip(HEADLIGHT_POSES, ("R", "L")):
        out.append(Instance(f"headlight_{tag}", "headlight", pose, "chassis"))
    table = rot_z(-drive)  # 傘歯車 1:1 で、車軸と逆向きに回る(walker の instances と同じ関係)
    out += [Instance("gear_table", "gear_table", table, "table"), Instance("socket_frame", "socket_frame", table, "table")]
    spin = rot_x(drive)
    out += [Instance(n, n, spin, "axle") for n in ("pinion", "sleeve", "shaft")]
    for sx, tag in ((1, "R"), (-1, "L")):
        flip = eye if sx > 0 else d.ROT180_Z
        wheel = translation(sx * WHEEL_X0, 0, 0) @ spin @ flip
        out += [
            Instance(f"wheel_rear_{tag}", "wheel_rear", wheel, "axle"),
            Instance(f"tire_rear_{tag}", "tire", wheel, "axle"),
            Instance(f"hubcap_rear_{tag}", "hubcap", wheel @ translation(TIRE_W, 0, 0), "axle"),
        ]
        knuckle = translation(sx * KINGPIN_X, WHEELBASE, 0) @ rot_z(steer)
        out.append(Instance(f"knuckle_{tag}", f"knuckle_{tag}", knuckle, f"steer_{tag}"))
        out.append(Instance(f"fender_front_{tag}", f"fender_front_{tag}", knuckle, f"steer_{tag}"))
        # C クリップはボスの下でキングピンの溝にはまる(厚みの向きを下へ)
        clip_z = BEAM_Z[0] - (d.GROOVE_W - d.CLIP_T) / 2
        clip_flat = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]])
        out.append(Instance(f"clip_{tag}", "clip", translation(sx * KINGPIN_X, WHEELBASE, clip_z) @ rot_z(steer) @ clip_flat, f"steer_{tag}"))
        out.append(Instance(f"clip_tie_{tag}", "clip", knuckle @ translation(0, TIE_ARM, KNUCKLE_Z[0] + STEER_ARM_T + d.clip_seat(TIE_PIN_CLEAR)) @ CLIP_UP, f"steer_{tag}"))
        front_wheel = knuckle @ translation(sx * (WHEEL_X0 - KINGPIN_X), 0, 0) @ flip
        out += [
            Instance(f"wheel_front_{tag}", "wheel_front", front_wheel, f"steer_{tag}"),
            Instance(f"tire_front_{tag}", "tire", front_wheel, f"steer_{tag}"),
            Instance(f"hubcap_front_{tag}", "hubcap", front_wheel @ translation(TIRE_W, 0, 0), f"steer_{tag}"),
            Instance(f"clip_wheel_{tag}", "clip", front_wheel @ translation(FRONT_BORE_X1 - 0.2 + d.clip_seat(FRONT_CLIP_CLEAR), 0, 0), f"steer_{tag}"),
        ]
    out.append(Instance("tie_rod", "tie_rod", translation(-TIE_ARM * math.sin(steer), TIE_ARM * math.cos(steer) - TIE_ARM, 0), "steer_R"))
    out.append(Instance("link", "link", link_pose(head_pitch, steer), "steer_R"))
    out.append(Instance("clip_drag", "clip", translation(KINGPIN_X, WHEELBASE, 0) @ rot_z(steer) @ translation(-DRAG_ARM, 0, KNUCKLE_Z[1] + d.clip_seat(DRAG_PIN_CLEAR)) @ CLIP_UP, "steer_R"))
    out.append(Instance("hat", "hat", head, "head"))
    out.append(Instance("hat_back", "hat_back", head, "head"))
    out.append(Instance("clip_j", "clip", head @ translation(ARM_X[1] + d.clip_seat(J_PIN_CLEAR), J_LOCAL[0], J_LOCAL[1]), "head"))
    if with_stackchan:
        seat = translation(0, 0, d.TABLE_Z1)
        out.append(Instance("sc_base", "sc_base", table @ seat, "table"))
        nod = translation(0, 0, d.SC_PITCH_AXIS_H) @ rot_x(head_pitch) @ translation(0, 0, -d.SC_PITCH_AXIS_H)
        out.append(Instance("sc_servo", "sc_servo", seat, "head"))
        for key in ("sc_head", "sc_core", "sc_face", "sc_eyes"):
            out.append(Instance(key, key, seat @ nod, "head"))
    return out


@lru_cache(maxsize=None)
def part_shape(key: str) -> Manifold:
    shape = part_defs()[key].build()
    if shape.status().name != "NoError" or shape.is_empty():
        raise RuntimeError(f"部品 {key} の形状が不正: {shape.status()}")
    return shape


def placed(instance: Instance) -> Manifold:
    return apply(part_shape(instance.part), instance.matrix)
