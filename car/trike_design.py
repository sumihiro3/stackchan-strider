"""スタックチャン逆三輪車(trike): 前の 2 輪で走り、うなずきで後ろの 1 輪を切る小さい車。

参考の台車(daisya.mp4)と同じく、頭の右側の四角い穴にピンを差したレバーが頭と一緒に回り、
太い引き棒を前後に押して、後ろの 1 輪のフォークを切る。殻やハットはない。

- 駆動部(デッキ、ターンテーブル歯車、ソケット枠、ピニオン、スリーブ、ヨーク)は walker の部品をそのまま使う。
- 駆動輪・六角シャフト・ホイールキャップ・C クリップは car_design の部品をそのまま使う。
- 引き棒は平行に前後へすべるだけ(上は縦長の穴、下は横長の穴でピンを受ける)。後輪の切れ角 δ は sin δ = 引き棒のずれ / FORK_ARM。

座標系は car_design と同じ(原点 = 駆動輪の車軸とヨー軸の交点、X 右、Y 前、Z 上、地面 z = -WHEEL_R)。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from manifold3d import Manifold

import car_design as base
from car_design import (  # noqa: F401  verify_car / build_car / render_car が使う名前
    AXIS_Z_TO_X, CLIP_UP, COLORS, GROUND_Z, HOLE_FIT, PRESS_FIT, ROOT, STEER_FIT, WHEEL_R, Instance, PartDef, apply, box,
    circle, cyl, cyl_x, hull2, plate_yz, polygon, rect, rot_x, rot_z, translation,
)

d = base.d
OUT = ROOT / "out" / "trike"
PUBLISH = ROOT / "print_trike"
PROJECT_NAME = "stackchan_trike"
PLATE_ROLES = (("gray", "Gray"), ("red", "Red"), ("black", "Black"))
PLATE_GROUP = {"hubcap": False}
VIEWS = {  # render_car.py のカメラ(位置 [mm]、注視点 [mm])
    "front34": ((230, 250, 150), (0, -15, 25)),
    "rear34": ((-200, -300, 170), (0, -35, 25)),
    "side": ((480, -20, 30), (0, -20, 30)),
    "top": ((0, -20, 520), (0, -20, 0)),
    "rear": ((0, -520, 40), (0, -40, 30)),
}
AXLE_LABELS = ("前の駆動輪", "後ろの 1 輪")  # verify_car.py の重さの配分の表示

# ---------------------------------------------------------------- 寸法
NOD_RANGE = (0.0, 30.0)  # 使ううなずきの範囲 [deg]
STEER_MAX = math.radians(30)
LEVER_R = 31.5  # うなずきの軸から J まで。J の内側の C クリップ(半径 4.6)が頭の上面(z 70.5)より上に来る最小に近い値
LEVER_A = math.radians(sum(NOD_RANGE) / 2)  # 直進のうなずきで J が軸の真上に来る向き
J_LOCAL = (LEVER_R * math.sin(LEVER_A), d.SC_PITCH_AXIS_H + LEVER_R * math.cos(LEVER_A))  # 頭のローカル (y, z)
# J のピンは引き棒と一体で、高さは変わらない。レバーの穴は軸から外向きに長い穴で、ピンはその中を外へずれる。
# そのため引き棒のずれは R tan(15° - うなずき)。端から端まで 2 R tan 15° = 16.9 mm
FORK_ARM = LEVER_R * math.tan(LEVER_A) / math.sin(STEER_MAX)  # うなずきの両端で ±STEER_MAX
J_SLOT_OUT = LEVER_R / math.cos(LEVER_A) - LEVER_R + 0.2  # レバーの長い穴を J から外へ伸ばす長さ
TRAIL_L = 60.0  # 駆動輪の車軸から後輪のキングピンまで(後輪が側板の後ろ y -39.5 に当たらない最小に近い値)
WHEELBASE = -TRAIL_L  # verify_car.py の重さの配分は「前の車軸 y = 0、もう一方 y = WHEELBASE」で計算する

# 頭の右側の穴(公式 MainBody STL を実測: LEGO Technic の梁と同じ直径 4.8 の丸穴。入口に直径 6.0・深さ 1 の座ぐり、
# 壁を x 27 → 19.8 まで貫いて中の空洞へ抜ける。中心 z 60.5、間隔 8)
HEAD_HOLE_Y = (5.7, -10.3)  # 3 つのうち前と後ろを使う(間隔 16 mm でねじれを止める)
HEAD_HOLE_Z = 60.5
HEAD_WALL_X = 19.95  # 頭の穴の壁の内面(公式 STL の実測。返しはこの 0.1 手前に掛ける)
# 割りピン(LEGO の摩擦ピンのように、切れ目で縮んで穴を通り、壁を抜けた先の返しが内側の面に掛かって抜けない)
HEAD_PIN = dict(r=2.2, ridge=0.4, ridge_l=0.8, lead=0.8, slit=1.2, slit_l=5.5)  # r 2.35・返し 0.35 では差すのが固すぎた
# LEGO Technic のピン(摩擦つき、長さ 2 ポッチ)で留めるレバーの穴: 梁と同じ直径 4.8・幅 8、頭の側の面につばの逃げ(直径 6.2、深さ 0.8)
LEVER_LEGO = True  # 組んだ姿(干渉の確認、ガイド)のレバーを LEGO のピンで留める版にする。False で割りピンの版
LEGO_HOLE = dict(r=2.45, len=8.0, cb_r=3.15, cb_d=0.8, boss_r=3.4)  # ボスの半径 3.6 でヨーク、4.0 で引き棒にも当たる
# LEGO のピンは前の穴だけ。後ろの穴の高さでは、ピンの外の端(頭の面から 8 mm)を引き棒が通る。
# 後ろは返しのない丸ピン(別に刷る lever_peg)で回り止めにする。レバーと一体だと、どちらの面を下にしてもサポートが要る
LEGO_PEG = dict(r=2.25, depth=5.0, lead=0.4)
LEVER_X = (d.SC_HEAD_HALF_W, d.SC_HEAD_HALF_W + 2.4)  # 頭の横の面に当てる板(J の穴の軸受けを兼ねる)
LEVER_CLEAR_R = d.EAR_R + 0.5  # ヨークの耳のパッド(軸から半径 13)を避ける。レバーはこの外だけにある
J_BOSS = dict(r=4.0, x0=LEVER_X[1] + 0.2)  # 引き棒の J のボス(縦の板からレバーの外面の手前まで)
LINK_X = (31.0, 35.0)  # 引き棒の縦の板(ヨークのパッドの外面 30.6 の外)
LINK_KNEE = (-46.0, 41.0)  # 引き棒の折れ目 (y, z)。前へ 7 mm ずれてもヨークのカラー(半径 44.7、上面 36.4)と腕の上を越える
J_PIN_CLEAR = J_BOSS["x0"] - (LEVER_X[0] - 0.2)  # ボスの面からクリップの溝まで(レバーの内面の 0.2 内。クリップは頭の上に出る)

# 後ろ
RW = 14.0  # 後輪の半径(輪ゴムを含めた設計の値。フォークの頭はこの 0.5 上)
RW_W = 8.0
# 後輪のプラスチックは RW より 1 mm 小さくし、輪ゴムの出る分をフォークの頭とのすきまに回す(RW のままだと輪ゴムが頭に当たった)
RW_RIM = 13.0
RW_GROOVE = dict(depth=0.4, w=2.0)  # 輪ゴム(内径 20 mm、太さ 1 mm ほど)の平らな溝。溝の底の直径 25.2 mm に 26% 伸ばしてはめる
RW_Z = GROUND_Z + RW  # 後輪の軸の高さ
FORK_LEG_X = (-RW_W / 2 - 0.3 - 5.0, -RW_W / 2 - 0.3)
FORK_HALF_Y = d.STUD_R  # フォークの前後の厚みの半分(ピンの半径と同じ)

# 小さい車輪(後輪、補助輪)の軸: 脚や側板と一体の溝つきピン(walker と同じ太さ)に通し、外で C クリップが留める。
# 一体なので軸が必ず直角に立ち、車輪が首を振らない
WHEEL_SEAT = 0.3  # 車輪の内面と、脚・側板の座のすきま
WHEEL_END_GAP = 0.2  # 車輪の外面とクリップの溝のすきま
# 補助輪: 側板の前の外面に付け、地面から AUX["lift"] 浮かせる(前にのめったときだけ着いて転がる)
AUX = dict(r=9.0, w=6.0, y=34.0, lift=2.0)
AUX_Z = GROUND_Z + AUX["lift"] + AUX["r"]

# 駆動輪(ホイールとタイヤの一体。右はシャフトと一体)
DISH_T = 2.5  # 皿の厚み。外の面(印刷でベッドに付く面)に寄せ、飾りの穴が橋にならないようにする
DRIVE_HOLES = dict(count=6, r=3.5, at=15.0)
NUT = dict(af=12.0, groove=1.2, depth=0.6)  # 外の面に彫る六角ナットの輪郭(飾り)
DRIVE_WALL = 1.2  # 左の駆動輪の六角穴の奥の壁。シャフトの先の溝つきピンだけが貫く
DRIVE_PIN_CLEAR = base.TIRE_W - base.TIGHT_X[1] + 0.2  # シャフトの六角の端からクリップの溝まで(左の車輪の外面 + 0.2)
# 左の駆動輪を空転させる(右だけで駆動する)。左右が 1 本の軸でつながっていると、曲がるとき内側と外側の車輪が同じ速さでしか
# 回れず(切れ角 30° で内側は外側の 0.28 倍が要る)、軽い後輪のほうが横に滑って曲がらなかった(2026-11-23 実機)
LEFT_FREE = True
LEFT_FREE_CLEAR = 0.25  # 空転する左の車輪の丸穴と、六角シャフトの角のすきま(半径)
DRIVE_CLIP = LEFT_FREE  # 左の駆動輪の抜け止めの C クリップ。圧入しないときは要る
CROWN_Z = (RW_Z + RW + 0.5, 2.1)  # フォークの頭(後輪の上 0.5)
BRIDGE_Z = (CROWN_Z[1] + 0.3, CROWN_Z[1] + 0.3 + 4.0)  # 後ろの橋(側板の腕の下に当て、下から押されても外れない)
ARM_Z = (BRIDGE_Z[1], BRIDGE_Z[1] + 5.0)  # 側板から後ろへ伸ばす腕
TAB_Z = (BRIDGE_Z[1] + 0.2, BRIDGE_Z[1] + 0.2 + 3.0)  # 引き棒の下の横板
TAB_X = (7.0, LINK_X[1])
TAB_HALF_Y = 4.0
GUIDE_X = (LINK_X[1] + 0.3, LINK_X[1] + 2.3)  # 横板の外端を受ける壁
BRIDGE_HALF_Y = 15.0
# 橋は側板の腕から下へ出した 2 本爪のスナップ(ヨークと同じ形)で留める。丸ピンの圧入では外せなかった
BRIDGE_SNAP = dict(width=6.0, thick=3.0, dy=0.0)

# おもり(10 円玉)
COIN_R, COIN_T, COIN_G = 23.5 / 2, 1.5, 4.5
COINS = 10  # 後輪にかかる重さの割合: 6 枚で 13%、10 枚で 19%、12 枚で 22%(重いほど後輪が滑りにくいが、駆動輪の重さは減る)
CUP = dict(y=-TRAIL_L - 28.0, wall=1.6, depth=12 * COIN_T + 1.0)  # 12 枚まで入る
LID = dict(t=1.2, gap=0.2, tab_t=1.2, tab_deg=50.0, count=3, barb=0.5, skirt=3.4)  # カップのふた(外から 3 本のツメで掛ける)
EXTRA_MASSES = [(f"10 円玉 {COINS} 枚", COINS * COIN_G, (0.0, CUP["y"], BRIDGE_Z[1] + COINS * COIN_T / 2))]


# ---------------------------------------------------------------- かじ取りの幾何
def _head_pose(theta: float) -> np.ndarray:
    return base._head_pose(theta)


def j_world(theta: float) -> np.ndarray:
    """レバーの J(長い穴の内端、引き棒の板の厚みの中央)の位置。"""
    return (_head_pose(theta) @ np.array([(LINK_X[0] + LINK_X[1]) / 2, J_LOCAL[0], J_LOCAL[1], 1.0]))[:3]


def _axis_world() -> np.ndarray:
    return (_head_pose(0.0) @ np.array([0.0, 0.0, d.SC_PITCH_AXIS_H, 1.0]))[:3]


def steering_geometry() -> tuple[float, float]:
    """(直進のうなずき角 [rad]、J の半径 [mm])。J が軸の真上に来るうなずきを直進にする。"""
    return LEVER_A, LEVER_R


def link_shift(theta: float) -> float:
    """直進の姿勢からの引き棒の前後のずれ [mm](正で前)。ピンは高さ R のまま、レバーの穴の向き(15° - うなずき)の線上にある。"""
    return float(LEVER_R * math.tan(LEVER_A - theta))


def steer_from_nod(theta: float) -> float:
    """うなずき角 [rad] から後輪の切れ角 [rad](上から見て反時計回りが正)。"""
    return math.asin(max(-1.0, min(1.0, link_shift(theta) / FORK_ARM)))


# ---------------------------------------------------------------- 部品
def _rail_section():
    """walker の吊り下げ版の側板の断面(YZ)。デッキとヨークのツメの角穴、スリーブの穴。"""
    deck_mid = (d.DECK_Z0 + d.DECK_Z1) / 2
    lug_mid = d.COLLAR_Z0 + d.COLLAR_LUG_T / 2
    nodes = [circle(13.0, 0, 0)]
    for sign in (1, -1):
        nodes += [circle(7.5, sign * d.DECK_TENON_Y, deck_mid), circle(7.5, sign * d.YOKE_TENON_Y, lug_mid)]
    section = hull2(*nodes)
    section += rect(-TRAIL_L - BRIDGE_HALF_Y, -d.DECK_TENON_Y, *ARM_Z)  # 後ろの橋を下から受ける腕
    section += hull2(circle(5.0, AUX["y"], AUX_Z), rect(AUX["y"] - 5.0, AUX["y"] + 5.0, d.DECK_Z0 - 1.0, d.DECK_Z0))  # 補助輪の柱
    section -= circle(d.SLEEVE_R + HOLE_FIT)
    for sign in (1, -1):
        section -= rect(sign * d.DECK_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.DECK_TENON_Y + d.TENON_W / 2 + 0.2, d.DECK_Z0 - 0.15, d.DECK_Z1 + 0.15)
        section -= rect(sign * d.YOKE_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.YOKE_TENON_Y + d.TENON_W / 2 + 0.2, d.COLLAR_Z0 - 0.15, d.COLLAR_Z0 + d.COLLAR_LUG_T + 0.15)
    return section


def _wheel_clear(width: float) -> float:
    """座の面から C クリップの溝までの長さ(座のすきま + 車輪の幅 + 外のすきま)。"""
    return WHEEL_SEAT + width + WHEEL_END_GAP


@lru_cache(maxsize=None)
def rail_part() -> Manifold:
    """右の側板。walker と同じ軸受け穴と角穴、後ろへ伸びて橋を受ける腕、前の補助輪の柱と溝つきピン。

    補助輪のピンを上へ立てるため、内面を下にして刷る。
    """
    rail = plate_yz(_rail_section(), d.PLATE_IN, d.PLATE_OUT)
    rail += cyl_x(d.STUD_R + 1.5, d.PLATE_OUT - 0.01, d.PLATE_OUT + WHEEL_SEAT, AUX["y"], AUX_Z)  # 補助輪の内面を受ける座
    rail += apply(d.grooved_pin(0.0, _wheel_clear(AUX["w"])), AXIS_Z_TO_X).translate((d.PLATE_OUT, AUX["y"], AUX_Z))
    rail += _bridge_snap(1)
    return rail


def _bridge_snap_frame(sx: int):
    """橋を留めるスナップの根元(側板の腕の下面)、向き(下)、爪の開く向き(前後。側板を寝かせて刷る層の中で開く)。

    側板は内面を下にして刷るので、ツメの内面を側板の内面にそろえてベッドに着ける(浮くとサポートが要る)。
    """
    x_mid = sx * (d.PLATE_IN + BRIDGE_SNAP["thick"] / 2)
    return (x_mid, -TRAIL_L + BRIDGE_SNAP["dy"], ARM_Z[0] + 0.01), (0, 0, -1), (0, 1, 0)


def _bridge_snap(sx: int) -> Manifold:
    return base.snap_tenon(*_bridge_snap_frame(sx), length=BRIDGE_Z[1] - BRIDGE_Z[0] + 0.01,
                           width=BRIDGE_SNAP["width"], thick=BRIDGE_SNAP["thick"])


@lru_cache(maxsize=None)
def bridge_part() -> Manifold:
    """後ろの橋。左右の側板の腕の下に当て、キングピンの穴、フォークの腕のピンが動く円弧の穴、引き棒の横板の受け、10 円玉のカップ。"""
    y0 = -TRAIL_L
    plate = rect(-d.PLATE_OUT, d.PLATE_OUT, y0 - BRIDGE_HALF_Y, y0 + BRIDGE_HALF_Y)
    cup_r = COIN_R + 0.2 + CUP["wall"]
    plate += hull2(circle(cup_r, 0, CUP["y"]), rect(-cup_r, cup_r, CUP["y"], y0 - BRIDGE_HALF_Y + 1.0))
    bridge = plate.extrude(BRIDGE_Z[1] - BRIDGE_Z[0]).translate((0, 0, BRIDGE_Z[0]))
    bridge -= cyl(d.STUD_R + STEER_FIT, BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1).translate((0, y0, 0))
    # フォークの腕のピンが ±30° 動く円弧の穴(半径方向と角度に 0.5 の余裕)
    r_in, r_out = FORK_ARM - d.STUD_R - 0.5, FORK_ARM + d.STUD_R + 0.5
    deg = math.degrees(STEER_MAX) + math.degrees(math.asin((d.STUD_R + 0.5) / FORK_ARM)) + 2.0
    sector = polygon([(0, 0)] + [(r_out * 1.2 * math.cos(math.radians(a)), r_out * 1.2 * math.sin(math.radians(a))) for a in np.linspace(-deg, deg, 25)])
    slot = (sector ^ (circle(r_out) - circle(r_in))) + circle(d.STUD_R + 0.5, r_in + d.STUD_R + 0.5, 0)
    bridge -= slot.extrude(10).translate((0, y0, BRIDGE_Z[0] - 1))
    # 引き棒の横板の外端を受ける壁(横板は前後に ±FORK_ARM/2 すべる)
    travel = FORK_ARM * math.sin(STEER_MAX)
    bridge += box(*GUIDE_X, y0 - travel - TAB_HALF_Y - 1.0, y0 + travel + TAB_HALF_Y + 1.0, BRIDGE_Z[1] - 0.01, TAB_Z[1] - 0.6)
    # 10 円玉のカップ
    bridge += cyl(cup_r, BRIDGE_Z[1] - 0.01, BRIDGE_Z[1] + CUP["depth"]).translate((0, CUP["y"], 0))
    bridge -= cyl(COIN_R + 0.2, BRIDGE_Z[1], BRIDGE_Z[1] + CUP["depth"] + 1).translate((0, CUP["y"], 0))
    bridge -= _lid_groove(cup_r)
    # 側板の腕から下りるスナップの角穴(爪の返しが橋の下面に掛かる)
    for sx in (1, -1):
        bridge -= base.snap_hole(*_bridge_snap_frame(sx), -1.0, BRIDGE_Z[1] - BRIDGE_Z[0] + 1.0,
                                 width=BRIDGE_SNAP["width"], thick=BRIDGE_SNAP["thick"])
    return bridge


def _cup_top() -> float:
    return BRIDGE_Z[1] + CUP["depth"]


def _ccw(points) -> "CrossSection":
    """点の並びの向きによらず、内側を埋めた断面にする(CrossSection は時計回りを空とみなす)。"""
    pts = np.asarray(points, dtype=np.float64)
    area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    return polygon(pts if area > 0 else pts[::-1])


def _lid_barb(r_out: float, depth: float, z_mid: float) -> Manifold:
    """カップの外の面の V 字の帯(z_mid を中心に上下 45°、r_out から内へ depth)。溝とツメの返しの両方に使う。"""
    ring = _ccw([(r_out + 0.01, z_mid - depth - 0.01), (r_out - depth, z_mid), (r_out + 0.01, z_mid + depth + 0.01)])
    return ring.revolve(96)


def _lid_groove(cup_r: float) -> Manifold:
    """ふたのツメの返しが落ちる、カップの外の面の V 溝。"""
    return _lid_barb(cup_r, LID["barb"], _cup_top() - LID["skirt"] + LID["barb"] + 0.4).translate((0, CUP["y"], 0))


@lru_cache(maxsize=None)
def cup_lid_part() -> Manifold:
    """10 円玉のカップのふた(設計の座標)。上の板と、外から下りる 3 本のツメ。ツメの先の V 字の返しがカップの溝に掛かる。

    板を下にして刷る(返しは上下とも 45° なので、サポートが要らない)。
    """
    cup_r = COIN_R + 0.2 + CUP["wall"]
    top = _cup_top()
    r_in = cup_r + LID["gap"]
    r_out = r_in + LID["tab_t"]
    lid = cyl(r_out, top + LID["gap"], top + LID["gap"] + LID["t"])
    skirt = cyl(r_out, top - LID["skirt"], top + LID["gap"] + 0.01) - cyl(r_in, top - LID["skirt"] - 1, top + LID["gap"] + 0.02)
    half = math.radians(LID["tab_deg"] / 2)
    keep = None
    for k in range(LID["count"]):
        a = 2 * math.pi * k / LID["count"] + math.pi / 2
        wedge = polygon([(0, 0)] + [(40 * math.cos(a + t), 40 * math.sin(a + t)) for t in np.linspace(-half, half, 9)])
        piece = wedge.extrude(LID["skirt"] + 2).translate((0, 0, top - LID["skirt"] - 1))
        keep = piece if keep is None else keep + piece
    z_barb = top - LID["skirt"] + LID["barb"] + 0.4
    tabs = (skirt + _lid_barb(r_in, LID["barb"], z_barb)) ^ keep
    return (lid + tabs).translate((0, CUP["y"], 0))


@lru_cache(maxsize=None)
def fork_part() -> Manifold:
    """後輪のフォーク(ローカル: キングピンの軸 = Z、原点 = 橋の下のキングピンの中心、z は設計の高さ)。

    片持ちの脚から後輪の溝つきピン(C クリップ)。頭の上から上向きのキングピン(橋を貫き C クリップ)と、
    右へ伸びる腕の先の上向きのピン(橋の円弧の穴と引き棒の横長の穴を貫き C クリップ)。
    前後の厚みをピンの太さ(2 × FORK_HALF_Y)にそろえ、横に寝かせて刷る。3 本のピンはどれもベッドに接して寝るので、サポートが要らない。
    """
    h = FORK_HALF_Y
    crown = rect(FORK_LEG_X[0], FORK_ARM + d.STUD_R + 1.0, -h, h)
    fork = crown.extrude(CROWN_Z[1] - CROWN_Z[0]).translate((0, 0, CROWN_Z[0]))
    fork += plate_yz(hull2(circle(h, 0, RW_Z), rect(-h, h, CROWN_Z[0], CROWN_Z[1])), *FORK_LEG_X)
    fork += cyl_x(d.STUD_R + 1.5, FORK_LEG_X[1] - 0.01, FORK_LEG_X[1] + WHEEL_SEAT, 0, RW_Z) ^ box(-20, 20, -h, h, -50, 50)  # 後輪の内面を受ける座
    fork += apply(d.grooved_pin(0.0, _wheel_clear(RW_W)), AXIS_Z_TO_X).translate((FORK_LEG_X[1], 0, RW_Z))
    fork += d.grooved_pin(CROWN_Z[1], BRIDGE_Z[1] + 0.2 - CROWN_Z[1])
    fork += d.grooved_pin(CROWN_Z[1], TAB_Z[1] + 0.2 - CROWN_Z[1]).translate((FORK_ARM, 0, 0))
    return fork


_R_TO_X = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)  # 断面 (r, x) を回した形を、軸 X へ


def _ring_x(r0: float, r1: float, x0: float, x1: float) -> Manifold:
    return cyl_x(r1, x0, x1) - cyl_x(r0, x0 - 1, x1 + 1)


def _holes_x(count: int, r: float, at: float, x0: float, x1: float) -> Manifold:
    return Manifold.compose([cyl_x(r, x0, x1, at * math.cos(2 * math.pi * k / count), at * math.sin(2 * math.pi * k / count)) for k in range(count)])


def _small_wheel(radius: float, width: float, holes: int) -> Manifold:
    """小さい車輪(ローカル: 軸 = X、内面 x = 0、外面 x = width)。角は 45°、皿は外の面に寄せて飾りの穴をあける。

    中心は溝つきピンが通る穴(すきま HOLE_FIT で回る)。外の面を下にして刷る。
    """
    c = min(2.0, radius * 0.15)
    pts = [(radius - c - 2.0, 0.0), (radius - c, 0.0), (radius, c), (radius, width - c), (radius - c, width), (radius - c - 2.0, width)]
    band = polygon(pts).revolve(96).transform(_R_TO_X)
    hub_r = d.STUD_R + HOLE_FIT + 2.2
    wheel = band + cyl_x(radius - c - 1.0, width - 2.0, width) + cyl_x(hub_r, 0, width)
    inner = radius - c - 2.0
    if holes and inner - hub_r > 3.0:
        wheel -= _holes_x(holes, (inner - hub_r) / 2 - 0.6, (inner + hub_r) / 2, width - 3.0, width + 1.0)
    hole = cyl_x(d.STUD_R + HOLE_FIT, -1.0, width + 1.0)
    # 穴の両端を 45° で面取りする。ベッドに着く側は 1 層目がつぶれて穴の縁がふくらみ、軸に当たって回りが重くなった
    r, ch = d.STUD_R + HOLE_FIT, 0.5
    cone = Manifold.cylinder(ch + 0.01, r + ch + 0.01, r, 48)
    hole += cone.transform(_R_TO_X).translate((-0.01, 0, 0))
    hole += cone.mirror((0, 0, 1)).transform(_R_TO_X).translate((width + 0.01, 0, 0))
    return wheel - hole


@lru_cache(maxsize=None)
def wheel_trike_part() -> Manifold:
    """後輪(ローカル: 軸 = X、内面 x = 0)。中央に輪ゴムの平らな溝。荷重が軽く横に滑るので、ゴムで食いつかせる。"""
    r0, hw = RW_RIM - RW_GROOVE["depth"], RW_GROOVE["w"] / 2
    groove = _ccw([(r0, RW_W / 2 - hw), (RW_RIM + 1.0, RW_W / 2 - hw), (RW_RIM + 1.0, RW_W / 2 + hw), (r0, RW_W / 2 + hw)])
    return _small_wheel(RW_RIM, RW_W, 5) - groove.revolve(96).transform(_R_TO_X)


@lru_cache(maxsize=None)
def wheel_aux_part() -> Manifold:
    """補助輪(ローカル: 軸 = X、内面 x = 0)。小さいので飾りの穴はあけない。"""
    return _small_wheel(AUX["r"], AUX["w"], 0)


def _drive_wheel_body() -> Manifold:
    """駆動輪の共通の形(ローカル: 軸 = X、内面 x = 0、外面 x = TIRE_W)。皿を外の面に寄せ、飾りの穴と六角ナットの彫り。"""
    w, c = base.TIRE_W, base.TIRE_CHAMFER
    rim_in = base.RIM_R - 2.0
    pts = [(rim_in, 0.0), (WHEEL_R - c, 0.0), (WHEEL_R, c), (WHEEL_R, w - c), (WHEEL_R - c, w), (rim_in, w)]
    band = polygon(pts).revolve(96).transform(_R_TO_X)
    g = base.GROOVE_DEPTH
    band -= polygon([(WHEEL_R - g, w / 2), (WHEEL_R + 1.0, w / 2 - g - 1.0), (WHEEL_R + 1.0, w / 2 + g + 1.0)]).revolve(96).transform(_R_TO_X)
    hub_r = base.HUB_R + 2.0
    wheel = band + cyl_x(rim_in + 0.5, w - DISH_T, w) + cyl_x(hub_r, base.HUB_X0 - base.WHEEL_X0, w)
    h = DRIVE_HOLES
    wheel -= _holes_x(h["count"], h["r"], h["at"], w - DISH_T - 1.0, w + 1.0)
    nut = d.hexagon(NUT["af"]) - d.hexagon(NUT["af"] - 2 * NUT["groove"])
    wheel -= apply(nut.extrude(NUT["depth"] + 1.0), AXIS_Z_TO_X).translate((w - NUT["depth"], 0, 0))
    return wheel


@lru_cache(maxsize=None)
def wheel_drive_shaft_part() -> Manifold:
    """右の駆動輪と六角シャフトの一体(ローカルは駆動輪と同じ)。外の面を下に、シャフトを立てて刷る。

    シャフトの左端の先は細い溝つきピンで、左の車輪の奥の壁を貫いて外へ出る(あとで C クリップを足せる)。
    付け根(ハブの内面)のすぐ内側でスリーブが支えるので、付け根に大きな曲げはかからない。
    """
    end = -(base.WHEEL_X0 + base.SHAFT_HALF)
    shaft = apply(d.hexagon(base.SHAFT_ACTUAL_AF).extrude(base.WHEEL_X0 + base.SHAFT_HALF + 1.0), AXIS_Z_TO_X).translate((end, 0, 0))
    pin = apply(d.grooved_pin(0.0, DRIVE_PIN_CLEAR), AXIS_Z_TO_X).mirror((1, 0, 0)).translate((end + 0.01, 0, 0))
    return _drive_wheel_body() + shaft + pin


@lru_cache(maxsize=None)
def wheel_drive_part() -> Manifold:
    """左の駆動輪。LEFT_FREE なら丸穴で空転し、C クリップで抜けを止める。
    そうでなければ、六角穴は大半を元のすきまで通し、外寄り TIGHT_X だけ圧入する(car_design の後輪と同じ)。

    六角穴の奥は壁で、シャフトの先の溝つきピンの穴だけをあける。クリップはこの壁の外の面に当たる。
    """
    wheel = _drive_wheel_body()
    loose = d.SHAFT_AF + d.SHAFT_BORE_CLEARANCE
    lead = Manifold.batch_hull([base._hex_bore_x(base.WHEEL_SOCKET_AF + 2 * base.WHEEL_SOCKET_LEAD, base.TIGHT_X[0] - 0.01, base.TIGHT_X[0]),
                                base._hex_bore_x(base.WHEEL_SOCKET_AF, base.TIGHT_X[0] + base.WHEEL_SOCKET_LEAD, base.TIGHT_X[0] + base.WHEEL_SOCKET_LEAD + 0.01)])
    if LEFT_FREE:
        # 丸穴にして空転させる。シャフトの角と、奥の壁を貫く丸いピンが軸受けになる
        wheel -= cyl_x(base.SHAFT_ACTUAL_AF / math.sqrt(3) + LEFT_FREE_CLEAR, -20.0, base.TIRE_W - DRIVE_WALL)
    else:
        wheel -= base._hex_bore_x(loose, -20.0, base.TIGHT_X[0]) + lead + base._hex_bore_x(base.WHEEL_SOCKET_AF, base.TIGHT_X[0], base.TIRE_W - DRIVE_WALL)
    return wheel - cyl_x(d.STUD_R + HOLE_FIT, 0.0, base.TIRE_W + 1.0)


@lru_cache(maxsize=None)
def lever_part() -> Manifold:
    """頭の右側のレバー(頭のローカル座標)。ピン 2 本を頭の横の四角い穴に差し、上の J の長い穴に引き棒のピンを受ける。

    ヨークの耳のパッドは軸から半径 13 なので、レバーはすべて半径 LEVER_CLEAR_R の外に置く(頭と一緒に回っても当たらない)。
    外の面を下にして刷る(頭に差すピンが上を向き、サポートが要らない)。
    """
    lever = _lever_plate()
    for y in HEAD_HOLE_Y:
        lever += _split_pin().translate((0, y, HEAD_HOLE_Z))
    return lever


def _lever_plate() -> Manifold:
    """レバーの板(頭の横の面に当てる。J の長い穴つき)。"""
    axis_z = d.SC_PITCH_AXIS_H
    out = (math.sin(LEVER_A), math.cos(LEVER_A))
    j_out = (J_LOCAL[0] + J_SLOT_OUT * out[0], J_LOCAL[1] + J_SLOT_OUT * out[1])
    nodes = [circle(4.0, y, HEAD_HOLE_Z) for y in HEAD_HOLE_Y] + [circle(J_BOSS["r"] + 0.5, *J_LOCAL), circle(J_BOSS["r"] + 0.5, *j_out)]
    section = hull2(*nodes) - circle(LEVER_CLEAR_R, 0, axis_z)
    r = d.STUD_R + STEER_FIT
    section -= hull2(circle(r, *J_LOCAL), circle(r, *j_out))
    return plate_yz(section, *LEVER_X)


@lru_cache(maxsize=None)
def lever_lego_part() -> Manifold:
    """LEGO Technic のピンで頭に留めるレバー(頭のローカル座標)。割りピンの代わりに、梁と同じ穴を 2 つ持つ。

    ピンの半分(8 mm)を受けるため、穴のまわりだけ外へ LEGO_HOLE["len"] の厚みのボスを出す。穴は外へ抜けていて、
    ピンの先の返しが外の面に掛かる。頭の側の面にはピンのつばの逃げを掘る(頭の穴の逃げと合わせて、つばが収まる)。
    """
    h, g = LEGO_HOLE, LEGO_PEG
    front, rear = HEAD_HOLE_Y
    lever = _lever_plate() + cyl_x(h["boss_r"], LEVER_X[0], LEVER_X[0] + h["len"], front, HEAD_HOLE_Z)
    lever -= cyl_x(h["r"], LEVER_X[0] - 1.0, LEVER_X[0] + h["len"] + 1.0, front, HEAD_HOLE_Z)
    lever -= cyl_x(h["cb_r"], LEVER_X[0] - 1.0, LEVER_X[0] + h["cb_d"], front, HEAD_HOLE_Z)
    return lever - cyl_x(g["r"] + PRESS_FIT, LEVER_X[0] - 1.0, LEVER_X[1] + 1.0, rear, HEAD_HOLE_Z)


@lru_cache(maxsize=None)
def lego_pin_part() -> Manifold:
    """LEGO Technic のピン(摩擦つき、長さ 2 ポッチ)のおおよその形(頭のローカル座標、前の穴)。印刷しない。干渉の確認とガイド用。

    つば(直径 5.8、厚さ 1.6)の真ん中が頭の横の面に来て、頭の穴とレバーの穴の逃げに半分ずつ入る。
    """
    x0, y, z = LEVER_X[0], HEAD_HOLE_Y[0], HEAD_HOLE_Z
    pin = cyl_x(2.4, x0 - 7.8, x0 + 7.8, y, z) + cyl_x(2.9, x0 - 0.8, x0 + 0.8, y, z)
    return pin - cyl_x(1.5, x0 - 8.0, x0 + 8.0, y, z)


@lru_cache(maxsize=None)
def lever_peg_part() -> Manifold:
    """LEGO 版のレバーの後ろの穴に外から押し込む丸ピン(頭のローカル座標。外の面にそろう)。先は頭の穴に 5 mm 入る。"""
    g = LEGO_PEG
    tip = LEVER_X[0] - g["depth"]
    profile = _ccw([(0.0, LEVER_X[1]), (g["r"] - g["lead"], LEVER_X[1]), (g["r"], LEVER_X[1] - g["lead"]),
                    (g["r"], tip + g["lead"]), (g["r"] - g["lead"], tip), (0.0, tip)])
    return profile.revolve(48).transform(_R_TO_X).translate((0, HEAD_HOLE_Y[1], HEAD_HOLE_Z))


def _split_pin() -> Manifold:
    """頭の穴に差す割りピン(軸 = X、中心 y = z = 0)。レバーの内面 LEVER_X[0] から -X へ伸びる。

    返しの掛かる面は壁の内面(HEAD_WALL_X)の 0.1 先。先は円すいにして、穴へ入れやすくする。
    """
    p = HEAD_PIN
    catch = HEAD_WALL_X - 0.1  # 返しの、抜けを止める面
    tip = catch - p["ridge_l"] - p["lead"]
    r, rr = p["r"], p["r"] + p["ridge"]
    profile = _ccw([(0.0, LEVER_X[0] + 0.01), (r, LEVER_X[0] + 0.01), (r, catch), (rr, catch), (rr, catch - p["ridge_l"]),
                       (r - 0.4, tip), (0.0, tip)])
    pin = profile.revolve(48).transform(_R_TO_X)
    return pin - box(tip - 1.0, tip + p["slit_l"], -rr - 1, rr + 1, -p["slit"] / 2, p["slit"] / 2)


@lru_cache(maxsize=None)
def link_part() -> Manifold:
    """引き棒(直進の姿勢、設計の座標)。J から後ろ下へ折れ目 LINK_KNEE まで、そこから橋へ下りる縦の板と、橋の上をすべる横板。

    上は内向き(-X)の溝つきピン(レバーの長い穴を貫き、内側で C クリップ)、横板には横長の穴(フォークの腕のピンが横へ動く分を逃がす)。
    縦の板の外面を下にして刷る(ピンと横板が上を向く)。
    """
    j = j_world(LEVER_A)
    y0 = -TRAIL_L
    foot = rect(y0 - TAB_HALF_Y, y0 + TAB_HALF_Y, TAB_Z[0], TAB_Z[0] + 8.0)
    knee = circle(3.5, *LINK_KNEE)
    bar = plate_yz(circle(4.0, j[1], j[2]) + hull2(circle(3.5, j[1], j[2]), knee) + hull2(knee, foot), *LINK_X)
    bar += cyl_x(J_BOSS["r"], J_BOSS["x0"], LINK_X[0] + 0.01, j[1], j[2])
    bar = bar.trim_by_plane((0, 0, 1), TAB_Z[0])
    tab = box(TAB_X[0], TAB_X[1], y0 - TAB_HALF_Y, y0 + TAB_HALF_Y, *TAB_Z)
    link = bar + tab
    r = d.STUD_R + STEER_FIT
    link += apply(d.grooved_pin(0.0, J_PIN_CLEAR), AXIS_Z_TO_X).mirror((1, 0, 0)).translate((J_BOSS["x0"] + 0.01, j[1], j[2]))
    x_in = FORK_ARM * math.cos(STEER_MAX) - r - 0.3
    link -= hull2(circle(r, x_in + r, y0), circle(r, FORK_ARM + 0.3, y0)).extrude(10).translate((0, 0, TAB_Z[0] - 1))
    return link


# ---------------------------------------------------------------- 部品表と姿勢
REUSED = ("deck", "gear_table", "socket_frame", "pinion", "sleeve")
SC_PARTS = ("sc_base", "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes")


def part_defs() -> dict[str, PartDef]:
    c = COLORS
    base_defs = base.part_defs()
    defs = [
        PartDef("rail_R", "右の側板(後ろへ腕、前に補助輪の柱)", rail_part, c["gray"], 1),
        PartDef("rail_L", "左の側板", lambda: rail_part().mirror((1, 0, 0)), c["gray"], 1),
        PartDef("yoke", "ヨーク(walker と同じ)", d.yoke_part, c["gray"], 1),
        PartDef("bridge", "後ろの橋(10 円玉のカップつき)", bridge_part, c["gray"], 1),
        PartDef("wheel_drive_shaft", "右の駆動輪と六角シャフトの一体", wheel_drive_shaft_part, c["black"], 1),
        PartDef("wheel_drive", "左の駆動輪(空転)" if LEFT_FREE else "左の駆動輪", wheel_drive_part, c["black"], 1),
        PartDef("fork", "後輪のフォーク", fork_part, c["red"], 1),
        PartDef("wheel_trike", "後輪", wheel_trike_part, c["black"], 1),
        PartDef("wheel_aux", "補助輪", wheel_aux_part, c["black"], 2),
        PartDef("lever", "頭の右のレバー(割りピン)", lever_part, c["red"], 1),
        PartDef("lever_lego", "頭の右のレバー(LEGO のピンで留める。lever の代わり)", lever_lego_part, c["red"], 1),
        PartDef("lever_peg", "LEGO 版のレバーの後ろの丸ピン", lever_peg_part, c["red"], 1),
        PartDef("lego_pin", "LEGO Technic のピン(摩擦つき、2 ポッチ。用意する)", lego_pin_part, "#d8d8d4", 1, printable=False),
        PartDef("link", "引き棒", link_part, c["red"], 1),
        PartDef("cup_lid", "10 円玉のカップのふた", cup_lid_part, c["gray"], 1),
    ]
    out = {p.key: p for p in defs}
    for key in REUSED + SC_PARTS:
        out[key] = base_defs[key]
    out["clip"] = PartDef("clip", base_defs["clip"].label, base_defs["clip"].build, base_defs["clip"].color, 7 if DRIVE_CLIP else 6)
    return out


def instances(drive: float = 0.0, head_pitch: float | None = None, with_stackchan: bool = True) -> list[Instance]:
    """drive: 駆動輪の車軸の回転角 [rad]、head_pitch: うなずき [rad](省略で直進の角度)。後輪の切れ角はうなずきから決まる。"""
    eye = np.eye(4)
    if head_pitch is None:
        head_pitch = LEVER_A
    steer = steer_from_nod(head_pitch)
    head = _head_pose(head_pitch)
    out = [Instance(n, n, eye, "chassis") for n in ("rail_R", "rail_L", "deck", "yoke", "bridge", "cup_lid")]
    table = rot_z(-drive)
    out += [Instance("gear_table", "gear_table", table, "table"), Instance("socket_frame", "socket_frame", table, "table")]
    spin = rot_x(drive)
    out += [Instance(n, n, spin, "axle") for n in ("pinion", "sleeve")]
    out.append(Instance("wheel_drive_shaft", "wheel_drive_shaft", translation(base.WHEEL_X0, 0, 0) @ spin, "axle"))
    left = translation(-base.WHEEL_X0, 0, 0) @ spin @ d.ROT180_Z
    out.append(Instance("wheel_drive", "wheel_drive", left, "axle"))
    if DRIVE_CLIP:
        out.append(Instance("clip_drive", "clip", left @ translation(base.TIGHT_X[1] + d.clip_seat(DRIVE_PIN_CLEAR), 0, 0), "axle"))
    # 補助輪(内面を側板の外面から 0.3 離す)
    for sx, tag in ((1, "R"), (-1, "L")):
        flip = eye if sx > 0 else d.ROT180_Z
        aux = translation(sx * (d.PLATE_OUT + WHEEL_SEAT), AUX["y"], AUX_Z) @ flip
        out.append(Instance(f"wheel_aux_{tag}", "wheel_aux", aux, "chassis"))
        clip_x = d.PLATE_OUT + d.clip_seat(_wheel_clear(AUX["w"]))
        out.append(Instance(f"clip_aux_{tag}", "clip", translation(sx * clip_x, AUX["y"], AUX_Z) @ flip, "chassis"))
    fork = translation(0, -TRAIL_L, 0) @ rot_z(steer)
    out.append(Instance("fork", "fork", fork, "steer"))
    out.append(Instance("wheel_trike", "wheel_trike", fork @ translation(-RW_W / 2, 0, RW_Z), "steer"))
    out.append(Instance("clip_trike", "clip", fork @ translation(FORK_LEG_X[1] + d.clip_seat(_wheel_clear(RW_W)), 0, RW_Z), "steer"))
    out.append(Instance("clip_king", "clip", fork @ translation(0, 0, CROWN_Z[1] + d.clip_seat(BRIDGE_Z[1] + 0.2 - CROWN_Z[1])) @ CLIP_UP, "steer"))
    out.append(Instance("clip_arm", "clip", fork @ translation(FORK_ARM, 0, CROWN_Z[1] + d.clip_seat(TAB_Z[1] + 0.2 - CROWN_Z[1])) @ CLIP_UP, "steer"))
    # 引き棒は前後にずれるだけ。うなずきの両端で J が下がる分は、縦長の穴の中でピンが下がる
    out.append(Instance("link", "link", translation(0, link_shift(head_pitch), 0), "link"))
    out.append(Instance("lever", "lever_lego" if LEVER_LEGO else "lever", head, "head"))
    if LEVER_LEGO:
        out.append(Instance("lever_peg", "lever_peg", head, "head"))
        out.append(Instance("lego_pin", "lego_pin", head, "head"))
    shift = link_shift(head_pitch)
    j = j_world(LEVER_A)
    out.append(Instance("clip_j", "clip", translation(J_BOSS["x0"] - d.clip_seat(J_PIN_CLEAR), j[1] + shift, j[2]) @ d.ROT180_Z, "link"))
    if with_stackchan:
        seat = translation(0, 0, d.TABLE_Z1)
        out.append(Instance("sc_base", "sc_base", table @ seat, "table"))
        out.append(Instance("sc_servo", "sc_servo", seat, "head"))
        for key in ("sc_head", "sc_core", "sc_face", "sc_eyes"):
            out.append(Instance(key, key, head, "head"))
    return out


@lru_cache(maxsize=None)
def part_shape(key: str) -> Manifold:
    shape = part_defs()[key].build()
    if shape.status().name != "NoError" or shape.is_empty():
        raise RuntimeError(f"部品 {key} の形状が不正: {shape.status()}")
    return shape


def placed(instance: Instance) -> Manifold:
    return apply(part_shape(instance.part), instance.matrix)


# build_car.py --one-color のプレート分け。駆動部の 5 部品は car500 とまったく同じ形なので、組んである car500 から外して使える
ONE_COLOR_PLATES = [("Trike new parts", None), ("Same as car500 (reuse)", ["deck", "gear_table", "socket_frame", "pinion", "sleeve"])]
ONE_COLOR_MARGIN = 10.0  # ベッドの端からの余白 [mm](端は糸を引きやすいので広く取る)

# build_car.py の印刷の向き(car と同じ名前の部品も、この車の形に合わせて上書きする)
PRINT_POSE_OVERRIDE = {
    "rail_R": ("X_UP", "内面を下(補助輪のピンを上へ立てる)", False),
    "rail_L": ("X_DOWN", "内面を下(補助輪のピンを上へ立てる)", False),
    "bridge": ("Z_UP", "板を下(ピン・壁・カップを上)", False),
    "wheel_drive_shaft": ("X_DOWN", "外の面を下、シャフトを立てる", False),
    "wheel_drive": ("X_DOWN", "外の面を下", False),
    "fork": ("Y_DOWN", "横に寝かせる(3 本のピンはベッドに接して寝る)", False),
    "wheel_trike": ("X_DOWN", "外の面を下", False),
    "wheel_aux": ("X_DOWN", "外の面を下", False),
    "lever": ("X_DOWN", "外の面を下(頭に差すピンを上)", False),
    "lever_lego": ("X_UP", "頭の側の面を下(ボスを上)", False),
    "lever_peg": ("X_DOWN", "立てる", False),
    "link": ("X_DOWN", "縦の板の外面を下(J のピンと横板を上)", False),
    "cup_lid": ("Z_DOWN", "板を下(ツメを上)", False),
}
ENGLISH_NAMES = {
    "bridge": "Rear bridge", "wheel_drive_shaft": "Drive wheel R + shaft", "wheel_drive": "Drive wheel L", "fork": "Rear fork",
    "wheel_trike": "Rear wheel", "wheel_aux": "Training wheel", "lever": "Head lever", "link": "Steering link", "cup_lid": "Coin cup lid", "lever_lego": "Head lever (LEGO pin)", "lever_peg": "Lever peg (LEGO version)",
}


_LEVER_PIN = "lever_peg" if LEVER_LEGO else "lever"  # 頭の穴に入っている、形のあるピン
FIT_CHECKS = [
    # (説明, 動かす部品, 止める部品, ずらす量 [mm])
    (f"{_LEVER_PIN} を 0.4 mm 前へずらすと頭の穴の壁に当たる(ガタが小さい)", _LEVER_PIN, "sc_head", (0, 0.4, 0)),
    (f"{_LEVER_PIN} を 0.4 mm 上へずらすと頭の穴の壁に当たる", _LEVER_PIN, "sc_head", (0, 0, 0.4)),
] + ([] if LEVER_LEGO else [
    ("lever を 0.3 mm 外へ引くと割りピンの返しが頭の壁の内面に当たる(抜けない)", "lever", "sc_head", (0.3, 0, 0)),
]) + [
    ("cup_lid を 0.3 mm 持ち上げるとツメの返しがカップの溝に当たる(外れない)", "cup_lid", "bridge", (0, 0, 0.3)),
    ("bridge を 0.3 mm 上げると側板の腕に当たる(下から押されても外れない)", "bridge", "rail_R", (0, 0, 0.3)),
    ("bridge を 0.3 mm 下げるとスナップの返しに当たる(落ちない)", "bridge", "rail_R", (0, 0, -0.3)),
    ("bridge を 0.4 mm 前へずらすとスナップに当たる(ガタが小さい)", "bridge", "rail_R", (0, 0.4, 0)),
    ("link の横板を 0.5 mm 外へずらすと橋の壁に当たる", "link", "bridge", (0.5, 0, 0)),
    ("wheel_drive_shaft を 0.6 mm 内へずらすとハブが rail_R に当たる(車軸が左右に動かない)", "wheel_drive_shaft", "rail_R", (-0.6, 0, 0)),
    ("wheel_drive を 0.6 mm 内へずらすとハブが rail_L に当たる", "wheel_drive", "rail_L", (0.6, 0, 0)),
]
