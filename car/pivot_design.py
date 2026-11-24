"""スタックチャンのピボット(pivot): 左右の駆動輪を 1 本の軸でつないで走り、頭を上げると右の駆動輪が浮いて、その場で旋回する。

逆三輪車(trike_design)は、後輪を切っても曲がらなかった。実物で分かったのは次の 2 つ。
- 左右の駆動輪を軸でつなぐと、まっすぐ進む。
- 片側の駆動輪だけが地面を押すと、押さない側を中心に、ほぼその場で回る。
参考の台車(動画)も舵は切らず、頭が水平のあいだは直進し、頭を上げるとその場で回る。同じ動きにする。
旋回の左右は、首を回す向き(前進・後退)で選ぶ。そのため仕組みは右の 1 組だけでよい。

仕組み(右の側板の内側。部品は隠さず、そのまま見せる):
- 頭の右のレバー: ヨークの耳の円板を囲む馬てい形の輪。LEGO Technic のピン 2 本で頭に留める。輪の開きは円板より狭いので、頭は上へ抜けない。
- レバーのピン P は、うなずきの軸の後ろにある(ユーザーの指示。顔の横に引き板を出さない)。頭を上げると P は下がり、
  押し板が、車体の下の「足の腕」の後ろの端を押し下げる。足の腕は側板の穴(車軸の少し前)を軸に回り、途中の足が下がって、右の駆動輪を浮かせる。
- 押す形なので、反動は頭を持ち上げる向きになる。P を軸から遠く(PIN_R)、足の腕の入力点を軸から遠くに置いて、押す力を小さくしてある。
覆いは付けない(フェンダーと前後の覆いを付けた形は、ゴツゴツして不採用になった)。

座標系は car_design と同じ(原点 = 駆動輪の車軸とヨー軸の交点、X 右、Y 前、Z 上、地面 z = -WHEEL_R)。
"""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np
from manifold3d import Manifold

import car_design as base
import trike_design as trike
from car_design import (  # noqa: F401  verify_car / build_car / render_car が使う名前
    AXIS_Z_TO_X, COLORS, GROUND_Z, HOLE_FIT, ROOT, WHEEL_R, Instance, PartDef, apply, box,
    circle, cyl_x, hull2, plate_yz, polygon, rect, rot_x, rot_z, translation,
)

d = base.d

# 駆動輪をヨークの下(x ±30〜40)へ寄せるため、タイヤの上(z 30)より上へ、ターンテーブルから上をまとめて上げる。
# 歯車の歯とかみ合いは動かさず、ターンテーブル歯車の首だけが長くなる。walker/ は書き換えず、ここで上書きする
TABLE_RAISE = 6.5
if not getattr(d, "_pivot_table_raised", False):
    for _name in ("TABLE_Z0", "BOSS_TOP", "TABLE_Z1", "FRAME_Z1", "COLLAR_Z0", "COLLAR_Z1"):
        setattr(d, _name, getattr(d, _name) + TABLE_RAISE)
    d.PITCH_Z = d.TABLE_Z1 + d.SC_PITCH_AXIS_H
    d._pivot_table_raised = True

OUT = ROOT / "out" / "pivot"
PUBLISH = ROOT / "print_pivot"
PROJECT_NAME = "stackchan_pivot"
PLATE_ROLES = (("gray", "Gray"), ("red", "Red"), ("black", "Black"))
PLATE_GROUP = {"hubcap": False}
VIEWS = {  # render_car.py のカメラ(位置 [mm]、注視点 [mm])
    "front34": ((230, 250, 150), (0, 0, 25)),
    "rear34": ((-200, -300, 170), (0, 0, 25)),
    "side": ((480, 0, 30), (0, 0, 30)),
    "top": ((0, 0, 520), (0, 0, 0)),
    "rear_low": ((70, -420, 0), (15, 0, -8)),  # 車体の下の足を見る
    "inside": ((-260, -330, 60), (20, -20, 10)),  # 左後ろから、右の側板の内側の仕組みを見る
}
AXLE_LABELS = ("駆動輪", "前の補助輪")  # verify_car.py の重さの配分の表示

# ---------------------------------------------------------------- 寸法
NOD_RANGE = (0.0, 30.0)  # 使ううなずきの範囲 [deg]
NEUTRAL = math.radians(NOD_RANGE[0])  # 直進のうなずき(頭が水平)。足は浮いている。頭を上げきる(NOD_RANGE[1])と足が下りる
AXIS_Z = d.TABLE_Z1 + d.SC_PITCH_AXIS_H  # うなずきの軸の高さ(軸は y = 0)

# 頭のレバー(馬てい形の輪)。寸法は、直進の姿勢で軸を原点にした (y, z)
RING_CLEAR = 0.5  # 輪と耳の円板のすきま(半径)。頭とヨークの位置のずれを見込む
RING_R = (d.EAR_R + RING_CLEAR, 20.5)
# 輪の開き(うなずきの範囲の真ん中の姿勢で、真下から左右へ)。耳の柱(幅 ±8)を、うなずきの両端でも避ける。
# 弦は約 22 mm で、円板の直径 26 mm より狭い
RING_OPEN = math.radians(56.0)
RING_MID = math.radians(sum(NOD_RANGE) / 2)
LEVER_X = (d.SC_HEAD_HALF_W, d.SC_HEAD_HALF_W + 3.4)  # 耳の円板(x 27.2〜30.6)と同じ厚み
# 軸からレバーのピン P まで。頭が水平のとき軸の真後ろ。30° 上げると 12 mm 下がる。
# 輪の外(半径 20.5)より遠くに置く: P の動きが大きいほど、押す力は小さくて済む(18 では 0.57 N、24 で 0.5 N 前後)
PIN_R = 24.0
PIN_SHOULDER_R = 3.5  # P の太い根元。押し板が内へずれるのを止める段

# 押し板は、レバーのピン P と足の腕の耳のピンのどちらにも外から(-X へ)差す。同じ向きなので、たわませずに付けられる
SEAT = 0.3  # 板どうしのすきま
LUG_X = (LEVER_X[1] + 0.6, LEVER_X[1] + 3.9)  # 足の腕の、押し板をつなぐ耳の外面の位置(押し板の位置は前の形と同じ)
LINK_T = 3.0
LINK_X = (LUG_X[1] + SEAT, LUG_X[1] + SEAT + LINK_T)
JOINT_FIT = 0.3  # 回るピンと穴のすきま(半径)。0.15 では、寝かせて刷ったピンが回らなかった

# 足の腕の軸の棒(車の座標の (y, z))。左右の側板に通して、両端を C クリップで留める。
# 前の形は右の側板だけで支える片持ちで、腕の後ろの端が左右に 2〜3 mm 振れた(計算)。
# 駆動輪のタイヤ(半径 30)の前、デッキの下。右前の補助輪とそのクリップより上
HINGE = (37.5, -3.0)
SHAFT_R = 3.0
HUB_TUBE_R = 5.5  # 足の腕が棒に通る筒。右の側板の内面まで伸ばし、腕が左右に振れないようにする
# 押し板がつながる点(車の座標)。タイヤの後ろ。前の形より 6.5 mm 高い(ターンテーブルと一緒に P も上がったので、押し板は同じ長さ)
ARM_IN = (-45.5, -3.0)
# 押し板の折れ目(直進の姿勢、車の座標の (y, z))。P から後ろへ下り、ここから足の腕の耳までまっすぐ下りる
LINK_KNEE = (ARM_IN[0], 44.5)
# 足の腕の板(x)。ピニオンの背面(22.3)と右の駆動輪の内面(29.6)のあいだを、駆動輪のハブ(半径 9)の下でくぐる
ARM_X = (23.0, 26.0)
ARM_KNEE = (12.0, -16.0)  # 板の折れ目(ハブの下)
# 二股の足(参考の台車と同じく、車体の下の真ん中で左右に分かれる)。足の腕の板に、ほぞで差して留める別の部品。
# 右の先だけが地面を押し、左の先は右より短く、車体が左へ傾いたときの倒れ止めになる。
# 駆動輪は横へすべりにくいので、回る中心は車軸の線の上の x = FORK_X に来る。真ん中(0)より回す力に余裕がある
FORK_Y = -20.0  # 二股の前後の位置(ピニオンの後ろ寄りの下)
FORK_X = 12.0  # 二股の先の |x|
FORK_T = 6.0  # 二股の部品の厚み(y)
FORK_BAR_Z = (-23.0, -15.0)  # 横棒の高さ。ほぞ(高さ TENON_W)より高く、肩が足の腕の内面に当たる
FORK_MID_Z = sum(FORK_BAR_Z) / 2
TENON_TIP = 2.0  # ほぞの返しの先の長さ。walker の形(2.8)のままでは右の駆動輪の内面に届くので短く切る
TIP_HALF_W = 3.0
FOOT_GAP = 3.3  # 直進のとき、右の先が地面から浮いている量。うなずき約 12° で地面に着く(ファームの旋回の角度の下限 10° の少し上)
# 左の先を右より短くする量。足を下ろすと車体は左の駆動輪のほうへ傾き、地面が左の先に近づく(1.5 では 0.5 mm しか残らない)
LEFT_TIP_RISE = 2.5
TIP_R_Z = GROUND_Z + FOOT_GAP  # 右の先の下端(直進の姿勢)
TIP_L_Z = TIP_R_Z + LEFT_TIP_RISE
FOOT_CLEAR = -FOOT_GAP  # verify 用: 直進のとき、足が地面の線より下がっている量(負の値 = 浮いている)

# 前後の支えは補助輪(転がる)。右の側板の前と、左の側板の後ろに 1 個ずつ。
# 足が車軸の後ろにあるので、重心(車軸の 3 mm 前)をはさんで反対の右前に支えが要る(左前では、地面を押す左の駆動輪に重さが残らない)。
# 突起だけの支えは、直進中ずっと地面をこする(重心が車軸の 2 mm 前にあり、前の支えに重さの 5% がかかる)ので、車輪にした。
# 倒れ止めの突起は付けない。足を下ろして車体が傾くと、左後ろの突起が地面に着いてこすった(計算)
AUX = dict(r=trike.AUX["r"], w=trike.AUX["w"], y=40.0, lift=1.0)  # 補助輪は trike と同じ部品。直進のとき地面から lift 浮く
AUX_Z = GROUND_Z + AUX["lift"] + AUX["r"]
AUX_CLEAR = SEAT + AUX["w"] + 0.2  # 座の面から C クリップの溝まで
WHEELBASE = AUX["y"]  # verify_car.py の重さの配分(駆動輪の車軸 y = 0、もう一方 = 左前の補助輪 y = WHEELBASE)

# 駆動輪はヨークの下。車軸には、内から ピニオン(背面 x = GEAR_BACK)→ 駆動輪のハブ → 駆動輪 → スリーブ → 側板 の順に並ぶ(左は鏡写し)。
# 傘歯車がピニオンを押す力は、ハブ・駆動輪・スリーブのつばを通って側板が受ける
AXLE_GAP = 0.2  # ピニオンの背面と駆動輪のハブのすきま
DRIVE_X0 = d.GEAR_BACK + AXLE_GAP + (base.WHEEL_X0 - base.HUB_X0)  # 駆動輪の内面 |x|(ハブは内面から 7.1 mm 内へ出る)
DRIVE_X1 = DRIVE_X0 + base.TIRE_W
SLEEVE_X0 = DRIVE_X1 + 0.1  # スリーブの内端
SHAFT_HALF = d.L0 - 0.3  # シャフトの端 |x|(スリーブの外端の手前)

EXTRA_MASSES: list = []


# ---------------------------------------------------------------- 仕組みの幾何
def _head_pose(theta: float) -> np.ndarray:
    return base._head_pose(theta)


def _pin(theta: float) -> tuple[float, float]:
    """レバーのピン P の (y, z)。直進で軸の真後ろ。頭を上げる(theta が増える)と下がる。"""
    a = theta - NEUTRAL
    return -PIN_R * math.cos(a), AXIS_Z - PIN_R * math.sin(a)


def _turn(point: tuple[float, float], phi: float) -> tuple[float, float]:
    """足の腕に付いた点 (y, z)(直進の姿勢)が、腕が phi [rad] 回ったときに来る位置。phi が正で、軸より後ろの点が下がる。"""
    y, z = point[0] - HINGE[0], point[1] - HINGE[1]
    return HINGE[0] + y * math.cos(phi) - z * math.sin(phi), HINGE[1] + y * math.sin(phi) + z * math.cos(phi)


def _arm_in(phi: float) -> tuple[float, float]:
    """足の腕が phi [rad] 回ったときの、押し板がつながる点の (y, z)。"""
    return _turn(ARM_IN, phi)


LINK_L = math.dist(_pin(NEUTRAL), _arm_in(0.0))  # 押し板の上の穴と下の穴の間隔


def arm_angle(theta: float) -> float:
    """うなずき theta [rad] での、足の腕の回転角 [rad]。押し板の長さが変わらない角度を二分法で求める。"""
    p = _pin(theta)
    lo, hi = math.radians(-20), math.radians(45)
    f = lambda phi: math.dist(p, _arm_in(phi)) - LINK_L  # noqa: E731  phi が増えると I が下がり、P から離れる
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def foot_drop(theta: float) -> float:
    """二股の右の先(地面を押す側)が、直進の位置から下がった量 [mm]。"""
    return TIP_R_Z - _turn((FORK_Y, TIP_R_Z), arm_angle(theta))[1]


def steering_geometry() -> tuple[float, float]:
    """(直進のうなずき角 [rad]、レバーのピンの半径 [mm])。"""
    return NEUTRAL, PIN_R


def steer_from_nod(theta: float) -> float:
    """verify_car.py の表示用。舵はないので、足の腕の回転角を返す。"""
    return arm_angle(theta)


# ---------------------------------------------------------------- 部品
def _bore_x(radius: float, x0: float, x1: float, y: float = 0.0, z: float = 0.0, chamfer: float = 0.4) -> Manifold:
    """X 向きの丸穴(x0〜x1 を貫く)。両端を 45° で面取りする(1 層目のつぶれで穴の縁がふくらみ、ピンに当たるのを防ぐ)。"""
    hole = cyl_x(radius, x0 - 1.0, x1 + 1.0)
    cone = Manifold.cylinder(chamfer + 0.01, radius + chamfer + 0.01, radius, 48)
    hole += cone.transform(trike._R_TO_X).translate((x0 - 0.01, 0, 0))
    hole += cone.mirror((0, 0, 1)).transform(trike._R_TO_X).translate((x1 + 0.01, 0, 0))
    return hole.translate((0, y, z))


def _pin_x(x0: float, clear: float, y: float, z: float) -> Manifold:
    """x0 から +X へ立つ溝つきピン(clear 先に C クリップの溝)。"""
    return apply(d.grooved_pin(0.0, clear), AXIS_Z_TO_X).translate((x0 - 0.01, y, z))


def _rail(aux_y: float) -> Manifold:
    """側板(右の向き)。walker と同じ軸受け穴と角穴に、補助輪のピンと足の腕の軸の棒の穴を足す。
    外形は、節をまとめて 1 つの凸の形にする(枝分かれさせない)。

    aux_y: 補助輪の前後の位置。補助輪のピンを上へ立てるため、内面を下にして刷る。
    """
    deck_mid = (d.DECK_Z0 + d.DECK_Z1) / 2
    lug_mid = d.COLLAR_Z0 + d.COLLAR_LUG_T / 2
    nodes = [circle(13.0, 0, 0), circle(5.0, aux_y, AUX_Z)]
    for sign in (1, -1):
        nodes += [circle(7.5, sign * d.DECK_TENON_Y, deck_mid), circle(7.5, sign * d.YOKE_TENON_Y, lug_mid)]
    nodes.append(circle(6.5, *HINGE))  # 足の腕の軸の棒
    section = hull2(*nodes)
    section -= circle(d.SLEEVE_R + HOLE_FIT)
    for sign in (1, -1):
        section -= rect(sign * d.DECK_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.DECK_TENON_Y + d.TENON_W / 2 + 0.2, d.DECK_Z0 - 0.15, d.DECK_Z1 + 0.15)
        section -= rect(sign * d.YOKE_TENON_Y - d.TENON_W / 2 - 0.2, sign * d.YOKE_TENON_Y + d.TENON_W / 2 + 0.2, d.COLLAR_Z0 - 0.15, d.COLLAR_Z0 + d.COLLAR_LUG_T + 0.15)
    rail = plate_yz(section, d.PLATE_IN, d.PLATE_OUT)
    rail += cyl_x(d.STUD_R + 1.5, d.PLATE_OUT - 0.01, d.PLATE_OUT + SEAT, aux_y, AUX_Z)  # 補助輪の内面を受ける座
    rail += _pin_x(d.PLATE_OUT, AUX_CLEAR, aux_y, AUX_Z)
    rail -= _bore_x(SHAFT_R + JOINT_FIT, d.PLATE_IN, d.PLATE_OUT, *HINGE)  # 足の腕の軸の棒が通る穴
    return rail


@lru_cache(maxsize=None)
def rail_r_part() -> Manifold:
    """右の側板。前に補助輪。"""
    return _rail(AUX["y"])


@lru_cache(maxsize=None)
def rail_l_part() -> Manifold:
    """左の側板。後ろに補助輪。"""
    return _rail(-AUX["y"]).mirror((1, 0, 0))


@lru_cache(maxsize=None)
def foot_part() -> Manifold:
    """足の腕(車の座標、直進の姿勢)。前の軸の棒から、右の駆動輪のハブの下をくぐって後ろへ伸びる板。

    途中に二股の足のほぞが入る角穴、後ろの端に押し板をつなぐ耳とピン。軸の筒は右の側板の内面まで伸びる。
    筒と耳のピンを上へ立てるため、内側の面を下にして刷る。
    """
    hub, knee = circle(6.0, *HINGE), circle(4.0, *ARM_KNEE)
    fork, rear = circle(6.0, FORK_Y, FORK_MID_Z), circle(4.5, *ARM_IN)
    section = hull2(hub, knee) + hull2(knee, fork) + hull2(fork, rear)
    half_y, half_z = FORK_T / 2 + 0.15, d.TENON_W / 2 + 0.2  # ほぞの角穴(側板のデッキの角穴と同じすきま)
    section -= rect(FORK_Y - half_y, FORK_Y + half_y, FORK_MID_Z - half_z, FORK_MID_Z + half_z)
    arm = plate_yz(section, *ARM_X)
    arm += cyl_x(HUB_TUBE_R, ARM_X[1] - 0.01, d.PLATE_IN - SEAT, *HINGE)
    arm -= _bore_x(SHAFT_R + JOINT_FIT, ARM_X[0], d.PLATE_IN - SEAT, *HINGE)
    arm += plate_yz(rear, ARM_X[1] - 0.01, LUG_X[1])
    return arm + _pin_x(LUG_X[1], SEAT + LINK_T + 0.2, *ARM_IN)


_XZ_TO_XYZ = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0]])  # 断面 (x, z) を y の向きに押し出した形を、車の座標へ


@lru_cache(maxsize=None)
def fork_part() -> Manifold:
    """二股の足(車の座標、直進の姿勢)。横棒の右の端のスナップほぞを足の腕の角穴に差して留める。右の先が地面を押す。

    地面に押されると、横棒が角穴の上の面を押す(力は面で受ける)。ほぞの肩が腕の内面に、返しが外面に掛かって、左右にも抜けない。
    断面が x–z の平らな板なので、y の面を下にして刷る。
    """
    bar_lo, bar_hi = FORK_BAR_Z
    root = ARM_X[0] - 0.1
    section = rect(-FORK_X - TIP_HALF_W, root, bar_lo, bar_hi)
    for x, tip in ((FORK_X, TIP_R_Z), (-FORK_X, TIP_L_Z)):
        section += hull2(rect(x - TIP_HALF_W, x + TIP_HALF_W, bar_lo, bar_lo + 1.0), circle(TIP_HALF_W, x, tip + TIP_HALF_W))
    # ほぞの割りは横棒まで延ばす(ほぞだけでは短く、返しが乗り越えるときに折れる)
    flex = 8.0
    section -= rect(root - flex, root, FORK_MID_Z - 0.8, FORK_MID_Z + 0.8)
    tenon_len = ARM_X[1] - ARM_X[0] + 0.2
    tenon = d._snap_tenon(d.TENON_W, tenon_len, 4.0) ^ rect(-d.TENON_W, d.TENON_W, -1.0, tenon_len + TENON_TIP)
    section += tenon.rotate(-90).translate((root, FORK_MID_Z))
    return apply(section.extrude(FORK_T), _XZ_TO_XYZ).translate((0, FORK_Y - FORK_T / 2, 0))


@lru_cache(maxsize=None)
def hinge_rod_part() -> Manifold:
    """足の腕の軸の棒(車の座標)。左右の側板を貫き、両端の溝に C クリップ。下の面を少し平らに削り、寝かせて刷る。"""
    x0 = d.PLATE_OUT + SEAT
    rod = cyl_x(SHAFT_R, -x0, x0, *HINGE)
    end = _pin_x(x0, 0.2, *HINGE)
    rod += end + end.mirror((1, 0, 0))
    return rod.trim_by_plane((0, 0, 1), HINGE[1] - SHAFT_R + 0.8)  # 両端のピン(半径 2.5)の下も平らになる


def _link_ends() -> tuple[tuple[float, float], tuple[float, float]]:
    """直進の姿勢での、押し板の上の穴(P)と下の穴(足の腕の耳のピン)の (y, z)。"""
    return _pin(NEUTRAL), _arm_in(0.0)


@lru_cache(maxsize=None)
def link_part() -> Manifold:
    """押し板(直進の姿勢、車の座標)。レバーのピン P から後ろへ下り、折れ目から足の腕の耳までまっすぐ下りる。

    両端とも丸穴の平らな板。寝かせて刷る。
    """
    top, bottom = _link_ends()
    bar = hull2(circle(4.0, *top), circle(3.5, *LINK_KNEE)) + hull2(circle(3.5, *LINK_KNEE), circle(4.0, *bottom))
    link = plate_yz(bar, *LINK_X)
    return link - _bore_x(d.STUD_R + JOINT_FIT, *LINK_X, *top) - _bore_x(d.STUD_R + JOINT_FIT, *LINK_X, *bottom)


def _rotated(section, angle: float):
    """直進の姿勢で描いた断面(軸が原点)を、頭のローカル座標へ直す。"""
    return section.rotate(-math.degrees(angle)).translate((0.0, d.SC_PITCH_AXIS_H))


def _lever(sx: int) -> Manifold:
    """頭のレバー(頭のローカル座標)。sx = 1 が右(ピンは後ろ)。-1 は左用(ピンは前)で、今は使わない。

    頭の側の面を下にして刷る(LEGO のピンの穴のボスと、ピン P が上を向く)。
    """
    h = trike.LEGO_HOLE
    reach = 60.0
    a, b = reach * math.sin(RING_OPEN), reach * math.cos(RING_OPEN)
    ring = circle(RING_R[1]) - circle(RING_R[0]) - trike._ccw([(0.0, 0.0), (-a, -b), (0.0, -1.5 * reach), (a, -b)])
    section = _rotated(ring, RING_MID)
    holes = [(y, trike.HEAD_HOLE_Z) for y in trike.HEAD_HOLE_Y]
    for y, z in holes:
        section += circle(h["boss_r"] + 1.0, y, z)
    # ピン P。直進の姿勢(頭が水平)で、右は軸の真後ろ、左は真前。輪の外へ出した腕の先にある
    py, pz = -sx * PIN_R * math.cos(NEUTRAL), d.SC_PITCH_AXIS_H + sx * PIN_R * math.sin(NEUTRAL)
    section += hull2(circle(4.5, py, pz), circle(4.5, py * (RING_R[1] - 4.0) / PIN_R, pz))
    section -= circle(RING_R[0], 0.0, d.SC_PITCH_AXIS_H)
    lever = plate_yz(section, *LEVER_X)
    for y, z in holes:
        boss = cyl_x(h["boss_r"], LEVER_X[0], LEVER_X[0] + h["len"], y, z)
        lever += boss - cyl_x(RING_R[0], LEVER_X[0] - 1.0, LEVER_X[0] + h["len"] + 1.0, 0.0, d.SC_PITCH_AXIS_H)
        lever -= cyl_x(h["r"], LEVER_X[0] - 1.0, LEVER_X[0] + h["len"] + 1.0, y, z)
        lever -= cyl_x(h["cb_r"], LEVER_X[0] - 1.0, LEVER_X[0] + h["cb_d"], y, z)
    lever += cyl_x(PIN_SHOULDER_R, LEVER_X[1] - 0.01, LINK_X[0] - 0.2, py, pz)
    lever += _pin_x(LINK_X[0] - 0.2, LINK_X[1] - LINK_X[0] + 0.4, py, pz)
    return lever if sx > 0 else lever.mirror((1, 0, 0))


@lru_cache(maxsize=None)
def lever_r_part() -> Manifold:
    return _lever(1)


def _lego_pins(sx: int) -> Manifold:
    """LEGO Technic のピン(摩擦つき、長さ 2 ポッチ)2 本のおおよその形(頭のローカル座標)。印刷しない。干渉の確認用。"""
    x0 = LEVER_X[0]
    pins = Manifold()
    for y in trike.HEAD_HOLE_Y:
        pin = cyl_x(2.4, x0 - 7.8, x0 + 7.8, y, trike.HEAD_HOLE_Z) + cyl_x(2.9, x0 - 0.8, x0 + 0.8, y, trike.HEAD_HOLE_Z)
        pins += pin - cyl_x(1.5, x0 - 8.0, x0 + 8.0, y, trike.HEAD_HOLE_Z)
    return pins if sx > 0 else pins.mirror((1, 0, 0))


@lru_cache(maxsize=None)
def wheel_drive_part() -> Manifold:
    """駆動輪(左右同じ。ローカルは trike の駆動輪と同じ: 軸 = X、内面 x = 0)。六角穴は貫通で、外寄り TIGHT_X だけ圧入する。"""
    loose = d.SHAFT_AF + d.SHAFT_BORE_CLEARANCE
    lead = Manifold.batch_hull([base._hex_bore_x(base.WHEEL_SOCKET_AF + 2 * base.WHEEL_SOCKET_LEAD, base.TIGHT_X[0] - 0.01, base.TIGHT_X[0]),
                                base._hex_bore_x(base.WHEEL_SOCKET_AF, base.TIGHT_X[0] + base.WHEEL_SOCKET_LEAD, base.TIGHT_X[0] + base.WHEEL_SOCKET_LEAD + 0.01)])
    wheel = trike._drive_wheel_body() - base._hex_bore_x(loose, -20.0, base.TIGHT_X[0]) - lead
    return wheel - base._hex_bore_x(base.WHEEL_SOCKET_AF, base.TIGHT_X[0], base.TIRE_W + 1.0)


@lru_cache(maxsize=None)
def shaft_part() -> Manifold:
    """六角シャフト。左のスリーブから右のスリーブまで通り、ピニオン・駆動輪・スリーブを貫く。寝かせて刷る。"""
    return base._hex_bore_x(base.SHAFT_ACTUAL_AF, -SHAFT_HALF, SHAFT_HALF)


@lru_cache(maxsize=None)
def pinion_part() -> Manifold:
    """ピニオン。walker の形から、左の側板まで伸びていた筒を切る(左の駆動輪がそこに入る)。"""
    return d.pinion_part().trim_by_plane((1, 0, 0), -d.GEAR_BACK)


@lru_cache(maxsize=None)
def sleeve_part() -> Manifold:
    """スリーブ(右の向き。左は 180° 回して使う)。walker の形の外寄りだけ。つばが側板の内面に当たり、駆動輪の外面を受ける。"""
    return d.sleeve_part().trim_by_plane((1, 0, 0), SLEEVE_X0)


@lru_cache(maxsize=None)
def deck_part() -> Manifold:
    """デッキ。walker の形から、駆動輪とそのハブが通る所を切り欠く。"""
    deck = d.deck_part()
    tire_y = math.sqrt(WHEEL_R**2 - d.DECK_Z0**2) + 1.0
    hub_y = base.HUB_R + 3.0
    for sx in (1, -1):
        for x0, x1, half_y in ((DRIVE_X0 - 1.0, DRIVE_X1 + 1.0, tire_y), (d.GEAR_BACK - 0.3, DRIVE_X1 + 1.0, hub_y)):
            lo, hi = sorted((sx * x0, sx * x1))
            deck -= box(lo, hi, -half_y, half_y, d.DECK_Z0 - 1.0, d.DECK_Z1 + 1.0)
    return deck


# ---------------------------------------------------------------- 部品表と姿勢
REUSED = ("gear_table", "socket_frame")
SC_PARTS = ("sc_base", "sc_servo", "sc_head", "sc_core", "sc_face", "sc_eyes")
CLIPS = 6  # レバーのピンと押し板、押し板と足の腕、足の腕の軸の棒の両端、補助輪 2 個


def part_defs() -> dict[str, PartDef]:
    c = COLORS
    base_defs = base.part_defs()
    defs = [
        PartDef("rail_R", "右の側板(前に補助輪)", rail_r_part, c["gray"], 1),
        PartDef("rail_L", "左の側板(後ろに補助輪)", rail_l_part, c["gray"], 1),
        PartDef("yoke", "ヨーク(walker と同じ)", d.yoke_part, c["gray"], 1),
        PartDef("deck", "デッキ(駆動輪の所を切り欠く)", deck_part, c["gray"], 1),
        PartDef("pinion", "ピニオン(左の筒を切る)", pinion_part, c["gray"], 1),
        PartDef("sleeve", "スリーブ(左右に 1 個ずつ)", sleeve_part, c["gray"], 2),
        PartDef("shaft", "六角シャフト", shaft_part, c["gray"], 1),
        PartDef("wheel_drive", "駆動輪(左右同じ。六角穴が貫通)", wheel_drive_part, c["black"], 2),
        PartDef("wheel_aux", "補助輪(trike と同じ。右前と左後ろ)", trike.wheel_aux_part, c["black"], 2),
        PartDef("lever_R", "頭の右のレバー", lever_r_part, c["red"], 1),
        PartDef("link", "押し板", link_part, c["red"], 1),
        PartDef("foot", "足の腕", foot_part, c["red"], 1),
        PartDef("fork", "二股の足(足の腕にほぞで差す)", fork_part, c["red"], 1),
        PartDef("hinge_rod", "足の腕の軸の棒", hinge_rod_part, c["red"], 1),
        PartDef("lego_pins_R", "LEGO Technic のピン(摩擦つき、2 ポッチ。用意する)", lambda: _lego_pins(1), "#d8d8d4", 1, printable=False),
    ]
    out = {p.key: p for p in defs}
    for key in REUSED + SC_PARTS:
        out[key] = base_defs[key]
    out["clip"] = PartDef("clip", base_defs["clip"].label, base_defs["clip"].build, base_defs["clip"].color, CLIPS)
    # 色は「車体が赤、動く所(レバー・押し板・足)がグレー、車輪が黒」(2026-11-24 ユーザーが決めた)。
    # 流用する部品はもとの色(車体 = グレー、動く所 = 赤)で定義されているので、ここで赤とグレーを入れ替える
    swap = {COLORS["gray"]: COLORS["red"], COLORS["red"]: COLORS["gray"]}
    return {k: dataclasses.replace(v, color=swap.get(v.color, v.color)) if v.printable else v for k, v in out.items()}


def instances(drive: float = 0.0, head_pitch: float | None = None, with_stackchan: bool = True) -> list[Instance]:
    """drive: 駆動輪の車軸の回転角 [rad]、head_pitch: うなずき [rad](省略で直進の角度)。足の腕と押し板はうなずきから決まる。"""
    eye = np.eye(4)
    if head_pitch is None:
        head_pitch = NEUTRAL
    head = _head_pose(head_pitch)
    half_turn = d.ROT180_Z
    out = [Instance(n, n, eye, "chassis") for n in ("rail_R", "rail_L", "deck", "yoke")]
    table = rot_z(-drive)
    out += [Instance("gear_table", "gear_table", table, "table"), Instance("socket_frame", "socket_frame", table, "table")]
    spin = rot_x(drive)
    out += [Instance(n, n, spin, "axle") for n in ("pinion", "shaft")]
    out.append(Instance("sleeve_R", "sleeve", spin, "axle"))
    out.append(Instance("sleeve_L", "sleeve", spin @ half_turn, "axle"))
    out.append(Instance("wheel_drive_R", "wheel_drive", translation(DRIVE_X0, 0, 0) @ spin, "axle"))
    out.append(Instance("wheel_drive_L", "wheel_drive", translation(-DRIVE_X0, 0, 0) @ spin @ half_turn, "axle"))
    for sx, tag in ((1, "R"), (-1, "L")):  # 補助輪は右前と左後ろ
        side = eye if sx > 0 else half_turn
        y = AUX["y"]  # 右は前。左は 180° 回すので後ろに来る
        out.append(Instance(f"wheel_aux_{tag}", "wheel_aux", side @ translation(d.PLATE_OUT + SEAT, y, AUX_Z), "chassis"))
        out.append(Instance(f"clip_aux_{tag}", "clip", side @ translation(d.PLATE_OUT + d.clip_seat(AUX_CLEAR), y, AUX_Z), "chassis"))
    out.append(Instance("hinge_rod", "hinge_rod", eye, "chassis"))
    rod_clip = d.PLATE_OUT + SEAT + d.clip_seat(0.2)
    out.append(Instance("clip_hinge_R", "clip", translation(rod_clip, *HINGE), "chassis"))
    out.append(Instance("clip_hinge_L", "clip", translation(-rod_clip, *HINGE) @ half_turn, "chassis"))
    phi = arm_angle(head_pitch)
    swing = translation(0, *HINGE) @ rot_x(phi) @ translation(0, -HINGE[0], -HINGE[1])
    out.append(Instance("foot", "foot", swing, "lift"))
    out.append(Instance("fork", "fork", swing, "lift"))
    # 押し板は、下の穴が足の腕の耳のピンに、上の穴が P に合うように動く
    (py0, pz0), (iy0, iz0) = _link_ends()
    (py, pz), (iy, iz) = _pin(head_pitch), _arm_in(phi)
    turn = math.atan2(pz - iz, py - iy) - math.atan2(pz0 - iz0, py0 - iy0)
    out.append(Instance("link", "link", translation(0, iy, iz) @ rot_x(turn) @ translation(0, -iy0, -iz0), "lift"))
    out.append(Instance("clip_in", "clip", translation(LUG_X[1] + d.clip_seat(SEAT + LINK_T + 0.2), iy, iz), "lift"))
    out.append(Instance("clip_pin", "clip", translation(LINK_X[0] - 0.2 + d.clip_seat(LINK_X[1] - LINK_X[0] + 0.4), py, pz), "head"))
    out.append(Instance("lever_R", "lever_R", head, "head"))
    out.append(Instance("lego_pins_R", "lego_pins_R", head, "head"))
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


# build_car.py --one-color のプレート分け。1 枚目はピボットだけの部品、2 枚目は駆動部と車輪
ONE_COLOR_PLATES = [("Pivot new parts", None),
                    ("Drive and wheels", ["deck", "gear_table", "socket_frame", "pinion", "sleeve", "shaft", "yoke", "wheel_drive", "wheel_aux"])]
ONE_COLOR_MARGIN = 10.0  # ベッドの端からの余白 [mm]

PRINT_POSE_OVERRIDE = {
    "rail_R": ("X_UP", "内面を下(補助輪のピンを上へ立てる)", False),
    "rail_L": ("X_DOWN", "内面を下(補助輪のピンを上へ立てる)", False),
    "wheel_drive": ("X_DOWN", "外の面を下", False),
    "pinion": ("X_DOWN", "背面を下、歯と筒を上", False),
    "sleeve": ("X_UP", "内側の端を下(つばの斜面が下から広がる)", False),
    "shaft": ("Z_UP", "六角の平らな面を下にして寝かせる", False),
    "deck": ("Z_UP", "そのまま(ボスを上)", False),
    "wheel_aux": ("X_DOWN", "外の面を下", False),
    "lever_R": ("X_UP", "頭の側の面を下(ピンを上へ立てる)", False),
    "link": ("X_UP", "寝かせる", False),
    "foot": ("X_UP", "内側の面を下(軸の筒と耳のピンを上へ立てる)", False),
    "fork": ("Y_DOWN", "後ろの面を下(平らな板)", False),
    "hinge_rod": ("Z_UP", "平らに削った面を下にして寝かせる", False),
}
ENGLISH_NAMES = {
    "rail_R": "Side rail R", "rail_L": "Side rail L", "wheel_aux": "Training wheel",
    "wheel_drive": "Drive wheel", "shaft": "Hex shaft", "sleeve": "Sleeve", "pinion": "Pinion", "deck": "Deck",
    "lever_R": "Head lever R", "link": "Push link", "foot": "Foot arm", "fork": "Forked foot", "hinge_rod": "Foot hinge rod",
}

FIT_CHECKS = [
    # (説明, 動かす部品, 止める部品, ずらす量 [mm])
    ("lever_R を 1.2 mm 上げるとヨークの耳に当たる(頭が上へ抜けない)", "lever_R", "yoke", (0, 0, 1.2)),
    ("lever_R を 0.8 mm 前へずらすとヨークの耳に当たる", "lever_R", "yoke", (0, 0.8, 0)),
    ("lever_R を 0.8 mm 後ろへずらすとヨークの耳に当たる", "lever_R", "yoke", (0, -0.8, 0)),
    ("link を 0.5 mm 内へずらすとレバーのピンの段に当たる", "link", "lever_R", (-0.5, 0, 0)),
    ("link を 0.5 mm 内へずらすと足の腕の耳に当たる", "link", "foot", (-0.5, 0, 0)),
    ("link を 0.6 mm 外へずらすと耳のピンの C クリップに当たる", "link", "clip_in", (0.6, 0, 0)),
    ("foot を 0.5 mm 外へずらすと軸の筒が rail_R の内面に当たる", "foot", "rail_R", (0.5, 0, 0)),
    ("fork を 0.3 mm 上げると角穴の上の面に当たる(地面に押されても面で受ける)", "fork", "foot", (0, 0, 0.3)),
    ("fork を 0.5 mm 外へずらすとほぞの肩が足の腕に当たる", "fork", "foot", (0.5, 0, 0)),
    ("fork を 0.5 mm 内へずらすとほぞの返しが足の腕に掛かる", "fork", "foot", (-0.5, 0, 0)),
    ("hinge_rod を 0.5 mm 右へずらすと左の C クリップが rail_L に当たる", "hinge_rod", "clip_hinge_L", (0.5, 0, 0)),
    ("hinge_rod を 0.5 mm 左へずらすと右の C クリップが rail_R に当たる", "hinge_rod", "clip_hinge_R", (-0.5, 0, 0)),
    # 車軸が左右に動かない: ピニオン・駆動輪・スリーブが、左右の側板のあいだに並んで詰まっている
    ("sleeve_R を 0.4 mm 外へずらすとつばが rail_R に当たる", "sleeve_R", "rail_R", (0.4, 0, 0)),
    ("sleeve_L を 0.4 mm 外へずらすとつばが rail_L に当たる", "sleeve_L", "rail_L", (-0.4, 0, 0)),
    ("sleeve_R を 0.3 mm 内へずらすと wheel_drive_R に当たる", "sleeve_R", "wheel_drive_R", (-0.3, 0, 0)),
    ("wheel_drive_R を 0.4 mm 内へずらすとハブがピニオンに当たる", "wheel_drive_R", "pinion", (-0.4, 0, 0)),
    ("wheel_drive_L を 0.4 mm 内へずらすとハブがピニオンの筒に当たる", "wheel_drive_L", "pinion", (0.4, 0, 0)),
]

def _peak_torque_per_newton() -> float:
    """足にかかる力 1 N あたりの、首のサーボの回す力 [N·mm] の最大(仕事の釣り合い: 足の下がる速さ ÷ うなずきの速さ)。

    足が地面に着いてから端までで調べる。足は着いた直後がいちばん速く下がるので、最大は着地のときになる(端の値で見ると 2 割ほど小さく出る)。
    """
    step = math.radians(0.5)
    angles = [math.radians(a) for a in np.arange(NOD_RANGE[0] + 0.5, NOD_RANGE[1] + 0.01, 0.5)]
    loaded = [a for a in angles if foot_drop(a) + FOOT_CLEAR >= 0.0]
    return max((foot_drop(a) - foot_drop(a - step)) / step for a in loaded)


# verify_car.py の力の表。足・地面を押す駆動輪の位置 (x, y) と、前後の支えの位置と浮き (x, y, 浮き)
_FOOT = (FORK_X, FORK_Y)
_AUX_X = d.PLATE_OUT + SEAT + AUX["w"] / 2
_SUPPORTS = [(_AUX_X, AUX["y"], AUX["lift"]), (-_AUX_X, -AUX["y"], AUX["lift"])]  # 補助輪(右前、左後ろ)
_RAIL_X = (d.PLATE_IN + d.PLATE_OUT) / 2
_PHI_MAX = arm_angle(math.radians(NOD_RANGE[1]))
_LOW_POINTS = [("二股の左の先", -FORK_X, FORK_Y, _turn((FORK_Y, TIP_L_Z), _PHI_MAX)[1] - GROUND_Z),
               ("足の腕の板の下", (ARM_X[0] + ARM_X[1]) / 2, FORK_Y, _turn((FORK_Y, FORK_MID_Z - 6.0), _PHI_MAX)[1] - GROUND_Z),
               ("押し板の下の端", (LINK_X[0] + LINK_X[1]) / 2, _arm_in(_PHI_MAX)[0], _arm_in(_PHI_MAX)[1] - 4.5 - GROUND_Z)]
_WHEEL_X = DRIVE_X0 + base.TIRE_W / 2
STATICS = dict(
    cases=[
        dict(name="頭を上げきって右の足を下ろす(うなずき 30°)", foot=_FOOT, wheel=(-_WHEEL_X, 0.0), lifted=(_WHEEL_X, 0.0), supports=_SUPPORTS,
             low_points=_LOW_POINTS),
    ],
    foot_below_ground=lambda: foot_drop(math.radians(NOD_RANGE[1])) + FOOT_CLEAR,  # 端までうなずいたとき、足が地面の線より下がる量
    torque_per_newton=lambda: _peak_torque_per_newton(),
)
