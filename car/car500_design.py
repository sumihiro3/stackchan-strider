"""スタックチャン・チョロQ風 FIAT 500: レトロカー(car_design.py)の駆動部とかじ取りを、短い車軸の間隔で使い回す小さい車。

前後の車軸の間隔を 105 → WHEELBASE_500 mm に縮め、長いボンネットの代わりに丸い鼻の 1 体の殻をかぶせる。
重さはほぼ駆動輪(後ろ)の真上にあるので、鼻の中に 10 円玉を入れる筒を作り、前輪にかかる重さを足す。
座標系・部品の持ち方は car_design.py と同じ(原点 = 後輪の車軸、地面 z = -WHEEL_R)。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold

import car_design as base

WHEELBASE_500 = 78.0  # 前輪を ±30° 切ったとき、タイヤとフェンダーが側板(y ≤ 39.5)と後輪に当たらない最小に近い値
base.configure(wheelbase=WHEELBASE_500, rail_taper_y=(34.0, 50.0))
base.FRONT_FENDER_DEG = (40.0, 150.0)  # 前のフェンダーの前端を短くして、鼻より前に出さない(部品を作る前に変える)

from car_design import *  # noqa: E402,F403  configure() の後で読む(間隔から決まる値を新しい値で受け取る)
from car_design import (  # noqa: E402
    _ball, _bumper_bar, _crowned, _dome, _hood_snap_frame, _largest, _top_z, ring_sector_x,
)

d = base.d
OUT = ROOT / "out" / "car500"  # noqa: F405
PUBLISH = ROOT / "print_car500"  # noqa: F405
PROJECT_NAME = "stackchan_car500"
COLORS = dict(base.COLORS, orange="#e8742a")
PLATE_ROLES = (("gray", "Gray"), ("orange", "Orange"), ("black", "Black"))
PLATE_GROUP = {"hat": False, "top500": False, "fender_front_R": True, "fender_front_L": True}  # build_car.py: 部品の少ないプレートを作らない
VIEWS = {  # render_car.py のカメラ(位置 [mm]、注視点 [mm])
    "front34": ((230, 260, 140), (0, 25, 15)),
    "rear34": ((-230, -250, 150), (0, 15, 15)),
    "side": ((480, 25, 20), (0, 25, 20)),
    "top": ((0, 26, 520), (0, 26, 0)),
    "front": ((0, 500, 30), (0, 25, 30)),
}

# ---------------------------------------------------------------- 殻の寸法
SIDE_X = 56.0  # 横の壁の外面 |x|(後輪の内面 58 の内側。側板の台の出っぱりを受ける)
BELT_Z = 50.0  # 窓の下の線(頭の下端 43.5 を隠す)
REAR_Y = -53.0  # 後ろの端
NOSE_Y = 108.0  # 鼻の先(前輪の前端と同じ)
SHELL_R = 12.0  # 殻の角の丸み
FRONT_FLOOR_Z = 12.5  # 前輪の間では殻の下をここまで上げる(横棒・ナックル・タイロッドの上)
NOSE_FLOOR_Z = 4.0  # タイロッドより前では下げて鼻を厚く見せる
COCKPIT_500 = dict(half_w=42.0, y0=-27.0, y1=47.0, radius=12.0)  # 上の開口(キャンバストップを開けたところ)。ハットの腕と引き棒を通す
WINDSHIELD_Y_500 = 49.2

# ---------------------------------------------------------------- おもり(10 円玉)
COIN_R, COIN_T, COIN_G = 23.5 / 2, 1.5, 4.5
COINS = 5
BALLAST = dict(y=89.0, z0=13.0, wall=1.6, plug=2.0, flange=1.2)
BALLAST_HOLE_R = COIN_R + 0.1
EXTRA_MASSES = [(f"10 円玉 {COINS} 枚", COINS * COIN_G, (0.0, BALLAST["y"], BALLAST["z0"] + BALLAST["plug"] + COINS * COIN_T / 2))]


def _shell_points():
    """殻の外形の球の中心(左右対称)。凸包が FIAT 500 の丸い箱になる。"""
    r = SHELL_R
    pts = [
        (SIDE_X - r, -40.0, -6.0 + r), (SIDE_X - r, 34.0, -6.0 + r),  # 下の角
        (SIDE_X - r, -34.0, BELT_Z - r), (SIDE_X - r, 40.0, BELT_Z - r),  # 窓の下の線(横の壁は縦。内面 54 がヨークのツメの先 53.4 の外)
        (32.0, REAR_Y + r, 4.0), (28.0, REAR_Y + r + 6.0, 24.0), (20.0, -40.0, BELT_Z - r - 2.0),  # 丸く下がるエンジンフード
        (24.0, 60.0, 30.0), (16.0, 82.0, 22.0),  # 前へ下がるボンネット
        (22.0, NOSE_Y - r, 8.0), (20.0, NOSE_Y - r - 2.0, 20.0), (26.0, 70.0, 4.0),  # 立った丸い鼻
    ]
    return mirror_x(pts), r  # noqa: F405


def shell_outer() -> Manifold:
    pts, r = _shell_points()
    return blob(pts, r)  # noqa: F405


def shell_inner(extra: float = 0.0) -> Manifold:
    pts, r = _shell_points()
    return blob(pts, r, SKIN + extra)  # noqa: F405


def _steer_envelope_local() -> Manifold:
    """前輪・タイヤ・キャップ・フェンダー・支柱が占める範囲(ナックルのローカル。余裕 1〜1.5 mm)。"""
    x0, x1 = 1.0, 22.5
    tire = cyl_x(WHEEL_R + 1.0, x0, x1)  # noqa: F405
    fender = cyl_x(FENDER_R + FENDER_T + 2.0, x0, x1) ^ box(x0 - 1, x1 + 1, -50, 50, 12.0, 60)  # noqa: F405
    return tire + fender


@lru_cache(maxsize=None)
def steer_clearance() -> Manifold:
    """前輪が ±STEER_MAX 振れる範囲(左右)。殻から抜いて、タイヤハウスにする。"""
    env = _steer_envelope_local()
    steps = np.linspace(-STEER_MAX, STEER_MAX, 13)  # noqa: F405
    right = Manifold()
    for a, b in zip(steps[:-1], steps[1:]):
        pa = apply(env, translation(KINGPIN_X, WHEELBASE, 0) @ rot_z(a))  # noqa: F405
        pb = apply(env, translation(KINGPIN_X, WHEELBASE, 0) @ rot_z(b))  # noqa: F405
        right += Manifold.batch_hull([pa, pb])
    return right + right.mirror((1, 0, 0))


@lru_cache(maxsize=None)
def link_clearance() -> Manifold:
    """引き棒(J → 折れ目 → ドラッグアームのピン)がうなずき 0〜30° で通る範囲。線分を半径 4.5 の球の凸包で包む。"""
    knee_local = np.array([sum(LINK_X) / 2, LINK_KNEE[0], LINK_KNEE[1], 1.0])  # noqa: F405
    poses = []
    for theta in np.radians(np.linspace(NOD_RANGE[0], NOD_RANGE[1], 11)):  # noqa: F405
        delta = steer_from_nod(theta)  # noqa: F405
        knee = (link_pose(theta, delta) @ knee_local)[:3]  # noqa: F405
        poses.append((j_world(theta), knee, k_world(delta)))  # noqa: F405
    out = Manifold()
    for (j0, n0, k0), (j1, n1, k1) in zip(poses[:-1], poses[1:]):
        for a0, b0, a1, b1 in ((j0, n0, j1, n1), (n0, k0, n1, k1)):
            out += Manifold.batch_hull([_ball(*p, 4.5) for p in (a0, b0, a1, b1)])
    return out



def _rear_fender_ring() -> Manifold:
    """後輪の上のふくらみ(右)。殻の横の壁から外へ張り出し、タイヤを覆う。"""
    section = _crowned(SIDE_X - 1.0, FENDER_X[1], FENDER_R, FENDER_T, 2.0)  # noqa: F405
    return ring_sector_x(section, *REAR_FENDER_DEG)  # noqa: F405


def ballast_tube() -> Manifold:
    """鼻の中の 10 円玉の筒(下が開いていて、ふたを差す)。上は殻の内面につなぐ。"""
    b = BALLAST
    tube = cyl(BALLAST_HOLE_R + b["wall"], b["z0"], 60.0).translate((0, b["y"], 0)) ^ shell_inner(-0.5)  # noqa: F405
    return tube - cyl(BALLAST_HOLE_R, b["z0"] - 1.0, b["z0"] + b["plug"] + COINS * COIN_T + 0.3).translate((0, b["y"], 0))  # noqa: F405


@lru_cache(maxsize=None)
def body500_part() -> Manifold:
    """1 体の殻。上は大きく開け(キャンバストップを開けた形)、前輪の振れる範囲と後輪の上をくり抜く。"""
    outer, inner = shell_outer(), shell_inner()
    body = outer - inner
    body = body.trim_by_plane((0, 0, 1), -6.0 + SKIN + 0.2)  # 底は開ける
    wb = WHEELBASE  # noqa: F405
    body -= box(-80, 80, 40.0, wb + TIE_ARM + 9.0, -20, FRONT_FLOOR_Z)  # noqa: F405  前輪の間は横棒・ナックル・タイロッドの上まで開ける
    body -= box(-80, 80, wb + TIE_ARM + 9.0, NOSE_Y + 5, -20, NOSE_FLOOR_Z)  # noqa: F405
    c = COCKPIT_500
    opening = hull2(circle(c["radius"], -c["half_w"] + c["radius"], c["y0"] + c["radius"]),  # noqa: F405
                    circle(c["radius"], c["half_w"] - c["radius"], c["y0"] + c["radius"]),  # noqa: F405
                    rect(-c["half_w"], c["half_w"], c["y0"] + c["radius"], c["y1"]))  # noqa: F405
    body -= opening.extrude(60).translate((0, 0, 30.0))
    # 後輪: 軸とハブを通し、上は外へ張り出したふくらみで覆う
    for sx in (1, -1):
        body -= cyl_x(FENDER_R - 1.0, 40.0 if sx > 0 else -80.0, 80.0 if sx > 0 else -40.0)  # noqa: F405
    ring = _rear_fender_ring()
    body += ring + ring.mirror((1, 0, 0))
    body -= steer_clearance()
    body -= link_clearance()
    body -= rails_clearance()  # noqa: F405
    # 側板の台の出っぱり(レトロカーと同じ位置)
    dt = DETENT  # noqa: F405
    for sx in (1, -1):
        for sy in (1, -1):
            body += _dome(sx * (SIDE_X - SKIN), sy * dt["y"], dt["z"], dt["r"], dt["h"], -sx)  # noqa: F405
    # 横棒の上へ下ろすブロックとツメ(レトロカーのボンネットと同じ。横棒と側板の台の床を貫く)
    x0, x1, _ = HOOD_BLOCK  # noqa: F405
    for sx in (1, -1):
        block = box(x0, x1, wb - 5.0, wb + 5.0, BEAM_Z[1] + 0.2, 60.0) ^ shell_inner(-0.5)  # noqa: F405
        body += block if sx > 0 else block.mirror((1, 0, 0))
    body += ballast_tube()
    # 後ろのバンパーのツメを受ける台(後ろの壁の内側)と角穴
    for sx in (1, -1):
        body += box(sx * BUMPER_STAY_X - 5.0, sx * BUMPER_STAY_X + 5.0, REAR_Y - 1.0, BUMPER_R_BOSS_Y_500, -6.0 + SKIN + 0.2, 12.0) ^ outer  # noqa: F405
        body -= snap_hole(*_bumper_r_frame(sx), -2.0, 14.0)  # noqa: F405
        tz = _top_z(outer, sx * TOP_PEG_500["x"], TOP_PEG_500["y"])  # noqa: F405
        body += cyl(TOP_PEG_500["r"], tz - 1.5, tz + TOP_PEG_500["h"]).translate((sx * TOP_PEG_500["x"], TOP_PEG_500["y"], 0))  # noqa: F405
    body -= body_decor_holes()
    # エンジンフードの横のスリット(5 本)と、後ろの角の丸いテールランプの溝
    cutter = Manifold()
    for k in range(5):
        z = 14.0 + 3.2 * k
        cutter += box(-14.0, 14.0, -80.0, -40.0, z - 0.7, z + 0.7)  # noqa: F405
    for sx in (1, -1):
        ring = circle(3.4, sx * 30.0, 22.0) - circle(2.2, sx * 30.0, 22.0)  # noqa: F405
        cutter += ring.extrude(40.0).transform(np.array([[1, 0, 0, 0], [0, 0, 1, -80.0], [0, 1, 0, 0]], dtype=np.float64))
    body = engrave(body, cutter, outer, blob(*_shell_points(), ENGRAVE))  # noqa: F405
    body = _largest(body)
    for sx in (1, -1):
        body += snap_tenon(*_hood_snap_frame(sx), HOOD_SNAP_LEN + 0.1).translate((0, 0, 0.1))  # noqa: F405
    return body



# ---------------------------------------------------------------- 飾り(グレー)と小物
def _front_surface_y(x: float, z: float) -> float:
    """殻の外面の、(x, z) での前の端の y。"""
    probe = shell_outer() ^ box(x - 0.05, x + 0.05, 0.0, 200.0, z - 0.05, z + 0.05)  # noqa: F405
    return probe.bounding_box()[4]


def _rear_surface_y(x: float, z: float) -> float:
    probe = shell_outer() ^ box(x - 0.05, x + 0.05, -200.0, 0.0, z - 0.05, z + 0.05)  # noqa: F405
    return probe.bounding_box()[1]


HEADLIGHT_X, HEADLIGHT_Z = 21.0, 21.0
BADGE_Z = 14.5  # ひげ形のモールと紋章の高さ
BADGE_PEG = dict(x=7.0, r=1.0, depth=4.0)
MIRROR_500 = dict(x=50.0, y=43.0, peg_r=1.5)
BUMPER_BAR_R = 2.2
BAR_Z = STAY_Z[0] + BUMPER_BAR_R  # noqa: F405  棒の中心(下面を腕の下面にそろえ、平らな面を下にして印刷する)
BUMPER_F_Y, BUMPER_R_Y, BUMPER_R_Z = NOSE_Y - 1.0, REAR_Y - BUMPER_BAR_R - 0.3, 4.0
BUMPER_R_BOSS_Y_500 = REAR_Y + 7.5  # 後ろの壁の内側の台の内面(ツメの返しが掛かる)
TOP_PEG_500 = dict(x=24.0, y=-47.0, r=1.0, h=3.5)
LID_GAP = 0.3  # ふたのつばと筒の下端のすきま


@lru_cache(maxsize=None)
def headlight_poses() -> list:
    """丸目のヘッドライトの姿勢(レトロカーと同じ部品。殻の前の面から半分ほど出す)。"""
    out = []
    for sx in (1, -1):
        y = _front_surface_y(sx * HEADLIGHT_X, HEADLIGHT_Z)
        out.append(translation(sx * HEADLIGHT_X, y - 3.5, HEADLIGHT_Z))  # noqa: F405
    return out


def _bar(points, r: float) -> Manifold:
    """点列を球の凸包でつないだ棒。"""
    out = Manifold()
    for a, b in zip(points[:-1], points[1:]):
        out += Manifold.batch_hull([_ball(*a, r), _ball(*b, r)])
    return out


@lru_cache(maxsize=None)
def badge_part() -> Manifold:
    """鼻のひげ形のモール(左右へ伸びて先が少し上がる)と、真ん中の紋章。裏のピン 2 本を鼻の穴に差す。"""
    pts = []
    for x in np.linspace(-13.0, 13.0, 11):
        z = BADGE_Z + 0.012 * x * x  # 先が上がる
        pts.append((x, _front_surface_y(x, z) + 0.3, z))
    strip = _bar(pts, 0.9)
    y0 = _front_surface_y(0.0, BADGE_Z)
    to_y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)
    crest = Manifold.cylinder(2.0, 4.0, 3.6, 48).transform(to_y).translate((0, y0 - 0.8, BADGE_Z))
    crest -= Manifold.cylinder(1.0, 2.6, 2.6, 48).transform(to_y).translate((0, y0 + 0.8, BADGE_Z))
    part = (strip + crest) - shell_outer()  # 裏を鼻の曲面に沿わせる
    for sx in (1, -1):
        y = _front_surface_y(sx * BADGE_PEG["x"], BADGE_Z + 0.012 * BADGE_PEG["x"] ** 2)
        part += cyl_y(BADGE_PEG["r"], y - BADGE_PEG["depth"], y + 0.5, sx * BADGE_PEG["x"], BADGE_Z + 0.012 * BADGE_PEG["x"] ** 2)
    return part


def cyl_y(r: float, y0: float, y1: float, x: float = 0.0, z: float = 0.0) -> Manifold:
    to_y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float64)
    return Manifold.cylinder(y1 - y0, r, r, 48).transform(to_y).translate((x, y0, z))


def _bumper_bar_500(half: float = 34.0, wrap: bool = True, r: float = BUMPER_BAR_R) -> Manifold:
    """細いバンパーの棒(ローカル: 中心が原点。wrap なら両端が -y へ回り込む)。"""
    if not wrap:
        return _bar([(-half, 0.0, 0.0), (half, 0.0, 0.0)], r)
    return _bar([(-half - 3.0, -5.0, 0.0), (-half, 0.0, 0.0), (half, 0.0, 0.0), (half + 3.0, -5.0, 0.0)], r)


def _bumper_f_frame(sx: float = 1.0):
    return (sx * BUMPER_STAY_X, WHEELBASE + 5.0 + 0.2, sum(STAY_Z) / 2), (0, -1, 0), (1, 0, 0)  # noqa: F405


def _bumper_r_frame(sx: float = 1.0):
    return (sx * BUMPER_STAY_X, BUMPER_R_Y + 2.0, BUMPER_R_Z + sum(STAY_Z) / 2), (0, 1, 0), (1, 0, 0)  # noqa: F405


@lru_cache(maxsize=None)
def bumper_front500_part() -> Manifold:
    """前のバンパー。腕 2 本が鼻の下を後ろへ伸び、先のツメが横棒を前から貫く(レトロカーと同じ留め方)。"""
    part = _bumper_bar_500().translate((0, BUMPER_F_Y, BAR_Z))
    for sx in (1, -1):
        part += box(sx * BUMPER_STAY_X - 2.5, sx * BUMPER_STAY_X + 2.5, WHEELBASE + 5.2, BUMPER_F_Y, STAY_Z[0], STAY_Z[1])  # noqa: F405
        part += snap_tenon(*_bumper_f_frame(sx), 10.0 + 0.2 + 0.2)  # noqa: F405
    return part


@lru_cache(maxsize=None)
def bumper_rear500_part() -> Manifold:
    """後ろのバンパー。短い腕の先のツメが、後ろの壁と内側の台を貫く。"""
    part = _bumper_bar_500(28.0, wrap=False).translate((0, BUMPER_R_Y, BUMPER_R_Z + BAR_Z))  # 後ろの角は丸く引っ込むので回り込ませない
    for sx in (1, -1):
        part += box(sx * BUMPER_STAY_X - 2.5, sx * BUMPER_STAY_X + 2.5, BUMPER_R_Y, BUMPER_R_Y + 2.0, BUMPER_R_Z + STAY_Z[0], BUMPER_R_Z + STAY_Z[1])  # noqa: F405
        part += snap_tenon(*_bumper_r_frame(sx), BUMPER_R_BOSS_Y_500 - (BUMPER_R_Y + 2.0) + 0.2)  # noqa: F405
    return part


@lru_cache(maxsize=None)
def windshield500_part() -> Manifold:
    """風防の枠(ガラスは入れない)。根元を殻の曲面に合わせ、縦の辺の下のピンを殻の上面の穴に差す。"""
    half, z0, z1 = 40.0, BELT_Z - 6.0, BELT_Z + 14.0
    frame = CrossSection.square((2 * half, z1 - z0)).translate((-half, z0)).offset(-0.01)
    frame = frame.offset(3.0, d.JoinType.Round) - CrossSection.square((2 * half - 6, z1 - z0 - 4)).translate((-half + 3, z0 + 3)).offset(2.0, d.JoinType.Round)
    plate = frame.extrude(2.4).transform(np.array([[1, 0, 0, 0], [0, 0, 1, WINDSHIELD_Y_500 - 1.2], [0, 1, 0, 0]], dtype=np.float64))
    plate -= shell_outer()
    for sx in (1, -1):
        tz = _top_z(shell_outer(), sx * WINDSHIELD_PEG["x"], WINDSHIELD_Y_500)  # noqa: F405
        plate += cyl(WINDSHIELD_PEG["r"], tz - WINDSHIELD_PEG["depth"], tz + 0.5).translate((sx * WINDSHIELD_PEG["x"], WINDSHIELD_Y_500, 0))  # noqa: F405
    return plate


@lru_cache(maxsize=None)
def top500_part() -> Manifold:
    """たたんだキャンバストップ(黒)。後ろのエンジンフードの上に載せ、ピン 2 本にかぶせる。"""
    y = TOP_PEG_500["y"]
    z = _top_z(shell_outer(), 0.0, y) + 3.0  # noqa: F405
    roll = Manifold.batch_hull([_ball(sx * 34.0, y, z, 6.0) for sx in (1, -1)] + [_ball(sx * 30.0, y - 4.0, z - 1.5, 5.0) for sx in (1, -1)])
    for x in (-18.0, 0.0, 18.0):  # たたんだ骨
        roll -= box(x - 0.6, x + 0.6, y - 20, y + 20, z + 4.0, z + 20)  # noqa: F405
    roll = roll - shell_outer()
    for sx in (1, -1):
        tz = _top_z(shell_outer(), sx * TOP_PEG_500["x"], y)  # noqa: F405
        roll -= cyl(TOP_PEG_500["r"] + PRESS_FIT, tz - 1.0, tz + TOP_PEG_500["h"] + 0.5).translate((sx * TOP_PEG_500["x"], y, 0))  # noqa: F405
    return roll


@lru_cache(maxsize=None)
def mirror500_part() -> Manifold:
    """右のドアミラー(設計の座標)。窓の下の線の上に立ち、丸い鏡が後ろを向く。下の棒を穴に差す。"""
    x, y = MIRROR_500["x"], MIRROR_500["y"]
    z = _top_z(shell_outer(), x, y)  # noqa: F405
    foot = cyl(2.6, z - 3.0, z + 1.0).translate((x, y, 0)) - shell_outer()  # noqa: F405
    peg = cyl(MIRROR_500["peg_r"], z - 5.0, z).translate((x, y, 0))  # noqa: F405
    stalk = Manifold.batch_hull([_ball(x, y, z + 0.5, 1.3), _ball(x + 3.0, y, z + 6.0, 1.1)])
    head = Manifold.sphere(4.5, 48).scale((1.0, 0.6, 0.85)).translate((x + 4.0, y, z + 9.0))
    head -= Manifold.sphere(3.6, 48).scale((1.0, 0.3, 0.8)).translate((x + 4.0, y - 2.4, z + 9.0))
    return foot + peg + stalk + head


@lru_cache(maxsize=None)
def ballast_lid_part() -> Manifold:
    """おもりの筒のふた(設計の座標)。差し込みの輪を筒の穴に押し込む。10 円玉はこの上に重ねる。"""
    b = BALLAST
    z0 = b["z0"]
    flange = cyl(BALLAST_HOLE_R + b["wall"], z0 - LID_GAP - b["flange"], z0 - LID_GAP)  # noqa: F405
    plug = cyl(BALLAST_HOLE_R - PRESS_FIT, z0 - LID_GAP - 0.01, z0 + b["plug"]) - cyl(BALLAST_HOLE_R - PRESS_FIT - 1.2, z0, z0 + b["plug"] + 1)  # noqa: F405
    return (flange + plug).translate((0, b["y"], 0))


def ballast_part() -> Manifold:
    """10 円玉の束(印刷しない。干渉と重さの確認用)。"""
    z0 = BALLAST["z0"] + BALLAST["plug"]
    return cyl(COIN_R, z0, z0 + COINS * COIN_T).translate((0, BALLAST["y"], 0))  # noqa: F405


def body_decor_holes() -> Manifold:
    """殻にあける、飾りを差す穴(ピンより PRESS_FIT だけ太い)。"""
    holes = Manifold()
    for pose in headlight_poses():
        holes += apply(headlight_socket(), pose)  # noqa: F405
    for sx in (1, -1):
        z = BADGE_Z + 0.012 * BADGE_PEG["x"] ** 2
        y = _front_surface_y(sx * BADGE_PEG["x"], z)
        holes += cyl_y(BADGE_PEG["r"] + PRESS_FIT, y - BADGE_PEG["depth"] - 0.5, y + 1.0, sx * BADGE_PEG["x"], z)  # noqa: F405
        tz = _top_z(shell_outer(), sx * WINDSHIELD_PEG["x"], WINDSHIELD_Y_500)  # noqa: F405
        holes += cyl(WINDSHIELD_PEG["r"] + PRESS_FIT, tz - WINDSHIELD_PEG["depth"] - 0.5, tz + 1.0).translate((sx * WINDSHIELD_PEG["x"], WINDSHIELD_Y_500, 0))  # noqa: F405
        mz = _top_z(shell_outer(), MIRROR_500["x"], MIRROR_500["y"])  # noqa: F405
        holes += cyl(MIRROR_500["peg_r"] + PRESS_FIT, mz - 6.0, mz + 1.0).translate((sx * MIRROR_500["x"], MIRROR_500["y"], 0))  # noqa: F405
    return holes


# ---------------------------------------------------------------- 部品表と姿勢
RETRO_ONLY = {"body_rear", "body_front", "windshield", "top", "molding_body_R", "molding_body_L", "molding_hood_R", "molding_hood_L",
              "mirror_R", "mirror_L", "fender_rear_R", "fender_rear_L", "bumper_front", "bumper_rear", "grille"}
RECOLOR = {"fender_front_R": "orange", "fender_front_L": "orange", "hubcap": "gray"}


def _own_parts() -> list:
    c = COLORS
    return [
        PartDef("body500", "殻(ボディ)", body500_part, c["orange"], 1),  # noqa: F405
        PartDef("headlight", "ヘッドライト(レトロカーと同じ)", headlight_part, c["gray"], 2),  # noqa: F405
        PartDef("badge", "鼻のひげのモールと紋章", badge_part, c["gray"], 1),  # noqa: F405
        PartDef("bumper_front500", "前のバンパー", bumper_front500_part, c["gray"], 1),  # noqa: F405
        PartDef("bumper_rear500", "後ろのバンパー", bumper_rear500_part, c["gray"], 1),  # noqa: F405
        PartDef("windshield500", "風防の枠", windshield500_part, c["gray"], 1),  # noqa: F405
        PartDef("top500", "たたんだキャンバストップ", top500_part, c["black"], 1),  # noqa: F405
        PartDef("mirror500_R", "右のドアミラー", mirror500_part, c["gray"], 1),  # noqa: F405
        PartDef("mirror500_L", "左のドアミラー", lambda: mirror500_part().mirror((1, 0, 0)), c["gray"], 1),  # noqa: F405
        PartDef("ballast_lid", "おもりの筒のふた", ballast_lid_part, c["gray"], 1),  # noqa: F405
        PartDef("ballast", f"10 円玉 {COINS} 枚(おもり)", ballast_part, "#b87333", 1, printable=False),  # noqa: F405
    ]


def part_defs() -> dict:
    defs = {}
    for key, p in base.part_defs().items():
        if key in RETRO_ONLY:
            continue
        if key in RECOLOR:
            p = PartDef(p.key, p.label, p.build, COLORS[RECOLOR[key]], p.count, p.printable)  # noqa: F405
        defs[key] = p
    for p in _own_parts():
        defs[p.key] = p
    return defs


def _own_instances() -> list:
    eye = np.eye(4)
    out = [Instance(n, n, eye, "chassis") for n in ("body500", "badge", "bumper_front500", "bumper_rear500", "windshield500", "top500",  # noqa: F405
                                                    "mirror500_R", "mirror500_L", "ballast_lid", "ballast")]
    out += [Instance(f"headlight_{t}", "headlight", pose, "chassis") for pose, t in zip(headlight_poses(), ("R", "L"))]  # noqa: F405
    return out


def instances(drive: float = 0.0, head_pitch: float | None = None, with_stackchan: bool = True) -> list:
    """car_design.instances() の駆動部・かじ取り・足まわり・ハット・スタックチャンに、この車の殻と飾りを足す。"""
    out = [i for i in base.instances(drive, head_pitch, with_stackchan) if i.part not in RETRO_ONLY and i.part != "headlight"]
    return out + _own_instances()


@lru_cache(maxsize=None)
def part_shape(key: str) -> Manifold:
    shape = part_defs()[key].build()
    if shape.status().name != "NoError" or shape.is_empty():
        raise RuntimeError(f"部品 {key} の形状が不正: {shape.status()}")
    return shape


def placed(instance) -> Manifold:
    return apply(part_shape(instance.part), instance.matrix)  # noqa: F405
