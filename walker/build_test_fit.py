"""平歯車の試し刷り用 STL の書き出し。

右側の駆動まわり 1 組(側板・スリーブ・六角シャフト・歯車カバー)と、平歯車とクランク歯車を
歯厚の減らし量 SPUR_BACKLASH を変えて書き出す。手で回して固さを比べ、本番の値を決めるためのもの。
使い方: python walker/build_test_fit.py [--backlash 0.2,0.3]
出力: out/test_fit/
"""
from __future__ import annotations

import sys

import design as d
from build import OUT, print_shape

COMMON = ("side_plate", "sleeve", "shaft", "cover")
GEARS = ("spur", "crank")


def _clear_caches() -> None:
    # 歯形は SPUR_BACKLASH を呼び出し時に読むが、部品は lru_cache に残るので作り直させる
    for value in vars(d).values():
        if hasattr(value, "cache_clear"):
            value.cache_clear()


def _export(key: str, path) -> dict:
    mesh = d.to_trimesh(print_shape(key))
    if not mesh.is_watertight or mesh.volume <= 0:
        raise RuntimeError(f"{path.name}: 水密でない、または裏返っている")
    mesh.export(path)
    return dict(file=path.name, volume_cm3=round(mesh.volume / 1000, 2))


def main() -> int:
    backlashes = [d.SPUR_BACKLASH]
    if "--backlash" in sys.argv:
        backlashes = [float(v) for v in sys.argv[sys.argv.index("--backlash") + 1].split(",")]
    out = OUT / "test_fit"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.stl"):
        old.unlink()
    rows = [_export(key, out / f"{key}.stl") for key in COMMON]
    default = d.SPUR_BACKLASH
    try:
        for backlash in backlashes:
            d.SPUR_BACKLASH = backlash
            _clear_caches()
            rows += [_export(key, out / f"{key}_bl{backlash:.2f}.stl") for key in GEARS]
    finally:
        d.SPUR_BACKLASH = default
        _clear_caches()
    for row in rows:
        print(f"{row['file']:28s} {row['volume_cm3']:6.2f} cm3")
    total = sum(row["volume_cm3"] for row in rows)
    print(f"合計 {total:.1f} cm3(PLA 1.24 g/cm3 で約 {total * 1.24:.0f} g、充填率で減る)  出力先 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
