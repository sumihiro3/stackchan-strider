"""コマンドラインの --model で、使う車の設計モジュールを選ぶ(car = レトロカー。トライクが部品を使う、trike = 逆三輪車、pivot = 片側を浮かせて旋回する車)。"""
from __future__ import annotations

import importlib
import sys

MODELS = {"car": "car_design", "trike": "trike_design", "pivot": "pivot_design"}


def model_name(argv: list[str] | None = None) -> str:
    """argv から --model の値を読む(argparse より先に設計モジュールを import するため)。省略時は car。"""
    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return "car"


def load(argv: list[str] | None = None):
    """--model に対応する設計モジュール。"""
    name = model_name(argv)
    if name not in MODELS:
        raise SystemExit(f"--model は {', '.join(MODELS)} のどれか: {name}")
    return importlib.import_module(MODELS[name])
