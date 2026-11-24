"""コマンドラインの --model で、使う車の設計モジュールを選ぶ(このブランチは pivot = ピボットだけ。car_design と trike_design は部品を借りる元として置いてある)。"""
from __future__ import annotations

import importlib
import sys

MODELS = {"pivot": "pivot_design"}


def model_name(argv: list[str] | None = None) -> str:
    """argv から --model の値を読む(argparse より先に設計モジュールを import するため)。省略時は pivot。"""
    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return "pivot"


def load(argv: list[str] | None = None):
    """--model に対応する設計モジュール。"""
    name = model_name(argv)
    if name not in MODELS:
        raise SystemExit(f"--model は {', '.join(MODELS)} のどれか: {name}")
    return importlib.import_module(MODELS[name])
