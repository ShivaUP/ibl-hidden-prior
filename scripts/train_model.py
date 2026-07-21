#!/usr/bin/env python3
"""Train entry point (Phase 4 smoke / later full runs).

Usage:
    python scripts/train_model.py --model standard --smoke
    python scripts/train_model.py --model pc --smoke
    python scripts/train_model.py --model bayes --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.loop import smoke_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train IBL prior models.")
    p.add_argument(
        "--model",
        required=True,
        choices=["standard", "pc", "bayes", "standard_rnn", "pc_rnn", "bayesian"],
    )
    p.add_argument("--condition", default="history_only")
    p.add_argument("--smoke", action="store_true", help="Tiny subset, few epochs.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambda-rt", type=float, default=0.2)
    p.add_argument("--hidden-size", type=int, default=64)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        epochs = args.epochs or 3
        max_trials = args.max_trials or 256
    else:
        epochs = args.epochs or 20
        max_trials = args.max_trials

    # Normalize aliases
    name = args.model
    if name == "standard_rnn":
        name = "standard"
    if name == "pc_rnn":
        name = "pc"
    if name == "bayesian":
        name = "bayes"

    result = smoke_train(
        name,
        ROOT,
        condition=args.condition,
        max_trials=max_trials,
        epochs=epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lambda_rt=args.lambda_rt,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
