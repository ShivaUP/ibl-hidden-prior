#!/usr/bin/env python3
"""07 — Train v2 model(s) on synthetic sessions.

Usage:
  python scripts/07_train_model.py --model tanh_bptt
  python scripts/07_train_model.py --model tanh_pc
  python scripts/07_train_model.py --model gru_pc --epochs 2
  python scripts/07_train_model.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_v2.train import ACTIVE_MODELS, save_checkpoint, train_model
from src.synthetic.schema import load_synthetic_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(ACTIVE_MODELS), default=None)
    p.add_argument("--all", action="store_true", help="Train all active models in order")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    if not args.all and not args.model:
        p.error("provide --model ID or --all")
    cfg = load_synthetic_config()
    models = list(ACTIVE_MODELS) if args.all else [args.model]
    results = []
    for mid in models:
        model, meta = train_model(mid, cfg, epochs=args.epochs, verbose=True)
        out = ROOT / cfg["paths"]["artifacts"] / "models" / mid
        path = save_checkpoint(model, meta, out)
        results.append(
            {
                "model": mid,
                "saved": str(path),
                "final_loss": meta["history"][-1]["loss"],
            }
        )
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
