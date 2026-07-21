#!/usr/bin/env python3
"""Phase 5: full split-aware training for all models × conditions.

Usage:
    python scripts/train_phase5.py
    python scripts/train_phase5.py --models standard bayes --conditions history_only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.loop import full_train_run, normalize_model_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 full training.")
    p.add_argument("--grid", type=Path, default=ROOT / "configs" / "grids" / "v1_rnn_grid.yaml")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text())
    train_eids = list(splits["train"])
    val_eids = list(splits["val"])
    test_eids = list(splits["test"])
    # Hard guarantee: never train on test
    assert not (set(train_eids) & set(test_eids))
    assert not (set(val_eids) & set(test_eids))

    models = args.models or grid["models"]
    conditions = args.conditions or grid["conditions"]
    max_epochs = args.max_epochs or int(grid["max_epochs"])
    patience = int(grid["patience"])
    batch_size = int(grid["batch_size"])
    default = grid["default_run"]

    stamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []

    for model in models:
        model = normalize_model_name(model)
        for condition in conditions:
            runs = [default]
            if condition == "history_only":
                runs = [default, *grid.get("history_only_extra", [])]
            for run in runs:
                result = full_train_run(
                    model,
                    condition,
                    ROOT,
                    train_eids=train_eids,
                    val_eids=val_eids,
                    hidden_size=int(run["hidden_size"]),
                    lr=float(run["lr"]),
                    lambda_rt=float(run["lambda_rt"]),
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    run_name=str(run["name"]),
                )
                bm = result["best_metrics"]
                rows.append(
                    {
                        "created_utc": stamp,
                        "model": model,
                        "condition": condition,
                        "run_name": run["name"],
                        "checkpoint": result["checkpoint"],
                        "best_epoch": bm.get("best_epoch"),
                        "val_choice_nll": bm.get("val_choice_nll"),
                        "val_rt_nll": bm.get("val_rt_nll"),
                        "val_choice_acc": bm.get("val_choice_acc"),
                        "val_loss": bm.get("val_loss"),
                        "train_choice_nll": bm.get("train_choice_nll"),
                        "train_rt_nll": bm.get("train_rt_nll"),
                        "hidden_size": run["hidden_size"],
                        "lr": run["lr"],
                        "lambda_rt": run["lambda_rt"],
                    }
                )

    out_dir = ROOT / "reports" / "behavior"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_selection_table.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Selection on history_only default runs
    hist = [r for r in rows if r["condition"] == "history_only" and r["run_name"] == "default"]
    hist_sorted = sorted(hist, key=lambda r: r["val_choice_nll"])
    best_choice = hist_sorted[0]["val_choice_nll"] if hist_sorted else None
    # Provisional epsilon: max gap among models or 0.05, whichever larger of small floor
    if len(hist_sorted) >= 2:
        spread = hist_sorted[-1]["val_choice_nll"] - hist_sorted[0]["val_choice_nll"]
        epsilon = max(0.02, min(0.05, spread + 0.01))
    else:
        epsilon = 0.05
    rt_vals = [r["val_rt_nll"] for r in hist_sorted if r["val_rt_nll"] is not None]
    rt_floor = max(rt_vals) + 0.5 if rt_vals else None  # permissive secondary floor

    selection = {
        "created_utc": stamp,
        "history_only_ranking": hist_sorted,
        "best_val_choice_nll": best_choice,
        "provisional_choice_epsilon": epsilon,
        "provisional_rt_secondary_floor": rt_floor,
        "note": (
            "Epsilon/RT floor are provisional from this Phase 5 cohort; "
            "written into frozen_v1.yaml for later behavior-matched neural compare."
        ),
    }
    sel_path = out_dir / "model_selection_summary.json"
    sel_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {sel_path}")
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
