#!/usr/bin/env python3
"""14 — Hidden-size capacity sweep for the RNN-family models.

Trains each RNN model (tanh_bptt, gru, optionally tanh_pc) at several hidden
sizes, then measures two things on the synthetic held-out set:

  1. Held-out choice accuracy (behavioral fit)
  2. Block-prior decodability (linear-decoder AUROC on the pre-stimulus state)

Plotting metric-vs-size reveals the *elbow*: the smallest hidden size where both
metrics plateau. Bigger than the elbow adds compute/parameters with no gain.

The frozen v1/v2 default is 48. Bayes is excluded (it ignores hidden_size).

Trainings run **in memory** — the frozen checkpoints in artifacts/v2/models are
NOT overwritten.

Usage
-----
  python scripts/14_hidden_size_sweep.py
  python scripts/14_hidden_size_sweep.py --sizes 8 16 32 48 64 96
  python scripts/14_hidden_size_sweep.py --model tanh_bptt --epochs 30
  python scripts/14_hidden_size_sweep.py --load-existing   # replot only

Output
------
  reports/v2/block_decoder/hidden_size_sweep.csv
  reports/v2/block_decoder/hidden_size_sweep.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import SyntheticBatch
from src.synthetic.schema import load_synthetic_config
from src.models_v2.train import train_model
from src.models_v2.rollout import accuracy_and_ce, rollout_closed_loop
from src.eval.block_decoder import (
    extract_labels_flat,
    extract_latent_states,
    fit_block_decoder,
)

DEFAULT_SIZES = [8, 16, 32, 48, 64]
DEFAULT_MODELS = ["tanh_bptt", "gru"]


def _load_heldout(cfg: dict) -> SyntheticBatch:
    path = ROOT / "data" / "processed" / "synthetic_v2" / "heldout_sessions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun: python scripts/05_build_synthetic_datasets.py"
        )
    z = np.load(path)
    phase = PhaseTicks.from_config(cfg)
    return SyntheticBatch(
        probability_left=z["probability_left"],
        p_right=z["p_right"],
        block_id=z["block_id"],
        side=z["side"],
        contrast=z["contrast"],
        phase=phase,
    )


def _n_params(model) -> int:
    return int(sum(v.size for v in model.parameters().values()))


def sweep_one(
    model_id: str,
    hidden_size: int,
    cfg: dict,
    batch: SyntheticBatch,
    *,
    epochs: int | None,
    folds: int,
    seed: int,
) -> dict:
    cfg_size = {**cfg, "hidden_size": hidden_size}

    model, _ = train_model(model_id, cfg_size, epochs=epochs, verbose=False)

    # Behavioral held-out accuracy
    roll = rollout_closed_loop(model, batch, cfg_size, model_id, seed=seed)
    acc = accuracy_and_ce(roll)

    # Prior decodability (linear decoder on pre-stimulus state)
    hidden = extract_latent_states(model, model_id, batch, cfg_size, seed=seed)
    labels = extract_labels_flat(batch)
    dec = fit_block_decoder(hidden, labels, binary=True, n_folds=folds)

    return {
        "model": model_id,
        "hidden_size": hidden_size,
        "n_params": _n_params(model),
        "heldout_accuracy": round(acc["accuracy"], 4),
        "choice_ce": round(acc["cross_entropy"], 4),
        "prior_auroc": round(dec.get("auroc_mean", float("nan")), 4),
        "prior_acc": round(dec["accuracy_mean"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hidden-size capacity sweep")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help=f"Hidden sizes to sweep (default: {DEFAULT_SIZES})")
    parser.add_argument("--model", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to sweep (default: {DEFAULT_MODELS}); bayes is excluded")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (fewer = faster, default: config)")
    parser.add_argument("--folds", type=int, default=5, help="CV folds for decoder (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed (default: 0)")
    parser.add_argument("--load-existing", action="store_true",
                        help="Skip training; replot from existing CSV")
    args = parser.parse_args()

    out_dir = ROOT / "reports" / "v2" / "block_decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hidden_size_sweep.csv"
    fig_path = out_dir / "hidden_size_sweep.png"

    if args.load_existing:
        if not csv_path.exists():
            print(f"No existing CSV at {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
    else:
        models = [m for m in args.model if m != "bayes"]
        if "bayes" in args.model:
            print("[sweep] note: bayes ignores hidden_size and is excluded")

        cfg = load_synthetic_config()
        batch = _load_heldout(cfg)

        print(f"\n=== Hidden-size sweep  |  models={models}  |  sizes={args.sizes} ===\n")
        rows = []
        for model_id in models:
            for size in args.sizes:
                print(f"[train] {model_id:<10} hidden={size:<4} ...", end="", flush=True)
                row = sweep_one(
                    model_id, size, cfg, batch,
                    epochs=args.epochs, folds=args.folds, seed=args.seed,
                )
                rows.append(row)
                print(f"  acc={row['heldout_accuracy']:.3f}  "
                      f"prior_AUROC={row['prior_auroc']:.3f}  "
                      f"params={row['n_params']}")

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path.relative_to(ROOT)}")

    # Plot
    from src.plot.phase10_figures import fig_hidden_size_sweep
    fig_hidden_size_sweep(csv_path, fig_path)
    print(f"Figure saved to:  {fig_path.relative_to(ROOT)}")

    # Elbow hint per model
    print("\n--- Elbow hint (smallest size within 0.5% of that model's best on both metrics) ---")
    for model_id, g in df.groupby("model"):
        g = g.sort_values("hidden_size")
        best_acc = g["heldout_accuracy"].max()
        best_auroc = g["prior_auroc"].max()
        elbow = None
        for _, r in g.iterrows():
            if r["heldout_accuracy"] >= best_acc - 0.005 and r["prior_auroc"] >= best_auroc - 0.005:
                elbow = int(r["hidden_size"])
                break
        print(f"  {model_id:<10} elbow ≈ {elbow}  "
              f"(best acc={best_acc:.3f}, best AUROC={best_auroc:.3f})")


if __name__ == "__main__":
    main()
