#!/usr/bin/env python3
"""12 — Logistic regression block prior decoder for all v2 models.

Extracts the pre-stimulus hidden state at each trial from every trained model,
then fits a logistic regression to classify the true block prior
(left-biased vs right-biased). Reports accuracy and AUROC per model.

Usage
-----
  python scripts/12_block_decoder.py                  # binary (left vs right)
  python scripts/12_block_decoder.py --multiclass     # 3-class (left/unbiased/right)
  python scripts/12_block_decoder.py --model tanh_bptt gru
  python scripts/12_block_decoder.py --folds 10 --C 0.1
  python scripts/12_block_decoder.py --by-tick        # decode at each within-trial tick

Output
------
  reports/v2/block_decoder/block_decoder_results.json
  reports/v2/block_decoder/block_decoder_by_tick.json   (with --by-tick)
  (summary printed to stdout)

Reference dataset
-----------------
  IBL Brain Wide Map (2025)
  https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import SyntheticBatch
from src.synthetic.schema import load_synthetic_config
from src.eval.block_decoder import decode_all_models, decode_all_models_by_tick


def _load_heldout(cfg: dict) -> SyntheticBatch:
    path = ROOT / "data" / "processed" / "synthetic_v2" / "heldout_sessions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\n"
            "Run: python scripts/05_build_synthetic_datasets.py"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Block prior LR decoder for all v2 models")
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        metavar="ID",
        help="Model IDs to decode (default: all in config)",
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="Use 3-class decoding (left / unbiased / right) instead of binary",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        metavar="C",
        help="Inverse LR regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for rollout (default: 0)",
    )
    parser.add_argument(
        "--by-tick",
        action="store_true",
        help="Decode block prior at each within-trial tick (temporal 'layer' analysis)",
    )
    args = parser.parse_args()

    cfg = load_synthetic_config()
    batch = _load_heldout(cfg)

    binary = not args.multiclass
    checkpoint_dir = ROOT / cfg["paths"]["artifacts"] / "models"
    out_dir = ROOT / cfg["paths"]["reports"] / "block_decoder"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.by_tick:
        _run_by_tick(cfg, batch, args, binary, checkpoint_dir, out_dir)
        return

    mode_label = "binary (left vs right)" if binary else "3-class (left/unbiased/right)"
    print(f"\n=== Block Prior Decoder  |  mode: {mode_label}  |  folds: {args.folds} ===\n")

    results = decode_all_models(
        cfg,
        batch,
        model_ids=args.model,
        checkpoint_dir=checkpoint_dir,
        binary=binary,
        n_folds=args.folds,
        C=args.C,
        seed=args.seed,
    )

    if not results:
        print("[block_decoder] No models decoded. Check that checkpoints exist.")
        sys.exit(1)

    # Save results
    out_path = out_dir / "block_decoder_results.json"

    # coef_mean is a nested list — json-safe already from fit_block_decoder
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path.relative_to(ROOT)}")

    # Plot
    from src.plot.phase10_figures import fig_block_decoder
    fig_path = out_dir / "block_decoder_comparison.png"
    fig_block_decoder(out_path, fig_path)
    print(f"Figure saved to:  {fig_path.relative_to(ROOT)}")

    # Print ranking table
    print("\n--- Ranking by AUROC (binary) or Accuracy (multiclass) ---")
    if binary:
        ranked = sorted(results.items(), key=lambda kv: kv[1].get("auroc_mean", 0), reverse=True)
        print(f"{'Model':<16}  {'Accuracy':>10}  {'AUROC':>8}")
        print("-" * 42)
        for mid, r in ranked:
            print(
                f"  {mid:<14}  {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}  "
                f"{r.get('auroc_mean', float('nan')):.3f}"
            )
    else:
        ranked = sorted(results.items(), key=lambda kv: kv[1].get("accuracy_mean", 0), reverse=True)
        print(f"{'Model':<16}  {'Accuracy':>10}")
        print("-" * 30)
        for mid, r in ranked:
            print(f"  {mid:<14}  {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}")


def _run_by_tick(cfg, batch, args, binary, checkpoint_dir, out_dir) -> None:
    """Tick-by-tick ('layer by layer') decoding across the trial."""
    mode_label = "binary (left vs right)" if binary else "3-class"
    print(f"\n=== Block Prior Decoder — tick by tick  |  mode: {mode_label}  |  folds: {args.folds} ===\n")

    results = decode_all_models_by_tick(
        cfg,
        batch,
        model_ids=args.model,
        checkpoint_dir=checkpoint_dir,
        binary=binary,
        n_folds=args.folds,
        C=args.C,
        seed=args.seed,
    )

    if not results:
        print("[block_decoder] No models decoded. Check that checkpoints exist.")
        sys.exit(1)

    out_path = out_dir / "block_decoder_by_tick.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path.relative_to(ROOT)}")

    from src.plot.phase10_figures import fig_block_decoder_by_tick
    fig_path = out_dir / "block_decoder_by_tick.png"
    fig_block_decoder_by_tick(out_path, fig_path)
    print(f"Figure saved to:  {fig_path.relative_to(ROOT)}")

    # Peak-tick summary table
    metric = "auroc_by_tick" if binary else "accuracy_by_tick"
    label = "AUROC" if binary else "Accuracy"
    print(f"\n--- Peak {label} tick per model ---")
    print(f"{'Model':<16}  {'Peak tick':>9}  {'Phase':>9}  {label:>7}")
    print("-" * 48)
    for mid, r in results.items():
        vals = r[metric]
        peak = int(np.argmax(vals))
        print(f"  {mid:<14}  {peak:>9}  {r['tick_phase'][peak]:>9}  {vals[peak]:.3f}")


if __name__ == "__main__":
    main()
