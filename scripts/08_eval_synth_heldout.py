#!/usr/bin/env python3
"""08 — Evaluate frozen models on synthetic held-out (closed-loop).

Usage:
  python scripts/08_eval_synth_heldout.py
  python scripts/08_eval_synth_heldout.py --model gru
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

from src.models_v2.rollout import (
    accuracy_and_ce,
    load_model,
    rollout_closed_loop,
    summarize_kyan_diagnostics,
)
from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import SyntheticBatch
from src.synthetic.schema import load_synthetic_config


def _load_heldout(cfg: dict) -> SyntheticBatch:
    path = ROOT / "data" / "processed" / "synthetic_v2" / "heldout_sessions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run python scripts/05_build_synthetic_datasets.py"
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


def eval_one(model_id: str, cfg: dict) -> dict:
    ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "model.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}; train first")
    model = load_model(model_id, ckpt)
    batch = _load_heldout(cfg)
    roll = rollout_closed_loop(model, batch, cfg, model_id)
    metrics = accuracy_and_ce(roll)
    # psychometric by block prior (coarse)
    true_p = roll["true_p_right"]
    for prior in (0.2, 0.5, 0.8):
        mask = np.isclose(true_p, prior)
        if mask.any():
            metrics[f"acc_block_p_right_{prior}"] = float(
                np.mean(roll["choice"][mask] == roll["side"][mask])
            )
    # zero-contrast prior probe (observed choice path)
    zc = np.isclose(roll["contrast"], 0.0)
    if zc.any():
        metrics["zero_contrast_p_choice_right_mean"] = float(
            np.mean(roll["p_choice_right"][zc])
        )
        metrics["zero_contrast_n"] = int(zc.sum())
    metrics["kyan_diagnostics"] = summarize_kyan_diagnostics(roll)

    out_dir = ROOT / cfg["paths"]["reports"] / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": "synth_heldout", "model_id": model_id, **metrics}
    (out_dir / f"synth_heldout_{model_id}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    # save roll for figures
    art = ROOT / cfg["paths"]["artifacts"] / "synthetic" / "heldout" / model_id
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art / "rollout.npz", **roll)
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=None,
        choices=["tanh_bptt", "tanh_pc", "gru", "bayes"],
    )
    args = p.parse_args()
    cfg = load_synthetic_config()
    models = [args.model] if args.model else list(cfg["models"])
    results = []
    for mid in models:
        try:
            results.append(eval_one(mid, cfg))
        except FileNotFoundError as exc:
            print(f"SKIP {mid}: {exc}", file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
