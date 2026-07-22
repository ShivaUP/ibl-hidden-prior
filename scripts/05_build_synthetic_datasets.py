#!/usr/bin/env python3
"""05 — Sample synthetic train/held-out sessions from synthetic_v2 config.

Usage:
  python scripts/05_build_synthetic_datasets.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic.generate import build_training_tensors, generate_sessions
from src.synthetic.schema import load_synthetic_config


def main() -> int:
    cfg = load_synthetic_config()
    seed = int(cfg["train"]["seed"])
    rng = np.random.default_rng(seed)
    n_trials = int(cfg["trials_per_session_default"])
    # Large pool for training epochs to resample from, plus held-out
    train_pool = generate_sessions(64, n_trials, cfg, rng)
    held = generate_sessions(
        int(cfg["eval"]["synth_sessions"]),
        int(cfg["eval"]["synth_trials"]),
        cfg,
        np.random.default_rng(int(cfg["eval"]["seed"])),
    )
    x_h, y_h = build_training_tensors(
        held, cfg, np.random.default_rng(int(cfg["eval"]["seed"]) + 1)
    )

    out = ROOT / "data" / "processed" / "synthetic_v2"
    out.mkdir(parents=True, exist_ok=True)
    # Note: processed/ is gitignored; manifests record how to rebuild
    np.savez_compressed(
        out / "heldout_sessions.npz",
        probability_left=held.probability_left,
        p_right=held.p_right,
        block_id=held.block_id,
        side=held.side,
        contrast=held.contrast,
        inputs=x_h,
        targets=y_h,
        n_steps=held.phase.n_steps,
        response_tick=held.phase.response_tick,
    )
    np.savez_compressed(
        out / "train_pool_meta.npz",
        probability_left=train_pool.probability_left,
        p_right=train_pool.p_right,
        block_id=train_pool.block_id,
        side=train_pool.side,
        contrast=train_pool.contrast,
        n_steps=train_pool.phase.n_steps,
    )
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials": n_trials,
        "n_steps_per_trial": train_pool.phase.n_steps,
        "heldout_sessions": int(held.side.shape[0]),
        "train_pool_sessions": int(train_pool.side.shape[0]),
        "phase": {
            "baseline": train_pool.phase.baseline_ticks,
            "go_offset": train_pool.phase.go_offset_from_stim_ticks,
            "response_from_go": train_pool.phase.response_offset_from_go_ticks,
            "feedback": train_pool.phase.feedback_ticks,
            "stim_duration": train_pool.phase.stim_duration_ticks,
            "n_steps": train_pool.phase.n_steps,
        },
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    man = {
        "created_utc": meta["created_utc"],
        "rebuild": [
            "python scripts/04_fit_synthetic_stats.py",
            "python scripts/05_build_synthetic_datasets.py",
        ],
        "outputs": {
            "heldout": "data/processed/synthetic_v2/heldout_sessions.npz",
            "train_pool_meta": "data/processed/synthetic_v2/train_pool_meta.npz",
            "meta": "data/processed/synthetic_v2/dataset_meta.json",
        },
        "phase": meta["phase"],
    }
    (ROOT / "data" / "manifests" / "synthetic_datasets_v2.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
