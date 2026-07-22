#!/usr/bin/env python3
"""09 — Frozen-weight transfer eval on real mouse tick sessions.

Usage:
  python scripts/09_eval_real_transfer.py
  python scripts/09_eval_real_transfer.py --model gru
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

from src.models_v2.rollout import load_model
from src.synthetic.schema import RIGHT, load_synthetic_config


def run_session(model, model_id: str, data: dict) -> dict:
    x = data["inputs"]
    n_trials = int(data["n_trials"])
    n_steps = int(data["n_steps"])
    resp = int(data["response_tick"])
    correct_side = data["correct_side"]
    mouse_choice = data["mouse_choice"]

    is_bayes = model_id == "bayes"
    state = model.zero_state(1)
    p_right = np.empty(n_trials)
    pred = np.empty(n_trials, dtype=np.int64)

    for t in range(n_trials):
        sl = slice(t * n_steps, (t + 1) * n_steps)
        trial = x[sl]
        for step in range(n_steps):
            xt = trial[step : step + 1]
            if is_bayes:
                if step == resp:
                    probs = model.probs(xt, state)
                    p_right[t] = float(probs[0, RIGHT])
                    pred[t] = int(np.argmax(probs[0]))
                state = model.step_prior(xt, state)
            else:
                state = model.step(xt, state)
                if step == resp:
                    probs = model.probs(state)
                    p_right[t] = float(probs[0, RIGHT])
                    pred[t] = int(np.argmax(probs[0]))

    acc_correct = float(np.mean(pred == correct_side))
    acc_mouse = float(np.mean(pred == mouse_choice))
    p_c = np.where(correct_side == RIGHT, p_right, 1.0 - p_right)
    p_m = np.where(mouse_choice == RIGHT, p_right, 1.0 - p_right)
    return {
        "n_trials": n_trials,
        "acc_vs_correct_side": acc_correct,
        "acc_vs_mouse_choice": acc_mouse,
        "ce_vs_correct_side": float(-np.mean(np.log(np.clip(p_c, 1e-12, 1.0)))),
        "ce_vs_mouse_choice": float(-np.mean(np.log(np.clip(p_m, 1e-12, 1.0)))),
    }


def eval_model(model_id: str, cfg: dict) -> dict:
    ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "model.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    model = load_model(model_id, ckpt)
    tick_dir = ROOT / "data" / "processed" / "real_v2_ticks"
    man_path = ROOT / "data" / "manifests" / "real_v2_ticks.json"
    if not man_path.exists():
        raise FileNotFoundError("Run python scripts/06_map_real_to_v2_ticks.py first")
    man = json.loads(man_path.read_text())
    per = []
    for s in man["sessions"]:
        z = np.load(ROOT / s["path"])
        data = {k: z[k] for k in z.files}
        m = run_session(model, model_id, data)
        m["eid"] = s["eid"]
        per.append(m)

    def mean_key(k: str) -> float:
        return float(np.mean([r[k] for r in per])) if per else float("nan")

    summary = {
        "stage": "real_transfer",
        "model_id": model_id,
        "n_sessions": len(per),
        "acc_vs_correct_side": mean_key("acc_vs_correct_side"),
        "acc_vs_mouse_choice": mean_key("acc_vs_mouse_choice"),
        "ce_vs_correct_side": mean_key("ce_vs_correct_side"),
        "ce_vs_mouse_choice": mean_key("ce_vs_mouse_choice"),
        "per_session": per,
    }
    out = ROOT / cfg["paths"]["reports"] / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"real_transfer_{model_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


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
            results.append(
                {k: v for k, v in eval_model(mid, cfg).items() if k != "per_session"}
            )
        except FileNotFoundError as exc:
            print(f"SKIP {mid}: {exc}", file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
