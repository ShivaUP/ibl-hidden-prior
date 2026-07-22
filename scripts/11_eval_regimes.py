#!/usr/bin/env python3
"""11 — Eval regimes on synth held-out AND real behavior.

Writes rollouts + metrics for each (domain × regime × model):
  artifacts/v2/{synthetic|real}/regimes/{regime}/{model}/rollout.npz
  reports/v2/metrics/{synth|real}_{regime}_{model}.json

Usage:
  python scripts/11_eval_regimes.py
  python scripts/11_eval_regimes.py --domain real --model gru
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
    REGIMES,
    accuracy_and_ce,
    accuracy_real,
    filter_batch_fixed_prior,
    load_model,
    pool_real_rollouts,
    rollout_closed_loop,
    rollout_real_session,
    summarize_kyan_diagnostics,
)
from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import SyntheticBatch, generate_sessions
from src.synthetic.schema import load_synthetic_config


def _load_or_build_heldout(cfg: dict) -> SyntheticBatch:
    path = ROOT / "data" / "processed" / "synthetic_v2" / "heldout_sessions.npz"
    phase = PhaseTicks.from_config(cfg)
    if path.exists():
        z = np.load(path)
        return SyntheticBatch(
            probability_left=z["probability_left"],
            p_right=z["p_right"],
            block_id=z["block_id"],
            side=z["side"],
            contrast=z["contrast"],
            phase=phase,
        )
    return generate_sessions(
        int(cfg["eval"]["synth_sessions"]),
        int(cfg["eval"]["synth_trials"]),
        cfg,
        np.random.default_rng(int(cfg["eval"]["seed"])),
    )


def eval_synth(model_id: str, regime: str, cfg: dict) -> dict:
    ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "model.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    model = load_model(model_id, ckpt)
    batch = _load_or_build_heldout(cfg)
    if regime == "fixed_prior":
        batch = filter_batch_fixed_prior(batch)
    roll = rollout_closed_loop(
        model, batch, cfg, model_id, seed=int(cfg["eval"]["seed"]), regime=regime
    )
    metrics = accuracy_and_ce(roll)
    metrics.update(
        {
            "domain": "synth",
            "regime": regime,
            "model_id": model_id,
            "stage": "synth_regime",
            "n_sessions": int(batch.side.shape[0]),
            "n_trials": int(batch.side.shape[1]),
        }
    )
    if regime != "fixed_prior":
        metrics["kyan_diagnostics"] = summarize_kyan_diagnostics(roll)
    else:
        metrics["kyan_diagnostics"] = {"note": "switch/history-gap N/A on fixed_prior"}

    out_m = ROOT / cfg["paths"]["reports"] / "metrics"
    out_m.mkdir(parents=True, exist_ok=True)
    (out_m / f"synth_{regime}_{model_id}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    art = ROOT / cfg["paths"]["artifacts"] / "synthetic" / "regimes" / regime / model_id
    art.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in roll.items() if k != "regime"}
    np.savez_compressed(art / "rollout.npz", **save, regime=np.asarray(regime))
    return metrics


def eval_real(model_id: str, regime: str, cfg: dict) -> dict:
    ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "model.npz"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    man_path = ROOT / "data" / "manifests" / "real_v2_ticks.json"
    if not man_path.exists():
        raise FileNotFoundError("Run python scripts/06_map_real_to_v2_ticks.py first")
    model = load_model(model_id, ckpt)
    phase = PhaseTicks.from_config(cfg)
    man = json.loads(man_path.read_text())
    rolls = []
    for s in man["sessions"]:
        z = np.load(ROOT / s["path"])
        data = {k: z[k] for k in z.files}
        try:
            rolls.append(
                rollout_real_session(model, data, cfg, model_id, phase, regime=regime)
            )
        except ValueError as exc:
            print(f"  skip eid {s['eid']}: {exc}", file=sys.stderr)
            continue
    if not rolls:
        raise RuntimeError(f"No real sessions for {model_id}/{regime}")
    pooled = pool_real_rollouts(rolls)
    metrics = accuracy_real(pooled)
    metrics.update(
        {
            "domain": "real",
            "regime": regime,
            "model_id": model_id,
            "stage": "real_regime",
            "n_sessions": len(rolls),
            "n_trials_pooled": int(pooled["valid"].sum()),
        }
    )
    if regime != "fixed_prior":
        metrics["kyan_diagnostics"] = summarize_kyan_diagnostics(pooled)
    else:
        metrics["kyan_diagnostics"] = {"note": "switch/history-gap N/A on fixed_prior"}

    out_m = ROOT / cfg["paths"]["reports"] / "metrics"
    out_m.mkdir(parents=True, exist_ok=True)
    (out_m / f"real_{regime}_{model_id}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    art = ROOT / cfg["paths"]["artifacts"] / "real" / "regimes" / regime / model_id
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art / "rollout.npz", **pooled, regime=np.asarray(regime))
    # also legacy alias for history_only
    if regime == "history_only":
        legacy = ROOT / cfg["paths"]["reports"] / "metrics" / f"real_transfer_{model_id}.json"
        legacy.write_text(
            json.dumps(
                {
                    "stage": "real_transfer",
                    "model_id": model_id,
                    **{k: metrics[k] for k in metrics if k.startswith("acc_") or k.startswith("ce_")},
                    "n_sessions": metrics["n_sessions"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, choices=["tanh_bptt", "tanh_pc", "gru", "bayes"])
    p.add_argument("--regime", default=None, choices=list(REGIMES))
    p.add_argument("--domain", default=None, choices=["synth", "real", "both"])
    args = p.parse_args()
    cfg = load_synthetic_config()
    models = [args.model] if args.model else list(cfg["models"])
    regimes = [args.regime] if args.regime else list(cfg.get("eval", {}).get("regimes", REGIMES))
    domains = (
        ["synth", "real"]
        if (args.domain is None or args.domain == "both")
        else [args.domain]
    )
    for mid in models:
        for reg in regimes:
            for domain in domains:
                try:
                    if domain == "synth":
                        m = eval_synth(mid, reg, cfg)
                    else:
                        m = eval_real(mid, reg, cfg)
                    print(
                        json.dumps(
                            {
                                "domain": domain,
                                "model_id": mid,
                                "regime": reg,
                                "accuracy": m.get("accuracy") or m.get("acc_vs_correct_side"),
                            }
                        )
                    )
                except Exception as exc:
                    print(f"SKIP {domain}/{mid}/{reg}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
