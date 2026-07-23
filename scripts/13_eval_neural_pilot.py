#!/usr/bin/env python3
"""13 — Neural prior VE pilot with frozen synth-trained v2 models.

Uses cached spike matrices when available (pilot eid). Model q_t comes from
`rollout_real_session` → belief / zero_evidence_p_right (history_only).

Behavioral training/scoring remains correct-side only. Mouse prior is used only
as the **neural readout target** (brain → prior axis), matching v1 Phase 8.

Usage:
  python scripts/13_eval_neural_pilot.py
  python scripts/13_eval_neural_pilot.py --eid 1191f865-b10a-45c8-9c48-24a980fd9402
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.models_v2.rollout import load_model, rollout_real_session
from src.neural.prior_readout import fit_prior_readout, model_explains_neural_prior
from src.synthetic.channels import PhaseTicks
from src.synthetic.mapper_real import encode_real_session
from src.synthetic.schema import load_synthetic_config

DEFAULT_EID = "1191f865-b10a-45c8-9c48-24a980fd9402"
REGIONS = ("MOs", "vlOFC_orbvl")
MODELS = ("tanh_bptt", "tanh_pc", "gru", "bayes")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eid", default=DEFAULT_EID)
    args = p.parse_args()
    cfg = load_synthetic_config()
    eid = args.eid
    stamp = datetime.now(timezone.utc).isoformat()

    cache = ROOT / "data" / "processed" / "neural" / eid
    trials_path = cache / "trials.parquet"
    if not trials_path.exists():
        print(
            f"Missing {trials_path}. Need Phase-8 cache or re-download spikes "
            "(see git show 02fcb76:scripts/eval_phase8_neural_pilot.py).",
            file=sys.stderr,
        )
        return 1

    trials = pd.read_parquet(trials_path)
    if "mouse_prior_hat" not in trials.columns:
        params, _ = fit_mouse_prior(trials, train_eids=[eid])
        trials = apply_mouse_prior(trials, params)
        trials.to_parquet(trials_path, index=False)

    phase = PhaseTicks.from_config(cfg)
    enc = encode_real_session(trials, phase)

    out_v2 = ROOT / "data" / "processed" / "neural_v2" / eid
    out_v2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_v2 / "real_ticks.npz", **enc)

    model_q: dict[str, np.ndarray] = {}
    for mid in MODELS:
        ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / mid / "model.npz"
        if not ckpt.exists():
            print(f"SKIP missing checkpoint {ckpt}", file=sys.stderr)
            continue
        model = load_model(mid, ckpt)
        roll = rollout_real_session(
            model, enc, cfg, mid, phase, regime="history_only"
        )
        q = np.asarray(roll["belief"][0], dtype=float)
        model_q[mid] = q
        pd.DataFrame(
            {
                "trial_index": trials["trial_index"].to_numpy(),
                "prior_q": q,
                "p_choice_right": roll["p_choice_right"][0],
            }
        ).to_parquet(out_v2 / f"{mid}_prior_q.parquet", index=False)

    mouse_prior = trials["mouse_prior_hat"].to_numpy(dtype=float)
    n = len(mouse_prior)
    ve_rows = []
    region_summaries = {}

    for region in REGIONS:
        npz_path = cache / f"{region}_counts.npz"
        if not npz_path.exists():
            region_summaries[region] = {"error": f"missing {npz_path.name}"}
            continue
        blob = np.load(npz_path, allow_pickle=True)
        counts = np.asarray(blob["counts"], dtype=float)
        if counts.shape[0] != n:
            # align by truncating to min length
            m = min(counts.shape[0], n)
            counts = counts[:m]
            mp = mouse_prior[:m]
            region_summaries[region] = {
                "warning": f"length mismatch counts={blob['counts'].shape[0]} trials={n}; truncated to {m}"
            }
        else:
            mp = mouse_prior
            region_summaries[region] = {}

        readout = fit_prior_readout(counts, mp)
        region_summaries[region].update(
            {
                "n_units": readout["n_units"],
                "n_trials": readout["n"],
                "ve_cv_mouse_prior": readout["ve_cv"],
                "corr_cv_mouse_prior": readout["corr_cv"],
            }
        )
        neural_oof = readout["oof_pred"]
        mask = readout["mask"]
        # oof_pred is length of finite subset; expand carefully
        # fit_prior_readout returns oof on masked rows only — length == mask.sum()
        neural_full = np.full(mask.shape[0], np.nan)
        neural_full[mask] = neural_oof[: int(mask.sum())]

        for mid, q in model_q.items():
            q_use = q[: mask.shape[0]]
            metrics = model_explains_neural_prior(neural_full, q_use)
            ve_rows.append(
                {
                    "eid": eid,
                    "region": region,
                    "model": mid,
                    "condition": "history_only",
                    "ve_raw": metrics["ve_raw"],
                    "ve_linear_recal": metrics["ve_linear_recal"],
                    "corr": metrics["corr"],
                    "n": metrics["n"],
                    "confirmatory": False,
                }
            )

    reports = ROOT / "reports" / "v2" / "neural"
    reports.mkdir(parents=True, exist_ok=True)
    ve_df = pd.DataFrame(ve_rows)
    ve_df.to_csv(reports / "ve_unmatched.csv", index=False)
    ve_df.to_csv(reports / "ve_unmatched_pilot.csv", index=False)

    summary = {
        "stage": "neural_pilot_v2",
        "created_utc": stamp,
        "eid": eid,
        "models": list(model_q.keys()),
        "regions": region_summaries,
        "primary_metric": "ve_linear_recal",
        "note": (
            "Model q_t = zero-evidence / belief from synth-trained v2 rollouts. "
            "Neural target axis = CV Ridge readout of mouse_prior_hat (v1 Phase 8)."
        ),
        "n_ve_rows": int(len(ve_df)),
    }
    (reports / "phase8_pilot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not ve_df.empty:
        print(ve_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
