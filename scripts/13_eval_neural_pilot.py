#!/usr/bin/env python3
"""13 — Neural prior VE on the shared behavior+neural cohort (full run).

Default: all eids in `shared_behavior_neural_eids.json` (same sessions as real
behavior transfer). Downloads spikes via ONE when not cached.

Usage:
  python scripts/13_eval_neural_pilot.py              # full cohort
  python scripts/13_eval_neural_pilot.py --eid <eid>  # single session
  python scripts/13_eval_neural_pilot.py --max-sessions 3  # smoke
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

from src.data.config import load_frozen_config
from src.data.inspect_trials import load_trials_for_eid
from src.data.processed_trials import build_processed_session
from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.models_v2.rollout import load_model, rollout_real_session
from src.neural.prior_readout import fit_prior_readout, model_explains_neural_prior
from src.neural.spikes import NeuralWindow, build_region_matrix
from src.synthetic.channels import PhaseTicks
from src.synthetic.mapper_real import encode_real_session
from src.synthetic.schema import load_synthetic_config

from src.neural.regions import NEURAL_REGIONS

REGIONS = NEURAL_REGIONS
MODELS = ("tanh_bptt", "tanh_pc", "gru", "gru_pc")
SHARED = ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json"


def make_one(cache_dir: Path):
    from one.api import ONE

    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def _load_or_build_trials(one, eid: str, rt_pct: tuple[float, float]) -> pd.DataFrame:
    pq = ROOT / "data" / "processed" / "trials" / f"{eid}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    raw = load_trials_for_eid(one, eid)
    proc = build_processed_session(eid, raw, rt_percentiles=rt_pct)
    pq.parent.mkdir(parents=True, exist_ok=True)
    proc.to_parquet(pq, index=False)
    return proc


def eval_one_eid(
    *,
    one,
    eid: str,
    cfg: dict,
    phase: PhaseTicks,
    window: NeuralWindow,
    rt_pct: tuple[float, float],
    models: list,
) -> tuple[list[dict], dict]:
    trials = _load_or_build_trials(one, eid, rt_pct)
    if "mouse_prior_hat" not in trials.columns:
        params, _ = fit_mouse_prior(trials, train_eids=[eid])
        trials = apply_mouse_prior(trials, params)

    cache = ROOT / "data" / "processed" / "neural" / eid
    cache.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(cache / "trials.parquet", index=False)

    enc = encode_real_session(trials, phase)
    out_v2 = ROOT / "data" / "processed" / "neural_v2" / eid
    out_v2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_v2 / "real_ticks.npz", **enc)

    model_q: dict[str, np.ndarray] = {}
    for mid in models:
        ckpt = ROOT / cfg["paths"]["artifacts"] / "models" / mid / "model.npz"
        if not ckpt.exists():
            continue
        model = load_model(mid, ckpt)
        roll = rollout_real_session(model, enc, cfg, mid, phase, regime="history_only")
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
    ve_rows: list[dict] = []
    region_info: dict = {}

    for region in REGIONS:
        npz_path = cache / f"{region}_counts.npz"
        try:
            if npz_path.exists():
                blob = np.load(npz_path, allow_pickle=True)
                counts = np.asarray(blob["counts"], dtype=float)
                n_units = int(blob["n_units"]) if "n_units" in blob.files else counts.shape[1]
            else:
                mat = build_region_matrix(one, eid, trials, region, window=window)
                if mat.n_units < 1:
                    raise RuntimeError(f"no units in {region}")
                counts = mat.counts
                n_units = mat.n_units
                np.savez_compressed(
                    npz_path,
                    counts=counts,
                    cluster_ids=mat.cluster_ids,
                    acronyms=mat.acronyms,
                    trial_index=mat.trial_index,
                    n_units=n_units,
                    window=mat.window,
                )
        except Exception as exc:  # noqa: BLE001
            region_info[region] = {"error": str(exc)}
            continue

        m = min(counts.shape[0], n)
        counts = counts[:m]
        mp = mouse_prior[:m]
        readout = fit_prior_readout(counts, mp)
        region_info[region] = {
            "n_units": readout["n_units"],
            "n_trials": readout["n"],
            "ve_cv_mouse_prior": readout["ve_cv"],
            "corr_cv_mouse_prior": readout["corr_cv"],
        }
        mask = readout["mask"]
        neural_full = np.full(mask.shape[0], np.nan)
        neural_full[mask] = readout["oof_pred"][: int(mask.sum())]

        for mid, q in model_q.items():
            metrics = model_explains_neural_prior(neural_full, q[: mask.shape[0]])
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
    return ve_rows, {"eid": eid, "regions": region_info, "n_trials": n}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eid", default=None, help="Single eid (default: full shared cohort)")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--manifest", default=str(SHARED.relative_to(ROOT)))
    args = p.parse_args()

    cfg = load_synthetic_config()
    fcfg = load_frozen_config()
    rt_pct = tuple(float(x) for x in fcfg["data"]["trial_inclusion"]["rt_percentile_trim"])
    phase = PhaseTicks.from_config(cfg)
    window = NeuralWindow(t_start=-0.1, t_end=0.3, align_event="stimOn_times")
    stamp = datetime.now(timezone.utc).isoformat()

    man = json.loads((ROOT / args.manifest).read_text())
    eids = [args.eid] if args.eid else list(man["eids"])
    if args.max_sessions is not None:
        eids = eids[: args.max_sessions]

    one = make_one(ROOT / fcfg["data"]["cache_dir"])
    models = [m for m in MODELS if (ROOT / cfg["paths"]["artifacts"] / "models" / m / "model.npz").exists()]

    all_rows: list[dict] = []
    session_summaries = []
    for i, eid in enumerate(eids, start=1):
        print(f"[{i}/{len(eids)}] neural VE {eid}", flush=True)
        try:
            rows, summary = eval_one_eid(
                one=one,
                eid=eid,
                cfg=cfg,
                phase=phase,
                window=window,
                rt_pct=rt_pct,
                models=models,
            )
            all_rows.extend(rows)
            session_summaries.append(summary)
            print(f"  rows={len(rows)} regions={list(summary['regions'].keys())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {eid}: {exc}", flush=True)
            session_summaries.append({"eid": eid, "error": str(exc)})

    reports = ROOT / "reports" / "v2" / "neural"
    reports.mkdir(parents=True, exist_ok=True)
    ve_df = pd.DataFrame(all_rows)
    ve_df.to_csv(reports / "ve_unmatched.csv", index=False)
    ve_df.to_csv(reports / "ve_unmatched_full.csv", index=False)

    # session-mean table for ranking
    if not ve_df.empty:
        mean_tbl = (
            ve_df.groupby(["region", "model"], as_index=False)
            .agg(
                ve_linear_recal=("ve_linear_recal", "mean"),
                ve_std=("ve_linear_recal", "std"),
                corr=("corr", "mean"),
                n_sessions=("eid", "nunique"),
                n_trials=("n", "sum"),
            )
        )
        mean_tbl.to_csv(reports / "ve_unmatched_session_mean.csv", index=False)
    else:
        mean_tbl = pd.DataFrame()

    summary = {
        "stage": "neural_full_shared_cohort_v2",
        "created_utc": stamp,
        "manifest": args.manifest,
        "n_eids_requested": len(eids),
        "n_eids_with_rows": int(ve_df["eid"].nunique()) if not ve_df.empty else 0,
        "models": models,
        "primary_metric": "ve_linear_recal",
        "aggregation": "per-session VE, then mean across sessions (ve_unmatched_session_mean.csv)",
        "note": (
            "Same sessions as real behavior transfer (shared_behavior_neural n=10 dual-region-first). "
            "Model q_t = synth-trained belief; neural axis = CV Ridge → mouse_prior_hat."
        ),
        "sessions": session_summaries,
        "n_ve_rows": int(len(ve_df)),
    }
    (reports / "phase8_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # keep phase8_pilot.json as pointer to full for compatibility
    (reports / "phase8_pilot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_rows": len(ve_df),
                "n_sessions": int(ve_df["eid"].nunique()) if not ve_df.empty else 0,
                "mean_table_rows": int(len(mean_tbl)),
            },
            indent=2,
        )
    )
    return 0 if not ve_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
