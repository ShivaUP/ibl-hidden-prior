#!/usr/bin/env python3
"""Phase 8 pilot: one neural eid → spike matrices, prior readout, unmatched VE.

Usage:
    python scripts/eval_phase8_neural_pilot.py
    python scripts/eval_phase8_neural_pilot.py --eid 1191f865-b10a-45c8-9c48-24a980fd9402
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

from src.data.config import load_frozen_config, repo_root
from src.data.bayes_features import build_bayes_table
from src.data.event_bins import BinConfig, build_condition_arrays
from src.data.inspect_trials import load_trials_for_eid
from src.data.processed_trials import build_processed_session
from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.eval.predict import predict_arrays
from src.neural.prior_readout import fit_prior_readout, model_explains_neural_prior
from src.neural.spikes import NeuralWindow, build_region_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eid",
        default="1191f865-b10a-45c8-9c48-24a980fd9402",
        help="Pilot neural-behavior eid (MOs+ORBvl preferred).",
    )
    return p.parse_args()


def make_one(cache_dir: Path):
    from one.api import ONE

    cache_dir.mkdir(parents=True, exist_ok=True)
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def main() -> int:
    args = parse_args()
    cfg = load_frozen_config()
    root = repo_root()
    stamp = datetime.now(timezone.utc).isoformat()
    eid = args.eid

    # Freeze peri-stim window under evaluation.neural
    neural_cfg = cfg.setdefault("evaluation", {}).setdefault("neural", {})
    window_cfg = neural_cfg.setdefault(
        "peri_stim_window_s",
        {"t_start": -0.1, "t_end": 0.3, "align_event": "stimOn_times"},
    )
    window = NeuralWindow(
        t_start=float(window_cfg.get("t_start", -0.1)),
        t_end=float(window_cfg.get("t_end", 0.3)),
        align_event=str(window_cfg.get("align_event", "stimOn_times")),
    )

    one = make_one(root / cfg["data"]["cache_dir"])
    print(f"Loading trials for {eid}")
    raw = load_trials_for_eid(one, eid)
    kept = build_processed_session(eid, raw)
    print(f"  kept {len(kept)} QC trials")

    # Mouse prior: fit on this session alone for pilot (document limitation)
    params, fit_info = fit_mouse_prior(kept, train_eids=[eid])
    kept = apply_mouse_prior(kept, params)

    out_dir = root / "data" / "processed" / "neural" / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_dir / "trials.parquet", index=False)

    regions = ["MOs", "vlOFC_orbvl"]
    region_summaries = {}
    ve_rows = []
    readout_by_region: dict[str, np.ndarray] = {}

    # Build model inputs for this eid (behavior-core checkpoints, OOD eval)
    bin_cfg = BinConfig(
        bin_size_s=cfg["rnn"]["bin_size_ms"] / 1000.0,
        max_bins=100,
        pad_bins_after_feedback=1,
    )
    rnn_payload = build_condition_arrays(kept, "history_only", cfg=bin_cfg)
    bayes_tab = build_bayes_table(kept, "history_only")
    model_preds: dict[str, pd.DataFrame] = {}
    for model in ("standard", "pc", "bayes"):
        print(f"Model prior inference: {model}")
        if model == "bayes":
            model_preds[model] = predict_arrays(
                root, model, "history_only", bayes_df=bayes_tab
            )
        else:
            model_preds[model] = predict_arrays(
                root, model, "history_only", rnn_payload=rnn_payload
            )
        model_preds[model].to_parquet(out_dir / f"{model}_prior_q.parquet", index=False)

    for region in regions:
        print(f"Building spike matrix: {region}")
        try:
            mat = build_region_matrix(one, eid, kept, region, window=window)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {region}: {exc}")
            region_summaries[region] = {"error": str(exc)}
            continue
        np.savez_compressed(
            out_dir / f"{region}_counts.npz",
            counts=mat.counts,
            cluster_ids=mat.cluster_ids,
            acronyms=mat.acronyms,
            trial_index=mat.trial_index,
            mouse_prior_hat=kept["mouse_prior_hat"].to_numpy(),
            window=np.array([window.t_start, window.t_end]),
        )
        readout = fit_prior_readout(mat.counts, kept["mouse_prior_hat"].to_numpy())
        region_summaries[region] = {
            "n_units": mat.n_units,
            "n_trials": mat.n_trials,
            "readout_ve_cv": readout["ve_cv"],
            "readout_corr_cv": readout["corr_cv"],
            "window": mat.window,
        }
        print(
            f"  units={mat.n_units} readout VE={readout['ve_cv']:.4f} "
            f"corr={readout['corr_cv']:.4f}"
        )
        full_neural = np.full(len(kept), np.nan)
        full_neural[np.where(readout["mask"])[0]] = readout["oof_pred"]
        readout_by_region[region] = full_neural

        for model, preds in model_preds.items():
            merged = kept[["trial_index"]].merge(preds, on="trial_index", how="inner")
            idx_map = {int(t): i for i, t in enumerate(kept["trial_index"])}
            npri = np.array(
                [full_neural[idx_map[int(t)]] for t in merged["trial_index"]], dtype=float
            )
            metrics = model_explains_neural_prior(npri, merged["prior_q"].to_numpy())
            ve_rows.append(
                {
                    "eid": eid,
                    "region": region,
                    "model": model,
                    "status": "ok_ood_behavior_core_checkpoint",
                    **metrics,
                }
            )
            print(
                f"  VE {model}/{region}: raw={metrics['ve_raw']:.4f} "
                f"lin={metrics['ve_linear_recal']:.4f} corr={metrics['corr']:.4f}"
            )

    # Patch frozen YAML neural window without full rewrite of comments.
    cfg_path = root / "configs" / "frozen_v1.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    if "peri_stim_window_s:" not in text:
        insert = (
            "    peri_stim_window_s:\n"
            f"      t_start: {window.t_start}\n"
            f"      t_end: {window.t_end}\n"
            f"      align_event: {window.align_event}\n"
            f"    pilot_eid: {eid}\n"
            "    unit_source: SpikeSortingLoader_or_alf\n"
            "    neural_behavior_pool_note: "
            "\"Strict core∩neural empty; use neural_behavior_pool for Phase 8.\"\n"
        )
        text = text.replace(
            "    emphasize_switch_window: true\n",
            "    emphasize_switch_window: true\n" + insert,
        )
        cfg_path.write_text(text, encoding="utf-8")


    report = {
        "created_utc": stamp,
        "eid": eid,
        "mouse_prior_fit": {"params": params.to_dict(), "fit_info": fit_info},
        "window": window.to_dict(),
        "regions": region_summaries,
        "ve_unmatched_attempted": ve_rows,
        "notes": [
            "Strict behavior-core∩neural is empty; this eid is from neural_behavior_pool.",
            "Model VE uses behavior-core checkpoints evaluated OOD on this neural eid.",
            "Confirmatory claims require Phase 9 behavior matching + preferably neural-pool training.",
        ],
    }
    rep_dir = root / "reports" / "neural"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "phase8_pilot.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if ve_rows:
        pd.DataFrame(ve_rows).to_csv(rep_dir / "ve_unmatched_pilot.csv", index=False)
    print(json.dumps(report, indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
