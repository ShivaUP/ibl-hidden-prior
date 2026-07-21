#!/usr/bin/env python3
"""Phase 7: mouse latent prior + model prior match (history-only).

Usage:
    python scripts/eval_phase7_priors.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.eval.predict import attach_trial_meta, predict_split
from src.eval.prior_match import (
    prior_match_metrics,
    switch_prior_mse,
    update_asymmetry_prior,
)


MODELS = ("standard", "pc", "bayes")


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text())
    train_eids = list(splits["train"])
    eval_eids = list(splits["val"]) + list(splits["test"])

    trials = pd.read_parquet(ROOT / "data" / "processed" / "trials" / "all_trials.parquet")
    # Fit mouse prior on train only (no leakage into held-out match scores)
    params, fit_info = fit_mouse_prior(trials, train_eids=train_eids)
    print("Fitted mouse prior:", params.to_dict(), fit_info)

    mouse_all = apply_mouse_prior(trials, params)
    mouse_dir = ROOT / "data" / "processed" / "mouse_prior"
    mouse_dir.mkdir(parents=True, exist_ok=True)
    mouse_path = mouse_dir / "history_only_mouse_prior.parquet"
    mouse_all[
        [
            "eid",
            "trial_index",
            "probabilityLeft",
            "stimulus_right",
            "choice_right",
            "mouse_prior_hat",
            "mouse_prior_choice_p_right",
            "block_switch",
        ]
    ].to_parquet(mouse_path, index=False)
    (mouse_dir / "params.json").write_text(
        json.dumps({"params": params.to_dict(), "fit_info": fit_info, "created_utc": stamp}, indent=2),
        encoding="utf-8",
    )

    # Sanity: correlates with oracle P(right)=1-probabilityLeft but is not identical
    oracle_right = 1.0 - mouse_all["probabilityLeft"].astype(float)
    corr_all = float(np.corrcoef(mouse_all["mouse_prior_hat"], oracle_right)[0, 1])
    mae_vs_oracle = float(np.mean(np.abs(mouse_all["mouse_prior_hat"] - oracle_right)))

    model_dir = ROOT / "data" / "processed" / "model_priors"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "reports" / "behavior"
    fig_dir = ROOT / "reports" / "figures" / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    match_rows = []
    for model in MODELS:
        print(f"Extracting prior_q for {model}")
        preds = predict_split(ROOT, model, "history_only", eval_eids)
        # also save full-core priors for downstream neural (all eids)
        preds_all = predict_split(
            ROOT, model, "history_only", list(trials["eid"].astype(str).unique())
        )
        preds_all[["eid", "trial_index", "prior_q", "p_right"]].to_parquet(
            model_dir / f"{model}_history_only.parquet", index=False
        )

        df = attach_trial_meta(preds, trials)
        df = df.merge(
            mouse_all[["eid", "trial_index", "mouse_prior_hat"]],
            on=["eid", "trial_index"],
            how="inner",
        )
        m = prior_match_metrics(df)
        s = switch_prior_mse(df)
        asym = update_asymmetry_prior(df)
        asym.to_csv(out_dir / f"prior_asymmetry_{model}.csv", index=False)

        match_rows.append(
            {
                "model": model,
                "split": "val+test",
                **m,
                **s,
                "asymmetry": asym.to_dict(orient="records"),
            }
        )

        # Scatter plot
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(df["mouse_prior_hat"], df["prior_q"], s=4, alpha=0.15, c="#4C78A8")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Mouse prior hat p_t")
        ax.set_ylabel(f"Model prior q_t ({model})")
        ax.set_title(f"Prior match ({model})\ncorr={m['corr']:.3f} rmse={m['rmse']:.3f}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"prior_scatter_{model}.png", dpi=120)
        plt.close(fig)

    match_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "asymmetry"} for r in match_rows]
    )
    match_path = out_dir / "prior_match.csv"
    match_df.to_csv(match_path, index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(match_df["model"], match_df["corr"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Corr(mouse prior, model q)")
    ax.set_title("History-only prior match (held-out)")
    fig.tight_layout()
    fig.savefig(fig_dir / "prior_match_corr_bars.png", dpi=120)
    plt.close(fig)

    summary = {
        "created_utc": stamp,
        "mouse_prior_params": params.to_dict(),
        "fit_info_train": fit_info,
        "corr_mouse_prior_vs_oracle_prior_right_all": corr_all,
        "mae_mouse_prior_vs_oracle_prior_right_all": mae_vs_oracle,
        "note_not_oracle": "Mouse prior is behavior-derived; not equal to probabilityLeft.",
        "match_heldout": match_rows,
        "artifacts": {
            "mouse_prior": str(mouse_path),
            "model_priors": str(model_dir),
            "prior_match_csv": str(match_path),
        },
    }
    summary_path = out_dir / "phase7_prior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {match_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"match": match_df.to_dict(orient="records"), "corr_vs_block": corr_all}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
