#!/usr/bin/env python3
"""Phase 6: held-out behavioral + switch-centered evaluation.

Uses val+test as evaluation sessions (never used for hyperparameter retuning here).
Primary scientific focus: history_only.

Usage:
    python scripts/eval_phase6.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.behavior_choice import choice_metrics, psychometric_table
from src.eval.behavior_rt import rt_by_strength_and_block, rt_metrics
from src.eval.predict import attach_trial_meta, predict_split
from src.eval.switch_centered import (
    asymmetry_table,
    extract_switch_windows,
    fit_adaptation_half_life,
    switch_trajectory,
)


MODELS = ("standard", "pc", "bayes")
CONDITIONS = ("history_only", "full_information", "fixed_prior")


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text())
    # Held-out evaluation: val + test (train excluded)
    eval_eids = list(splits["val"]) + list(splits["test"])
    trials = pd.read_parquet(ROOT / "data" / "processed" / "trials" / "all_trials.parquet")

    out_dir = ROOT / "reports" / "behavior"
    fig_dir = ROOT / "reports" / "figures" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    psycho_parts = []
    rt_parts = []
    switch_summaries = []

    for condition in CONDITIONS:
        for model in MODELS:
            ckpt = ROOT / "artifacts" / "models" / model / condition / "default.pt"
            if not ckpt.exists():
                print(f"[skip] missing {ckpt}")
                continue
            print(f"Eval {model}/{condition} on {len(eval_eids)} held-out eids")
            preds = predict_split(ROOT, model, condition, eval_eids)
            df = attach_trial_meta(preds, trials)
            if condition == "fixed_prior":
                df = df.loc[df["probabilityLeft"].round(4) == 0.5].copy()

            cm = choice_metrics(df)
            rm = rt_metrics(df)
            metric_rows.append(
                {
                    "created_utc": stamp,
                    "model": model,
                    "condition": condition,
                    "split": "val+test",
                    **cm,
                    **{f"rt_{k}" if not k.startswith("rt_") and k != "n_trials" else k: v for k, v in rm.items() if k != "n_trials"},
                    "rt_nll": rm["rt_nll"],
                    "rt_median_mouse": rm["rt_median_mouse"],
                    "rt_median_model": rm["rt_median_model"],
                    "log_rt_mae": rm["log_rt_mae"],
                }
            )

            pt = psychometric_table(df, by_block=True)
            pt["model"] = model
            pt["condition"] = condition
            psycho_parts.append(pt)

            rtb = rt_by_strength_and_block(df)
            rtb["model"] = model
            rtb["condition"] = condition
            rt_parts.append(rtb)

            if condition == "history_only":
                sw = extract_switch_windows(df)
                if len(sw) == 0:
                    print("  [warn] no switches passed QC window")
                    continue
                traj_all = switch_trajectory(sw, prefer_strict=False)
                traj_strict = switch_trajectory(sw, prefer_strict=True)
                asym = asymmetry_table(sw)
                adapt_mouse = fit_adaptation_half_life(traj_all, "p_align_mouse")
                adapt_model = fit_adaptation_half_life(traj_all, "p_align_model")

                sw.to_parquet(out_dir / f"switch_trials_{model}_history_only.parquet", index=False)
                traj_all.to_csv(out_dir / f"switch_traj_{model}_all.csv", index=False)
                traj_strict.to_csv(out_dir / f"switch_traj_{model}_strict.csv", index=False)
                asym.to_csv(out_dir / f"switch_asymmetry_{model}.csv", index=False)

                n_sw = int((sw["rel_trial"] == 0).sum())
                n_relaxed = int(sw.loc[sw["rel_trial"] == 0, "relaxed_qc"].sum())
                switch_summaries.append(
                    {
                        "model": model,
                        "n_switches_all": n_sw,
                        "n_switches_relaxed": n_relaxed,
                        "n_switches_strict": n_sw - n_relaxed,
                        "adapt_mouse": adapt_mouse,
                        "adapt_model": adapt_model,
                        "asymmetry": asym.to_dict(orient="records"),
                    }
                )

                # Plot trajectories
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.axvline(0, color="k", ls="--", lw=1)
                ax.plot(traj_all["rel_trial"], traj_all["p_align_mouse"], "o-", label="mouse")
                ax.plot(traj_all["rel_trial"], traj_all["p_align_model"], "s-", label=model)
                ax.set_xlabel("Trials from block switch")
                ax.set_ylabel("P(aligned with new block)")
                ax.set_title(f"Switch-centered ({model}, history-only, all QC)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / f"switch_traj_{model}.png", dpi=120)
                plt.close(fig)

                # Psychometric plot for this model
                fig, ax = plt.subplots(figsize=(6, 4))
                overall = pt.loc[pt["slice"] == "all"]
                ax.plot(overall["signed_contrast"], overall["p_right_mouse"], "o-", label="mouse")
                ax.plot(overall["signed_contrast"], overall["p_right_model"], "s-", label=model)
                ax.set_xlabel("Signed contrast")
                ax.set_ylabel("P(right)")
                ax.set_title(f"Psychometric ({model}, history-only held-out)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / f"psychometric_{model}_history_only.png", dpi=120)
                plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "heldout_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if psycho_parts:
        pd.concat(psycho_parts, ignore_index=True).to_csv(
            out_dir / "psychometrics.csv", index=False
        )
    if rt_parts:
        pd.concat(rt_parts, ignore_index=True).to_csv(
            out_dir / "rt_by_strength_block.csv", index=False
        )

    # History-only comparison bar chart
    hist = metrics_df.loc[metrics_df["condition"] == "history_only"]
    if len(hist):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].bar(hist["model"], hist["choice_nll"], color=["#4C78A8", "#F58518", "#54A24B"])
        axes[0].set_ylabel("Choice NLL (held-out)")
        axes[0].set_title("History-only choice fit")
        axes[1].bar(hist["model"], hist["rt_nll"], color=["#4C78A8", "#F58518", "#54A24B"])
        axes[1].set_ylabel("RT NLL (held-out)")
        axes[1].set_title("History-only RT fit")
        fig.tight_layout()
        fig.savefig(fig_dir / "heldout_history_only_bars.png", dpi=120)
        plt.close(fig)

    summary = {
        "created_utc": stamp,
        "eval_eids": eval_eids,
        "metrics_path": str(metrics_path),
        "history_only_metrics": hist.to_dict(orient="records") if len(hist) else [],
        "switch_summaries": switch_summaries,
        "notes": [
            "Evaluation on val+test only; no hyperparameter retuning.",
            "Switch window -10..+30; prefer 10/20, allow 8/16 flagged as relaxed.",
            "Sensitivity: compare switch_traj_*_all.csv vs *_strict.csv.",
        ],
    }
    summary_path = out_dir / "phase6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"history_only": summary["history_only_metrics"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
