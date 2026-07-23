#!/usr/bin/env python3
"""14 — Behavior-matched neural VE + survival on the shared cohort (multi-session).

Matching uses real history_only CE on the **same** shared cohort (from script 11).
VE aggregation: session-mean `ve_linear_recal`; survival via session bootstrap.

Usage:
  python scripts/14_eval_neural_matched.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neural.behavior_match import MatchConfig, filter_ve_to_matched, select_behavior_matched
from src.neural.survival import holm_correct

MODELS = ("tanh_bptt", "tanh_pc", "gru", "bayes")


def _behavior_metrics_df() -> pd.DataFrame:
    rows = []
    for mid in MODELS:
        path = ROOT / "reports" / "v2" / "metrics" / f"real_history_only_{mid}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        ce = float(m.get("cross_entropy", m.get("ce_vs_correct_side", np.nan)))
        rows.append(
            {
                "model": mid,
                "condition": "history_only",
                "choice_nll": ce,
                "rt_nll": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _session_bootstrap_advantage(
    ve_df: pd.DataFrame,
    region: str,
    best: str,
    second: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap over sessions: mean(VE_best) - mean(VE_second)."""

    d = ve_df.loc[ve_df["region"] == region]
    # pivot eid × model
    wide = d.pivot_table(index="eid", columns="model", values="ve_linear_recal", aggfunc="mean")
    if best not in wide.columns or second not in wide.columns:
        return {"error": "missing model columns", "n": 0}
    sub = wide[[best, second]].dropna()
    n = len(sub)
    if n < 3:
        return {"n": n, "delta": float("nan"), "p_boot": float("nan")}
    obs = float(sub[best].mean() - sub[second].mean())
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    eids = sub.index.to_numpy()
    for i in range(n_boot):
        idx = rng.choice(eids, size=n, replace=True)
        samp = sub.loc[idx]
        deltas[i] = float(samp[best].mean() - samp[second].mean())
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    if abs(obs) < 1e-15:
        p = 1.0
    else:
        p = float(np.mean(deltas * np.sign(obs) <= 0.0))
        p = min(1.0, 2.0 * p)
    return {
        "n": int(n),
        "delta": obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_boot": p,
        "ve_cand": float(sub[best].mean()),
        "ve_base": float(sub[second].mean()),
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()
    out = ROOT / "reports" / "v2" / "neural"
    out.mkdir(parents=True, exist_ok=True)

    held = _behavior_metrics_df()
    if held.empty:
        print("Missing real_history_only metrics; run scripts/11_eval_regimes.py --domain real", file=sys.stderr)
        return 1

    match_cfg = MatchConfig(choice_epsilon=0.05, rt_nll_floor=1e9, choice_primary=True)
    match = select_behavior_matched(held, condition="history_only", cfg=match_cfg)
    match["metric_source"] = "shared cohort real_history_only_*.json CE"
    match["created_utc"] = stamp
    match["notes"] = [
        "Choice-primary ε-ball on shared behavior+neural cohort CE.",
        "RT floor non-binding.",
        "Neural VE aggregated across the same shared cohort sessions.",
    ]
    (out / "behavior_matched_models.json").write_text(json.dumps(match, indent=2), encoding="utf-8")
    print(f"Matched: {match['matched_models']}")

    ve_path = out / "ve_unmatched_full.csv"
    if not ve_path.exists():
        ve_path = out / "ve_unmatched.csv"
    if not ve_path.exists():
        print("Missing ve_unmatched; run scripts/13_eval_neural_pilot.py", file=sys.stderr)
        return 1
    ve_all = pd.read_csv(ve_path)
    ve_unmatched = ve_all.copy()
    ve_unmatched["confirmatory"] = False
    ve_matched = filter_ve_to_matched(ve_all, match["matched_models"])
    ve_unmatched.to_csv(out / "ve_unmatched.csv", index=False)
    ve_matched.to_csv(out / "ve_matched.csv", index=False)

    # session-mean for ranking
    mean_all = (
        ve_all.groupby(["region", "model"], as_index=False)
        .agg(ve_linear_recal=("ve_linear_recal", "mean"), corr=("corr", "mean"), n_sessions=("eid", "nunique"))
    )
    mean_matched = mean_all[mean_all["model"].isin(match["matched_models"])].copy()
    mean_matched.to_csv(out / "ve_matched_session_mean.csv", index=False)

    ranking = []
    survival_rows = []
    for region, g in mean_matched.groupby("region"):
        g2 = g.sort_values("ve_linear_recal", ascending=False)
        ranking.append(
            {
                "region": region,
                "ranking": g2[["model", "ve_linear_recal", "corr", "n_sessions"]].to_dict(orient="records"),
                "best_matched_model": str(g2.iloc[0]["model"]) if len(g2) else None,
                "best_matched_ve": float(g2.iloc[0]["ve_linear_recal"]) if len(g2) else float("nan"),
            }
        )
        models = [r["model"] for r in ranking[-1]["ranking"]]
        if len(models) >= 2:
            res = _session_bootstrap_advantage(ve_matched, region, models[0], models[1])
            survival_rows.append(
                {
                    "region": region,
                    "test": "matched_best_vs_second_session_bootstrap",
                    "best_model": models[0],
                    "second_model": models[1],
                    **res,
                }
            )
        elif len(models) == 1:
            survival_rows.append(
                {
                    "region": region,
                    "test": "single_matched_model",
                    "model": models[0],
                    "ve_obs": float(g2.iloc[0]["ve_linear_recal"]),
                }
            )

    pvals = [float(r["p_boot"]) for r in survival_rows if "p_boot" in r and np.isfinite(r.get("p_boot", np.nan))]
    if pvals:
        adjusted = holm_correct(pvals)
        j = 0
        for row in survival_rows:
            if "p_boot" in row and np.isfinite(row.get("p_boot", np.nan)):
                row["p_holm"] = float(adjusted[j])
                row["survive_alpha_05"] = bool(adjusted[j] < 0.05)
                j += 1

    (out / "survival_tests.json").write_text(json.dumps(survival_rows, indent=2), encoding="utf-8")
    pd.DataFrame(survival_rows).to_csv(out / "survival_tests.csv", index=False)
    summary = {
        "stage": "neural_matched_full_v2",
        "created_utc": stamp,
        "matched_models": match["matched_models"],
        "excluded_models": match["excluded_models"],
        "ranking": ranking,
        "survival": survival_rows,
        "n_ve_rows": int(len(ve_all)),
        "n_sessions": int(ve_all["eid"].nunique()) if len(ve_all) else 0,
    }
    (out / "phase9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"matched": match["matched_models"], "n_sessions": summary["n_sessions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
