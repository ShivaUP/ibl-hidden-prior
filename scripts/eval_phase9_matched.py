#!/usr/bin/env python3
"""Phase 9: behavior-matched neural comparison + survival tests.

Uses held-out history_only choice/RT for matching, Phase 8 pilot VE for
confirmatory tables, and trial-bootstrap survival within each region.

Usage:
    python scripts/eval_phase9_matched.py
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

from src.data.config import load_frozen_config, repo_root
from src.neural.behavior_match import MatchConfig, filter_ve_to_matched, select_behavior_matched
from src.neural.survival import bootstrap_ve_advantage, holm_correct


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()
    cfg = load_frozen_config()
    root = repo_root()
    out = root / "reports" / "neural"
    out.mkdir(parents=True, exist_ok=True)

    neural_cfg = cfg["evaluation"]["neural"]
    match_cfg_yaml = cfg["evaluation"]["behavior_matching"]
    match_cfg = MatchConfig(
        choice_epsilon=float(match_cfg_yaml["choice_epsilon_ball"]),
        rt_nll_floor=float(match_cfg_yaml["rt_secondary_floor"]),
        choice_primary=bool(match_cfg_yaml["choice_primary"]),
    )

    # Prefer held-out metrics for matching (Phase 6); fall back to val ranking
    held = pd.read_csv(root / "reports" / "behavior" / "heldout_metrics.csv")
    match = select_behavior_matched(held, condition="history_only", cfg=match_cfg)
    match["metric_source"] = "heldout_metrics.csv history_only"
    match["created_utc"] = stamp
    match["notes"] = [
        "Choice-primary: models outside ε-ball are excluded even if neural VE is higher.",
        "RT floor is secondary (rt_nll <= floor).",
        "Neural VE is OOD (behavior-core checkpoints on neural_behavior_pool eid).",
    ]
    match_path = out / "behavior_matched_models.json"
    match_path.write_text(json.dumps(match, indent=2), encoding="utf-8")
    print(f"Matched models: {match['matched_models']}")
    print(f"Excluded: {match['excluded_models']}")

    ve_path = out / "ve_unmatched_pilot.csv"
    if not ve_path.exists():
        print(f"Missing {ve_path}; run eval_phase8_neural_pilot.py first")
        return 1
    ve_all = pd.read_csv(ve_path)
    ve_unmatched = ve_all.copy()
    ve_unmatched["confirmatory"] = False
    ve_matched = filter_ve_to_matched(ve_all, match["matched_models"])

    ve_unmatched.to_csv(out / "ve_unmatched.csv", index=False)
    ve_matched.to_csv(out / "ve_matched.csv", index=False)

    # Rank matched models by mean ve_linear_recal per region
    ranking = []
    for region, g in ve_matched.groupby("region"):
        g2 = g.sort_values("ve_linear_recal", ascending=False)
        ranking.append(
            {
                "region": region,
                "ranking": g2[["model", "ve_linear_recal", "corr"]].to_dict(orient="records"),
                "best_matched_model": str(g2.iloc[0]["model"]),
                "best_matched_ve": float(g2.iloc[0]["ve_linear_recal"]),
            }
        )

    # Survival tests
    survival_rows = []
    pilot_eid = neural_cfg.get("pilot_eid") or "1191f865-b10a-45c8-9c48-24a980fd9402"
    neural_dir = root / "data" / "processed" / "neural" / pilot_eid

    from src.neural.prior_readout import fit_prior_readout

    for region_rank in ranking:
        region = region_rank["region"]
        models = [r["model"] for r in region_rank["ranking"]]
        npz = neural_dir / f"{region}_counts.npz"
        if not npz.exists():
            survival_rows.append({"region": region, "error": f"missing {npz.name}"})
            continue
        blob = np.load(npz, allow_pickle=True)
        counts = blob["counts"]
        mouse_prior = blob["mouse_prior_hat"]
        readout = fit_prior_readout(counts, mouse_prior)
        neural = np.full(len(mouse_prior), np.nan)
        neural[np.where(readout["mask"])[0]] = readout["oof_pred"]
        trials = pd.read_parquet(neural_dir / "trials.parquet")

        if len(models) >= 2:
            cand, base = models[0], models[1]
            pcand = pd.read_parquet(neural_dir / f"{cand}_prior_q.parquet")
            pbase = pd.read_parquet(neural_dir / f"{base}_prior_q.parquet")
            m = trials[["trial_index"]].copy()
            m["neural"] = neural
            m = m.merge(
                pcand[["trial_index", "prior_q"]].rename(columns={"prior_q": "q_cand"}),
                on="trial_index",
            )
            m = m.merge(
                pbase[["trial_index", "prior_q"]].rename(columns={"prior_q": "q_base"}),
                on="trial_index",
            )
            boot = bootstrap_ve_advantage(
                m["neural"].to_numpy(),
                m["q_cand"].to_numpy(),
                m["q_base"].to_numpy(),
                n_boot=2000,
                seed=0,
            )
            survival_rows.append(
                {
                    "region": region,
                    "test": "matched_best_vs_second",
                    "candidate": cand,
                    "baseline": base,
                    "metric": "ve_linear_recal",
                    "eid": pilot_eid,
                    **boot,
                }
            )
        elif len(models) == 1:
            # Only one matched model: test whether its VE > 0
            model = models[0]
            preds = pd.read_parquet(neural_dir / f"{model}_prior_q.parquet")
            m = trials[["trial_index"]].copy()
            m["neural"] = neural
            m = m.merge(preds[["trial_index", "prior_q"]], on="trial_index")
            y = m["neural"].to_numpy()
            q = m["prior_q"].to_numpy()
            mask = np.isfinite(y) & np.isfinite(q)
            y, q = y[mask], q[mask]
            rng = np.random.default_rng(0)
            n_boot = 2000
            deltas = np.empty(n_boot)

            def _ve_lin(u, qq):
                if np.std(qq) < 1e-12:
                    pred = np.full_like(u, u.mean())
                else:
                    coef = np.polyfit(qq, u, 1)
                    pred = coef[0] * qq + coef[1]
                sst = np.sum((u - u.mean()) ** 2)
                if sst <= 1e-12:
                    return 0.0
                return float(1.0 - np.sum((u - pred) ** 2) / sst)

            obs = _ve_lin(y, q)
            for i in range(n_boot):
                idx = rng.integers(0, len(y), size=len(y))
                deltas[i] = _ve_lin(y[idx], q[idx])
            ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
            p = float(np.mean(deltas <= 0.0))
            survival_rows.append(
                {
                    "region": region,
                    "test": "matched_ve_gt_zero",
                    "candidate": model,
                    "baseline": "zero",
                    "metric": "ve_linear_recal",
                    "eid": pilot_eid,
                    "n": int(len(y)),
                    "ve_cand": obs,
                    "ve_base": 0.0,
                    "delta": obs,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_two_sided": min(1.0, 2.0 * p),
                    "p_one_sided_gt0": p,
                }
            )
        else:
            survival_rows.append(
                {"region": region, "test": "n/a", "note": "no matched models in VE table"}
            )

    # Also document unmatched bayes vs matched standard (exploratory, not confirmatory)
    exploratory = []
    if "bayes" in set(ve_all["model"]) and "standard" in set(ve_all["model"]):
        for region in ve_all["region"].unique():
            exploratory.append(
                {
                    "region": region,
                    "note": "exploratory_only_bayes_excluded_by_choice_eps",
                    "bayes_ve": float(
                        ve_all.loc[
                            (ve_all.region == region) & (ve_all.model == "bayes"),
                            "ve_linear_recal",
                        ].iloc[0]
                    ),
                    "standard_ve": float(
                        ve_all.loc[
                            (ve_all.region == region) & (ve_all.model == "standard"),
                            "ve_linear_recal",
                        ].iloc[0]
                    ),
                }
            )

    # Holm across regions (among tests that have p-values)
    pvals = [r["p_two_sided"] for r in survival_rows if "p_two_sided" in r and np.isfinite(r["p_two_sided"])]
    if pvals:
        adj = holm_correct(pvals)
        j = 0
        for r in survival_rows:
            if "p_two_sided" in r and np.isfinite(r.get("p_two_sided", np.nan)):
                r["p_holm"] = adj[j]
                r["survive_alpha_05"] = bool(adj[j] < 0.05)
                j += 1

    survival = {
        "created_utc": stamp,
        "pilot_eid": pilot_eid,
        "matched_models": match["matched_models"],
        "excluded_from_confirmatory": match["excluded_models"],
        "region_rankings_matched": ranking,
        "tests": survival_rows,
        "exploratory_unmatched_bayes_vs_standard": exploratory,
        "multiple_comparison": "holm_across_regions",
        "limitations": [
            "Single-session trial bootstrap (not session permutation); expand neural pool for session-level tests.",
            "OOD checkpoints: models trained on behavior-core, not neural_behavior_pool.",
            "Only standard enters held-out choice ε-ball; pc/bayes excluded from confirmatory claims.",
            "Unmatched bayes may show higher VE but cannot enter confirmatory claims.",
        ],
    }
    (out / "survival_tests.json").write_text(json.dumps(survival, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(survival_rows).to_csv(out / "survival_tests.csv", index=False)

    summary = {
        "created_utc": stamp,
        "matched_models": match["matched_models"],
        "excluded_models": match["excluded_models"],
        "best_heldout_choice": match["best_model"],
        "matched_region_winners": {r["region"]: r["best_matched_model"] for r in ranking},
        "survival": survival_rows,
        "artifacts": {
            "behavior_matched_models": str(match_path),
            "ve_matched": str(out / "ve_matched.csv"),
            "ve_unmatched": str(out / "ve_unmatched.csv"),
            "survival_tests": str(out / "survival_tests.json"),
        },
    }
    (out / "phase9_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
