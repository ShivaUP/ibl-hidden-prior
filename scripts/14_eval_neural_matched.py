#!/usr/bin/env python3
"""14 — Behavior-matched neural VE + survival (v2).

Matches models on real history_only **cross-entropy** (correct-side), ε-ball.
RT floor is non-binding in v2 (rt_nll set to 0 for all models).

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
from src.neural.prior_readout import fit_prior_readout, model_explains_neural_prior
from src.neural.survival import bootstrap_ve_advantage, holm_correct
from src.synthetic.schema import load_synthetic_config

MODELS = ("tanh_bptt", "tanh_pc", "gru", "bayes")
DEFAULT_EID = "1191f865-b10a-45c8-9c48-24a980fd9402"


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
                "rt_nll": 0.0,  # RT not used in v2 selection
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()
    cfg = load_synthetic_config()
    out = ROOT / "reports" / "v2" / "neural"
    out.mkdir(parents=True, exist_ok=True)

    held = _behavior_metrics_df()
    if held.empty:
        print(
            "Missing reports/v2/metrics/real_history_only_*.json — run scripts/11_eval_regimes.py --domain real",
            file=sys.stderr,
        )
        return 1

    # choice ε from v1 freeze; RT floor effectively off
    match_cfg = MatchConfig(choice_epsilon=0.05, rt_nll_floor=1e9, choice_primary=True)
    match = select_behavior_matched(held, condition="history_only", cfg=match_cfg)
    match["metric_source"] = "reports/v2/metrics/real_history_only_*.json CE (correct side)"
    match["created_utc"] = stamp
    match["notes"] = [
        "Choice-primary ε-ball on real history_only cross-entropy.",
        "RT floor non-binding in v2.",
        "Neural VE uses synth-trained model belief on neural pilot eid.",
    ]
    (out / "behavior_matched_models.json").write_text(
        json.dumps(match, indent=2), encoding="utf-8"
    )
    print(f"Matched: {match['matched_models']}")
    print(f"Excluded: {match['excluded_models']}")

    ve_path = out / "ve_unmatched_pilot.csv"
    if not ve_path.exists():
        ve_path = out / "ve_unmatched.csv"
    if not ve_path.exists():
        print("Missing ve_unmatched; run scripts/13_eval_neural_pilot.py first", file=sys.stderr)
        return 1
    ve_all = pd.read_csv(ve_path)
    ve_unmatched = ve_all.copy()
    ve_unmatched["confirmatory"] = False
    ve_matched = filter_ve_to_matched(ve_all, match["matched_models"])
    ve_unmatched.to_csv(out / "ve_unmatched.csv", index=False)
    ve_matched.to_csv(out / "ve_matched.csv", index=False)

    ranking = []
    for region, g in ve_matched.groupby("region"):
        g2 = g.sort_values("ve_linear_recal", ascending=False)
        ranking.append(
            {
                "region": region,
                "ranking": g2[["model", "ve_linear_recal", "corr"]].to_dict(orient="records"),
                "best_matched_model": str(g2.iloc[0]["model"]) if len(g2) else None,
                "best_matched_ve": float(g2.iloc[0]["ve_linear_recal"]) if len(g2) else float("nan"),
            }
        )

    pilot_eid = DEFAULT_EID
    neural_dir = ROOT / "data" / "processed" / "neural" / pilot_eid
    neural_v2 = ROOT / "data" / "processed" / "neural_v2" / pilot_eid
    trials = pd.read_parquet(neural_dir / "trials.parquet")
    mouse_prior = trials["mouse_prior_hat"].to_numpy(dtype=float)

    survival_rows = []
    for region_rank in ranking:
        region = region_rank["region"]
        models = [r["model"] for r in region_rank["ranking"]]
        npz = neural_dir / f"{region}_counts.npz"
        if not npz.exists() or not models:
            survival_rows.append({"region": region, "error": "missing counts or no matched models"})
            continue
        counts = np.asarray(np.load(npz)["counts"], dtype=float)
        m = min(counts.shape[0], len(mouse_prior))
        readout = fit_prior_readout(counts[:m], mouse_prior[:m])
        mask = readout["mask"]
        neural_full = np.full(mask.shape[0], np.nan)
        neural_full[mask] = readout["oof_pred"][: int(mask.sum())]

        # Build per-trial VE sources for bootstrap: use model q series
        q_by_model = {}
        for mid in models:
            pq = neural_v2 / f"{mid}_prior_q.parquet"
            if pq.exists():
                q_by_model[mid] = pd.read_parquet(pq)["prior_q"].to_numpy(dtype=float)[:m]

        if len(q_by_model) == 1:
            mid = models[0]
            # VE > 0 survival via bootstrap of ve_linear_recal
            from src.neural.prior_readout import variance_explained

            a = neural_full
            b = q_by_model[mid][: len(a)]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            rng = np.random.default_rng(0)
            boots = []
            n = len(a)
            for _ in range(1000):
                idx = rng.integers(0, n, size=n)
                aa, bb = a[idx], b[idx]
                if np.std(bb) < 1e-12:
                    boots.append(np.nan)
                    continue
                coef = np.polyfit(bb, aa, 1)
                boots.append(variance_explained(aa, coef[0] * bb + coef[1]))
            boots = np.asarray(boots, dtype=float)
            p_gt0 = float(np.mean(boots <= 0))
            survival_rows.append(
                {
                    "region": region,
                    "test": "matched_ve_gt_zero",
                    "model": mid,
                    "ve_obs": float(region_rank["best_matched_ve"]),
                    "p_boot": p_gt0,
                    "ci_low": float(np.nanpercentile(boots, 2.5)),
                    "ci_high": float(np.nanpercentile(boots, 97.5)),
                }
            )
        elif len(q_by_model) >= 2:
            best = models[0]
            second = models[1]
            # use existing helper if signatures match
            try:
                res = bootstrap_ve_advantage(
                    neural_full,
                    q_by_model[best][: len(neural_full)],
                    q_by_model[second][: len(neural_full)],
                    n_boot=1000,
                    seed=0,
                )
                survival_rows.append(
                    {
                        "region": region,
                        "test": "matched_best_vs_second",
                        "best_model": best,
                        "second_model": second,
                        "delta": float(res.get("delta", np.nan)),
                        "ci_low": float(res.get("ci_low", np.nan)),
                        "ci_high": float(res.get("ci_high", np.nan)),
                        "p_boot": float(res.get("p_two_sided", np.nan)),
                        "ve_cand": float(res.get("ve_cand", np.nan)),
                        "ve_base": float(res.get("ve_base", np.nan)),
                    }
                )
            except TypeError:
                survival_rows.append(
                    {
                        "region": region,
                        "test": "matched_best_vs_second",
                        "best_model": best,
                        "second_model": second,
                        "error": "bootstrap_ve_advantage signature mismatch — inspect src/neural/survival.py",
                    }
                )

    # Holm across regions for p_boot / p fields
    pvals = []
    for row in survival_rows:
        if "p_boot" in row:
            pvals.append(float(row["p_boot"]))
        elif "p" in row:
            pvals.append(float(row["p"]))
        else:
            pvals.append(float("nan"))
    if pvals and np.any(np.isfinite(pvals)):
        adjusted = holm_correct(np.asarray(pvals, dtype=float))
        for row, p_h in zip(survival_rows, adjusted):
            row["p_holm"] = float(p_h) if np.isfinite(p_h) else float("nan")
            row["survive_alpha_05"] = bool(np.isfinite(p_h) and p_h < 0.05)

    (out / "survival_tests.json").write_text(json.dumps(survival_rows, indent=2), encoding="utf-8")
    pd.DataFrame(survival_rows).to_csv(out / "survival_tests.csv", index=False)

    summary = {
        "stage": "neural_matched_v2",
        "created_utc": stamp,
        "matched_models": match["matched_models"],
        "excluded_models": match["excluded_models"],
        "ranking": ranking,
        "survival": survival_rows,
        "config_paths": cfg["paths"],
    }
    (out / "phase9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"matched": match["matched_models"], "n_survival": len(survival_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
