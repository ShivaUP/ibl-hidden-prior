#!/usr/bin/env python3
"""15 — Neural comparison figures (v2).

Usage:
  python scripts/15_make_neural_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_COLORS = {
    "tanh_bptt": "#4c72b0",
    "tanh_pc": "#55a868",
    "gru": "#c44e52",
    "bayes": "#8172b3",
}
MODEL_LABELS = {
    "tanh_bptt": "tanh BPTT",
    "tanh_pc": "tanh PC",
    "gru": "GRU",
    "bayes": "Bayes",
}


def main() -> int:
    neural = ROOT / "reports" / "v2" / "neural"
    fig_dir = ROOT / "reports" / "v2" / "figures" / "neural"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ve_u = neural / "ve_unmatched.csv"
    ve_m = neural / "ve_matched.csv"
    if not ve_u.exists():
        print(f"Missing {ve_u}; run scripts/13_eval_neural_pilot.py", file=sys.stderr)
        return 1

    u = pd.read_csv(ve_u)
    m = pd.read_csv(ve_m) if ve_m.exists() else pd.DataFrame()
    regions = sorted(u["region"].unique())
    models = [x for x in ("tanh_bptt", "tanh_pc", "gru", "bayes") if x in set(u["model"])]

    fig, axes = plt.subplots(1, max(len(regions), 1), figsize=(4.2 * max(len(regions), 1), 4.2), sharey=True)
    if len(regions) <= 1:
        axes = [axes]
    for ax, region in zip(axes, regions):
        uu = u.loc[u["region"] == region].set_index("model")
        mm = m.loc[m["region"] == region].set_index("model") if len(m) else pd.DataFrame()
        x = np.arange(len(models))
        vals_u = [float(uu.loc[mo, "ve_linear_recal"]) if mo in uu.index else np.nan for mo in models]
        vals_m = [
            float(mm.loc[mo, "ve_linear_recal"]) if (len(mm) and mo in mm.index) else np.nan
            for mo in models
        ]
        ax.bar(x - 0.18, vals_u, width=0.35, label="all models (unmatched)", color="#9ecae1")
        ax.bar(x + 0.18, vals_m, width=0.35, label="behavior-matched only", color="#2171b5")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(mo, mo) for mo in models], rotation=15)
        ax.set_title(region)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("VE of neural prior by model q\n(linear recalibration)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Neural comparison (pilot)\n"
        "Does the model's latent prior explain population activity in MOs / vlOFC?",
        fontsize=11,
    )
    fig.tight_layout()
    out1 = fig_dir / "neural_ve_unmatched_vs_matched.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # Survival
    surv = neural / "survival_tests.csv"
    out2 = fig_dir / "survival_tests.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    if surv.exists():
        df = pd.read_csv(surv)
        if "ve_obs" in df.columns and "region" in df.columns:
            colors = ["#54A24B" if bool(s) else "#cccccc" for s in df.get("survive_alpha_05", [False] * len(df))]
            ax.bar(df["region"], df["ve_obs"], color=colors)
            if "ci_low" in df.columns:
                yerr = np.vstack([df["ve_obs"] - df["ci_low"], df["ci_high"] - df["ve_obs"]])
                ax.errorbar(df["region"], df["ve_obs"], yerr=yerr, fmt="none", ecolor="k", capsize=4)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("Matched VE (obs)")
            ax.set_title("Survival: green = Holm p < 0.05")
        elif "delta" in df.columns:
            ax.bar(df["region"], df["delta"], color="#54A24B")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("VE delta (best − second)")
            ax.set_title("Matched survival tests")
        else:
            ax.text(0.5, 0.5, df.to_string(), ha="center", va="center", fontsize=7, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Run scripts/14_eval_neural_matched.py", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    guide = fig_dir / "README.md"
    guide.write_text(
        "# Neural figures (v2)\n\n"
        "- `neural_ve_unmatched_vs_matched.png` — primary VE bars per region.\n"
        "  Light = all models; dark = only behavior-matched (CE within ε of best).\n"
        "- `survival_tests.png` — does matched VE advantage survive bootstrap + Holm?\n\n"
        "Model q is synth-trained belief on the neural pilot session.\n"
        "Neural axis is CV Ridge readout of behavior-derived mouse prior (v1 Phase 8).\n",
        encoding="utf-8",
    )
    print(json.dumps({"figures": [str(out1.relative_to(ROOT)), str(out2.relative_to(ROOT))]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
