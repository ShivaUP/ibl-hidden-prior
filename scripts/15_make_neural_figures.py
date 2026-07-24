#!/usr/bin/env python3
"""15 — Neural comparison figures (shared cohort; all-model VE + survival).

Behavior-matched dual bars are retired from the primary display.

Usage:
  python scripts/15_make_neural_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot.v2_style import (
    MODEL_COLORS,
    PASTEL,
    apply_style,
    pad_ylim_for_labels,
    save_figure,
    style_axes_title,
    style_suptitle,
    style_ylabel,
)

MODEL_LABELS = {
    "tanh_bptt": "tanh BPTT",
    "tanh_pc": "tanh PC",
    "gru": "GRU",
    "gru_pc": "GRU PC",
    "bayes": "Bayes",
}

REGION_LABELS = {
    "MOs": "MOs",
    "vlOFC_orbvl": "vlOFC",
    "ACAd": "ACAd",
    "MOp": "MOp",
}


def _region_label(region: str) -> str:
    return REGION_LABELS.get(region, region)


def main() -> int:
    apply_style()
    neural = ROOT / "reports" / "v2" / "neural"
    fig_dir = ROOT / "reports" / "v2" / "figures" / "neural"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ve_u = neural / "ve_unmatched.csv"
    if not ve_u.exists():
        print(f"Missing {ve_u}", file=sys.stderr)
        return 1

    u = pd.read_csv(ve_u)

    def sess_mean(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["region", "model"], as_index=False)
            .agg(
                ve=("ve_linear_recal", "mean"),
                ve_std=("ve_linear_recal", "std"),
                n=("eid", "nunique"),
            )
        )

    uu = sess_mean(u)
    prefer = ["MOs", "vlOFC_orbvl", "ACAd", "MOp"]
    regions = [r for r in prefer if r in set(uu["region"])] + sorted(
        set(uu["region"]) - set(prefer)
    )
    models = [x for x in ("tanh_bptt", "tanh_pc", "gru", "gru_pc") if x in set(uu["model"])]

    n_reg = max(len(regions), 1)
    ncols = min(4, n_reg)
    nrows = int(np.ceil(n_reg / ncols))
    fig_w = max(5.4 * ncols, 11.0)
    fig_h = 5.2 * nrows + 1.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.12, wspace=0.30, hspace=0.40)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n_reg:]:
        ax.axis("off")

    all_vals: list[float] = []
    all_errs: list[float] = []
    for ax, region in zip(axes_flat, regions):
        g_u = uu.loc[uu["region"] == region].set_index("model")
        x = np.arange(len(models))
        vals_u = [float(g_u.loc[mo, "ve"]) if mo in g_u.index else np.nan for mo in models]
        err_u = []
        for mo in models:
            if mo in g_u.index and g_u.loc[mo, "n"] > 1:
                err_u.append(1.96 * float(g_u.loc[mo, "ve_std"]) / np.sqrt(g_u.loc[mo, "n"]))
            else:
                err_u.append(0.0)
        all_vals.extend(vals_u)
        all_errs.extend(err_u)
        bar_colors = [MODEL_COLORS.get(mo, PASTEL["blue"]) for mo in models]
        ax.bar(
            x,
            vals_u,
            width=0.65,
            yerr=err_u,
            capsize=3,
            color=bar_colors,
            edgecolor=PASTEL["ink"],
            linewidth=0.8,
            ecolor=PASTEL["ink"],
            alpha=0.95,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS.get(mo, mo) for mo in models],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        n_sess = int(g_u["n"].max()) if len(g_u) else 0
        style_axes_title(ax, f"{_region_label(region)}  (n={n_sess} sessions)", pad=8)
        ax.axhline(0, color=PASTEL["ink"], lw=0.6)

    pad_ylim_for_labels(axes_flat[0], all_vals, all_errs, headroom=0.08)
    ymin, ymax = axes_flat[0].get_ylim()
    axes_flat[0].set_ylim(max(ymin, -0.04), ymax)
    style_ylabel(
        axes_flat[0],
        "VE of neural prior by model belief\n(linear recal; mean ± 95% CI)",
    )
    style_suptitle(
        fig,
        "Neural prior alignment — all models (shared cohort)",
        y=0.96,
    )
    out1 = fig_dir / "neural_ve_unmatched_vs_matched.png"
    save_figure(fig, out1)
    # Alias without the outdated matched wording
    out1b = fig_dir / "neural_ve_by_model.png"
    save_figure(fig, out1b)
    plt.close(fig)

    surv = neural / "survival_tests.csv"
    out2 = fig_dir / "survival_tests.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.18)
    if surv.exists():
        df = pd.read_csv(surv)
        if "delta" in df.columns and "region" in df.columns:
            order = [r for r in prefer if r in set(df["region"])] + [
                r for r in df["region"].tolist() if r not in prefer
            ]
            # preserve order unique
            seen = set()
            order_u = []
            for r in order:
                if r not in seen and r in set(df["region"]):
                    order_u.append(r)
                    seen.add(r)
            df = df.set_index("region").loc[order_u].reset_index()
            labels = [_region_label(r) for r in df["region"]]
            colors = [
                PASTEL["green"] if bool(s) else PASTEL["gray"]
                for s in df.get("survive_alpha_05", [False] * len(df))
            ]
            x = np.arange(len(df))
            ax.bar(
                x,
                df["delta"],
                color=colors,
                edgecolor=PASTEL["ink"],
                linewidth=0.8,
                width=0.65,
            )
            if "ci_low" in df.columns:
                yerr = np.vstack([df["delta"] - df["ci_low"], df["ci_high"] - df["delta"]])
                ax.errorbar(
                    x, df["delta"], yerr=yerr, fmt="none", ecolor=PASTEL["ink"], capsize=4
                )
            vals = df["delta"].to_numpy(dtype=float)
            lo = float(np.nanmin(df["ci_low"])) if "ci_low" in df.columns else float(np.nanmin(vals))
            hi = float(np.nanmax(df["ci_high"])) if "ci_high" in df.columns else float(np.nanmax(vals))
            span = max(hi - lo, 1e-3)
            ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.axhline(0, color=PASTEL["ink"], lw=0.7)
            style_ylabel(ax, "VE gap (best − second model)")
            style_axes_title(
                ax,
                "Does the top model’s neural edge “survive”?\n"
                "green = yes after session-bootstrap + Holm across regions",
            )
            for i, row in df.iterrows():
                mark = "✓" if bool(row.get("survive_alpha_05")) else "ns"
                ax.text(
                    i,
                    float(row["delta"]) + 0.02 * span,
                    mark,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=PASTEL["ink"],
                )
    style_suptitle(fig, "Survival test of neural VE advantages (all models)", y=0.97)
    save_figure(fig, out2)
    plt.close(fig)

    print(
        {
            "figures": [
                str(out1.relative_to(ROOT)),
                str(out1b.relative_to(ROOT)),
                str(out2.relative_to(ROOT)),
            ]
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
