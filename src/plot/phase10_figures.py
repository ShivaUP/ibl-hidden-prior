"""Minimal Phase 10 figure panels from saved report tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_psychometrics(psych_csv: Path, out: Path) -> None:
    df = pd.read_csv(psych_csv)
    d = df.loc[df["condition"] == "history_only"] if "condition" in df.columns else df
    fig, ax = plt.subplots(figsize=(6, 4))
    if {"signed_contrast", "p_right_mouse"}.issubset(d.columns):
        mouse = d.groupby("signed_contrast", as_index=False)["p_right_mouse"].mean()
        ax.plot(mouse["signed_contrast"], mouse["p_right_mouse"], "ko-", label="mouse", ms=5)
        if "model" in d.columns and "p_right_model" in d.columns:
            for model, g in d.groupby("model"):
                agg = g.groupby("signed_contrast", as_index=False)["p_right_model"].mean()
                ax.plot(agg["signed_contrast"], agg["p_right_model"], "--", label=model)
    else:
        ax.text(0.5, 0.5, "psychometrics schema unexpected", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Signed contrast")
    ax.set_ylabel("P(choice right)")
    ax.set_title("Psychometrics (history-only)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_heldout_choice(held_csv: Path, out: Path) -> None:
    df = pd.read_csv(held_csv)
    d = df.loc[df["condition"] == "history_only"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(d["model"], d["choice_nll"], color=["#4C78A8", "#F58518", "#54A24B"][: len(d)])
    ax.set_ylabel("Held-out choice NLL")
    ax.set_title("History-only held-out choice")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_prior_match(prior_csv: Path, out: Path) -> None:
    df = pd.read_csv(prior_csv)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(df["model"], df["corr"], color=["#4C78A8", "#F58518", "#54A24B"][: len(df)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Corr(mouse prior, model q)")
    ax.set_title("History-only prior match")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_switch(traj_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for model in ("standard", "pc", "bayes"):
        path = traj_dir / f"switch_traj_{model}_all.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rel_trial" not in df.columns:
            continue
        ycol = "p_align_model" if "p_align_model" in df.columns else (
            "p_align_mouse" if "p_align_mouse" in df.columns else None
        )
        if ycol is None:
            continue
        g = df.groupby("rel_trial", as_index=False)[ycol].mean()
        ax.plot(g["rel_trial"], g[ycol], label=model)
        plotted = True
    # mouse from any file
    for model in ("standard", "pc", "bayes"):
        path = traj_dir / f"switch_traj_{model}_all.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "p_align_mouse" in df.columns:
                g = df.groupby("rel_trial", as_index=False)["p_align_mouse"].mean()
                ax.plot(g["rel_trial"], g["p_align_mouse"], "k--", label="mouse", lw=2)
            break
    if not plotted:
        ax.text(0.5, 0.5, "no switch traj", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Trials from block switch")
    ax.set_ylabel("P(choice aligned with new block)")
    ax.set_title("Switch-centered updating")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_neural_ve(ve_unmatched: Path, ve_matched: Path, out: Path) -> None:
    u = pd.read_csv(ve_unmatched)
    m = pd.read_csv(ve_matched)
    regions = sorted(u["region"].unique())
    models = ["standard", "pc", "bayes"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, region in zip(axes, regions):
        uu = u.loc[u["region"] == region].set_index("model")
        mm = m.loc[m["region"] == region].set_index("model") if len(m) else pd.DataFrame()
        x = np.arange(len(models))
        vals_u = [float(uu.loc[mo, "ve_linear_recal"]) if mo in uu.index else np.nan for mo in models]
        ax.bar(x - 0.18, vals_u, width=0.35, label="unmatched", color="#9ecae1")
        vals_m = [
            float(mm.loc[mo, "ve_linear_recal"]) if (len(mm) and mo in mm.index) else np.nan
            for mo in models
        ]
        ax.bar(x + 0.18, vals_m, width=0.35, label="matched", color="#2171b5")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(region)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("VE (linear recal)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Neural prior-readout VE (pilot eid)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_survival(surv_csv: Path, out: Path) -> None:
    df = pd.read_csv(surv_csv)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if "delta" in df.columns and "region" in df.columns:
        ax.bar(df["region"], df["delta"], color="#54A24B")
        if "ci_low" in df.columns:
            yerr = np.vstack(
                [
                    df["delta"] - df["ci_low"],
                    df["ci_high"] - df["delta"],
                ]
            )
            ax.errorbar(df["region"], df["delta"], yerr=yerr, fmt="none", ecolor="k", capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("VE delta (matched)")
        ax.set_title("Survival tests (Holm-corrected elsewhere)")
    else:
        ax.text(0.5, 0.5, "no survival rows", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
