"""Minimal Phase 10 figure panels from saved report tables."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def fig_hidden_size_sweep(csv_path: Path, out: Path) -> None:
    """Two-panel capacity sweep: held-out accuracy and prior AUROC vs hidden size.

    Reveals the 'elbow' — the smallest hidden size where behavioral accuracy and
    block-prior decodability plateau. A vertical marker shows the frozen default (48).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    MODEL_LABELS = {"tanh_bptt": "Tanh BPTT", "tanh_pc": "Tanh PC-CA", "gru": "GRU"}
    COLORS = {"tanh_bptt": "#4C78A8", "tanh_pc": "#F58518", "gru": "#54A24B"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    sizes_all = sorted(df["hidden_size"].unique())

    for model_id, g in df.groupby("model"):
        g = g.sort_values("hidden_size")
        color = COLORS.get(model_id, "#888888")
        label = MODEL_LABELS.get(model_id, model_id)
        ax1.plot(g["hidden_size"], g["heldout_accuracy"], "-o", color=color,
                 label=label, ms=6, lw=1.8)
        ax2.plot(g["hidden_size"], g["prior_auroc"], "-o", color=color,
                 label=label, ms=6, lw=1.8)

    for ax, ycol, title, ylab in [
        (ax1, "heldout_accuracy", "Held-out choice accuracy", "Accuracy"),
        (ax2, "prior_auroc", "Block-prior decodability", "AUROC (left vs right)"),
    ]:
        if 48 in sizes_all:
            ax.axvline(48, color="grey", ls="--", lw=0.9, alpha=0.8)
            ymin = df[ycol].min()
            ax.text(48, ymin, " default=48", fontsize=7, color="grey",
                    va="bottom", ha="left", rotation=90)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes_all)
        ax.set_xticklabels([str(s) for s in sizes_all], fontsize=8)
        ax.set_xlabel("Hidden size (units)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("Hidden-size capacity sweep — find the elbow", fontsize=12, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_block_decoder_by_tick(results_json: Path, out: Path) -> None:
    """Line plot: block-prior decodability (AUROC) at each within-trial tick.

    One line per model. The x-axis is the within-trial time step; background
    shading marks the trial phase (baseline / stim / go / response / feedback),
    revealing which processing stage best encodes the block prior.
    """
    import json

    with open(results_json) as f:
        results = json.load(f)

    MODEL_LABELS = {
        "tanh_bptt": "Tanh BPTT",
        "tanh_pc":   "Tanh PC-CA",
        "gru":       "GRU",
        "bayes":     "Bayes",
    }
    COLORS = {
        "tanh_bptt": "#4C78A8",
        "tanh_pc":   "#F58518",
        "gru":       "#54A24B",
        "bayes":     "#E45756",
    }
    PHASE_COLORS = {
        "baseline": "#F2F2F2",
        "stim":     "#DDEBF7",
        "go":       "#FFF2CC",
        "response": "#FCE4D6",
        "feedback": "#E2EFDA",
    }

    order = [m for m in ("tanh_bptt", "tanh_pc", "gru", "bayes") if m in results]
    if not order:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    ref = results[order[0]]
    n_steps = ref["n_steps"]
    tick_phase = ref["tick_phase"]
    binary = ref.get("binary", True)
    metric_key = "auroc_by_tick" if binary else "accuracy_by_tick"
    std_key = "auroc_std_by_tick" if binary else "accuracy_std_by_tick"
    ylabel = "AUROC (block: left vs right)" if binary else "Accuracy (3-class)"

    x = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Phase background shading
    seen_phases = set()
    for tick in range(n_steps):
        ph = tick_phase[tick]
        ax.axvspan(tick - 0.5, tick + 0.5, color=PHASE_COLORS.get(ph, "#FFFFFF"),
                   zorder=0, label=ph if ph not in seen_phases else None)
        seen_phases.add(ph)

    # One line per model
    for m in order:
        vals = np.array(results[m][metric_key], dtype=float)
        stds = np.array(results[m].get(std_key, [0.0] * n_steps), dtype=float)
        color = COLORS.get(m, "#888888")
        ax.plot(x, vals, "-o", color=color, label=MODEL_LABELS.get(m, m),
                ms=5, lw=1.8, zorder=3)
        ax.fill_between(x, vals - stds, vals + stds, color=color, alpha=0.15, zorder=2)
        # Mark peak tick
        peak = int(np.argmax(vals))
        ax.scatter([peak], [vals[peak]], s=110, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=4)

    ax.axhline(0.5, color="grey", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{tick_phase[i]}" for i in range(n_steps)], fontsize=8)
    ax.set_xlabel("Within-trial tick (recurrent step)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Block prior decodability by within-trial tick\n"
                 "(open circles = per-model peak tick)")
    ax.spines[["top", "right"]].set_visible(False)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=2, loc="lower right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_neural_vs_model_prior(
    agg_df: pd.DataFrame,
    model_json: Path,
    out: Path,
) -> None:
    """Compare block-prior decodability: real brain regions vs model hidden layers.

    Both are decoded with the *same* logistic-regression pipeline, so AUROC is
    directly comparable. Brain regions are grouped by anatomy (thalamus /
    midbrain / hindbrain / context); artificial models are shown as a separate
    group.
    """
    import json

    THAL = {"CL", "SPF"}
    MID = {"SCm", "MRN", "SNr", "RPF", "NPC"}
    HIND = {"GRN", "IRN", "SOC", "VII", "TRN", "FOTU"}
    GROUP_COLORS = {
        "thalamus": "#9467BD",
        "midbrain": "#D62728",
        "hindbrain": "#FF7F0E",
        "context": "#7F7F7F",
        "model": "#1F77B4",
    }
    MODEL_LABELS = {
        "tanh_bptt": "Tanh BPTT",
        "tanh_pc": "Tanh PC-CA",
        "gru": "GRU",
        "bayes": "Bayes",
    }

    def _group(region: str) -> str:
        if region in THAL:
            return "thalamus"
        if region in MID:
            return "midbrain"
        if region in HIND:
            return "hindbrain"
        return "context"

    entries = []  # (label, auroc, sem, group)
    if agg_df is not None and not agg_df.empty:
        for _, r in agg_df.iterrows():
            entries.append((str(r["region"]), float(r["auroc_mean"]),
                            float(r.get("auroc_sem", 0.0)), _group(str(r["region"]))))

    if Path(model_json).exists():
        with open(model_json) as f:
            model_results = json.load(f)
        for mid, res in model_results.items():
            if isinstance(res, dict) and "auroc_mean" in res:
                entries.append((MODEL_LABELS.get(mid, mid), float(res["auroc_mean"]),
                                float(res.get("auroc_std", 0.0)), "model"))

    if not entries:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    group_order = {"thalamus": 0, "midbrain": 1, "hindbrain": 2, "context": 3, "model": 4}
    entries.sort(key=lambda e: (group_order[e[3]], -e[1]))

    labels = [e[0] for e in entries]
    aurocs = [e[1] for e in entries]
    sems = [e[2] for e in entries]
    colors = [GROUP_COLORS[e[3]] for e in entries]
    y = np.arange(len(entries))

    fig, ax = plt.subplots(figsize=(9, max(4, len(entries) * 0.42 + 1)))
    ax.barh(y, aurocs, xerr=sems, color=colors, capsize=3,
            error_kw={"elinewidth": 1.2}, height=0.66, zorder=2)
    ax.axvline(0.5, color="grey", lw=0.9, ls="--", zorder=1)

    n_neural = sum(1 for e in entries if e[3] != "model")
    if 0 < n_neural < len(entries):
        ax.axhline(n_neural - 0.5, color="black", lw=0.8, ls=":", alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Block-prior AUROC (left vs right, same LR decoder)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Block-prior decodability:\nreal brain regions vs artificial model hidden states",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g.capitalize())
        for g in ("thalamus", "midbrain", "hindbrain", "context", "model")
        if any(e[3] == g for e in entries)
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", title="Source")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_neural_block_decoder(
    session_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    out: Path,
) -> None:
    """Two-panel figure for the neural region block-prior decoder.

    Panel 1 — Horizontal bar chart: mean AUROC ± SEM per region,
               coloured by region group (thalamus / midbrain / hindbrain / context).
               Chance line at 0.5. Session-level dots overlaid.

    Panel 2 — Strip plot: per-session AUROC dots per region,
               ordered to match panel 1. Shows session-to-session variability.
    """
    from src.neural.regions import CHOICE_REGIONS, CONTEXT_REGIONS

    THAL  = {"CL", "SPF"}
    MID   = {"SCm", "MRN", "SNr", "RPF", "NPC"}
    HIND  = {"GRN", "IRN", "SOC", "VII", "TRN", "FOTU"}
    CTX   = set(CONTEXT_REGIONS.keys())

    GROUP_COLORS = {
        "thalamus":  "#9467BD",
        "midbrain":  "#D62728",
        "hindbrain": "#FF7F0E",
        "context":   "#7F7F7F",
    }

    def _group(region: str) -> str:
        if region in THAL:  return "thalamus"
        if region in MID:   return "midbrain"
        if region in HIND:  return "hindbrain"
        return "context"

    if agg_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    # Sort by descending mean AUROC
    agg = agg_df.sort_values("auroc_mean", ascending=True).reset_index(drop=True)
    regions = agg["region"].tolist()
    colors  = [GROUP_COLORS[_group(r)] for r in regions]
    y       = np.arange(len(regions))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, len(regions) * 0.45 + 1.5)),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # ---- Panel 1: mean AUROC ± SEM ----------------------------------------
    bars = ax1.barh(y, agg["auroc_mean"], xerr=agg["auroc_sem"],
                    color=colors, capsize=4, error_kw={"elinewidth": 1.4},
                    height=0.6, zorder=2)
    # Session dots
    for i, region in enumerate(regions):
        sess = session_df.loc[session_df["region"] == region, "auroc"].dropna()
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(sess))
        ax1.scatter(sess, i + jitter, color="white", edgecolors=colors[i],
                    s=18, zorder=3, linewidths=0.9, alpha=0.85)
    ax1.axvline(0.5, color="grey", lw=0.9, ls="--", label="chance")
    ax1.set_yticks(y)
    ax1.set_yticklabels(regions, fontsize=9)
    ax1.set_xlabel("AUROC (block prior: left vs right)")
    ax1.set_title("Block prior decoder — real neural data\n(pre-stimulus window, good units)")
    ax1.set_xlim(0.3, 1.0)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    # Annotate n_sessions
    for i, row in agg.iterrows():
        ax1.text(ax1.get_xlim()[1] - 0.01, i, f"n={int(row['n_sessions'])}",
                 va="center", ha="right", fontsize=7, color="dimgrey")

    # ---- Panel 2: strip plot -----------------------------------------------
    for i, region in enumerate(regions):
        sess = session_df.loc[session_df["region"] == region, "auroc"].dropna().to_numpy()
        if len(sess) == 0:
            continue
        jitter = np.random.default_rng(i + 100).uniform(-0.2, 0.2, len(sess))
        ax2.scatter(sess, i + jitter, color=colors[i], s=22, alpha=0.75, zorder=2)
        ax2.plot([np.mean(sess)], [i], "k|", ms=12, mew=1.5, zorder=3)
    ax2.axvline(0.5, color="grey", lw=0.9, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels(regions, fontsize=9)
    ax2.set_xlabel("Per-session AUROC")
    ax2.set_title("Session variability")
    ax2.set_xlim(0.3, 1.0)
    ax2.spines[["top", "right"]].set_visible(False)

    # Legend for groups
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g.capitalize())
        for g in ("thalamus", "midbrain", "hindbrain", "context")
        if any(_group(r) == g for r in regions)
    ]
    ax1.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="grey", ls="--", label="chance (0.5)")
    ], fontsize=8, loc="lower right")

    fig.suptitle(
        "Block Prior Decoding from Real Neural Data\n"
        "IBL BWM regions (choice-selective, IBL 2025 Nature Fig 5)",
        fontsize=11, y=1.02,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)



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


def fig_block_decoder(results_json: Path, out: Path) -> None:
    """Three-panel comparison plot for the block prior LR decoder.

    Panel 1 — Accuracy (mean ± std across CV folds, dot-per-fold overlay)
    Panel 2 — AUROC (binary only; same layout)
    Panel 3 — Normalized confusion matrices side-by-side (one per model)
    """
    import json
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    with open(results_json) as f:
        results = json.load(f)

    MODEL_LABELS = {
        "tanh_bptt": "Tanh BPTT",
        "tanh_pc":   "Tanh PC-CA",
        "gru":       "GRU",
        "bayes":     "Bayes",
    }
    COLORS = {
        "tanh_bptt": "#4C78A8",
        "tanh_pc":   "#F58518",
        "gru":       "#54A24B",
        "bayes":     "#E45756",
    }
    order = [m for m in ("tanh_bptt", "tanh_pc", "gru", "bayes") if m in results]
    labels = [MODEL_LABELS.get(m, m) for m in order]
    colors = [COLORS.get(m, "#888888") for m in order]
    binary = results[order[0]].get("binary", True)
    n_models = len(order)

    # Top-level GridSpec: [acc | auroc | confusion×n_models]
    n_left_panels = 2 if binary else 1
    fig = plt.figure(figsize=(4.5 * (n_left_panels + 1) + 0.5, 4.4))
    gs = GridSpec(1, n_left_panels + 1, figure=fig, wspace=0.38)

    x = np.arange(n_models)

    def _bar_panel(ax, means, stds, fold_lists, ylabel, title):
        ax.bar(x, means, yerr=stds, color=colors, capsize=5,
               error_kw={"elinewidth": 1.5}, width=0.55, zorder=2)
        for i, folds in enumerate(fold_lists):
            jitter = np.linspace(-0.12, 0.12, len(folds))
            ax.scatter(i + jitter, folds, color="white", edgecolors=colors[i],
                       s=28, zorder=3, linewidths=1.2)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- Panel 1: Accuracy ------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _bar_panel(
        ax1,
        [results[m]["accuracy_mean"] for m in order],
        [results[m]["accuracy_std"]  for m in order],
        [results[m]["accuracy_folds"] for m in order],
        "Accuracy",
        "Block decoding accuracy\n(left vs right, 5-fold CV)",
    )

    # ---- Panel 2: AUROC ---------------------------------------------------
    if binary:
        ax2 = fig.add_subplot(gs[0, 1])
        _bar_panel(
            ax2,
            [results[m].get("auroc_mean", float("nan")) for m in order],
            [results[m].get("auroc_std", 0.0) for m in order],
            [results[m].get("auroc_folds", []) for m in order],
            "AUROC",
            "Block decoding AUROC\n(left vs right, 5-fold CV)",
        )
        cm_col = 2
    else:
        cm_col = 1

    # ---- Panel 3: Normalized confusion matrices ----------------------------
    n_classes = len(results[order[0]]["confusion_matrix"])
    class_labels = ["Left", "Right"] if n_classes == 2 else ["Left", "Unbiased", "Right"]

    gs_cm = GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs[0, cm_col], wspace=0.4)
    for idx, m in enumerate(order):
        cm = np.array(results[m]["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        sub = fig.add_subplot(gs_cm[0, idx])
        sub.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        sub.set_xticks(range(n_classes))
        sub.set_xticklabels(class_labels, fontsize=7)
        sub.set_yticks(range(n_classes))
        sub.set_yticklabels(class_labels if idx == 0 else [""] * n_classes, fontsize=7)
        sub.set_xlabel("Predicted", fontsize=7)
        if idx == 0:
            sub.set_ylabel("True", fontsize=7)
        sub.set_title(MODEL_LABELS.get(m, m), fontsize=8, color=colors[idx], pad=3)
        for r in range(n_classes):
            for c in range(n_classes):
                sub.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if cm_norm[r, c] > 0.6 else "black")

    # Title for confusion panel
    cm_ax = fig.add_subplot(gs[0, cm_col])
    cm_ax.set_title("Normalized confusion matrices", pad=36, fontsize=10)
    cm_ax.axis("off")

    fig.suptitle("Block Prior Linear Decoder — All Models", fontsize=11, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
