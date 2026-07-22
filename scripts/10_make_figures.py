#!/usr/bin/env python3
"""10 — Make v2 figures (synth + real, per-regime multipanel + comparisons).

Layout
------
  by_model/{model}/{synth|real}/{regime}/multipanel_diagnostics.png
  comparison/{synth|real}_{regime}_accuracy.png
  comparison/{synth|real}_{regime}_history_gap.png
  comparison/synth_vs_real_{regime}_accuracy.png
  comparison/synth_vs_real_{regime}_history_gap.png

Usage:
  python scripts/11_eval_regimes.py   # first, if rollouts missing
  python scripts/10_make_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_v2.rollout import REGIMES, switch_centered_zero_evidence
from src.synthetic.schema import RIGHT, load_synthetic_config

DOMAINS = ("synth", "real")


def _rollout_path(cfg: dict, domain: str, regime: str, model_id: str) -> Path:
    base = ROOT / cfg["paths"]["artifacts"]
    if domain == "synth":
        path = base / "synthetic" / "regimes" / regime / model_id / "rollout.npz"
        if not path.exists() and regime == "history_only":
            path = base / "synthetic" / "heldout" / model_id / "rollout.npz"
        return path
    return base / "real" / "regimes" / regime / model_id / "rollout.npz"


def _metrics_path(cfg: dict, domain: str, regime: str, model_id: str) -> Path:
    return ROOT / cfg["paths"]["reports"] / "metrics" / f"{domain}_{regime}_{model_id}.json"


def _true_p_right(roll) -> np.ndarray:
    files = set(roll.files)
    if "true_p_right" in files:
        return roll["true_p_right"]
    return 1.0 - roll["probability_left"]


def _p_choice(roll) -> np.ndarray:
    files = set(roll.files)
    if "p_choice_right" in files:
        return roll["p_choice_right"]
    return roll["p_right"]


def _zero_ev(roll) -> np.ndarray:
    files = set(roll.files)
    if "zero_evidence_p_right" in files:
        return roll["zero_evidence_p_right"]
    return roll["belief"]


def _valid_mask(roll, true_p: np.ndarray) -> np.ndarray:
    files = set(roll.files)
    if "valid" in files:
        return np.asarray(roll["valid"], dtype=bool)
    return np.isfinite(true_p)


def _plot_psychometric(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    p_choice = _p_choice(roll)
    side = roll["side"]
    contrast = roll["contrast"]
    valid = _valid_mask(roll, true_p)
    signed = np.where(side == RIGHT, contrast, -contrast)
    signed = np.where(valid, signed, np.nan)
    unique_signed = np.unique(np.round(signed[np.isfinite(signed)], 5))

    if regime == "fixed_prior":
        means, xs = [], []
        for value in unique_signed:
            mask = valid & np.isclose(signed, value)
            if mask.any():
                xs.append(float(value))
                means.append(float(np.nanmean(p_choice[mask])))
        if xs:
            ax.plot(xs, means, marker="o", color="#0072b2", label="block P(right)=0.5")
    else:
        for prior, color in ((0.2, "#d55e00"), (0.8, "#0072b2")):
            means, xs = [], []
            for value in unique_signed:
                mask = valid & np.isclose(true_p, prior) & np.isclose(signed, value)
                if mask.any():
                    xs.append(float(value))
                    means.append(float(np.nanmean(p_choice[mask])))
            if xs:
                ax.plot(
                    xs,
                    means,
                    marker="o",
                    label=f"block P(right)={prior:.1f}",
                    color=color,
                )
        # Real: also show mouse psychometric as thin dashed reference
        if domain == "real" and "mouse_choice" in set(roll.files):
            mouse = roll["mouse_choice"]
            mouse_right = (mouse == RIGHT).astype(np.float64)
            for prior, color in ((0.2, "#d55e00"), (0.8, "#0072b2")):
                means, xs = [], []
                for value in unique_signed:
                    mask = valid & np.isclose(true_p, prior) & np.isclose(signed, value)
                    if mask.any():
                        xs.append(float(value))
                        means.append(float(np.nanmean(mouse_right[mask])))
                if xs:
                    ax.plot(
                        xs,
                        means,
                        linestyle="--",
                        linewidth=1,
                        color=color,
                        alpha=0.55,
                        label=f"mouse P(right)={prior:.1f}",
                    )

    ax.axhline(0.5, color="0.7", linewidth=1)
    ax.axvline(0.0, color="0.7", linewidth=1)
    ax.set(
        title=f"Psychometric ({domain}, {regime})",
        xlabel="Signed contrast (left −, right +)",
        ylabel="P(choice right)",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False, fontsize=8)


def _plot_switch(ax, roll, regime: str) -> None:
    if regime == "fixed_prior":
        ax.text(
            0.5,
            0.5,
            "N/A (no block switches)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set(
            title="Belief adaptation (N/A)",
            xlabel="Trials relative to switch",
            ylabel="P(right) zero evidence",
        )
        return
    curve = switch_centered_zero_evidence(roll, before=20, after=30)
    ax.plot(curve["offsets"], curve["low_to_high"], color="#0072b2", label="0.2 → 0.8")
    ax.plot(curve["offsets"], curve["high_to_low"], color="#d55e00", label="0.8 → 0.2")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        title="Belief adaptation around block switches",
        xlabel="Trials relative to switch",
        ylabel="P(right) with zero sensory evidence",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False)


def _example_session_index(roll, true_p: np.ndarray) -> int:
    valid = _valid_mask(roll, true_p)
    counts = valid.sum(axis=1)
    return int(np.argmax(counts))


def per_model_domain_regime_figure(
    model_id: str, domain: str, regime: str, cfg: dict, fig_root: Path
) -> Path | None:
    hist_path = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "train_history.json"
    roll_path = _rollout_path(cfg, domain, regime, model_id)
    if not hist_path.exists() or not roll_path.exists():
        print(
            f"SKIP {domain}/{model_id}/{regime}: missing history or rollout",
            file=sys.stderr,
        )
        return None

    hist = json.loads(hist_path.read_text())["history"]
    roll = np.load(roll_path)
    true_p = _true_p_right(roll)
    zero_ev = _zero_ev(roll)
    valid = _valid_mask(roll, true_p)
    sess = _example_session_index(roll, true_p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color="#3b6ea8")
    ylab = "PC energy / step" if model_id == "tanh_pc" else "Response cross-entropy"
    ax.set(title="Training (synth)", xlabel="Epoch", ylabel=ylab)

    _plot_psychometric(axes[0, 1], roll, regime, domain)
    _plot_switch(axes[1, 0], roll, regime)

    ax = axes[1, 1]
    trials = np.arange(true_p.shape[1])
    v = valid[sess]
    tp = np.where(v, true_p[sess], np.nan)
    ze = np.where(v, zero_ev[sess], np.nan)
    ax.step(trials, tp, where="post", color="black", linewidth=1.5, label="true block P(right)")
    ax.plot(trials, ze, color="#8e44ad", alpha=0.9, label="model zero-evidence preference")
    if domain == "real" and "mouse_choice" in set(roll.files):
        # running mean of mouse right-choice in a window for visual reference
        mouse = roll["mouse_choice"][sess]
        mr = np.where(v & (mouse >= 0), (mouse == RIGHT).astype(float), np.nan)
        ax.plot(trials, mr, color="0.55", alpha=0.35, linewidth=0.8, label="mouse choice (1=R)")
    ax.set(
        title=f"Example session ({domain}, {regime})",
        xlabel="Trial",
        ylabel="Probability right",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Hidden-prior diagnostics — {model_id} — {domain} — {regime}", fontsize=13
    )
    out = fig_root / "by_model" / model_id / domain / regime
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multipanel_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # Legacy aliases for history_only synth
    if domain == "synth" and regime == "history_only":
        legacy = fig_root / "by_model" / model_id
        legacy.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, legacy / "multipanel_diagnostics.png")
        legacy_reg = fig_root / "by_model" / model_id / regime
        legacy_reg.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, legacy_reg / "multipanel_diagnostics.png")

    return path


def _load_metric_rows(cfg: dict, domain: str, regime: str) -> list[dict]:
    rows = []
    for mid in cfg["models"]:
        path = _metrics_path(cfg, domain, regime, mid)
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def _bar_accuracy(rows: list[dict], domain: str, regime: str, out: Path) -> Path | None:
    if not rows:
        return None
    names = [r["model_id"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    if domain == "real":
        x = np.arange(len(rows))
        w = 0.35
        ax.bar(
            x - w / 2,
            [r.get("acc_vs_correct_side", r.get("accuracy", np.nan)) for r in rows],
            w,
            label="vs correct side",
            color="#4c72b0",
        )
        ax.bar(
            x + w / 2,
            [r.get("acc_vs_mouse_choice", np.nan) for r in rows],
            w,
            label="vs mouse choice",
            color="#dd8452",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(frameon=False)
    else:
        ax.bar(names, [r.get("accuracy", np.nan) for r in rows], color="#4c72b0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{domain} accuracy — {regime}")
    path = out / f"{domain}_{regime}_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _history_gap(row: dict) -> float:
    g = (
        row.get("kyan_diagnostics", {})
        .get("counterfactual_zero_evidence_choice_probability", {})
        .get("history_gap")
    )
    return float(g) if g is not None else float("nan")


def _bar_history_gap(rows: list[dict], domain: str, regime: str, out: Path) -> Path | None:
    if not rows or regime == "fixed_prior":
        return None
    names = [r["model_id"] for r in rows]
    gaps = [_history_gap(r) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, gaps, color="#55a868")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel("history gap (high − low)")
    ax.set_title(f"{domain} zero-evidence history gap — {regime}")
    path = out / f"{domain}_{regime}_history_gap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _synth_vs_real_accuracy(cfg: dict, regime: str, out: Path) -> Path | None:
    models = list(cfg["models"])
    synth, real = [], []
    names = []
    for mid in models:
        sp = _metrics_path(cfg, "synth", regime, mid)
        rp = _metrics_path(cfg, "real", regime, mid)
        if not (sp.exists() and rp.exists()):
            continue
        s = json.loads(sp.read_text())
        r = json.loads(rp.read_text())
        names.append(mid)
        synth.append(s.get("accuracy", np.nan))
        real.append(r.get("acc_vs_correct_side", r.get("accuracy", np.nan)))
    if not names:
        return None
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, synth, w, label="synth held-out", color="#4c72b0")
    ax.bar(x + w / 2, real, w, label="real (vs correct)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Synth vs real accuracy — {regime}")
    ax.legend(frameon=False)
    path = out / f"synth_vs_real_{regime}_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _synth_vs_real_history_gap(cfg: dict, regime: str, out: Path) -> Path | None:
    if regime == "fixed_prior":
        return None
    models = list(cfg["models"])
    synth, real = [], []
    names = []
    for mid in models:
        sp = _metrics_path(cfg, "synth", regime, mid)
        rp = _metrics_path(cfg, "real", regime, mid)
        if not (sp.exists() and rp.exists()):
            continue
        names.append(mid)
        synth.append(_history_gap(json.loads(sp.read_text())))
        real.append(_history_gap(json.loads(rp.read_text())))
    if not names:
        return None
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, synth, w, label="synth held-out", color="#55a868")
    ax.bar(x + w / 2, real, w, label="real", color="#c44e52")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("history gap (high − low)")
    ax.set_title(f"Synth vs real history gap — {regime}")
    ax.legend(frameon=False)
    path = out / f"synth_vs_real_{regime}_history_gap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _comparison_switch_curves(
    cfg: dict, domain: str, regime: str, out: Path
) -> Path | None:
    if regime == "fixed_prior":
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "tanh_bptt": "#4c72b0",
        "tanh_pc": "#55a868",
        "gru": "#c44e52",
        "bayes": "#8172b3",
    }
    any_curve = False
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        curve = switch_centered_zero_evidence(roll, before=20, after=30)
        color = colors.get(mid, None)
        ax.plot(
            curve["offsets"],
            curve["low_to_high"],
            color=color,
            label=f"{mid} 0.2→0.8",
        )
        ax.plot(
            curve["offsets"],
            curve["high_to_low"],
            color=color,
            linestyle="--",
            label=f"{mid} 0.8→0.2",
        )
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set(
        title=f"Switch-centered zero-evidence — {domain} — {regime}",
        xlabel="Trials relative to switch",
        ylabel="P(right) zero evidence",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    path = out / f"{domain}_{regime}_switch_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def comparison_figures(cfg: dict, fig_root: Path) -> list[Path]:
    out = fig_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    regimes = list(cfg.get("eval", {}).get("regimes", REGIMES))

    for regime in regimes:
        for domain in DOMAINS:
            rows = _load_metric_rows(cfg, domain, regime)
            for fn in (_bar_accuracy, _bar_history_gap):
                p = fn(rows, domain, regime, out)
                if p:
                    paths.append(p)
            p = _comparison_switch_curves(cfg, domain, regime, out)
            if p:
                paths.append(p)

        for fn in (_synth_vs_real_accuracy, _synth_vs_real_history_gap):
            p = fn(cfg, regime, out)
            if p:
                paths.append(p)

    # Legacy real_transfer alias from history_only real metrics
    rows = _load_metric_rows(cfg, "real", "history_only")
    if rows:
        p = _bar_accuracy(rows, "real", "history_only", out)
        if p:
            legacy = out / "real_transfer_accuracy.png"
            shutil.copy(p, legacy)
            paths.append(legacy)
    return paths


def main() -> int:
    cfg = load_synthetic_config()
    fig_root = ROOT / cfg["paths"]["figures"]
    fig_root.mkdir(parents=True, exist_ok=True)
    regimes = list(cfg.get("eval", {}).get("regimes", REGIMES))
    made: list[str] = []
    for mid in cfg["models"]:
        for domain in DOMAINS:
            for regime in regimes:
                p = per_model_domain_regime_figure(mid, domain, regime, cfg, fig_root)
                if p:
                    made.append(str(p.relative_to(ROOT)))
    for p in comparison_figures(cfg, fig_root):
        made.append(str(p.relative_to(ROOT)))
    catalog = {
        "figures": made,
        "regenerate": (
            "python scripts/11_eval_regimes.py && python scripts/10_make_figures.py"
        ),
        "notes": {
            "by_model/{model}/{synth|real}/{regime}/multipanel_diagnostics.png": (
                "2×2 Kyan diagnostics per model × domain × regime. "
                "Real psychometrics include dashed mouse curves."
            ),
            "comparison/{synth|real}_{regime}_accuracy.png": "Per-domain accuracy bars.",
            "comparison/{synth|real}_{regime}_history_gap.png": (
                "Zero-evidence history gap (N/A for fixed_prior)."
            ),
            "comparison/{synth|real}_{regime}_switch_curves.png": (
                "All-model switch-centered overlay."
            ),
            "comparison/synth_vs_real_{regime}_*.png": (
                "Side-by-side synth held-out vs real for the same regime."
            ),
        },
    }
    (fig_root / "figure_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
