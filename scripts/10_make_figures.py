#!/usr/bin/env python3
"""10 — Make v2 figures (per-model multipanel + comparison).

Usage:
  python scripts/10_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic.schema import load_synthetic_config


def _load_metrics(reports: Path, stage: str, model_id: str) -> dict | None:
    path = reports / "metrics" / f"{stage}_{model_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def per_model_figure(model_id: str, cfg: dict, fig_root: Path) -> Path | None:
    reports = ROOT / cfg["paths"]["reports"]
    hist_path = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "train_history.json"
    roll_path = (
        ROOT / cfg["paths"]["artifacts"] / "synthetic" / "heldout" / model_id / "rollout.npz"
    )
    if not hist_path.exists() or not roll_path.exists():
        print(f"SKIP figure {model_id}: missing history or rollout", file=sys.stderr)
        return None

    hist = json.loads(hist_path.read_text())["history"]
    roll = np.load(roll_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # 1 training curve
    ax = axes[0, 0]
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color="#1f4e79")
    ax.set_title("Training loss / PC energy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    # 2 psychometric-ish: P(right) vs signed contrast by block
    ax = axes[0, 1]
    signed = np.where(roll["side"] == 1, roll["contrast"], -roll["contrast"])
    for prior, color in ((0.2, "#c44e52"), (0.5, "#4c72b0"), (0.8, "#55a868")):
        m = np.isclose(roll["probability_left"], prior)
        if not m.any():
            continue
        # bin by signed contrast
        levels = sorted(set(np.round(signed[m].ravel(), 5)))
        xs, ys = [], []
        for lv in levels:
            mm = m & np.isclose(signed, lv)
            if mm.any():
                xs.append(lv)
                ys.append(float(np.mean(roll["p_right"][mm])))
        ax.plot(xs, ys, "o-", label=f"pL={prior}", color=color)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title("Psychometric by block prior")
    ax.set_xlabel("signed contrast")
    ax.set_ylabel("P(right)")
    ax.legend(fontsize=8)

    # 3 switch-centered zero-evidence belief
    ax = axes[1, 0]
    pleft = roll["probability_left"]
    belief = roll["belief"]
    switches = []
    for s in range(pleft.shape[0]):
        for t in range(1, pleft.shape[1]):
            if not np.isclose(pleft[s, t], pleft[s, t - 1]):
                switches.append((s, t))
    window = np.arange(-10, 21)
    curves = []
    for s, t0 in switches:
        for d in window:
            tt = t0 + int(d)
            if 0 <= tt < belief.shape[1] and np.isclose(roll["contrast"][s, tt], 0.0):
                curves.append((int(d), float(belief[s, tt])))
    if curves:
        by_d: dict[int, list[float]] = {}
        for d, v in curves:
            by_d.setdefault(d, []).append(v)
        xs = sorted(by_d)
        ys = [float(np.mean(by_d[d])) for d in xs]
        ax.plot(xs, ys, color="#1f4e79")
    ax.axvline(0, color="gray", ls="--")
    ax.set_title("Switch-centered zero-contrast belief")
    ax.set_xlabel("trials from switch")
    ax.set_ylabel("P(right)")

    # 4 example session prior vs belief
    ax = axes[1, 1]
    s0 = 0
    ax.plot(1.0 - pleft[s0], label="true P(right)", color="#c44e52", alpha=0.8)
    ax.plot(belief[s0], label="model belief", color="#1f4e79", alpha=0.8)
    ax.set_title("Example session prior vs belief")
    ax.set_xlabel("trial")
    ax.set_ylabel("P(right)")
    ax.legend(fontsize=8)

    fig.suptitle(f"v2 diagnostics — {model_id}", fontsize=12)
    fig.tight_layout()
    out = fig_root / "by_model" / model_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multipanel_diagnostics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def comparison_figures(cfg: dict, fig_root: Path) -> list[Path]:
    reports = ROOT / cfg["paths"]["reports"]
    models = list(cfg["models"])
    paths = []

    # synth ranking
    rows = []
    for mid in models:
        m = _load_metrics(reports, "synth_heldout", mid)
        if m:
            rows.append(m)
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r["model_id"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        ax.bar(names, acc, color="#4c72b0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.set_title("Held-out synth accuracy ranking")
        out = fig_root / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        p = out / "synth_heldout_accuracy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # real transfer side-by-side
    rows = []
    for mid in models:
        m = _load_metrics(reports, "real_transfer", mid)
        if m:
            rows.append(m)
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(rows))
        w = 0.35
        ax.bar(x - w / 2, [r["acc_vs_correct_side"] for r in rows], w, label="vs correct side")
        ax.bar(x + w / 2, [r["acc_vs_mouse_choice"] for r in rows], w, label="vs mouse choice")
        ax.set_xticks(x)
        ax.set_xticklabels([r["model_id"] for r in rows])
        ax.set_ylim(0, 1)
        ax.set_title("Real transfer accuracy")
        ax.legend()
        out = fig_root / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        p = out / "real_transfer_accuracy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    cfg = load_synthetic_config()
    fig_root = ROOT / cfg["paths"]["figures"]
    fig_root.mkdir(parents=True, exist_ok=True)
    made = []
    for mid in cfg["models"]:
        p = per_model_figure(mid, cfg, fig_root)
        if p:
            made.append(str(p.relative_to(ROOT)))
    for p in comparison_figures(cfg, fig_root):
        made.append(str(p.relative_to(ROOT)))
    catalog = {
        "figures": made,
        "regenerate": "python scripts/10_make_figures.py",
        "notes": {
            "by_model/*/multipanel_diagnostics.png": (
                "Panels: train curve, psychometric-by-block, switch zero-contrast belief, "
                "example session prior vs belief (Kyan-style)."
            ),
            "comparison/synth_heldout_accuracy.png": "Primary synth ranking.",
            "comparison/real_transfer_accuracy.png": (
                "Secondary transfer: correct-side vs mouse-choice."
            ),
        },
    }
    (fig_root / "figure_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
