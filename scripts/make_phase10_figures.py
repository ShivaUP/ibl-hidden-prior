#!/usr/bin/env python3
"""Phase 10: regenerate required figure panels from report tables.

Usage:
    python scripts/make_phase10_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot.phase10_figures import (
    fig_heldout_choice,
    fig_neural_ve,
    fig_prior_match,
    fig_psychometrics,
    fig_survival,
    fig_switch,
)


def main() -> int:
    beh = ROOT / "reports" / "behavior"
    neu = ROOT / "reports" / "neural"
    out = ROOT / "reports" / "figures" / "phase10"
    out.mkdir(parents=True, exist_ok=True)

    fig_heldout_choice(beh / "heldout_metrics.csv", out / "heldout_choice_nll.png")
    if (beh / "psychometrics.csv").exists():
        fig_psychometrics(beh / "psychometrics.csv", out / "psychometrics.png")
    fig_prior_match(beh / "prior_match.csv", out / "prior_match.png")
    fig_switch(beh, out / "switch_centered.png")
    fig_neural_ve(neu / "ve_unmatched.csv", neu / "ve_matched.csv", out / "neural_ve_unmatched_vs_matched.png")
    fig_survival(neu / "survival_tests.csv", out / "survival_tests.png")

    print(f"Wrote figures under {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
