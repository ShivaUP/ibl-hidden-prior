# How to read v2 figures

## Scorecards (separate folder)
- Path: `scorecards/{domain}_{regime}_scorecard.png`
- Full guide: `scorecards/SCORECARD_GUIDE.md`
- Real domain = shared behavior+neural cohort (same sessions as neural VE).
- Start here for model ranking numbers.

## Multipanels (`by_model/...`)
- **Synth:** psychometric + switch = averages over held-out synthetic sessions.
- **Real:** one color per shared-cohort session; bottom-right = best session by accuracy.

## Switch boards (`comparison/*_switch_board.png`)
- Left: preference after 0.2→0.8 switches.
- Right: preference after 0.8→0.2 switches.

## Overall vs peri-switch (`comparison/*_overall_vs_switch_correctness.png`)
- Three bars per model (order: tanh BPTT → tanh PC → GRU → GRU PC): overall, 0.2→0.8 (−30…+30), 0.8→0.2 (−30…+30).
- Stars: paired Wilcoxon overall vs each switch window, Holm-adjusted within the panel.
- `fixed_prior`: overall only.

## Synth vs real boards (`comparison/synth_vs_real_*`)
- Same metric and regime: synth held-out vs real transfer.

All behavioral scores use **correct stimulus side**, not mouse choice.
All figures export at **DPI 600** via `save_figure`.
