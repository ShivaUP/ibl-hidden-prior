# Spec: overall vs switch-window correctness boards

Frozen 2026-07-23 (grill).

## Objective

For every domain × regime, compare session-level **overall** correctness to peri-switch correctness, and test whether they differ.

## Bars (per model)

Order always: `tanh_bptt → tanh_pc → gru → gru_pc`.

| Regime | Bars |
|---|---|
| `history_only`, `full_information` | overall · 0.2→0.8 (−30…+30) · 0.8→0.2 (−30…+30) |
| `fixed_prior` | overall only |

Switch windows: eligible genuine 0.2↔0.8 switches only; per session = mean correctness over all trials in the window(s) of that direction.

## Statistics

- Unit: session.
- Contrasts: overall vs 0.2→0.8; overall vs 0.8→0.2 (paired Wilcoxon signed-rank, two-sided).
- Multiple testing: Holm within each panel (≤8 tests).
- Display: 95% CI error bars; significance brackets/stars for Holm-adjusted p < 0.05 / 0.01 / 0.001.

## Colors

Global twin-complement palette in `src/plot/v2_style.py`:
- tanh BPTT ↔ tanh PC (blue ↔ amber)
- GRU ↔ GRU PC (rose ↔ teal)

Within-model bar shades: darker = overall, mid = 0.2→0.8, lighter = 0.8→0.2.

## Deliverables

- `comparison/{synth\|real}_{regime}_overall_vs_switch_correctness.png` (6 boards)
- Metrics JSON under `reports/v2/metrics/overall_vs_switch_correctness.json`
- All figure saves via `save_figure` / `SAVE_DPI=600`
