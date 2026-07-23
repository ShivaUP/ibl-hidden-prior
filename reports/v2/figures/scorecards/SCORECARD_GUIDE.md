# How to read scorecards

Location: `reports/v2/figures/scorecards/{synth|real}_{regime}_scorecard.png`

Each figure has **reading text on top** and two bar panels below. Panel titles are neutral (`Accuracy`, `History gap`); interpretation is only in the text block.

## Header fields

- **synth / real** — which evaluation domain.
- **regime** — `history_only`, `full_information`, or `fixed_prior` (defined in the text).

## Left panel: Accuracy

- Definition: fraction of trials where the model’s discrete choice equals the **correct stimulus side**.
- Not scored against mouse choice.
- Black outline: model with the highest accuracy on this plot.

## Right panel: History gap

- Definition: mean zero-evidence P(choice=right) in true 0.8 blocks minus that in true 0.2 blocks.
- Zero-evidence: counterfactual probe with sensory contrast held at 0.
- Near 0: little differential prior use across blocks.
- Large positive: model more often prefers right in right-biased blocks than in left-biased blocks.
- Undefined on `fixed_prior` (no 0.2/0.8 blocks).
- Black outline: largest |gap| among models.

## Related figures (not scorecards)

- Switch timing: `comparison/*_switch_board.png`
- Synth vs real transfer: `comparison/synth_vs_real_*_board.png`
- Per-model diagnostics: `by_model/.../multipanel_diagnostics.png`
