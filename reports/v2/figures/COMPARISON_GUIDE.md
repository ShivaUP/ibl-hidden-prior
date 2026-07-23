# How to read v2 figures

## Scorecards (separate folder)
- Path: `scorecards/{domain}_{regime}_scorecard.png`
- Full guide: `scorecards/SCORECARD_GUIDE.md`
- Start here for model ranking numbers.

## Multipanels (`by_model/...`)
- **Synth:** psychometric + switch = averages over held-out synthetic sessions.
- **Real:** one color per of the 10 core sessions; bottom-right = best session by accuracy.

## Switch boards (`comparison/*_switch_board.png`)
- Left: preference after 0.2→0.8 switches.
- Right: preference after 0.8→0.2 switches.

## Synth vs real boards (`comparison/synth_vs_real_*`)
- Same metric and regime: synth held-out vs real transfer.

All behavioral scores use **correct stimulus side**, not mouse choice.
