# Handoff — Phases 7–10 complete (v1 report shipped)

**Date:** 2026-07-21

## Status

v1 pipeline through Phase 10 is in place. Primary write-up: `reports/v1_report.md`.

## Headline results (post choice-encoding fix)

| Question | Answer |
|---|---|
| Best choice model | **standard** (held-out NLL 0.169) |
| Best mouse-prior match | **bayes** (corr 0.80) |
| Neural after behavior match | **Only standard matched** (ε=0.05); its VE>0 survives Holm in MOs & vlOFC on pilot eid. Unmatched bayes/pc look stronger neurally but are excluded. |

## Phase 9

- Matched set: `reports/neural/behavior_matched_models.json` → `["standard"]`
- Confirmatory VE: `ve_matched.csv` (standard only)
- Exploratory: `ve_unmatched.csv` (all three)
- Survival: `survival_tests.json` (VE>0 for standard; Holm across regions)

## Phase 10

- Figures: `reports/figures/phase10/*.png`
- Report: `reports/v1_report.md`
- Index: `reports/v1_artifact_index.md`

## Remaining optional upgrades (not blocking v1)

1. Process full neural_behavior_pool (25 eids) + retrain on pool splits  
2. Session-level permutation survival  
3. Tighten ε or report sensitivity at ε=0.07 (would include pc)

```bash
cd /Users/shivamahdian/Desktop/Performance && source .venv/bin/activate
python -m pytest tests/ -q
python scripts/eval_phase9_matched.py
python scripts/make_phase10_figures.py
```
