# Session handoff — Phase 7 + choice-encoding fix

**Date:** 2026-07-21  
**Status:** Phase 7 complete after critical encoding fix; Phases 8–10 remain.

## Critical fix (blocks all prior conclusions)

IBL ALF `choice` is **spatial**: `-1 = right`, `+1 = left` (verified: agreement with `feedbackType` × stim side = 1.0).

Previously the pipeline used `choice_right = (choice == 1)`, which **inverted** targets and `prev_choice_right`. Psychometrics and “good” choice NLL were fitting the flipped label.

**Fixed in:**
- `src/data/processed_trials.py`
- `src/data/features.py` (`prev_choice_right`)
- `src/data/inspect_trials.py` docs
- `docs/spec.md` §7.1

Then: rebuilt processed data → retrained all models → re-ran Phase 6 + 7.

## Phase 7 deliverables

| Artifact | Path |
|---|---|
| Mouse prior module | `src/eval/mouse_prior.py` |
| Prior-match metrics | `src/eval/prior_match.py` |
| Script | `scripts/eval_phase7_priors.py` |
| Mouse \(\hat p_t\) | `data/processed/mouse_prior/` |
| Model \(q_t\) | `data/processed/model_priors/` |
| Match table | `reports/behavior/prior_match.csv` |
| Summary | `reports/behavior/phase7_prior_summary.json` |
| Figures | `reports/figures/phase7/` |

### Mouse prior (train-fit)

- Leaky update of P(right) from experienced stim sides; \(\alpha \approx 0.20\)
- Logistic: \(\beta_{\text{contrast}} \approx +2.14\), \(\beta_{\text{prior}} \approx +1.49\) (signs now correct)
- Corr(\(\hat p_t\), oracle P(right)=1−probabilityLeft) ≈ **0.80**; not identical (MAE ≈ 0.14)

### Held-out history-only prior match (val+test)

| Model | corr(mouse, q) | RMSE | switch MSE |
|---|---:|---:|---:|
| **bayes** | **0.799** | **0.235** | **0.054** |
| pc | 0.209 | 0.293 | 0.085 |
| standard | −0.073 | 0.307 | 0.094 |

### Behavior (held-out history-only, post-fix)

| Model | choice NLL | acc |
|---|---:|---:|
| **standard** | **0.169** | **0.928** |
| pc | 0.238 | 0.894 |
| bayes | 0.548 | 0.734 |

**Interpretation seed:** standard still wins choice prediction; explicit Bayes wins mouse-prior match. Standard’s \(q_t\) readout is weakly/negatively aligned with the behavior-derived prior despite excellent choice fit.

## Stops still in force

- Do not treat `probabilityLeft` as mouse prior in history-only.
- Neural claims require Phases 8–9 and behavior matching.
- Do not train on test eids for selection.

## Next: Phase 8

Neural-intersect (MOs / vlOFC), prior-readout VE using these \(q_t\) definitions only.

```bash
cd /Users/shivamahdian/Desktop/Performance && source .venv/bin/activate
python -m pytest tests/ -q
python scripts/eval_phase7_priors.py
```
