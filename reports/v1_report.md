# v1 report — latent prior updating in the IBL biased-block task

**Config:** `configs/frozen_v1.yaml` (`frozen_v1`)  
**Models:** standard RNN · predictive-coding RNN · Bayesian online inference  
**Primary condition:** history-only  

---

## 1. Answers to the core questions

### Which model best explains trial-by-trial choice?

**Standard RNN.** Held-out history-only choice NLL: standard **0.169** ≫ pc 0.238 ≫ bayes 0.548 (acc 0.93 / 0.89 / 0.73).

Figure: `reports/figures/phase10/heldout_choice_nll.png`

### Which model best matches the mouse subjective prior?

**Bayesian model.** Corr(mouse \(\hat p_t\), model \(q_t\)): bayes **0.80** ≫ pc 0.21 ≫ standard −0.07.

Mouse prior is behavior-derived (leaky stim-side update, α≈0.20), correlated with oracle P(right)≈0.80 but **not** equal to `probabilityLeft`.

Figure: `reports/figures/phase10/prior_match.png`

### Does any neural advantage survive behavior matching?

**Only the standard RNN is behavior-matched** under the frozen ε=0.05 choice ball on held-out history-only metrics. PC (ΔNLL≈0.069) and Bayes (ΔNLL≈0.378) are excluded from confirmatory neural claims despite Bayes having the highest unmatched vlOFC VE.

For the single matched model (standard), trial-bootstrap VE vs 0 survives Holm correction in both MOs and vlOFC on the pilot neural eid (exploratory multi-model ranking is unmatched-only).

Figures: `reports/figures/phase10/neural_ve_unmatched_vs_matched.png`, `survival_tests.png`

---

## 2. Critical encoding note

IBL ALF `choice`: **−1 = right**, **+1 = left**. An early inverted `choice_right` encoding was fixed; all numbers in this report are **post-fix**.

---

## 3. Behavior (history-only)

| Model | Choice NLL | Acc | Prior corr | Switch prior MSE |
|---|---:|---:|---:|---:|
| standard | 0.169 | 0.928 | −0.073 | 0.094 |
| pc | 0.238 | 0.894 | 0.209 | 0.085 |
| bayes | 0.548 | 0.734 | 0.799 | 0.054 |

Psychometrics / switch: `reports/figures/phase10/psychometrics.png`, `switch_centered.png`

---

## 4. Neural (Phase 8–9)

### Cohort

- Strict `behavior-core ∩ (MOs|ORBvl)`: **empty** (behavior-first core had no ephys).
- `neural_behavior_pool`: 25 BWM ROI sessions pass almost-perfect QC.
- Pilot eid (MOs+ORBvl): `1191f865-b10a-45c8-9c48-24a980fd9402`
- Peri-stim window: **[−0.1, 0.3) s** from `stimOn`

### Neural prior readout (CV Ridge → mouse prior)

| Region | Units | Readout VE |
|---|---:|---:|
| MOs | 48 | 0.11 |
| vlOFC | 572 | 0.38 |

### Unmatched vs matched VE (linear recal of \(q_t\) → neural prior)

| Region | standard | pc | bayes | Matched? |
|---|---:|---:|---:|---|
| MOs | 0.10 | **0.27** | 0.10 | standard only |
| vlOFC | 0.21 | 0.32 | **0.39** | standard only |

### Behavior matching (held-out)

- Best choice NLL: standard 0.169  
- ε-ball: ΔNLL ≤ 0.05 → **standard only**  
- RT floor (`rt_nll` ≤ 2.03): all three pass; choice is binding  

### Survival (matched standard VE > 0; Holm across regions)

| Region | VE | 95% CI | Holm p | Survive α=0.05 |
|---|---:|---|---:|---|
| MOs | 0.104 | [0.069, 0.144] | ~0 | yes |
| vlOFC | 0.212 | [0.174, 0.252] | ~0 | yes |

Artifacts: `reports/neural/behavior_matched_models.json`, `ve_matched.csv`, `ve_unmatched.csv`, `survival_tests.json`

---

## 5. Limitations / open risks

1. Neural results are **one pilot session**, OOD (models trained on behavior-core).
2. Session-level permutation not yet possible; trial bootstrap only.
3. Expanding to full `neural_behavior_pool` + retraining would strengthen confirmatory claims.
4. Spec risks R1–R8: GRU/PC/Bayes forms frozen in YAML as used; ε and RT floor frozen from Phase 5/6; peri-stim window frozen after pilot load.

---

## 6. Reproduce

```bash
cd /Users/shivamahdian/Desktop/Performance && source .venv/bin/activate
python -m pytest tests/ -q
python scripts/eval_phase6.py
python scripts/eval_phase7_priors.py
python scripts/eval_phase8_neural_pilot.py
python scripts/eval_phase9_matched.py
python scripts/make_phase10_figures.py
```

Manifests: `data/manifests/behavior_core_eids.json`, `neural_intersect_eids.json`, `splits.json`
