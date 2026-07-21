# Handoff — Phase 7 complete + Phase 8.1–8.4 pilot

**Date:** 2026-07-21

## Critical encoding fix (invalidates pre-fix numbers)

IBL `choice`: **−1 = right**, **+1 = left**. Fixed `choice_right` / `prev_choice_right`, rebuilt data, retrained, re-ran Phase 6–7.

## Phase 7 (done)

- Mouse prior: leaky P(right), α≈0.20; corr vs oracle P(right)≈0.80
- Held-out prior match: **bayes 0.80** ≫ pc 0.21 ≫ standard −0.07
- Held-out choice: **standard 0.17 NLL** ≫ pc 0.24 ≫ bayes 0.55

## Phase 8 status

### P8.1 Neural intersect
- Strict `behavior-core ∩ (MOs|ORBvl BWM)` = **empty** (documented blocker)
- Expanded **neural_behavior_pool**: 25/30 QC’d ROI BWM eids pass almost-perfect gates
- Manifest: `data/manifests/neural_intersect_eids.json`

### P8.2–P8.4 Pilot (one eid with MOs+ORBvl)
- Eid: `1191f865-b10a-45c8-9c48-24a980fd9402`
- Window frozen: stimOn **[−0.1, 0.3) s** in `configs/frozen_v1.yaml`
- Neural prior readout (CV Ridge → mouse \(\hat p\)):
  - MOs: 48 units, VE≈0.11
  - vlOFC: 572 units, VE≈0.38
- Unmatched model VE (OOD behavior-core checkpoints; primary = `ve_linear_recal`):

| Region | standard | pc | bayes |
|---|---:|---:|---:|
| MOs | 0.10 | **0.27** | 0.10 |
| vlOFC | 0.21 | 0.32 | **0.39** |

Reports: `reports/neural/phase8_pilot.json`, `ve_unmatched_pilot.csv`

**STOP:** No confirmatory neural advantage claims until Phase 9 matching. Prefer retraining on neural-behavior pool before strong claims.

## Next (Phase 9 / expand 8)
1. Process full neural_behavior_pool; retrain or fine-tune on pool splits
2. Behavior-matched VE + survival tests
3. Phase 10 report

```bash
cd /Users/shivamahdian/Desktop/Performance && source .venv/bin/activate
python scripts/eval_phase7_priors.py
python scripts/build_neural_intersect.py --max-qc 30
python scripts/eval_phase8_neural_pilot.py
```
