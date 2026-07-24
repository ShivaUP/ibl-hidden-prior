# Real behavioral + neural evaluation (v2)

## Shared cohort (locked)

**Behavior transfer and neural VE use the same sessions.**

- Manifest: `data/manifests/shared_behavior_neural_eids.json` (n=8)
- Primary ROIs: MOs, ORBvl (vlOFC), ACAd, MOp — see [`docs/NEURAL_REGIONS.md`](NEURAL_REGIONS.md)
- Selection: greedy ROI set-cover over the four primary regions
- Coverage report: `reports/v2/neural/neural_intersect_summary.json`
- Legacy behavior-only list (no ephys): `data/manifests/behavior_core_eids.json` — **not** used for v2 real+neural claims

Rebuild:
```bash
python scripts/12_build_neural_intersect.py --max-sessions 20
python scripts/03_build_processed_trials.py
python scripts/06_map_real_to_v2_ticks.py
python scripts/11_eval_regimes.py --domain real
python scripts/13_eval_neural_pilot.py          # all ROIs present in each session
python scripts/14_eval_neural_matched.py
python scripts/15_make_neural_figures.py
python scripts/10_make_figures.py
```

## Scoring (behavior)

Train and test only on **correct stimulus side** (not mouse choice).  
On real rollouts, mouse action/reward are history **inputs** only.

## Neural comparison

Regions (Allen): MOs, ORBvl, ACAd, MOp (primary analysis scope). Active models: tanh_bptt, tanh_pc, gru, gru_pc.

1. Spike counts (peri-stim) → CV Ridge → mouse \(\hat p_t\) = neural prior readout \(n_t\)
2. Model belief \(q_t\) on the **same** trials
3. Primary: `ve_linear_recal` of \(n_t\) by \(q_t\), then **mean across sessions that have that region**
4. Behavior match: CE ε-ball on this cohort’s history_only metrics
5. Survival: session bootstrap of matched VE advantage + Holm across regions

## Why not the old behavior-core 10?

Those sessions have **no Neuropixels insertions**, so neural VE is impossible.

## Paper links

- Prior map: https://www.nature.com/articles/s41586-025-09226-1
- BWM activity map: https://www.nature.com/articles/s41586-025-09235-0
- Standardized behavior: https://elifesciences.org/articles/63711
