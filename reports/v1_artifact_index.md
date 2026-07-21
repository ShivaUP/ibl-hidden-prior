# v1 artifact index

Created: 2026-07-21T23:15:08.393073+00:00

## Checksums (sha256 prefix)

| path | sha256_16 | bytes |
|---|---|---:|
| `configs/frozen_v1.yaml` | c71c189f3503f8e1 | 4289 |
| `docs/spec.md` | a65bef1980d17221 | 15381 |
| `docs/plan.md` | cb69b4ec4d96fbcd | 19529 |
| `data/manifests/behavior_core_eids.json` | 0341fb7578aa2f7c | 4888 |
| `data/manifests/neural_intersect_eids.json` | 44ebc8598b3d61fd | 4904 |
| `data/manifests/splits.json` | 551e81a46ef87da7 | 682 |
| `reports/behavior/heldout_metrics.csv` | 6217ae4dcbd26722 | 2509 |
| `reports/behavior/prior_match.csv` | 2f17594963b56383 | 378 |
| `reports/behavior/phase7_prior_summary.json` | b1ec9a03e4ad06aa | 4803 |
| `reports/neural/behavior_matched_models.json` | dd6d07157da62d2f | 1565 |
| `reports/neural/ve_matched.csv` | cf53115a75dc2b78 | 378 |
| `reports/neural/ve_unmatched.csv` | 55cc8b8e3ed8db66 | 981 |
| `reports/neural/survival_tests.json` | 5a6a0d73312d9a25 | 2729 |
| `reports/neural/phase8_pilot.json` | 541dd884bdc07ade | 3338 |
| `reports/neural/phase9_summary.json` | 91fb6d50143d21b9 | 1745 |
| `reports/v1_report.md` | 4d6e6bcf3aeec49a | 4121 |
| `reports/figures/phase10/heldout_choice_nll.png` | 57454773fc0638d3 | 15592 |
| `reports/figures/phase10/neural_ve_unmatched_vs_matched.png` | 58f11240bebea02a | 28383 |
| `reports/figures/phase10/prior_match.png` | bf22d8995314c3e7 | 16568 |
| `reports/figures/phase10/psychometrics.png` | 9b0f21bea4e21e00 | 44014 |
| `reports/figures/phase10/survival_tests.png` | b55304de694f600d | 18371 |
| `reports/figures/phase10/switch_centered.png` | 1394992734dd72b6 | 78552 |

## Reproduce

```bash
python -m pytest tests/ -q
python scripts/eval_phase6.py
python scripts/eval_phase7_priors.py
python scripts/eval_phase8_neural_pilot.py
python scripts/eval_phase9_matched.py
python scripts/make_phase10_figures.py
```

## Risks R1–R8

- **R1:** Frozen: GRU standard; PC tanh residual form in YAML
- **R2:** Frozen: leaky online prior with feature evidence nudge
- **R3:** Frozen: choice_epsilon=0.05, rt_nll_floor=2.03
- **R4:** Frozen after pilot: peri_stim [-0.1, 0.3) stimOn
- **R5:** Partially frozen via event_delta_audit in YAML
- **R6:** Mitigated by almost-perfect session gate
- **R7:** Frozen: alpha-grid + logistic mouse prior in src/eval/mouse_prior.py
- **R8:** Behavior-core n=10; neural pool n=25 pass QC
