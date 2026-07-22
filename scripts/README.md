# Scripts (v2 pipeline)

All user-facing entrypoints are numbered. Run in order unless a later step’s inputs already exist.

| # | Script | Purpose |
|---|---|---|
| 00 | `00_smoke_one_connection.py` | ONE/openalyx connectivity smoke test |
| 01 | `01_run_session_qc.py` | QC candidates; pin `behavior_core_eids.json` |
| 02 | `02_audit_event_deltas.py` | Event-time deltas → phase-tick medians |
| 03 | `03_build_processed_trials.py` | Behavior-core → `data/processed/trials/` |
| 04 | `04_fit_synthetic_stats.py` | Empirical synth stats + `configs/synthetic_v2.yaml` |
| 05 | `05_build_synthetic_datasets.py` | Sample synth train pool + held-out |
| 06 | `06_map_real_to_v2_ticks.py` | Real trials → shared tick tensors |
| 07 | `07_train_model.py` | Train models (`--all`). BPTT/GRU/Bayes: 60×24×929; **PC: 60×24×240** (stable prior learning) |
| 08 | `08_eval_synth_heldout.py` | Legacy single held-out eval (prefer `11`) |
| 09 | `09_eval_real_transfer.py` | Legacy real transfer (prefer `11 --domain real`) |
| 10 | `10_make_figures.py` | Multipanels + comparisons for synth×real×regime (after `11`) |
| 11 | `11_eval_regimes.py` | Synth + real regimes: `history_only` / `full_information` / `fixed_prior` |

## Typical full run

```bash
source .venv/bin/activate

python scripts/00_smoke_one_connection.py
python scripts/01_run_session_qc.py
python scripts/02_audit_event_deltas.py
python scripts/03_build_processed_trials.py
python scripts/04_fit_synthetic_stats.py
python scripts/05_build_synthetic_datasets.py
python scripts/06_map_real_to_v2_ticks.py
python scripts/07_train_model.py --all
python scripts/11_eval_regimes.py          # synth + real × all regimes
python scripts/10_make_figures.py
```

Figure layout after `10`:
- `reports/v2/figures/by_model/{model}/{synth|real}/{regime}/multipanel_diagnostics.png`
- `reports/v2/figures/comparison/{synth|real}_{regime}_{accuracy|history_gap|switch_curves}.png`
- `reports/v2/figures/comparison/synth_vs_real_{regime}_{accuracy|history_gap}.png`

If manifests / `synthetic_v2.yaml` / trials already exist locally, you can start at **04** or **05**.

## Removed (v1-only)

These scripts were deleted when v2 became the active path:

- `train_model.py`, `train_phase5.py`
- `eval_phase6.py` … `eval_phase9_matched.py`
- `make_phase10_figures.py`
- `build_neural_intersect.py`
- `inspect_ibl_trial_fields.py`
- `build_processed_datasets.py` (replaced by `03_build_processed_trials.py`; no longer builds RNN bins / Bayes tables)

v1 reports under `reports/` remain as historical artifacts.
