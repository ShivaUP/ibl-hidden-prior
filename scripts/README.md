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
| 07 | `07_train_model.py` | Train `tanh_bptt` / `tanh_pc` / `gru` / `bayes` (`--all`). Full empirical-length sessions; truncated BPTT chunks. Default **60×24×929 ≈ 1.34M** trial-exposures (~3.9× Kyan). |


| 08 | `08_eval_synth_heldout.py` | **Primary** closed-loop synth ranking |
| 09 | `09_eval_real_transfer.py` | **Secondary** frozen real-behavior transfer |
| 10 | `10_make_figures.py` | Per-model multipanel + comparison figures |

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
python scripts/08_eval_synth_heldout.py
python scripts/09_eval_real_transfer.py
python scripts/10_make_figures.py
```

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
