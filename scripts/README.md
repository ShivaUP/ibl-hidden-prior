# Scripts (v2 pipeline)

All user-facing entrypoints are numbered. Run in order unless a later step’s inputs already exist.

| # | Script | Purpose |
|---|---|---|
| 00 | `00_smoke_one_connection.py` | ONE/openalyx connectivity smoke test |
| 01 | `01_run_session_qc.py` | QC candidates; pin `behavior_core_eids.json` |
| 02 | `02_audit_event_deltas.py` | Event-time deltas → phase-tick medians |
| 03 | `03_build_processed_trials.py` | Shared behavior+neural cohort → trials (default) |
| 04 | `04_fit_synthetic_stats.py` | Empirical synth stats + `configs/synthetic_v2.yaml` |
| 05 | `05_build_synthetic_datasets.py` | Sample synth train pool + held-out |
| 06 | `06_map_real_to_v2_ticks.py` | Shared cohort → tick tensors |
| 07 | `07_train_model.py` | Train models (`--all`). Active: tanh_bptt, tanh_pc, gru, gru_pc. Shared schedule 60×24×929; corrected PC (32 inference rounds) |
| 08 | `08_eval_synth_heldout.py` | Legacy single held-out eval (prefer `11`) |
| 09 | `09_eval_real_transfer.py` | Legacy real transfer (prefer `11 --domain real`) |
| 10 | `10_make_figures.py` | Multipanels + readable model scorecards / switch boards (after `11`) |
| 11 | `11_eval_regimes.py` | Synth + real regimes: `history_only` / `full_information` / `fixed_prior` |
| 12 | `12_build_neural_intersect.py` | Rebuild expanded-ROI BWM∩behavior pool + shared cohort |
| 13 | `13_eval_neural_pilot.py` | Full neural VE on shared cohort (all ROIs present per session) |
| 14 | `14_eval_neural_matched.py` | Behavior-matched VE + session-bootstrap survival (Holm) |
| 15 | `15_make_neural_figures.py` | Neural VE / survival figures |
| 16 | `16_plot_mlp_switch_block_decoding.py` | Two-panel MLP switch block decoding (4 models + 4 ROIs) |

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
python scripts/12_build_neural_intersect.py
python scripts/13_eval_neural_pilot.py
python scripts/14_eval_neural_matched.py
python scripts/15_make_neural_figures.py
python scripts/16_plot_mlp_switch_block_decoding.py
```

Figure layout after `10`:
- `reports/v2/figures/scorecards/{synth|real}_{regime}_scorecard.png` ← ranking numbers + precise reading text
- `reports/v2/figures/scorecards/SCORECARD_GUIDE.md`
- `reports/v2/figures/by_model/{model}/{synth|real}/{regime}/multipanel_diagnostics.png`
- `reports/v2/figures/comparison/{synth|real}_{regime}_switch_board.png`
- `reports/v2/figures/comparison/synth_vs_real_{regime}_board.png`
- Guide: `reports/v2/figures/COMPARISON_GUIDE.md`

Real eval scope: `docs/REAL_EVAL.md`. Neural: `reports/v2/neural/` + `reports/v2/figures/neural/`.

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
