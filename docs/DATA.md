# Data & artifacts (not in git)

This repository **does not** ship raw IBL downloads, processed tensors, or trained checkpoints. Those are regenerable and too large for GitHub.

## What *is* tracked

| Path | Why |
|---|---|
| `data/manifests/` | Session eid lists, QC scores, synth stats |
| `configs/` | `synthetic_v2.yaml`, frozen encodings |
| `reports/` | v1 write-up (historical); v2 metrics/figures after runs |
| `src/`, `scripts/`, `tests/` | Code |

## What is ignored

| Path | Approx size locally | Rebuild |
|---|---|---|
| `data/raw/` | ~GBs (ONE cache) | automatic on first download |
| `data/processed/` | tens of MB+ | `03_build_processed_trials.py`, `05_…`, `06_…` |
| `artifacts/` | checkpoints | `07_train_model.py` |
| `.venv/` | ~GBs | recreate from `requirements.txt` |

## Rebuild (v2)

See [`scripts/README.md`](../scripts/README.md) for the full numbered catalog.

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

Manifests: `behavior_core_eids.json`, `synthetic_stats_v2.json`, `synthetic_datasets_v2.json`, `real_v2_ticks.json`.

Public ONE endpoint: `https://openalyx.internationalbrainlab.org` (password `international` — used in scripts).

## Legacy v1

v1 mouse-supervised / neural scripts were **removed** from `scripts/`. Historical tables and figures remain under `reports/` (`v1_report.md`, `v1_artifact_index.md`). Neural analyses are **out of v2** success criteria.
