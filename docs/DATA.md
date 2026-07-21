# Data & artifacts (not in git)

This repository **does not** ship raw IBL downloads, processed tensors, or trained checkpoints. Those are regenerable and too large for GitHub.

## What *is* tracked

| Path | Why |
|---|---|
| `data/manifests/` | Session eid lists, QC scores, train/val/test splits |
| `configs/` | Frozen v1 encodings and matching rules |
| `reports/` | Tables, figures, and the v1 write-up |
| `src/`, `scripts/`, `tests/` | Code |

## What is ignored

| Path | Approx size locally | Rebuild |
|---|---|---|
| `data/raw/` | ~GBs (ONE cache) | automatic on first download |
| `data/processed/` | tens of MB | `python scripts/build_processed_datasets.py` |
| `artifacts/` | checkpoints | `python scripts/train_phase5.py` |
| `.venv/` | ~GBs | recreate from `requirements.txt` |

## Minimal online → local path

```bash
git clone https://github.com/ShivaUP/ibl-hidden-prior.git
cd ibl-hidden-prior
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Phase 1: QC + pin behavior-core (downloads via openalyx)
python scripts/run_session_qc.py

# Phase 3: processed trials / RNN bins / Bayes tables
python scripts/build_processed_datasets.py

# Phase 5–10: train, eval, figures (see reports/v1_report.md)
python scripts/train_phase5.py --conditions history_only
python scripts/eval_phase6.py
python scripts/eval_phase7_priors.py
# neural optional:
# python scripts/build_neural_intersect.py --max-qc 30
# python scripts/eval_phase8_neural_pilot.py
python scripts/eval_phase9_matched.py
python scripts/make_phase10_figures.py
```

Public ONE endpoint (no personal Alyx account required for open data):

`https://openalyx.internationalbrainlab.org` (password `international` — already used in scripts).

## Note on neural analyses

Strict `behavior-core ∩ ephys` was empty; neural results use `neural_behavior_pool` in `data/manifests/neural_intersect_eids.json`. Spike downloads are large and stay under `data/raw/`.
