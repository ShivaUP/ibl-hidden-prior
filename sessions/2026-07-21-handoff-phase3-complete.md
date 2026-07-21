# Session handoff — IBL Hidden Prior Modeling

**Date (UTC):** 2026-07-21  
**Workspace:** `/Users/shivamahdian/Desktop/Performance`  
**Handoff purpose:** Continue from completed Phase 3 into **Phase 4** (shared model I/O + smoke training), then behavior training/eval as in the plan.  
**Produced via:** `skills/hand-off.md` (saved under `sessions/` per user request).

---

## 1. One-sentence status

Grilling is frozen into `docs/spec.md` / `docs/plan.md` / `configs/frozen_v1.yaml`; Phases **0–3** are done (almost-perfect behavior-core, event-bin + Bayesian datasets, session splits); **next work is Phase 4 model interfaces + smoke train** — do not jump to neural or full training sweeps yet.

---

## 2. Source-of-truth documents (do not re-derive)

| Path | Role |
|------|------|
| `README.md` | Scientific framing + event-based RNN design (ground truth for model trio) |
| `AGENTS.md` | Agent rules; explicitly defers to README if conflict |
| `docs/spec.md` | Frozen v1 scientific/tech decisions from grilling |
| `docs/plan.md` | Phased execution with task IDs, DoD, stop gates |
| `configs/frozen_v1.yaml` | Machine-readable freeze (ε / λ / architectures still `pilot_pending`) |

**Model trio (README / AGENTS):** Standard RNN · Predictive-coding RNN · Bayesian. **Not** meta-RL in v1.

---

## 3. What this conversation accomplished

### 3.1 Process

1. Read README + AGENTS; inspected empty scaffolding repo.
2. Ran **grill-me** one question at a time; user accepted recommendations with refinements.
3. Fixed AGENTS to match README (removed meta-RL).
4. Built inspection tooling; installed stack into `.venv`; inspected real IBL sessions.
5. Froze QC / encodings from data + grilling; wrote spec + plan + YAML.
6. Executed plan Phases **0 → 3** (stop before full model training).

### 3.2 Key grilling decisions (pointers only — full text in `docs/spec.md`)

- Dataset: public ONE / openalyx; behavior-core first; neural-intersect later (MOs, vlOFC).
- Almost-perfect session QC: ≥95% trials pass timing/choice rules; ≥90% after RT percentile trim; else **drop whole session**.
- RT: `response − goCue` (stimOn fallback); log-normal / log-Gaussian; binary `prev_fast_rt`.
- `contrast_high`: high `{0.25, 0.5, 1.0}`, low `{0, 0.0625, 0.125}` (observed sessions had no 0.5).
- Reward channel: `feedbackType == +1` only; no separate `incorrect` channel.
- RNN: 100 ms bins; **bin 0 = `stimOn_times`**.
- Targets: choice primary + downweighted RT (`λ < 1`, numeric still pilot).
- Eval: switch window −10…+30; prefer 10/20 pre/post, allow 8/16 flagged + sensitivity.
- Mouse prior: behavior-derived latent (not `probabilityLeft`); oracle only in full-info.
- Neural v1: MOs + vlOFC; gate = variance explained by model \(q_t\); behavior-match choice-first.
- Non-goals: meta-RL, VISp-as-success, multibit contrast, separate incorrect channel.

### 3.3 Implementation completed (Phases 0–3)

**Phase 0 — scaffolding**

- Layout: `src/{data,models,train,eval,neural,plot}`, `scripts/`, `configs/`, `data/`, `reports/`, `docs/`, `tests/`
- `requirements.txt` pinned from venv
- `configs/frozen_v1.yaml` present

**Phase 1 — download + QC**

- `src/data/qc.py`, `scripts/run_session_qc.py`, `scripts/smoke_one_connection.py`
- 25 candidates → 15 pass almost-perfect → **10** pinned in `data/manifests/behavior_core_eids.json`
- Messy sessions from early inspection correctly failed the gate

**Phase 2 — freeze encodings + event audit**

- `src/data/features.py` + tests
- `scripts/audit_event_deltas.py` → `reports/inspection/event_deltas.md`
- Pooled medians (approx): go−stim ≈ 16 ms; resp−go ≈ 0.41 s; stimOff−stim ≈ 1.48 s  
  → stim + go-cue both land in **bin 0** at 100 ms

**Phase 3 — datasets**

- `src/data/processed_trials.py`, `event_bins.py`, `bayes_features.py`
- `scripts/build_processed_datasets.py`
- **9610** QC trials across 10 eids; **873** fixed-prior; **186** switches
- RNN bins + Bayesian tables for `history_only` / `full_information` / `fixed_prior`
- Splits: **6 train / 2 val / 2 test** sessions (`data/manifests/splits.json`)
- Unit tests: QC, features, event-bin leakage (bin 0: stim on; response/reward off until events)

---

## 4. Artifact map (where things live)

### Manifests & processed data

- `data/manifests/candidates_raw.json`
- `data/manifests/qc_scores.csv`
- `data/manifests/behavior_core_eids.json`
- `data/manifests/splits.json`
- `data/processed/trials/` (`all_trials.parquet` + per-eid)
- `data/processed/rnn_bins/{history_only,full_information,fixed_prior}.pkl` (+ `*_index.csv`)
- `data/processed/bayes_trials/{...}.parquet`
- `data/raw/one_cache/` — ONE download cache

### Reports

- `reports/inspection/ibl_trial_fields_summary.{txt,json}`
- `reports/inspection/event_deltas.{md,json}`
- `reports/inspection/smoke_one_connection.json`
- `reports/qc/qc_summary.txt`, `qc_scores.json`, `processed_datasets.json`

### Scripts (entry points)

```bash
source .venv/bin/activate
python src/data/inspect_trials.py
python scripts/smoke_one_connection.py
python scripts/run_session_qc.py --n-candidates 25 --max-core 10
python scripts/audit_event_deltas.py
python scripts/build_processed_datasets.py
python -m pytest tests/ -q
```

### Core library modules

- `src/data/config.py` — load frozen YAML
- `src/data/inspect_trials.py` — field inspection (runnable)
- `src/data/qc.py` — trial/session almost-perfect QC
- `src/data/features.py` — derived binary/history features
- `src/data/processed_trials.py` — cleaned trial tables + switches
- `src/data/event_bins.py` — 100 ms binary sequences
- `src/data/bayes_features.py` — trial-level Bayesian matrices

**Empty / not started:** `src/models/`, `src/train/`, `src/eval/`, `src/neural/`, `src/plot/` (packages exist; no training code yet).

---

## 5. How QC picks “best” data (plain language)

1. **Trial good?** Finite ordered times, positive RT, choice ±1, then RT in session 1st–99th percentile.  
2. **Session good?** ≥400 choice trials; left+right bias blocks; priors in `{0.2,0.5,0.8}`; ≥95% pass trial rules; ≥90% after RT trim — else **drop session**.  
3. **Core set:** among passers, rank by cleanliness; keep top 10.

Details: `docs/spec.md` §4, `src/data/qc.py`, `configs/frozen_v1.yaml` → `data.session_inclusion` / `trial_inclusion`.

---

## 6. Environment notes

- OS in this session: macOS (darwin); user rules also mention Windows — activate `.venv` accordingly.
- Python 3.12 venv at `.venv/`; key pins in `requirements.txt` (`ONE-api`, `ibllib`, `pandas`, `numpy`, `torch`, etc.).
- ONE public Alyx: use **documented public openalyx credentials** from IBL docs (do not commit secrets; treat any local overrides as sensitive).
- User rule: do **not** invent new pip installs without asking unless the user explicitly requests install (this session user did ask once to install libs).

---

## 7. Explicit stop gates (from plan)

- **After Phase 3 (current):** do not start real/full training until Phase 3 validation is accepted; smoke train only in Phase 4.
- **After Phase 2:** encoding changes require YAML version bump + rebuild from Phase 3.
- Neural work only after behavior milestones (Phases 5–7 before 8–9).

---

## 8. Next session — recommended work order

Tailored focus: **Phase 4 → early Phase 5 setup**.

1. **P4.1** `src/models/interfaces.py` — shared `ModelOutputs`, `extract_latent_prior`, dataset adapters for RNN bins vs Bayesian tables.  
2. **P4.2–P4.4** Minimal Standard RNN, PC-RNN, Bayesian stubs + smoke train on tiny subset; save under `artifacts/smoke/`.  
3. **P4.5** `scripts/train_model.py --model {standard,pc,bayes} --smoke`.  
4. Freeze open architecture choices (GRU vs LSTM; PC equations sketch; Bayesian form) into `configs/frozen_v1.yaml` before full sweeps.  
5. Only then Phase 5 full history-only / full-info / fixed-prior training on the 6/2/2 split.

**Do not** in the next session unless asked: rewrite README wholesale; expand to brain-wide neural; loosen almost-perfect QC; add meta-RL.

---

## 9. Open items still `pilot_pending` in YAML

- `outputs.lambda_rt`
- `evaluation.behavior_matching.choice_epsilon_ball` / `rt_secondary_floor`
- `architecture_open.standard_rnn_cell` / `predictive_coding_equations` / `bayesian_form`
- Exact neural peri-event window (Phase 8)
- Alyx revision string pinning (document on next manifest refresh)

See `docs/spec.md` §10 risks R1–R8.

---

## 10. Known pitfalls for the next agent

- `inspect_trials.py` is the **runnable** inspector; do not expect side effects from importing alone.
- Early inspection eids with broken RT/order are **not** in behavior-core by design.
- `prev_*` features are computed on full session then filtered to QC-pass trials (order preserved before filter).
- RNN sequences are **ragged** (pickled list of arrays), not a single dense tensor.
- `oracle_prior_right`: 1 iff `probabilityLeft < 0.5` (right-favoring block).
- Seed doc eid may pass QC but not always make top-10 by cleanliness ranking.

---

## 11. Suggested skills

Invoke these when relevant in the next session:

| Skill | When |
|-------|------|
| `.cursor/skills/grill-me/SKILL.md` | Any reopen of unresolved design choices (ε, λ, PC/Bayesian form) — one question at a time |
| `skills/hand-off.md` | End of next session; write a new file under `sessions/` |
| `.cursor/skills/create-rule/SKILL.md` | If team wants durable Cursor rules mirroring Phase 4+ conventions |
| Workspace rules | Always apply: `.cursor/rules/project.mdc`, `planning.mdc`, `python-research.mdc`, `AGENTS.md` |

---

## 12. Minimal “start here” checklist for a fresh agent

1. Read `docs/spec.md` + `docs/plan.md` Phase 4 section.  
2. Skim `configs/frozen_v1.yaml` and `data/manifests/behavior_core_eids.json` / `splits.json`.  
3. `source .venv/bin/activate` && `python -m pytest tests/ -q`.  
4. Confirm processed artifacts exist under `data/processed/`.  
5. Implement Phase 4 interfaces + smoke trains only.

---

*End of handoff. Do not duplicate `docs/spec.md` or `docs/plan.md` into future handoffs — update those files if decisions change, and point here only for session narrative + next steps.*
