# Execution plan (v1)

**Status:** Frozen after grilling (2026-07-20).  
**Companion:** `docs/spec.md`  
**Constraint:** No large complexity jumps. Behavior milestones before neural. First milestone = small download + QC/preprocess.

Dependency notation: task `P{phase}.{task}` depends on listed IDs. Subtasks are `P{phase}.{task}.{sub}`.

---

## Phase order (summary)

| Phase | Name | Stop before next when… |
|-------|------|-------------------------|
| 0 | Repo + freeze scaffolding | Config skeleton + folder layout exist |
| 1 | Small download + session QC | Almost-perfect behavior-core manifest pinned |
| 2 | Preprocessing freeze + derived fields | `configs/frozen_v1.yaml` complete; event-delta audit done |
| 3 | Dataset builders | RNN bins + Bayesian tables + unit tests on 1–3 eids |
| 4 | Model interfaces + toy train | Shared I/O contract; smoke train on tiny subset |
| 5 | Full behavioral training | Three models trained under all conditions |
| 6 | Behavioral evaluation + switch analyses | Metrics + figures for behavior frozen |
| 7 | Subjective prior estimator + model prior match | Prior-match tables frozen |
| 8 | Neural load + alignment | VE tables for MOs/vlOFC |
| 9 | Behavior-matched neural comparison | Survival tests + matched ranking |
| 10 | Report + archive | `reports/v1_report.md` + figure set complete |

---

## Phase 0 — Repo + freeze scaffolding

**Goal:** Make the repository ready for reproducible work without training models.

### Task P0.1 — Directory layout
- **Depends on:** none
- **Subtasks:**
  - P0.1.1 Create `src/{data,models,train,eval,neural,plot}`, `scripts/`, `configs/`, `data/{raw,processed,manifests}`, `reports/`, `docs/`
  - P0.1.2 Ensure `src/` is importable
- **Inputs:** `AGENTS.md` layout
- **Outputs:** empty package dirs with `__init__.py` where needed
- **DoD:** layout matches AGENTS; `python -c "import src"` works
- **Locations:** repo root as above

### Task P0.2 — Config skeleton
- **Depends on:** P0.1
- **Subtasks:**
  - P0.2.1 Create `configs/frozen_v1.yaml` with placeholder sections: inclusion, channels, RT, QC, switch, outputs, neural, matching
  - P0.2.2 Document that YAML is the machine-readable freeze (values filled in Phases 1–2)
- **Inputs:** `docs/spec.md`
- **Outputs:** `configs/frozen_v1.yaml`
- **DoD:** file parses; all sections present (may contain `TODO` until Phase 2)

### Task P0.3 — Requirements pin
- **Depends on:** none
- **Subtasks:**
  - P0.3.1 Write `requirements.txt` from current venv pins used for inspection (`ONE-api`, `ibllib`, `pandas`, `numpy`, …)
- **Inputs:** working `.venv`
- **Outputs:** `requirements.txt`
- **DoD:** fresh install instructions documented in README or docs note

**Phase 0 validation**
- [ ] Layout exists
- [ ] Config skeleton parses
- [ ] Inspection script still runs

**STOP:** Do not download full cohorts until Phase 0 validation passes.

---

## Phase 1 — Small download + session QC

**Goal:** Pin a tiny reproducible behavior cohort with almost-perfect timing QC.

### Task P1.1 — ONE connectivity smoke test
- **Depends on:** P0.1, P0.3
- **Subtasks:**
  - P1.1.1 Script: connect to openalyx, load 1 known eid trials table
  - P1.1.2 Write cache under `data/raw/one_cache/`
- **Inputs:** public ONE credentials
- **Outputs:** cached trials for 1 eid; log snippet in `reports/inspection/`
- **DoD:** trials columns include core fields from spec
- **Locations:** `scripts/smoke_one_connection.py` or extend `src/data/inspect_trials.py`

### Task P1.2 — Candidate eid search
- **Depends on:** P1.1
- **Subtasks:**
  - P1.2.1 Search biasedChoiceWorld (or project-confirmed protocol) eids
  - P1.2.2 Save raw candidate list with search metadata (date, query)
- **Inputs:** ONE API
- **Outputs:** `data/manifests/candidates_raw.json`
- **DoD:** ≥ N candidates (N documented; start small, e.g. 20–50)

### Task P1.3 — Per-session QC scoring
- **Depends on:** P1.2
- **Subtasks:**
  - P1.3.1 Implement trial rules (finite times, monotonicity, RT>0, choice ±1)
  - P1.3.2 Compute pre-percentile pass rate and post-percentile retained fraction
  - P1.3.3 Apply session gates (≥95% / ≥90%)
  - P1.3.4 Require bias blocks present and ≥400 choice trials (or document waiver)
- **Inputs:** candidate eids; `docs/spec.md` QC
- **Outputs:** `data/manifests/qc_scores.csv`; plots optional
- **DoD:** each candidate has pass/fail reason codes
- **Locations:** `src/data/qc.py`, `scripts/run_session_qc.py`

### Task P1.4 — Freeze behavior-core manifest
- **Depends on:** P1.3
- **Subtasks:**
  - P1.4.1 Write `data/manifests/behavior_core_eids.json` with eid list, QC summary, download UTC, Alyx/ONE revision if available
  - P1.4.2 Start with a **small** core (e.g. 5–15 sessions) that all pass almost-perfect QC
- **Inputs:** `qc_scores.csv`
- **Outputs:** pinned behavior-core manifest
- **DoD:** re-running QC on manifest yields 100% session pass

**Phase 1 validation**
- [ ] Manifest eids reload identically from cache
- [ ] All manifest sessions meet ≥95%/≥90% gates
- [ ] Contrast levels and `probabilityLeft` set summarized

**STOP:** Do not design event bins until behavior-core is pinned. Do not train models.

---

## Phase 2 — Preprocessing freeze + derived fields

**Goal:** Freeze all project-derived encodings and write final `configs/frozen_v1.yaml`.

### Task P2.1 — Event-delta audit
- **Depends on:** P1.4
- **Subtasks:**
  - P2.1.1 For each core eid, compute distributions of `goCue-stimOn`, `response-goCue`, `feedback-response`, `stimOff-stimOn`
  - P2.1.2 Propose 100 ms phase map consistent with bin0=`stimOn`
- **Inputs:** behavior-core trials
- **Outputs:** `reports/inspection/event_deltas.md` (+ optional json)
- **DoD:** median delays documented; pathological sessions already excluded by Phase 1

### Task P2.2 — Derived field builders
- **Depends on:** P2.1
- **Subtasks:**
  - P2.2.1 Implement abs contrast, `contrast_high`, `stimulus_right`, RT, `prev_*` features
  - P2.2.2 Implement `reward` from `feedbackType==+1`
  - P2.2.3 Unit tests on known rows from inspection eid
- **Inputs:** raw trials; spec rules
- **Outputs:** `src/data/features.py`; tests under `tests/` or `src/data/test_*.py`
- **DoD:** tests cover contrast set and reward coding

### Task P2.3 — Freeze YAML
- **Depends on:** P2.1, P2.2
- **Subtasks:**
  - P2.3.1 Fill `configs/frozen_v1.yaml` with final numeric rules (channels, QC, RT, switch window, λ placeholder, ε placeholder)
  - P2.3.2 Cross-check every field against `docs/spec.md`
- **Inputs:** event audit; feature code
- **Outputs:** complete `configs/frozen_v1.yaml`
- **DoD:** no `TODO` remaining in required sections (ε/λ may be `pilot_pending` only if explicitly labeled)

**Phase 2 validation**
- [ ] YAML loads
- [ ] Feature builder reproduces inspection contrast counts on a frozen eid
- [ ] Spec ↔ YAML consistency checklist signed off in a short note

**STOP:** No dataset builder changes to encoding rules after this freeze without a version bump (`frozen_v1` → `frozen_v1.1`).

---

## Phase 3 — Dataset builders

**Goal:** Build reproducible processed datasets for RNNs and Bayesian model.

### Task P3.1 — Trial-level processed tables
- **Depends on:** P2.3
- **Subtasks:**
  - P3.1.1 Write cleaned trial parquet/csv per eid + concatenated table
  - P3.1.2 Attach condition labels (fixed-prior slice mask; full-info; history-only)
  - P3.1.3 Attach switch indices
- **Inputs:** raw cache; frozen config
- **Outputs:** `data/processed/trials/`
- **DoD:** schema docstring; row counts match QC retained trials

### Task P3.2 — RNN event-bin builder
- **Depends on:** P3.1
- **Subtasks:**
  - P3.2.1 Allocate bins from stimOn; length rule frozen (fixed max bins or until feedback+pad)
  - P3.2.2 Paint binary channels per frozen timing map
  - P3.2.3 Assert no current-trial reward/response leakage at bin 0 beyond allowed channels
  - P3.2.4 Save arrays + index maps (eid, trial_id, condition)
- **Inputs:** processed trials; config
- **Outputs:** `data/processed/rnn_bins/`
- **DoD:** golden test: 1 trial’s bin0 has stim channels on; response/reward off until their events
- **Locations:** `src/data/event_bins.py`, `scripts/build_rnn_bins.py`

### Task P3.3 — Bayesian trial-level builder
- **Depends on:** P3.1
- **Subtasks:**
  - P3.3.1 Emit matched causal feature matrix per condition
  - P3.3.2 Save targets (choice, RT) and oracle prior for full-info
- **Inputs:** processed trials
- **Outputs:** `data/processed/bayes_trials/`
- **DoD:** feature causality checklist automated

### Task P3.4 — Splits
- **Depends on:** P3.2, P3.3
- **Subtasks:**
  - P3.4.1 Session-level train/val/test split (frozen seed)
  - P3.4.2 Ensure switch-eval sessions/blocks are not used for hyperparameter selection leaking into switch metrics
- **Inputs:** manifests
- **Outputs:** `data/manifests/splits.json`
- **DoD:** no eid overlap across splits

**Phase 3 validation**
- [ ] Rebuild from scratch yields checksum-identical processed artifacts (or documented float tolerance)
- [ ] Leakage tests pass
- [ ] Small subset loaders work in < few minutes

**STOP:** Do not start real training until Phase 3 validation passes.

---

## Phase 4 — Model interfaces + toy train

**Goal:** Shared I/O contract and smoke-test training on a tiny subset.

### Task P4.1 — Shared interfaces
- **Depends on:** P3.4
- **Subtasks:**
  - P4.1.1 Define `ModelOutputs` (choice logits/probs, RT params, prior \(q_t\), optional diagnostics)
  - P4.1.2 Define `extract_latent_prior(model, batch) -> q_t`
  - P4.1.3 Define dataset adapters for RNN bins vs Bayesian tables
- **Inputs:** spec fairness contract
- **Outputs:** `src/models/interfaces.py`
- **DoD:** all three model stubs implement interface

### Task P4.2 — Standard RNN stub + smoke train
- **Depends on:** P4.1
- **Subtasks:**
  - P4.2.1 Implement minimal GRU/LSTM + choice/RT heads
  - P4.2.2 Train 1–5 epochs on tiny subset; save checkpoint
- **Inputs:** `data/processed/rnn_bins/` tiny slice
- **Outputs:** `src/models/standard_rnn.py`; `artifacts/smoke/standard_rnn.pt`
- **DoD:** finite losses; choice probs in (0,1)

### Task P4.3 — Predictive-coding RNN stub + smoke train
- **Depends on:** P4.1, P4.2 (shared train loop preferred)
- **Subtasks:**
  - P4.3.1 Implement PC update cell with same I/O as standard RNN
  - P4.3.2 Smoke train identically
- **Outputs:** `src/models/pc_rnn.py`; smoke checkpoint
- **DoD:** same batch shapes as standard RNN

### Task P4.4 — Bayesian stub + smoke fit
- **Depends on:** P4.1
- **Subtasks:**
  - P4.4.1 Implement simplest explicit online prior update + choice/RT observation model
  - P4.4.2 Fit on tiny subset
- **Outputs:** `src/models/bayesian.py`; smoke artifact
- **DoD:** exposes \(q_t\); predicts choice/RT

### Task P4.5 — Model-agnostic train/eval harness
- **Depends on:** P4.2, P4.3, P4.4
- **Subtasks:**
  - P4.5.1 Single training entry with model name switch
  - P4.5.2 Shared loss \(\mathcal{L}_{choice}+\lambda\mathcal{L}_{RT}\)
- **Outputs:** `src/train/loop.py`, `scripts/train_model.py`
- **DoD:** `train_model.py --model {standard,pc,bayes} --smoke` works for all three

**Phase 4 validation**
- [ ] Interface tests pass
- [ ] Smoke trains finish without NaNs
- [ ] Latent prior extraction returns finite arrays

**STOP:** Freeze architecture choices (R1/R2) into YAML before Phase 5 full sweeps.

---

## Phase 5 — Full behavioral training

**Goal:** Train all models on all conditions with held-out session splits.

### Task P5.1 — Hyperparameter grid (shared RNN grid)
- **Depends on:** P4.5, Phase 4 STOP freeze
- **Subtasks:**
  - P5.1.1 Define small shared grid (hidden size, LR, λ, dropout)
  - P5.1.2 Early stop on val **choice NLL**
- **Outputs:** `configs/grids/v1_rnn_grid.yaml`
- **DoD:** grid size documented (prefer ≤12 runs per model×condition)

### Task P5.2 — Train standard RNN (all conditions)
- **Depends on:** P5.1
- **Subtasks:** run fixed / full-info / history-only; log metrics; save best checkpoints
- **Outputs:** `artifacts/models/standard_rnn/{condition}/`
- **DoD:** each condition has best checkpoint + val choice LL

### Task P5.3 — Train PC RNN (all conditions)
- **Depends on:** P5.1
- **Outputs:** `artifacts/models/pc_rnn/{condition}/`
- **DoD:** same as P5.2

### Task P5.4 — Fit Bayesian (all conditions)
- **Depends on:** P4.5
- **Outputs:** `artifacts/models/bayes/{condition}/`
- **DoD:** same as P5.2

### Task P5.5 — Select behavior candidates
- **Depends on:** P5.2–P5.4
- **Subtasks:**
  - P5.5.1 Rank by held-out history-only choice LL
  - P5.5.2 Record RT secondary metrics; set provisional ε-ball / RT floor after looking at spread (then freeze in YAML)
- **Outputs:** `reports/behavior/model_selection_table.csv`
- **DoD:** ε and RT floor written into `configs/frozen_v1.yaml`

**Phase 5 validation**
- [ ] All condition×model runs logged
- [ ] No training on test eids
- [ ] Checkpoints reload

**STOP:** Do not compute neural metrics yet. Do not change encodings.

---

## Phase 6 — Behavioral evaluation + switch analyses

**Goal:** Produce the behavioral figure/metric set from the spec.

### Task P6.1 — Choice + psychometric evaluation
- **Depends on:** P5.5
- **Subtasks:** held-out LL, accuracy, pseudo-\(R^2\); psychometrics overall and by block
- **Outputs:** `src/eval/behavior_choice.py`; `reports/behavior/psychometrics.*`
- **DoD:** tables + plots for three models × conditions

### Task P6.2 — RT evaluation
- **Depends on:** P5.5
- **Outputs:** `src/eval/behavior_rt.py`; RT summary figures
- **DoD:** secondary RT LL reported beside choice

### Task P6.3 — Switch-centered analyses
- **Depends on:** P5.5
- **Subtasks:**
  - P6.3.1 Detect switches; apply −10…+30 window
  - P6.3.2 Prefer ≥10/≥20; allow ≥8/≥16 flagged
  - P6.3.3 Sensitivity analysis excluding relaxed switches
  - P6.3.4 Asymmetry + adaptation half-life with bootstrap CIs
- **Outputs:** `src/eval/switch_centered.py`; switch figures
- **DoD:** mouse + 3 models trajectories plotted; sensitivity table exists

**Phase 6 validation**
- [ ] Metric checklist from spec §8.1 complete
- [ ] Sensitivity reported
- [ ] Hyperparameters not retuned using switch-eval leakage

**STOP:** Behavioral conclusions draftable before prior/neural phases.

---

## Phase 7 — Subjective prior estimator + model prior match

**Goal:** History-only mouse latent prior and model–mouse match scores.

### Task P7.1 — Fit mouse prior estimator
- **Depends on:** P3.1, P6.1 (behavior tables)
- **Subtasks:**
  - P7.1.1 Implement compact latent-bias / history model (IBL prior-paper template)
  - P7.1.2 Export trial-wise \(\hat{p}_t\) on history-only retained trials
- **Outputs:** `src/eval/mouse_prior.py`; `data/processed/mouse_prior/`
- **DoD:** \(\hat{p}_t\) in (0,1); correlates with block structure without equating to oracle

### Task P7.2 — Extract model priors \(q_t\)
- **Depends on:** P5.5, P4.1
- **Subtasks:** run `extract_latent_prior` for each model on history-only
- **Outputs:** `data/processed/model_priors/`
- **DoD:** aligned trial indices with mouse prior

### Task P7.3 — Prior match metrics
- **Depends on:** P7.1, P7.2, P6.3
- **Subtasks:** correlation/RMSE; switch-window MSE; asymmetry
- **Outputs:** `reports/behavior/prior_match.*`
- **DoD:** table comparing three models

**Phase 7 validation**
- [ ] Oracle prior not used as mouse target in history-only
- [ ] Match metrics reproducible from scripts

**STOP:** Neural alignment uses these \(q_t\) definitions only.

---

## Phase 8 — Neural load + alignment

**Goal:** Load MOs / vlOFC data for neural-intersect and compute primary VE.

### Task P8.1 — Neural-intersect manifest
- **Depends on:** P1.4
- **Subtasks:**
  - P8.1.1 Query sessions with usable units in MOs and vlOFC/orbvl
  - P8.1.2 Intersect with behavior-core
- **Outputs:** `data/manifests/neural_intersect_eids.json`
- **DoD:** non-empty or explicitly document blocker

### Task P8.2 — Neural preprocessing
- **Depends on:** P8.1
- **Subtasks:** unit QC; trial-aligned matrices; freeze peri-event window in YAML
- **Outputs:** `data/processed/neural/`; update YAML neural window
- **DoD:** one eid end-to-end documented

### Task P8.3 — Region prior-related readout
- **Depends on:** P8.2, P7.1
- **Subtasks:** decode/regress neural activity to mouse \(\hat{p}_t\) or construct prior axis; freeze method
- **Outputs:** `src/neural/prior_readout.py`
- **DoD:** trial-wise neural prior signal

### Task P8.4 — Model–neural VE (unmatched)
- **Depends on:** P8.3, P7.2
- **Subtasks:** VE and correlation for each model \(q_t\); full and switch window
- **Outputs:** `reports/neural/ve_unmatched.*`
- **DoD:** MOs and vlOFC tables exist

**Phase 8 validation**
- [ ] VISp not required
- [ ] Primary metric is VE; correlation companion only

**STOP:** Do not claim neural advantage until Phase 9 matching.

---

## Phase 9 — Behavior-matched neural comparison

**Goal:** Confirmatory matched ranking + survival tests.

### Task P9.1 — Build matched set
- **Depends on:** P5.5, P6.1, P6.2
- **Subtasks:** apply choice ε-ball + RT floor; list included/excluded checkpoints
- **Outputs:** `reports/neural/behavior_matched_models.json`
- **DoD:** choice-primary rule enforced in code asserts

### Task P9.2 — Matched VE comparison
- **Depends on:** P9.1, P8.4
- **Outputs:** `reports/neural/ve_matched.*`
- **DoD:** only matched models in confirmatory table

### Task P9.3 — Survival tests
- **Depends on:** P9.2
- **Subtasks:** paired bootstrap or session permutation; Holm across regions
- **Outputs:** `reports/neural/survival_tests.*`
- **DoD:** 95% CIs and corrected decisions reported

**Phase 9 validation**
- [ ] Unmatched vs matched both shown
- [ ] No neural claim for models outside choice ε-ball

**STOP:** Results ready for report writing.

---

## Phase 10 — Report + figures archive

**Goal:** Ship v1 artifacts contract.

### Task P10.1 — Figure generation scripts
- **Depends on:** P6–P9
- **Subtasks:** implement minimum panel set from spec §9
- **Outputs:** `src/plot/`; `reports/figures/`
- **DoD:** all required panels regenerated from scripts (no manual-only figures)

### Task P10.2 — Write `reports/v1_report.md`
- **Depends on:** P10.1
- **Subtasks:** behavior summary; switch; prior match; matched neural; limitations/risks
- **Outputs:** `reports/v1_report.md`
- **DoD:** answers the three core README questions with pointers to figures/tables

### Task P10.3 — Archive freeze
- **Depends on:** P10.2
- **Subtasks:** tag config version; list artifact checksums; note open risks R1–R8 status
- **Outputs:** `reports/v1_artifact_index.md`
- **DoD:** third party can reproduce from manifests + config + scripts

**Phase 10 validation**
- [ ] Deliverables checklist in `docs/spec.md` §9 all checked
- [ ] Non-goals not accidentally in scope

---

## Cross-cutting engineering rules

1. Prefer small scripts in `scripts/` calling `src/` functions.
2. Notebooks only for exploration; no notebook as system of record.
3. Every phase writes logs under `reports/`.
4. Any change to encodings after Phase 2 requires config version bump and re-run from Phase 3.
5. Do not install undocumented dependencies without updating `requirements.txt`.

---

## Immediate next action (after this plan)

Start **Phase 0 → Phase 1**: create `configs/frozen_v1.yaml` skeleton (if not present), then expand QC tooling beyond the inspection script to build the almost-perfect `behavior_core` manifest.
