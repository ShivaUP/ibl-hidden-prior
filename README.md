# IBL Hidden Prior Modeling

**Active science path: v2** (synth-primary, real-transfer secondary). Spec: [`docs/spec_v2.md`](docs/spec_v2.md).  
**Legacy v1** (mouse-supervised ranking + neural pilot): [`reports/v1_report.md`](reports/v1_report.md), [`docs/spec.md`](docs/spec.md).  
**Data is not in git:** [`docs/DATA.md`](docs/DATA.md).

---

## v2 objective (frozen 2026-07-21)

Train **only on synthetic** IBL-like sessions; **primary** ranking on held-out synth; **secondary** frozen transfer to real mouse behavior (no mouse fine-tune). Neural VE uses four primary prior-related ROIs (see [`docs/NEURAL_REGIONS.md`](docs/NEURAL_REGIONS.md)).

### Current results snapshot (v2, shared cohort n=8; post PC amendment)

**Cohort:** 8 almost-perfect behavior QC sessions with Neuropixels coverage whose **union** spans MOs, ORBvl (vlOFC), ACAd, and MOp. Same sessions for real behavior transfer and neural VE.

**History-only ranking (correctness vs correct stimulus side):**

| Model | Synth held-out | Real transfer | Real history gap |
|---|---:|---:|---:|
| GRU PC | 0.759 | **0.836** | 0.186 |
| GRU | 0.807 | 0.833 | **0.411** |
| tanh BPTT | **0.808** | 0.815 | 0.405 |
| tanh PC | 0.763 | 0.793 | 0.217 |

**Neural (session-mean VE):** among all models, GRU > tanh BPTT > GRU PC > tanh PC. Survival (best vs second, session bootstrap + Holm; no behavior-matching filter): survives in MOs, vlOFC, and MOp (not ACAd).

**Switch-centered correctness (real history-only, trials 0–15 post-switch):** GRU PC leads both 0.2→0.8 (~0.820) and 0.8→0.2 (~0.829); see `comparison/real_history_only_accuracy_to_switch_story.png`.

**Overall vs peri-switch boards:** `comparison/{synth|real}_{regime}_overall_vs_switch_correctness.png` — per model: overall · 0.2→0.8 (−30…+30) · 0.8→0.2 (−30…+30); Wilcoxon+Holm vs overall; `fixed_prior` = overall only. Model order: tanh BPTT → tanh PC → GRU → GRU PC (twin-complement colors). All figures save at DPI 600.

**MLP switch-block decoding (`scripts/16_…`):** three panels (−30…+30). A1 synth latents: BPTT best (GRU ~0.913, tanh BPTT ~0.907; PC ~0.81). A2 real mouse prior (~0.904) vs model \(q_t\): GRU ~0.921, tanh BPTT ~0.900; PC ~0.78. A3 neural û: MOs/vlOFC above chance; ACAd near chance. Figure: `figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png`.

**Reports:** [`reports/v2/CURRENT_STATUS_ARTICLE.docx`](reports/v2/CURRENT_STATUS_ARTICLE.docx), [`reports/v2/METHODS_DETAILED.docx`](reports/v2/METHODS_DETAILED.docx), [`docs/REAL_EVAL.md`](docs/REAL_EVAL.md).

### Model glossary

| ID | Test-time | Training |
|---|---|---|
| `tanh_bptt` | Vanilla tanh RNN (Kyan-adapted) | BPTT |
| `tanh_pc` | **Identical** tanh RNN | Corrected predictive-coding **credit assignment** (`PC_V2_CORRECTED.py` recipe) |
| `gru` | GRU (Shrijana-adapted) | BPTT |
| `gru_pc` | **Identical** GRU | Gate-aware PC credit assignment (same corrected recipe) |

`bayes` is legacy only (not in the active v2 comparison).

**PC schedule (locked):** 60 epochs × 24 sessions × 929 trials (same exposure as BPTT models); 32 inference rounds; `output_precision=0.025`; nudge-normalized local updates.

### Numbered runbook

Full script catalog: [`scripts/README.md`](scripts/README.md).

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Data prep (skip if manifests + processed trials already local)
python scripts/00_smoke_one_connection.py
python scripts/01_run_session_qc.py
python scripts/02_audit_event_deltas.py
python scripts/03_build_processed_trials.py

# Synth task + models
python scripts/04_fit_synthetic_stats.py
python scripts/05_build_synthetic_datasets.py
python scripts/06_map_real_to_v2_ticks.py
python scripts/07_train_model.py --all
python scripts/08_eval_synth_heldout.py       # primary ranking
python scripts/09_eval_real_transfer.py       # secondary transfer
python scripts/10_make_figures.py
```

Config: [`configs/synthetic_v2.yaml`](configs/synthetic_v2.yaml).  
**Tick / channel schema (empirical):** [`docs/TICK_AND_CHANNELS.md`](docs/TICK_AND_CHANNELS.md).  
Outputs: `artifacts/v2/`, `reports/v2/`.

### Figure catalog (v2)

| Path | What it answers |
|---|---|
| `reports/v2/figures/by_model/{model}/{synth\|real}/{regime}/multipanel_diagnostics.png` | Train curve; psychometric by block; switch-centered zero-evidence; example session (all prior levels 0.2/0.5/0.8) |
| `reports/v2/figures/scorecards/{synth\|real}_{regime}_scorecard.png` | Correctness + history gap (± 95% CI) |
| `reports/v2/figures/comparison/{synth\|real}_{regime}_correctness_by_prior.png` | Correctness by block prior + balanced score |
| `reports/v2/figures/comparison/{synth\|real}_{regime}_switch_board.png` | Switch directions across models |
| `reports/v2/figures/comparison/synth_vs_real_{regime}_board.png` | Synth vs real transfer |
| `reports/v2/figures/neural/neural_ve_unmatched_vs_matched.png` | Neural VE by region (MOs, ORBvl, ACAd, MOp) |
| `reports/v2/figures/neural/survival_tests.png` | Behavior-matched survival (session bootstrap + Holm) |
| `reports/v2/figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png` | MLP block decoding around switches: A1 synth latents · A2 mouse prior + model belief · A3 neural û |

Regenerate figures after eval: `python scripts/10_make_figures.py`, `python scripts/15_make_neural_figures.py`, and `python scripts/16_plot_mlp_switch_block_decoding.py`.  
Regenerate DOCX reports: `python scripts/make_v2_docx_reports.py`.

### Directory map (v2)

- `src/synthetic/` — channels, generator, real mapper  
- `src/models_v2/` — tanh / PC-CA / GRU / Bayes  
- `scripts/00_`…`16_` — pipeline entrypoints (see [`scripts/README.md`](scripts/README.md))  
- v1 train/eval/neural scripts **removed**; historical results stay under `reports/`

---

## Quick start (clone)

```bash
git clone https://github.com/ShivaUP/ibl-hidden-prior.git
cd ibl-hidden-prior
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the **v2 numbered runbook** above. Details: [`scripts/README.md`](scripts/README.md), [`docs/DATA.md`](docs/DATA.md).
---

## Scientific motivation

Adaptive behavior requires more than a fixed stimulus-response mapping because environmental statistics change over time and animals must update beliefs while preserving useful prior knowledge. The IBL biased-block task is a strong testbed for this problem because the hidden prior changes across blocks without being explicitly signaled to the mouse, forcing the animal to infer current bias from trial history and feedback.

This project focuses on the mechanistic question of how that hidden prior is computed, not only where it is represented. The working comparison is between learned recurrent dynamics, explicit latent-state inference, and prediction-error-driven recurrent dynamics.

---

## Core questions

Which model best reproduces behavior across the task and near block switches?

- Standard task-trained RNN
- Predictive-coding RNN
- Bayesian model

In the history-only condition, which model best matches how the mouse updates its subjective prior around block switches?

In prior-related regions (MOs, ORBvl, ACAd, MOp), which model best matches neural prior readouts, and does that advantage survive after behavior matching?

---

## Conditions to model

**Fixed prior:** only unbiased 50–50 blocks are used; this serves as a constrained control condition. In practice, these are trials or blocks with probabilityLeft = 0.5 in the IBL trials data.

**Full-information:** the true current block prior is provided to the model; this is the oracle condition.

**History-only:** the model must infer the hidden prior from past trials and feedback; this is the primary scientific condition.

**Switch-centered evaluation:** performance is evaluated specifically around block transitions, where belief updating is most informative.

These conditions must be kept strictly separate during preprocessing, training, and evaluation.

---

## Event-based trial representation

RNN-family models use an event-based within-trial representation rather than a single flat vector per trial. Each trial is discretized into 100 ms time bins aligned to current-trial stimulus onset, and input channels can turn on or off depending on the time step within the trial.

The base within-trial phases are:

- Stimulus presentation
- Delay / accumulation
- Response window
- Feedback / reward

The exact mapping from recorded IBL event times to these phases must be verified from stimOn_times, goCue_times, response_times, and feedback_times before the final preprocessing contract is frozen.

---

## Binary channel convention

For RNN-family models, each input channel is binary at each time step:

- 0 means the channel is off or inactive at that time step.
- 1 means the channel is on or active at that time step.

The trial representation is therefore a sequence of binary vectors. The intended interpretation is:

- stimulus_right = 1 means the current stimulus is on the right side; 0 means not-right, which will usually correspond to left in a valid stimulus trial.
- contrast_high = 1 means the current stimulus belongs to the high-contrast category defined by preprocessing; 0 means it does not.
- response_window = 1 means the animal is currently allowed to respond.
- reward = 1 means reward or positive feedback is present at that time step; 0 means no reward signal at that time step.

Previous-trial channels such as prev_choice_right = 1 or prev_correct = 1 can already be active at the current-trial stimulus time because those variables are known from the completed previous trial.

If later analyses require richer sensory coding, this binary scheme can be extended, but version 1 assumes binary on/off channels for the recurrent models.

---

## Model inputs by family and condition

### 1. Standard RNN

The standard RNN receives a sequence of 100 ms binary input vectors across the current trial.

#### Standard RNN — common channels across all conditions

These channels are available in every condition, provided they are causally available at that time step:

- stimulus_right
- contrast_high
- delay_phase
- response_window
- response_made
- reward
- prev_choice_right
- prev_correct
- prev_fast_rt or another binary previous-RT encoding
- optional trial_start or session_start indicator

At current-trial stimulus onset, the model is allowed to use:

- current-trial stimulus side,
- current-trial stimulus contrast,
- previous-trial choice,
- previous-trial correctness,
- previous-trial RT summary,

because these are all available before the current decision is made.

At current-trial stimulus onset, the model is not allowed to use:

- current-trial response,
- current-trial correctness,
- current-trial reward,
- future trial information.

#### Standard RNN — Fixed prior condition

**Inputs:**

- all common channels above
- no block-prior channel

**Restriction:**

only trials from unbiased probabilityLeft = 0.5 blocks are included.

**Interpretation:**

this is a constrained control in which the model does not need to infer changing block bias because the dataset slice is unbiased.

#### Standard RNN — Full-information condition

**Inputs:**

- all common channels above
- oracle_prior_right

Here oracle_prior_right = 1 means the true current block prior favors rightward stimuli, and 0 means it does not. This channel is externally provided from the true block context.

#### Standard RNN — History-only condition

**Inputs:**

- all common channels above
- no oracle prior channel

**Interpretation:**

the model must infer the hidden prior from previous choice, previous correctness, previous RT summary, and recent stimulus history carried in the recurrent state.

---

### 2. Predictive-coding RNN

The predictive-coding RNN uses the same external input channels as the standard RNN, so comparisons remain fair. What changes is the internal computation: the recurrent state is updated through explicit prediction-error-style dynamics rather than only through a generic recurrent update.

#### Predictive-coding RNN — common channels across all conditions

Same external channels as the standard RNN:

- stimulus_right
- contrast_high
- delay_phase
- response_window
- response_made
- reward
- prev_choice_right
- prev_correct
- prev_fast_rt
- optional trial_start or session_start

#### Predictive-coding RNN — Fixed prior condition

**Inputs:**

- all common channels above
- no block-prior channel
- only unbiased probabilityLeft = 0.5 blocks

#### Predictive-coding RNN — Full-information condition

**Inputs:**

- all common channels above
- oracle_prior_right

#### Predictive-coding RNN — History-only condition

**Inputs:**

- all common channels above
- no oracle prior channel

**Interpretation:**

the model must infer hidden prior from history and feedback, but its hidden-state updates are explicitly organized around prediction and prediction error.

---

### 3. Bayesian model

The Bayesian model uses the same causal information content as the RNNs, but usually in a trial-level formulation rather than 100 ms within-trial bins. It does not need binary event channels unless a within-trial Bayesian version is later introduced.

#### Bayesian model — trial-level common inputs

At current trial t, the model may use:

- current-trial stimulus side
- current-trial stimulus contrast
- previous choice
- previous correctness / previous reward
- previous reaction-time summary

These variables are causally available at current-trial stimulus time because they come either from the current stimulus or from completed trial t−1.

#### Bayesian model — Fixed prior condition

**Inputs:**

- current stimulus side
- current stimulus contrast
- previous choice
- previous correctness
- previous RT summary

**Restriction:**

fit and evaluate only on unbiased probabilityLeft = 0.5 blocks.

#### Bayesian model — Full-information condition

**Inputs:**

- current stimulus side
- current stimulus contrast
- previous choice
- previous correctness
- previous RT summary
- true current block prior

**Interpretation:**

the prior is given explicitly rather than inferred.

#### Bayesian model — History-only condition

**Inputs:**

- current stimulus side
- current stimulus contrast
- previous choice
- previous correctness
- previous RT summary
- no true block prior

**Interpretation:**

the model must infer the current latent prior from past evidence and outcomes.

---

## Time-step meaning of channels

The intended time-step logic for RNN-family models is:

**Stimulus presentation step:**
stimulus_right, contrast_high, prev_choice_right, prev_correct, and previous-RT channels may be active.

**Delay step:**
stimulus channels may turn off, while a delay or accumulation channel can remain active.

**Response-window step:**
response_window = 1 while the mouse is allowed to answer.

**Response step:**
response_made = 1 only when the response event occurs.

**Feedback step:**
reward = 1 only when positive feedback is delivered; otherwise it remains 0. Negative outcome can either remain implicit as reward = 0 or be represented by a separate binary channel such as incorrect = 1, depending on the final preprocessing decision.

This representation enforces causal structure because channels only turn on when that information is truly available in the task.

---

## Outputs

### Primary outputs

- Choice probability
- Reaction time or response latency
- Psychometric behavior
- Inferred latent prior

### Secondary outputs

- Previous-choice prediction, if useful diagnostically
- Previous-reward prediction, if useful diagnostically
- Latent strategy or engagement state, if useful for interpretation

---

## Behavioral analyses

- Full-session choice prediction
- Psychometric alignment
- Reaction-time fit
- Switch-centered adaptation curves
- Block-switch asymmetry
- Update-rate estimation
- Subjective-prior estimation in the history-only condition

---

## Neural analyses

### Regions of interest (belief / prior updating)

Full write-up: [`docs/NEURAL_REGIONS.md`](docs/NEURAL_REGIONS.md).

Sources:
- Findling et al., Nature 2025 (subjective prior map): https://www.nature.com/articles/s41586-025-09226-1
- IBL BWM, Nature 2025: https://www.nature.com/articles/s41586-025-09235-0
- IBL standardized behavior, eLife 2021: https://elifesciences.org/articles/63711

**Locked ROIs:** MOs, ORBvl (vlOFC), ACAd, MOp (primary analysis scope).

Shared cohort: **n=8** sessions; union covers all four ROIs (MOs in all 8; ORBvl≈5; ACAd≈4; MOp≈3). Per-region VE uses sessions that contain that region.

### Neural comparisons (implemented)

- Peri-stimulus spike counts → CV Ridge → mouse prior readout
- Model belief explains neural prior via `ve_linear_recal` (session-mean)
- Behavior-matched CE ε-ball; session-bootstrap survival + Holm across regions

---

## Data and preprocessing requirements

The project uses the public IBL dataset through ONE / SessionLoader, and preprocessing must begin by verifying the available trial-event structure from the released trial fields. The relevant trial datasets include contrastLeft, contrastRight, choice, feedbackType, probabilityLeft, stimOn_times, goCue_times, response_times, and feedback_times.

Before any model implementation, the preprocessing pipeline must:

1. Load candidate sessions and trial tables.
2. Verify event ordering and missingness.
3. Confirm the temporal relationship among stimulus onset, response window, response, and feedback.
4. Define the 100 ms binned event representation for RNN-family models.
5. Define the matched causal trial-level representation for the Bayesian model.
6. Save fixed processed datasets and manifests for reproducibility.

---

## Project plan

1. Data selection and preprocessing
2. Event-structure verification in the IBL trials data
3. Finalization of binary input channels and their time-step logic
4. Behavioral feature extraction
5. Model specification
6. Behavioral fitting and evaluation
7. Latent prior readout
8. Neural decoding and alignment
9. Behavior-matched model comparison
10. Mechanistic interpretation

---

## Deliverables for version 1

- Reproducible preprocessing pipeline
- Session filtering and manifest files
- Event-binned dataset construction code for RNN-family models
- Trial-level dataset construction code for Bayesian modeling
- Model training code
- Behavioral evaluation scripts
- Switch-centered analysis scripts
- Latent extraction utilities
- Neural alignment analysis
- Figures and a short written report

---

## Working principle

> The project is already in grilling mode, and the current source of truth is the event-based specification above. Earlier planning assumptions that treated the task mainly as a static trial-level input problem for all models are now superseded for the RNN-family models.

Do not start implementation until the following assumptions are explicit and documented:

- the exact event timing structure verified from the dataset,
- the final binary channel list,
- the exact definition of high versus non-high contrast,
- the handling of incorrect outcome as either absence of reward or a separate binary channel,
- and the exact distinction among fixed prior, full-information, and history-only conditions.
