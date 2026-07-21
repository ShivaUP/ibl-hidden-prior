# Project specification (v1 freeze)

**Status:** Frozen after grilling (2026-07-20).  
**Source of truth hierarchy:** `README.md` (scientific framing) → this file (v1 decisions) → `configs/frozen_v1.yaml` (machine-readable freeze; to be created in Phase 0/1).  
**Models (README ground truth):** (1) standard task-trained RNN, (2) predictive-coding RNN, (3) Bayesian / explicit online-inference model.  
**Out of scope for v1:** meta-RL; full brain-wide fishing; VISp as success criterion.

---

## 1. Project objective

Determine which model class best explains mouse trial-by-trial updating of a hidden block prior in the IBL biased-block decision task:

1. a **standard task-trained RNN**,
2. a **predictive-coding RNN**,
3. a **Bayesian / explicit online-inference model**,

under fair, behavior-first evaluation, with secondary neural alignment in prior-related regions (**MOs**, **vlOFC / orbvl**), and a confirmatory test of whether any neural advantage survives **behavior matching**.

---

## 2. Scientific motivation

Adaptive behavior requires updating beliefs when environmental statistics change. In the IBL biased-block task, the hidden prior (`probabilityLeft`) changes across blocks without being explicitly signaled, so the animal must infer current bias from trial history and feedback.

This project asks **how** that hidden prior is computed (mechanistic model class), not only **where** it is represented. Comparison axes:

- learned recurrent dynamics (standard RNN),
- prediction-error-organized recurrent dynamics (predictive-coding RNN),
- explicit latent-state inference (Bayesian model).

Primary scientific condition: **history-only** (no oracle prior channel). Fixed-prior and full-information conditions are controls.

---

## 3. Scope and non-goals

### In scope (v1)

- Public IBL behavior via ONE / `SessionLoader`, with later neural join for MOs and vlOFC/orbvl.
- Event-based 100 ms RNN inputs aligned so **timestep 0 = stimulus onset (`stimOn_times`)**.
- Trial-level Bayesian inputs with matched causal information.
- Behavior prediction (choice + downweighted RT).
- Switch-centered belief updating and subjective-prior matching.
- Neural prior-readout alignment and behavior-matched neural comparison.
- Reproducible manifests, frozen config, scripts, figures, short report.

### Non-goals (v1)

- Meta-RL RNN.
- Full brain-wide map as a required deliverable.
- VISp (or other regions) as a v1 success criterion (optional control only).
- Continuous / multi-bit contrast coding beyond binary `contrast_high`.
- Separate binary `incorrect` channel (incorrect = absence of reward).
- Treating true `probabilityLeft` as the mouse’s subjective prior in history-only analyses.

---

## 4. Dataset assumptions

### Source

- Public Alyx / ONE: `https://openalyx.internationalbrainlab.org`.
- Behavior tables via `SessionLoader.load_trials()` (fallback: `one.load_object(..., 'trials')`).
- Pin download date / Alyx revision in manifests.

### Nested scopes

1. **Behavior-core:** almost-perfect QC sessions (see QC below).
2. **Neural-intersect:** behavior-core ∩ sessions with usable units in **MOs** and/or **vlOFC / orbvl**.

### Session inclusion (behavior-core)

A session is eligible only if:

1. Public ONE access and complete trials table including at least:  
   `contrastLeft`, `contrastRight`, `choice`, `feedbackType`, `probabilityLeft`, `stimOn_times`, `goCue_times`, `response_times`, `feedback_times`.
2. Biased-block protocol with `probabilityLeft ∈ {0.2, 0.5, 0.8}` present.
3. At least one left-bias and one right-bias block in addition to any 0.5 blocks.
4. ≥ 400 completed choice trials before fine QC (soft gate; tighten only with documented reason).
5. **Almost-perfect timing QC (hard gate):**
   - ≥ **95%** of completed choice trials pass trial rules 1–4 below (before RT percentile trim).
   - After RT percentile trim, ≥ **90%** of completed choice trials remain.
   - Otherwise **drop the entire session** (do not keep messy sessions with heavy trial dropping).

### Trial inclusion (within retained sessions)

A trial enters training / RT scoring only if:

1. Finite `stimOn_times`, `goCue_times`, `response_times`, `feedback_times`.
2. Monotonic: `stimOn ≤ goCue ≤ response ≤ feedback`.
3. `RT = response_times - goCue_times > 0` (fallback definition only if goCue missing: `response - stimOn`).
4. `choice ∈ {-1, +1}` (exclude no-go from fit; may count for QC only).
5. RT within session **[1st, 99th]** percentile among trials that already passed 1–4.

### Empirically verified from inspection (3 public sessions)

- Absolute contrasts observed: `{0, 0.0625, 0.125, 0.25, 1.0}` (no `0.5` in those sessions).
- Stim side encoded cleanly via NaN pattern on `contrastLeft` / `contrastRight`.
- Some sessions fail almost-perfect timing QC (many nonpositive RTs / non-monotonic events) and must be excluded under the hard gate.

### Flagged assumptions

- Contrast set may include `0.5` in other sessions; `contrast_high` rule still uses `{0.25, 0.5, 1.0}` as high if present.
- Exact Alyx revision string is pinned at first full manifest build (not invented here).

---

## 5. Model definitions

### Shared scientific role

All three models predict **choice** and **RT**, expose a **latent prior readout** \(q_t\) in history-only, and are scored with the same behavioral and (when matched) neural metrics.

### 5.1 Standard task-trained RNN

- Input: sequence of 100 ms binary vectors for the current trial; **bin 0 aligned to `stimOn_times`**.
- Dynamics: generic recurrent update (GRU or LSTM; choice fixed in frozen config at implementation time).
- Trained end-to-end on choice + downweighted RT.
- Latent prior: explicit scalar readout from hidden state (trained or linearly decoded; method frozen in config).

### 5.2 Predictive-coding RNN

- **Same external channels and binning** as the standard RNN.
- Dynamics: recurrent state updated with explicit **prediction / prediction-error** structure (not only a generic RNN cell).
- Same targets and evaluation interface.
- Latent prior: explicit scalar readout analogous to the standard RNN.

### 5.3 Bayesian / explicit online-inference model

- Trial-level (not forced into 100 ms bins).
- Same causal information content as RNNs at decision time.
- Maintains an explicit latent prior / belief state updated from history and feedback.
- Outputs choice probability and RT distribution parameters.
- Latent prior \(q_t\): the model’s explicit belief (e.g. posterior mean of P(right) or P(left)—side convention frozen in config).

### Fairness contract

**Held fixed across all three**

- Causal information (no future leakage).
- Conditions: fixed-prior / full-info / history-only (strictly separate).
- Targets: choice + downweighted RT.
- Trial/session QC; split protocol; metrics; switch window.

**Held fixed across the two RNNs**

- 100 ms bins; bin 0 = stim onset.
- Binary channel set and timing rules.
- Shared capacity / optimization search grid and early-stopping on val choice NLL.

**Allowed to differ**

- Internal dynamics (generic vs PC vs Bayesian inference).
- Bayesian need not use binary event channels.
- Internal hyperparameters specific to PC or Bayesian forms.
- Internal parameterization of prior readout (must still expose comparable \(q_t\)).

---

## 6. Task conditions

| Condition | Prior information | Dataset slice | Role |
|-----------|-------------------|---------------|------|
| **Fixed prior** | No block-prior channel | Only `probabilityLeft = 0.5` trials/blocks | Constrained control |
| **Full-information** | Oracle prior channel / true prior feature | All QC trials | Oracle control |
| **History-only** | No oracle prior | All QC trials; prior must be inferred | **Primary scientific condition** |
| **Switch-centered eval** | (evaluation mode) | Switches in `probabilityLeft` | Belief-updating focus |

Conditions must not be mixed during preprocessing labels, training runs, or metric aggregation.

---

## 7. Inputs and outputs

### 7.1 Official IBL fields (raw)

Defined by IBL (not this project):  
`contrastLeft`, `contrastRight`, `choice` (−1 right / +1 left / 0 no-go; IBL ALF spatial), `feedbackType` (+1/−1/0), `rewardVolume`, `probabilityLeft`, `stimOn_times`, `goCue_times`, `response_times`, `feedback_times`, plus optional `stimOff_times`, `firstMovement_times`, etc.

Project binary target: `choice_right = 1` iff `choice == -1`.

### 7.2 RNN binary channels (project-derived)

At each 100 ms step, channels are 0/1. **Timestep 0 = stim onset.**

Common channels (all conditions), when causally available:

- `stimulus_right` — 1 if `contrastRight` finite
- `contrast_high` — 1 if abs contrast ∈ `{0.25, 0.5, 1.0}`; else 0 for `{0, 0.0625, 0.125}`
- `delay_phase` — active after stim-related onset window / until response window per frozen phase map
- `response_window` — 1 while response is allowed (from goCue through response, per frozen map)
- `response_made` — 1 only in the bin containing the response event
- `reward` — 1 iff `feedbackType == +1` (else 0); **no separate incorrect channel**
- `prev_choice_right`, `prev_correct`, `prev_fast_rt` (prev RT below session median)
- optional `trial_start` / `session_start`

Full-information only: `oracle_prior_right` derived from true `probabilityLeft` (binarization rule frozen in config).  
History-only / fixed-prior: no oracle prior channel.

**Causal rule at stim onset:** may use current stim side/contrast and previous-trial summary; may **not** use current response, current correctness/reward, or future trials.

### 7.3 Bayesian trial-level inputs

At trial \(t\): current stim side & contrast; previous choice; previous correctness/reward; previous RT summary; plus true block prior only in full-information.

### 7.4 RT definition

- Primary: `RT = response_times - goCue_times`
- Fallback: `response_times - stimOn_times` if goCue missing
- Likelihood: log-normal or Gaussian on `log(RT)`
- Input history: binary `prev_fast_rt` only (no current-trial RT as input)

### 7.5 Outputs

**Primary training targets**

- Choice: P(left/right)
- RT: distributional / transformed continuous target  
  Loss: \(\mathcal{L} = \mathcal{L}_{choice} + \lambda \mathcal{L}_{RT}\) with \(\lambda < 1\)

**Derived readouts**

- Psychometrics
- Latent prior \(q_t\) (history-only)

**Optional diagnostic heads:** off by default in v1.

---

## 8. Evaluation criteria

### 8.1 Behavioral (primary scientific focus: history-only; also report controls)

1. **Choice:** held-out log-likelihood and accuracy; also normalized pseudo-\(R^2\). **Primary for model selection.**
2. **Psychometrics:** P(right) vs signed contrast, overall and by block; bias, slope, lapse. Pool for plots; per-session/subject for tests.
3. **RT:** held-out RT log-likelihood (secondary); median/IQR by stim strength and block.
4. **Switch-centered adaptation:** mouse and model trajectories for P(choice aligned with new block) and inferred prior.
5. **Block-switch asymmetry:** 0.2→0.8 vs 0.8→0.2.
6. **Adaptation rate:** exponential/logistic fit; half-life (trials) with bootstrap CIs and GOF.

**Switch window**

- Align to first trial of new `probabilityLeft`.
- Window: **−10 to +30** trials.
- Prefer ≥10 pre / ≥20 post after QC; allow ≥8 / ≥16 but **flag** and report sensitivity excluding relaxed switches.
- Do not tune hyperparameters on post-switch trials that will be used for switch-centered evaluation.
- P(choice aligned with new block) = model probability for the side favored by the new block.

### 8.2 Subjective prior (history-only)

- **Mouse prior** \(\hat{p}_t\): latent behavioral estimate from choice history and feedback (IBL prior-paper template). **Not** true `probabilityLeft`.
- **Oracle prior:** true `probabilityLeft` — full-information only.
- **Model–mouse match:** correlation / RMSE of \(q_t\) vs \(\hat{p}_t\); switch-centered trajectory MSE; update asymmetry.

### 8.3 Neural (v1)

- **Regions:** MOs and vlOFC/orbvl only (VISp optional control).
- **Sessions:** neural-intersect.
- **Primary gate:** trial-wise **variance explained** of region prior-related neural readout by model \(q_t\) (history-only; emphasize −10…+30).
- **Companion:** trial-wise correlation.
- **Report-only:** RSA, trajectory geometry, asymmetry.

### 8.4 Behavior-matched neural comparison (confirmatory)

1. On held-out history-only data, form a behavior-matched set: choice within \(\epsilon\) of best held-out choice LL/accuracy, **and** RT above a secondary floor (choice primary; RT cannot replace choice).
2. Per class: best choice-LL checkpoint; include in confirmatory neural claims only if inside global choice \(\epsilon\)-ball.
3. Among matched models, compare prior-readout VE in MOs and vlOFC separately.
4. Advantage survives if paired bootstrap or session-level permutation 95% CI for VE difference excludes 0 after Holm correction across regions.
5. Always report unmatched (exploratory) and matched (confirmatory) rankings.

---

## 9. Deliverables for v1

### Docs

- `docs/spec.md` (this file)
- `docs/plan.md`
- `reports/v1_report.md`

### Config

- `configs/frozen_v1.yaml` — single machine-readable freeze for inclusion, channels, RT, QC, switch window, outputs, neural rules

### Data

- Manifests: behavior-core; neural-intersect
- `data/processed/` trial tables, RNN event-bin datasets, Bayesian trial-level tables

### Code

- Download / QC / preprocess; event-bin builder; Bayesian table builder
- Train/eval for three models; latent prior extraction
- Behavioral + switch analyses; neural VE + matching; figure scripts

### Figures (minimum)

- Psychometrics by block; RT summaries
- Switch-centered choice + prior trajectories (mouse + 3 models)
- Prior-match scores
- Neural VE unmatched vs matched; survival-test summary

---

## 10. Open questions and risks

| ID | Item | Status |
|----|------|--------|
| R1 | Exact GRU vs LSTM for standard RNN; exact PC equations | Open — freeze in `frozen_v1.yaml` before training Phase |
| R2 | Exact Bayesian generative form (beta-Bernoulli vs HMM block prior, etc.) | Open — choose simplest explicit online prior update matching causal inputs |
| R3 | Exact \(\epsilon\) for choice-matching band and RT floor | Open numeric — set from pilot behavioral runs, then freeze |
| R4 | Exact peri-stim / pre-choice neural window and unit QC | Open — freeze after first neural load |
| R5 | Phase map durations (`delay_phase`, `response_window`) in 100 ms bins | Partially constrained by events; freeze after event-delta audit on behavior-core |
| R6 | Sessions with poor timing sync | Mitigated by almost-perfect session gate; may shrink N |
| R7 | Mouse prior estimator hyperparameters | Open — prefer compact published-style latent bias model; freeze before neural compare |
| R8 | Alyx revision / eid list size | Open until first manifest build |

**Risk posture:** Prefer shrinking scope and freezing configs over inventing missing neuroscience details in code.

---

## 11. Milestone order (contract)

1. Small download + QC  
2. Preprocessing freeze (`configs/frozen_v1.yaml`)  
3. Dataset builders  
4. Model training  
5. Behavioral evaluation  
6. Latent prior readout  
7. Neural matching  
8. Report + figures  

**First practical milestone:** small data download + QC/preprocessing pipeline (no model training).
