# Within-trial time steps and input channels (v2)

**Source of truth:** `configs/synthetic_v2.yaml` (fitted by `scripts/04_fit_synthetic_stats.py` from behavior-core trials + `reports/inspection/event_deltas.json`).  
**Code:** `src/synthetic/channels.py` (`PhaseTicks`, `paint_trial`).  
**Stats dump:** `data/manifests/synthetic_stats_v2.json`.

This document describes the **empirical** tick schedule and channel coding used for synthetic training and real-trial mapping. It is **not** Kyan’s fixed 8-step toy schedule; phase *order* matches Kyan, durations come from IBL medians.

---

## 1. Binning convention

| Quantity | Value | Origin |
|---|---|---|
| Tick / bin size | **100 ms** | Project freeze (same as v1) |
| Alignment | Tick 0 = start of **baseline** before stim | Not stim-aligned index 0 (differs from v1 RNN bins) |
| Loss / choice readout | Only on **response tick** | Same idea as Kyan step 6 |

Empirical event medians (behavior-core, seconds) → integer ticks via `round(Δt / 0.1)` with floors:

| Event delta | Median (s) | Tick rule used |
|---|---|---|
| `goCue − stimOn` | ~0.016 | → **0** ticks (same bin as stim) |
| `response − goCue` | ~0.413 | → **4** ticks |
| `feedback − response` | ~0.0001 | → **2** feedback ticks (minimum hold) |
| `stimOff − stimOn` | ~1.48 | → **15** stim-duration ticks (clipped by trial length) |
| Pre-stim baseline | (design) | **2** ticks (fixed; not an IBL event) |

---

## 2. Phase tick layout (current freeze)

From `phase_ticks` in `synthetic_v2.yaml`:

```
baseline_ticks                 = 2
go_offset_from_stim_ticks      = 0
response_offset_from_go_ticks  = 4
feedback_ticks                 = 2
stim_duration_ticks            = 15
```

Derived indices (`PhaseTicks`):

| Symbol | Formula | Current value |
|---|---|---|
| `stim_start` | `baseline_ticks` | **2** |
| `go_tick` | `stim_start + go_offset` | **2** |
| `response_tick` | `go_tick + response_offset` | **6** |
| `feedback_start` | `response_tick + 1` | **7** |
| `n_steps` | `feedback_start + feedback_ticks` | **9** |
| `stim_end_exclusive` | `min(stim_start + stim_duration, n_steps)` | **9** |

### Tick-by-tick schedule (one trial)

| Tick (0-index) | Phase | What is on |
|---|---|---|
| 0–1 | Baseline | All channels **0** (recurrent state carries history) |
| 2 | Stim + go (same bin) | Visual side×magnitude; `go_cue=1` |
| 3–5 | Stim continues | Visual held; go off |
| 6 | **Response / readout** | Visual **off** on this tick (Kyan convention); **loss target = correct stim side** |
| 7–8 | Feedback | `action_left` or `action_right`; `rewarded` xor `not_rewarded` |

Because `stim_duration_ticks=15` but `n_steps=9`, stimulus is clipped to the trial: visual is painted on ticks `[2,3,4,5]` (skipping response tick 6).

---

## 3. Input channels (length 7)

Order is fixed in `CHANNEL_NAMES` / `N_INPUTS`:

| Index | Name | Type | Meaning |
|---|---|---|---|
| 0 | `visual_right` | continuous | Contrast magnitude on **right** (else 0) |
| 1 | `visual_left` | continuous | Contrast magnitude on **left** (else 0) |
| 2 | `go_cue` | binary | 1 only on `go_tick` |
| 3 | `action_left` | binary | Chosen/teacher action left (feedback ticks) |
| 4 | `action_right` | binary | Chosen/teacher action right (feedback ticks) |
| 5 | `rewarded` | binary | Correct outcome on feedback ticks |
| 6 | `not_rewarded` | binary | Incorrect outcome on feedback ticks |

### Sensory coding (locked)

For stimulus side \(s\) and absolute contrast \(c\):

- Right stim: `(visual_right, visual_left) = (c, 0)`
- Left stim: `(visual_right, visual_left) = (0, c)`

Synth training may add Gaussian noise (`sensory_noise_std_synth: 0.15`) to the two visual channels. **Real transfer:** noise off.

### Feedback coding

On feedback ticks only:

- Exactly one of `action_left` / `action_right` is 1.
- Exactly one of `rewarded` / `not_rewarded` is 1 (`rewarded` iff action == correct side).

Train: teacher-forced action with configurable error rate (`training_feedback_error_rate: 0.2`).  
Synth eval: closed-loop (model’s own action).  
Real transfer: mouse action + mouse reward.

---

## 4. Session / block generative distributions (from behavior)

Fitted from behavior-core pooled trials (`synthetic_stats_v2.json`):

### Block prior (`probabilityLeft`)

- Levels: `{0.2, 0.5, 0.8}` (IBL). Generator uses \(P(\text{stim right}) = 1 - \texttt{probabilityLeft}\).
- Session start: empirically almost always **0.5** in this cohort (`session_start_probability_left`).
- Transitions: empirical row-stochastic matrix over \(\{0.2,0.5,0.8\}\) (see YAML `block_transition_probability_left`). In this fit, 0.2↔0.8 dominate after leaving 0.5.

### Block length

- Empirical PMF over lengths clipped to **[10, 100]** trials (`block_length.values` / `probabilities`).
- Median length ≈ **45** trials.

### Contrast

Observed levels and frequencies in behavior-core (this fit; no 0.5 contrast in these sessions):

| Contrast | Empirical P |
|---|---|
| 0.0 | ~0.197 |
| 0.0625 | ~0.202 |
| 0.125 | ~0.213 |
| 0.25 | ~0.195 |
| 1.0 | ~0.194 |

### Session length

- Synth default: median completed trials ≈ **929** (`trials_per_session_default`).
- Train walks the **full** session in BPTT chunks of 32 trials (60 epochs × 24 sessions).

---

## 5. Diagnostics that depend on this schedule

Kyan/Shrijana multipanel figures need fields produced by `rollout_closed_loop`:

1. **Psychometric:** `p_choice_right` vs signed contrast, stratified by true block `P(right) ∈ {0.2, 0.8}`.
2. **Switch belief:** **counterfactual** `zero_evidence_p_right` — from the post-baseline state, run to the response tick with **visual = 0** but **go cue on**; average around switches (−20…+30), split 0.2→0.8 vs 0.8→0.2.
3. **Example session:** step plot of true `P(right)` vs `zero_evidence_p_right` (not the stim-path choice probability).

Recompute rollouts after code changes:

```bash
python scripts/08_eval_synth_heldout.py
python scripts/10_make_figures.py
```

---

## 7. PC training note

`tanh_pc` (credit-assignment) is unstable / fails to learn block history when trained on full empirical session length (~929 trials). Config uses **`pc_trials_per_session: 240`** (Kyan-scale) with the same empirical phase ticks and block/contrast stats. BPTT/GRU/Bayes still use 929.

## 8. Eval regimes (figures)

| Regime | Definition in v2 |
|---|---|
| `history_only` | Closed-loop; no oracle (primary) |
| `full_information` | Same dynamics; **eval-time** additive logit bias from log prior odds (`fi_oracle_logit_gain`) |
| `fixed_prior` | Eval only on trials with true \(P(\text{right})\approx 0.5\) |

Per-regime multipanels: `reports/v2/figures/by_model/{model}/{regime}/multipanel_diagnostics.png`  
Regenerate: `python scripts/11_eval_regimes.py && python scripts/10_make_figures.py`

Left stimulus, contrast `0.25`, correct left action (no sensory noise), current phase ticks:

| Tick | Event | VR | VL | go | aL | aR | rew | ¬rew | target |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| 1 | baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| 2 | stim+go | 0 | 0.25 | 1 | 0 | 0 | 0 | 0 | — |
| 3 | stim | 0 | 0.25 | 0 | 0 | 0 | 0 | 0 | — |
| 4 | stim | 0 | 0.25 | 0 | 0 | 0 | 0 | 0 | — |
| 5 | stim | 0 | 0.25 | 0 | 0 | 0 | 0 | 0 | — |
| 6 | response | 0 | 0 | 0 | 0 | 0 | 0 | 0 | LEFT |
| 7 | feedback | 0 | 0 | 0 | 1 | 0 | 1 | 0 | — |
| 8 | feedback | 0 | 0 | 0 | 1 | 0 | 1 | 0 | — |

Regenerate a live table anytime:

```bash
python -c "from src.synthetic.channels import PhaseTicks, paint_trial, CHANNEL_NAMES; from src.synthetic.schema import LEFT, load_synthetic_config; p=PhaseTicks.from_config(load_synthetic_config()); x,y=paint_trial(side=LEFT,contrast=0.25,action=LEFT,rewarded=True,phase=p); print('n_steps',p.n_steps); print(' '.join(CHANNEL_NAMES)); print(x); print('targets',y)"
```
