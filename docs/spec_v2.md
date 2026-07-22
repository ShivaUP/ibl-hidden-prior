# Project specification (v2 freeze)

**Status:** Frozen (user-confirmed 2026-07-21). Do not change without a YAML/spec bump.  
**Supersedes for primary science:** mouse-supervised v1 model ranking.  
**v1 artifacts** remain under `reports/` as legacy; do not delete without note.

---

## 1. Objective

Train models **only on synthetic IBL-like sessions**, compare them on **held-out synthetic data from the same generator**, then evaluate **frozen** weights on **real mouse behavior** via a shared trial mapper.

### Success criteria

| Priority | Criterion |
|---|---|
| **Primary** | Fair ranking on synthetic held-out: choice CE/accuracy, psychometric-by-block, zero-evidence / pre-stim prior probes, switch-centered adaptation |
| **Secondary** | Real-behavior transfer (no fine-tune): score vs **correct side** and vs **mouse choice**; psychometrics + switches |
| **Not v2 success** | Neural VE / MOs–vlOFC claims (parked) |

---

## 2. Model set (replace v1 trio)

| ID | Model | Source | Test-time computation | Training |
|---|---|---|---|---|
| `tanh_bptt` | Vanilla tanh RNN | Adapt Kyan standard | tanh RNN | BPTT |
| `tanh_pc` | Same tanh RNN | Adapt Kyan PC agent | **Identical** tanh RNN | Predictive-coding **credit assignment** (local inference + local synaptic updates) |
| `gru` | GRU | Adapt Shrijana | GRU cell | BPTT |
| `bayes` | Explicit online prior | New / port of project Bayes idea | Explicit \(q_t\) + stimulus readout | Likelihood / BPTT-through-params (**no** Bayes+CA twin) |

**PC naming:** `tanh_pc` = credit-assignment training, **not** v1 PE-dynamics cell.

**Adaptation rule:** Kyan/Shrijana code is a **reference**, not frozen. Rewrite as needed for empirical ticks, shared schema, and project layout.

**Parked:** v1 PyTorch GRU / PE-PC / mouse-fit Bayes as primary; Bayesian+CA; meta-RL; neural-as-gate.

---

## 3. Synthetic task (closer to real than Kyan)

### Session / block structure (empirical)

Fit generator stats from **behavior-core** (and document any QC-pass extension):

- Priors \(\in \{0.2, 0.5, 0.8\}\)
- Block **lengths** and **transition structure** from data (not forced 0.2↔0.8 alternation only)
- Contrast levels `{0, 0.0625, 0.125, 0.25, 0.5, 1.0}` with **empirical frequencies**
- Session length from empirical n_trials distribution

Synthetic ≠ replay of real eid sequences (keeps transfer meaningful).

### Within-trial (empirical phase ticks)

- Tick = **100 ms**
- Phase **order** (Kyan causal order): baseline → stim → delay/blank → go → response → feedback
- Phase **durations** = integer ticks from empirical event-delta medians/distributions (`go−stim`, `resp−go`, `fb−resp`, stim duration, etc.)
- Loss / readout only on **response-phase** tick(s)
- Optional synth sensory noise on visual channels; **no** noise when encoding real trials

### Sensory / channel coding

**Visual (locked):** side × magnitude (Kyan-style), not binary high/low, not one-hot-only:

- Right, contrast \(c\): `[c, 0]` on `(visual_right, visual_left)`
- Left, contrast \(c\): `[0, c]`

**Other channels (shared):** `go_cue`, `action_left`, `action_right`, `rewarded`, `not_rewarded` (explicit incorrect channel allowed in v2 synth schema).

---

## 4. Train / eval / transfer protocol

| Stage | Protocol |
|---|---|
| Synth train | Target = **correct stim side**; teacher-forced action/±reward (+ configurable error rate) |
| Synth held-out | Closed-loop feedback (model’s own action); primary ranking |
| Real transfer | Frozen weights; map ALF trials → same tick/channel schema; feedback ticks use **mouse** action/outcome; score **correct side** and **mouse choice** |
| Fine-tune on mice | **Out of v2** |
| RT head / joint RT loss | **Out of v2 primary** |

### Fairness

- Shared generator, schema, splits protocol, choice objective
- RNN hidden size default **48**; Bayes = explicit state + comparable readout
- Session-contiguous state for RNN-family main condition
- Intentional difference: BPTT vs PC-CA for the tanh pair only

---

## 5. Repository organization (v2)

### Numbered scripts

All user-facing scripts under `scripts/` use a **numeric prefix** matching pipeline order. Canonical list: `scripts/README.md`.

| Prefix | Role |
|---|---|
| `00_` | ONE connectivity smoke |
| `01_` | Session QC → behavior-core manifest |
| `02_` | Event-delta audit (phase ticks) |
| `03_` | Processed trials (behavior-core) |
| `04_` | Fit empirical synth stats + `synthetic_v2.yaml` |
| `05_` | Build synthetic datasets |
| `06_` | Map real behavior → shared tick tensors |
| `07_` | Train models (`--model` or `--all`) |
| `08_` | Eval synthetic held-out (primary) |
| `09_` | Eval real transfer (secondary) |
| `10_` | Make figures |

v1 phase scripts are **removed**, not archived under `09_legacy_*`.

### Outputs & naming

```
artifacts/v2/
  models/{model_id}/...
  synthetic/{split}/...
reports/v2/
  metrics/{stage}_{model_id}.json
  tables/...
  figures/
    overview/...
    by_model/{model_id}/...
    comparison/...
```

- `model_id` ∈ `{tanh_bptt, tanh_pc, gru, bayes}`
- Stage ∈ `{synth_heldout, real_transfer}`
- No silent overwrites: stamp or `run_id` in config when needed

### Figures (explicit contract)

**Per-model multi-panel figure** (one PNG/PDF per model, several panels):

Include **Kyan-style diagnostics** (adapted to our generator):

1. Training curve (loss / PC energy as applicable)
2. Psychometric by block prior
3. Switch-centered zero-evidence (or pre-stim) belief
4. Example session: true prior vs model belief

**Comparison figures** (multi-model panels or grouped bars):

- Held-out synth accuracy / CE ranking
- History-gap / prior calibration ranking
- Real transfer: correct-side vs mouse-choice side-by-side

**README obligation:** for **each** figure file, document: path, panels, what question it answers, which script regenerates it.

---

## 6. README requirements (thorough)

README (v2 section) must include:

1. Scientific objective (synth-primary, real-transfer-secondary)
2. Model glossary (especially PC = credit assignment)
3. Directory map
4. End-to-end runbook with **numbered scripts in order**
5. Config pointers (`configs/synthetic_v2.yaml`, etc.)
6. Figure catalog (every figure)
7. What is legacy v1 vs v2
8. Data: what is gitignored; how to rebuild (`docs/DATA.md` updated)

---

## 7. Implementation order

1. Data prep: smoke → QC → event deltas → processed trials (`00`–`03`)  
2. Empirical stats + generator (`04`–`05`)  
3. Real mapper (`06`)  
4. Adapt/port four models + train (`07`)  
5. Synth held-out eval (`08`)  
6. Real transfer eval (`09`)  
7. Figure suite (`10`)  
8. README + `docs/spec_v2.md` + `scripts/README.md`

---

## 8. Open risks

| ID | Risk |
|---|---|
| V2-R1 | Empirical phase ticks lengthen sequences → NumPy BPTT/PC cost; may need truncation policy |
| V2-R2 | Bayes fairness vs hidden-48 RNNs (document capacity, don’t fake equality) |
| V2-R3 | Real mapper: stim/go in same 100 ms bin — phase painting rules must be explicit |
| V2-R4 | Teacher-forced vs mouse feedback distribution shift on transfer |
| V2-R5 | Script renumber vs old docs/links — v1 scripts deleted; update handoffs that still cite them |

---

## 9. Grill lock summary

- Objective: **A** (synth primary, real transfer secondary)  
- Models: tanh BPTT, tanh PC-CA, GRU, **Bayes only** (no Bayes+CA)  
- Generator: empirical blocks/contrasts/sessions  
- Within-trial: empirical 100 ms phase ticks  
- Contrast: side×magnitude `[c,0]`/`[0,c]`  
- Protocol: correct-side train; transfer scores correct-side + mouse choice; no mouse FT; no RT primary  
- Fairness: shared schema; Kyan/Shrijana **adapted**  
- Deliverables: numbered scripts; thorough README; explicit figure naming; per-model multipanel + Kyan diagnostics  

**Freeze confirmed.** Implementation follows §7; deviations require YAML/spec bump.
