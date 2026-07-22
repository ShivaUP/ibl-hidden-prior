# Handoff — v2 synth pipeline + scope correction (grill paused)

**Date:** 2026-07-21  
**Repo:** `/Users/shivamahdian/Desktop/Performance`  
**Next session focus:** Finish amending the freeze (resume grill), then restore v1-style analysis ladder on synth-trained models; retrain at full 60×24×929 exposure if current checkpoints are short runs.

---

## One-line status

v2 **synth-trained** model stack (`tanh_bptt` / `tanh_pc` / `gru` / `bayes`) and numbered scripts `00`–`10` exist; early synth/real accuracy figures exist; user corrected that **belief-updating, regimes, and neural comparison remain in scope** — grill for the amended freeze is **paused** mid-way.

---

## Do not re-decide (already locked this session)

| Topic | Decision |
|---|---|
| Training | Synth-only; empirical 100 ms phase ticks; side×magnitude contrast; no mouse fine-tune |
| Models | `tanh_bptt`, `tanh_pc` (= PC **credit assignment**, not v1 PE cell), `gru`, `bayes` only |
| Exposure | **60 epochs × 24 sessions × 929 trials** ≈ 1.34M trial-exposures (~3.9× Kyan); full session walked in BPTT chunks of 32; ticks unchanged |
| Corrected science objective (grill **A**) | **Primary** = real **history-only** behavior + belief-updating; synth held-out = sanity; neural + behavior-matched = **secondary restored** (not parked) |
| Regimes (grill **A**) | `history_only` / `full_information` / `fixed_prior` as **eval-time** conditions; one synth training task |
| Prior match (grill **A**) | Keep v1 mouse \(\hat{p}_t\) + model \(q_t\); switch −10…+30; asymmetry |
| Neural (grill **A**) | MOs + vlOFC; VE of neural prior readout by \(q_t\); behavior-matched confirmatory |
| RT (grill **A**) | Out of training and model selection; optional descriptive mouse RT only |

**Paused before:** deliverables / script restoration plan, writing amended `docs/spec_v2.md`, and implementing analysis scripts `11+`.

---

## Artifacts to read (do not duplicate)

| Path | Role |
|---|---|
| `docs/spec_v2.md` | Current freeze text — **partially wrong** (still says neural parked / synth-primary ranking). Must be amended after grill completes |
| `docs/spec.md` | v1 analysis contract to restore under synth-trained models |
| `configs/synthetic_v2.yaml` | Empirically fitted ticks + **60/24/929** train budget |
| `scripts/README.md` | Numbered `00`–`10` catalog |
| `src/models_v2/` | NumPy models + full-session chunked `train.py` |
| `src/synthetic/` | Generator, channels, real mapper |
| `reports/v2/` | Early metrics/figures (may be from **pre-full-exposure** or short trains — verify before citing) |
| `reports/v1_report.md` | Legacy v1 results; scripts cited there were **deleted** |

---

## What shipped this session

1. Synth pipeline: fit stats → datasets → four models → synth eval → real transfer → figures  
2. Scripts renumbered **`00`–`10`**; v1 phase train/eval/neural scripts **removed** (must be **rebuilt** for restored analyses, not resurrected as mouse-supervised v1)  
3. `03_build_processed_trials.py` = trials only (no v1 RNN bins / Bayes tables)  
4. Train bugfix: was truncating to one 24-trial window; now walks **full** session in chunks  
5. User asked how to read accuracy plots: see below

### Accuracy meaning (for next agent / user)

- **Held-out synth accuracy:** \(P(\text{argmax choice} = \text{correct stim side})\) on closed-loop synth.  
- **Real transfer — vs correct side:** same vs true stim side on real trials.  
- **Real transfer — vs mouse choice:** vs mouse left/right.  
Not CE, not prior match, not neural VE.

---

## Open / next actions (in order)

1. **Resume grill** (`skills` / `.cursor/skills/grill-me`): deliverables + numbered scripts for regimes / prior / neural / behavior-match; then write **amended** `docs/spec_v2.md` and get explicit freeze confirm.  
2. **Retrain** at full budget if checkpoints are short:  
   `python scripts/07_train_model.py --all`  
   then `05` (if held-out needs 48 sess) → `08` → `09` → `10`.  
3. **Restore analysis ladder** on frozen synth models (new scripts, e.g. `11_`…):  
   - regime eval (history / oracle / fixed)  
   - mouse prior + switch-centered match  
   - neural intersect + VE + behavior-matched survival  
4. Update README / `docs/DATA.md` after freeze amendment.  
5. Do **not** treat current ranking figures as final science until full exposure + belief/neural eval exist.

### Suggested commands

```bash
cd /Users/shivamahdian/Desktop/Performance && source .venv/bin/activate
python -c "from src.models_v2.train import exposure_summary; from src.synthetic.schema import load_synthetic_config; print(exposure_summary(load_synthetic_config()))"
python scripts/05_build_synthetic_datasets.py   # if eval sess count changed
python scripts/07_train_model.py --all            # long
python scripts/08_eval_synth_heldout.py
python scripts/09_eval_real_transfer.py
python scripts/10_make_figures.py
```

---

## Suggested skills

| Skill | When |
|---|---|
| `.cursor/skills/grill-me/SKILL.md` | Resume one-question freeze amendment (deliverables next) |
| `skills/hand-off.md` | End of next session |
| `.cursor/skills/create-rule/SKILL.md` | Only if project rules need update after amended freeze |

---

## Risks / traps

- `docs/spec_v2.md` still says neural parked — **do not implement from that text without amendment**.  
- Deleting v1 scripts ≠ deleting science requirements; rebuild under synth-trained interface.  
- PC is **credit assignment**; do not confuse with v1 PE-dynamics.  
- IBL `choice`: −1 = right (already fixed in v1 path; keep in mappers).  
- Full 60×24×929 NumPy PC will be slow — expect long wall time.
