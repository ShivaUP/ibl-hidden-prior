# Real behavioral evaluation (v2)

## Scoring rule (locked)

**Train and test only on correctness** — the correct stimulus side.

- Training (synth): cross-entropy vs correct side.
- Synth held-out / regimes: accuracy + CE vs correct side.
- Real transfer: accuracy + CE vs correct side **only**.
- **Not** scored against mouse choice or mouse P(right).
- Figures do **not** overlay mouse psychometrics or mouse choices.

Model curves such as “zero-evidence P(right)” are the **network’s** preference under no stimulus — a diagnostic of learned prior use — not a mouse-matching target.

## What the network is tested on

Frozen synth-trained models are evaluated on the **behavior-core** set:

- Manifest: `data/manifests/behavior_core_eids.json` → `data/manifests/real_v2_ticks.json`
- **10 QC-gated sessions** (not the full IBL corpus)
- Weights are never fine-tuned on mice

## History input on real sessions (not a score)

On real rollouts, action/reward **input channels** still follow what happened in that session (mouse action + outcome). That is history the network sees so its state can update. It is **not** the training or ranking target.

Eval: `python scripts/11_eval_regimes.py --domain real`

## How metrics are pooled

| Metric | Pooling | Sensitive to different block *orders*? |
|---|---|---|
| Accuracy / CE vs correct side | All valid trials across the 10 sessions | **No** |
| Psychometric by block prior | Trials conditioned on prior + signed contrast | **No** |
| Switch-centered zero-evidence | Switches aligned at \(t=0\) | **No** |
| Example session panel | One session (longest), raw trial index | Illustrative only |

## What is *not* claimed

- Not matching mouse choice or mouse subjective prior.
- Not a subject-balanced “all mice” claim.
- Not absolute-time averages of belief across sessions.
