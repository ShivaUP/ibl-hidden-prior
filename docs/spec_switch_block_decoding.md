# Spec: MLP switch-centered block decoding (three panels)

Frozen decisions (2026-07-23 grilling; layout A locked).

## Objective

Compare how well an MLP can decode true biased-block identity (`P(right)=0.2` vs `0.8`) around genuine block switches from:

1. **A1 — History-only synth model latents** (capacity / latent readability).
2. **A2 — Real shared cohort scalars** — mouse subjective prior \(\hat p_t\) and each model’s zero-evidence belief \(q_t\) (primary Q2 three-way probe).
3. **A3 — Real neural prior readouts** — CV Ridge OOF \(\hat u_t\) by primary ROI.

## Scope

- Regime: **history_only** only.
- Models: `tanh_bptt`, `tanh_pc`, `gru`, `gru_pc`.
- Neural ROIs: `MOs`, `vlOFC_orbvl`, `ACAd`, `MOp` (`NEURAL_REGIONS`).
- Shared cohort: `data/manifests/shared_behavior_neural_eids.json` (n=8; same order as real tick / rollout sessions).
- Switch window: **−30 … +30** trials (isolated genuine 0.2↔0.8 switches only).
- Decoder: one-hidden-layer **MLP** (same family as existing switch-block decode tooling).

## Assumptions

- A1 features = concatenated zero-current-evidence latent trajectory (all within-trial ticks × hidden size); probe zeros visual/action/reward channels and keeps only the go cue.
- A2 features = scalar trial-level series: `mouse_prior_hat` (behavior-fit) and model `belief` / `zero_evidence_p_right` from `artifacts/v2/real/regimes/history_only/{model}/rollout.npz`.
- A3 features = scalar trial-level OOF neural prior readout (`fit_prior_readout`), one curve per ROI.
- A1 uncertainty = ±1 sample SD across available task-model seeds (canonical checkpoint if no seed replicates).
- A2–A3 uncertainty = ±1 sample SD across leave-one-session-out held-out sessions.

## Deliverables

- `scripts/16_plot_mlp_switch_block_decoding.py`
- `src/models_v2/block_decode.py`
- Figure: `reports/v2/figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png`
- Copy: project-root `mlp_rnn_vs_pc_switch_decoding.png`
- Metrics under `reports/v2/switch_block_decoding/`

## Open risks

- Few neural sessions per ROI → wide session SD and sparse far-window switches.
- Scalar A2/A3 features make the MLP nearly a calibrated threshold; kept for interface parity with A1 and for direct mouse vs model vs neural comparison.
- Without `model_seed_replicates/`, A1 shading collapses to zero width.
