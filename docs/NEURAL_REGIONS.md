# Neural regions of interest (belief / prior updating)

## Papers

| Paper | Link | Role |
|---|---|---|
| Findling et al., Nature 2025 | https://www.nature.com/articles/s41586-025-09226-1 | Primary source for subjective-prior anatomy |
| IBL BWM, Nature 2025 | https://www.nature.com/articles/s41586-025-09235-0 | BWM context; prior analyses in companion |
| IBL behavior, eLife 2021 | https://elifesciences.org/articles/63711 | Task + behavioral prior use |

## Locked primary ROIs (analysis scope)

| Spec name | Allen | Why |
|---|---|---|
| `MOs` | MOs | Secondary motor; strong prior decoding |
| `vlOFC_orbvl` | ORBvl | Ventrolateral OFC; high-level prior site |
| `ACAd` | ACAd | Dorsal ACC; named high-level cortical prior site |
| `MOp` | MOp | Primary motor partner of MOs |

Optional / deferred (documented in code, not in default VE): CP, VISp, LGd, SCm, GRN, ORBm, PL, ILA.

## Cohort

Shared behavior+neural sessions maximize **union** coverage of the four primary ROIs.
No single session has all four; per-region VE uses sessions that contain that region.

Rebuild (small cohort):
```bash
python scripts/12_build_neural_intersect.py --max-sessions 8 --skip-download-qc
```
