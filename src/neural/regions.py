"""Region acronym helpers for neural analyses.

v1 regions: MOs, vlOFC/ORBvl.
Paper-highlighted choice-selective regions (IBL et al. 2025, Nature):
  Thalamus: CL, SPF
  Midbrain:  SCm, MRN, SNr, RPF, NPC
  Hindbrain / pons / medulla / cerebellar nuclei: GRN, IRN, SOC, VII, TRN, FOTU
"""

from __future__ import annotations

# Allen CCF acronyms used by IBL channels/clusters tables.
# Spec region "vlOFC / orbvl" maps to Allen ORBvl (ventrolateral orbitofrontal).
REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "MOs": ("MOs",),
    "vlOFC_orbvl": ("ORBvl",),
}

# Optional control
REGION_ALIASES_OPTIONAL: dict[str, tuple[str, ...]] = {
    "VISp": ("VISp",),
}

# ---- Paper-highlighted choice-selective regions (IBL 2025 Nature Fig 5) ----
# Keys are display names; values are Allen CCF acronym prefixes to match.
CHOICE_REGIONS: dict[str, tuple[str, ...]] = {
    # Thalamus
    "CL":   ("CL",),
    "SPF":  ("SPF",),
    # Midbrain
    "SCm":  ("SCm",),
    "MRN":  ("MRN",),
    "SNr":  ("SNr",),
    "RPF":  ("RPF",),
    "NPC":  ("NPC",),
    # Hindbrain / pons / medulla / cerebellar nuclei
    "GRN":  ("GRN",),
    "IRN":  ("IRN",),
    "SOC":  ("SOC",),
    "VII":  ("VII",),
    "TRN":  ("TRN",),
    "FOTU": ("FOTU",),
}

# ---- Project locked ROIs (primary analysis scope) --------------------------
# Frontal / motor cortical regions. MOs + ORBvl were frozen in v1; ACAd + MOp
# added as locked ROIs for the shared-cohort analysis. Allen CCF acronyms.
PRIMARY_ROIS: dict[str, tuple[str, ...]] = {
    "MOs":   ("MOs",),    # secondary motor
    "ORBvl": ("ORBvl",),  # ventrolateral orbitofrontal = vlOFC
    "ACAd":  ("ACAd",),   # anterior cingulate, dorsal
    "MOp":   ("MOp",),    # primary motor
}

# Comparison / context regions
CONTEXT_REGIONS: dict[str, tuple[str, ...]] = {
    "MOs":  ("MOs",),
    "VISp": ("VISp",),
}

# Primary ROIs first, then secondary subcortical choice regions, then controls.
ALL_DECODE_REGIONS: dict[str, tuple[str, ...]] = {
    **PRIMARY_ROIS,
    **CHOICE_REGIONS,
    "VISp": ("VISp",),
}


def acronyms_for_spec_region(spec_name: str) -> tuple[str, ...]:
    if spec_name in REGION_ALIASES:
        return REGION_ALIASES[spec_name]
    if spec_name in REGION_ALIASES_OPTIONAL:
        return REGION_ALIASES_OPTIONAL[spec_name]
    return (spec_name,)


def unit_in_region(acronym: str, spec_region: str) -> bool:
    """True if a channel/cluster acronym belongs to the spec region."""
    a = str(acronym)
    for target in acronyms_for_spec_region(spec_region):
        if a == target or a.startswith(target):
            return True
    return False


def unit_in_any_decode_region(acronym: str) -> str | None:
    """Return the first matching decode-region key, or None."""
    a = str(acronym)
    for region, prefixes in ALL_DECODE_REGIONS.items():
        for p in prefixes:
            if a == p or a.startswith(p):
                return region
    return None
