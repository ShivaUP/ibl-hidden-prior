"""Region acronym helpers for neural analyses (belief / prior updating ROIs).

Primary analysis scope (v2 locked): MOs, ORBvl (vlOFC), ACAd, MOp — the four
highest-priority sites for subjective prior / belief updating from Findling et al.
Nature 2025 and related IBL maps.

Broader ROIs remain listed as optional for future expansion.

Papers:
- Findling et al., Nature 2025: https://www.nature.com/articles/s41586-025-09226-1
- IBL BWM, Nature 2025: https://www.nature.com/articles/s41586-025-09235-0
- IBL behavior, eLife 2021: https://elifesciences.org/articles/63711
"""

from __future__ import annotations

# Primary ROIs used by neural eval (analysis scope).
REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "MOs": ("MOs",),
    "vlOFC_orbvl": ("ORBvl",),
    "ACAd": ("ACAd",),
    "MOp": ("MOp",),
}

# Optional / deferred ROIs (documented; not in default neural VE loop).
REGION_ALIASES_OPTIONAL: dict[str, tuple[str, ...]] = {
    "CP": ("CP",),
    "VISp": ("VISp",),
    "LGd": ("LGd",),
    "SCm": ("SCm",),
    "GRN": ("GRN",),
    "ORBm": ("ORBm",),
    "PL": ("PL",),
    "ILA": ("ILA",),
}

NEURAL_REGIONS: tuple[str, ...] = tuple(REGION_ALIASES.keys())

REGION_TIERS: dict[str, str] = {
    "MOs": "primary_cortex",
    "vlOFC_orbvl": "primary_cortex",
    "ACAd": "primary_cortex",
    "MOp": "motor",
    "CP": "striatum",
    "VISp": "early_sensory",
    "LGd": "early_sensory",
    "SCm": "midbrain",
    "GRN": "hindbrain",
    "ORBm": "prefrontal_family",
    "PL": "prefrontal_family",
    "ILA": "prefrontal_family",
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


def atlas_query_acronyms() -> dict[str, list[str]]:
    """Flatten primary aliases for Alyx / BWM atlas queries."""
    return {name: list(acrs) for name, acrs in REGION_ALIASES.items()}
