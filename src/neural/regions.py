"""Region acronym helpers for v1 neural analyses (MOs, vlOFC/ORBvl)."""

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
