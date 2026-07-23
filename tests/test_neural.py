"""Tests for neural helpers."""

from __future__ import annotations

import numpy as np

from src.neural.prior_readout import fit_prior_readout, variance_explained
from src.neural.regions import unit_in_region


def test_unit_in_region():
    assert unit_in_region("MOs", "MOs")
    assert unit_in_region("ORBvl", "vlOFC_orbvl")
    assert unit_in_region("ACAd", "ACAd")
    assert unit_in_region("MOp", "MOp")
    assert not unit_in_region("VISp", "MOs")


def test_variance_explained_perfect():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(variance_explained(y, y) - 1.0) < 1e-9


def test_fit_prior_readout_smoke():
    rng = np.random.default_rng(0)
    true = rng.random(80)
    counts = np.column_stack([true + 0.05 * rng.normal(size=80), rng.normal(size=80)])
    out = fit_prior_readout(counts, true, n_splits=4)
    assert out["n"] == 80
    assert out["ve_cv"] > 0.5
