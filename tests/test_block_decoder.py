"""Tests for the block prior logistic regression decoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.block_decoder import (
    BLOCK_LEFT,
    BLOCK_RIGHT,
    BLOCK_UNBIASED,
    fit_block_decoder,
    label_blocks,
)


def test_label_blocks_known_values():
    p = np.array([[0.2, 0.5, 0.8], [0.2, 0.8, 0.5]])
    labels = label_blocks(p)
    assert labels[0, 0] == BLOCK_LEFT
    assert labels[0, 1] == BLOCK_UNBIASED
    assert labels[0, 2] == BLOCK_RIGHT
    assert labels[1, 0] == BLOCK_LEFT
    assert labels[1, 1] == BLOCK_RIGHT


def test_label_blocks_unknown_returns_minus_one():
    p = np.array([0.3, 0.6, 0.9])
    labels = label_blocks(p)
    assert np.all(labels == -1)


def test_label_blocks_tolerant():
    p = np.array([0.19, 0.21, 0.49, 0.51, 0.79, 0.81])
    labels = label_blocks(p, atol=0.05)
    expected = [BLOCK_LEFT, BLOCK_LEFT, BLOCK_UNBIASED, BLOCK_UNBIASED, BLOCK_RIGHT, BLOCK_RIGHT]
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("binary", [True, False])
def test_fit_block_decoder_perfect_signal(binary):
    """Decoder should achieve near-100% accuracy with perfectly separable features."""
    rng = np.random.default_rng(0)
    n = 300
    # Left block: feature ~ N(-3, 0.3), right block: feature ~ N(+3, 0.3)
    h_left = rng.normal(-3.0, 0.3, (n, 4))
    h_right = rng.normal(+3.0, 0.3, (n, 4))
    h_unbiased = rng.normal(0.0, 0.3, (n, 4))

    if binary:
        hidden = np.vstack([h_left, h_right])
        labels = np.array([BLOCK_LEFT] * n + [BLOCK_RIGHT] * n)
    else:
        hidden = np.vstack([h_left, h_unbiased, h_right])
        labels = np.array([BLOCK_LEFT] * n + [BLOCK_UNBIASED] * n + [BLOCK_RIGHT] * n)

    result = fit_block_decoder(hidden, labels, binary=binary, n_folds=3)
    assert result["accuracy_mean"] > 0.95, f"Expected >0.95, got {result['accuracy_mean']}"
    if binary:
        assert result["auroc_mean"] > 0.99, f"Expected AUROC >0.99, got {result['auroc_mean']}"


def test_fit_block_decoder_chance_signal(binary=True):
    """Decoder should be near chance with uninformative features."""
    rng = np.random.default_rng(1)
    n = 200
    hidden = rng.normal(0.0, 1.0, (n * 2, 1))  # no signal
    labels = np.array([BLOCK_LEFT] * n + [BLOCK_RIGHT] * n)

    result = fit_block_decoder(hidden, labels, binary=True, n_folds=3)
    # Should be close to 0.5 chance — allow generous margin
    assert result["accuracy_mean"] < 0.65, (
        f"Expected near-chance accuracy, got {result['accuracy_mean']}"
    )


def test_fit_block_decoder_excludes_unbiased_in_binary():
    """Binary mode should drop unbiased trials (label=1) automatically."""
    rng = np.random.default_rng(2)
    n = 100
    h_left = rng.normal(-2.0, 0.5, (n, 2))
    h_unbiased = rng.normal(0.0, 0.5, (n, 2))
    h_right = rng.normal(+2.0, 0.5, (n, 2))
    hidden = np.vstack([h_left, h_unbiased, h_right])
    labels = np.array([BLOCK_LEFT] * n + [BLOCK_UNBIASED] * n + [BLOCK_RIGHT] * n)

    result = fit_block_decoder(hidden, labels, binary=True, n_folds=3)
    assert result["n_samples"] == 2 * n  # unbiased dropped
    assert result["accuracy_mean"] > 0.85


def test_fit_block_decoder_returns_required_keys():
    rng = np.random.default_rng(3)
    hidden = rng.normal(0, 1, (60, 8))
    labels = np.array([BLOCK_LEFT] * 30 + [BLOCK_RIGHT] * 30)
    result = fit_block_decoder(hidden, labels, binary=True, n_folds=3)

    required = {
        "n_samples", "n_features", "n_folds", "binary",
        "accuracy_mean", "accuracy_std", "accuracy_folds",
        "confusion_matrix", "coef_mean", "intercept_mean",
        "auroc_mean", "auroc_std", "auroc_folds",
    }
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"
