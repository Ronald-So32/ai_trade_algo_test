"""Tests for VolatilityRegimeClassifier."""
import pytest
import numpy as np
import pandas as pd

from qrt.regime.volatility_regime import VolatilityRegimeClassifier, REGIME_LABELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_returns(sample_returns):
    """Aggregate sample asset returns into a single 'market' return series."""
    return sample_returns.mean(axis=1)


@pytest.fixture
def fitted_classifier(market_returns):
    clf = VolatilityRegimeClassifier(window=21, min_periods=5)
    clf.fit(market_returns)
    return clf


# ---------------------------------------------------------------------------
# test_regime_states
# ---------------------------------------------------------------------------

def test_regime_states(fitted_classifier, market_returns):
    """
    The classifier must produce exactly 4 distinct states (0-3) in
    a sufficiently long return series.
    """
    result = fitted_classifier.predict(market_returns, return_probabilities=False)

    observed_regimes = set(result["regime"].dropna().astype(int).unique())
    assert len(observed_regimes) == 4, (
        f"Expected 4 regime states, got {len(observed_regimes)}: {observed_regimes}"
    )
    assert observed_regimes == {0, 1, 2, 3}, (
        f"Unexpected regime codes: {observed_regimes}"
    )


# ---------------------------------------------------------------------------
# test_regime_probabilities_sum
# ---------------------------------------------------------------------------

def test_regime_probabilities_sum(fitted_classifier, market_returns):
    """
    For every date with a valid regime, the four probability columns must
    sum to 1.0 (within floating-point tolerance).
    """
    result = fitted_classifier.predict(market_returns, return_probabilities=True)

    prob_cols = [f"prob_{v}" for v in REGIME_LABELS.values()]
    for col in prob_cols:
        assert col in result.columns, f"Missing probability column: {col}"

    # Drop rows where any prob is NaN (early warm-up period)
    valid = result[prob_cols].dropna()
    assert len(valid) > 0, "No valid probability rows found"

    row_sums = valid[prob_cols].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-9, (
        f"Regime probabilities do not sum to 1. Max deviation: {(row_sums - 1.0).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# test_transition_matrix_rows
# ---------------------------------------------------------------------------

def test_transition_matrix_rows(fitted_classifier, market_returns):
    """
    Empirical transition matrix rows must sum to 1.0 (each row is a
    discrete probability distribution over next-state regimes).
    """
    result = fitted_classifier.predict(market_returns, return_probabilities=False)
    regimes = result["regime"].dropna().astype(int)

    if len(regimes) < 10:
        pytest.skip("Not enough regime observations to build a transition matrix")

    # Build empirical transition matrix
    n_states = 4
    trans = np.zeros((n_states, n_states), dtype=float)
    regime_values = regimes.values
    for t in range(len(regime_values) - 1):
        from_state = regime_values[t]
        to_state = regime_values[t + 1]
        trans[from_state, to_state] += 1.0

    # Normalise rows (skip rows with zero visits)
    row_sums = trans.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 0:
            row_prob_sum = trans[i].sum() / row_sums[i]  # always 1 before normalise
            assert abs(trans[i].sum() - row_sums[i]) < 1e-9

    # After normalisation check
    trans_norm = np.where(
        row_sums[:, None] > 0,
        trans / row_sums[:, None],
        0.0,
    )
    visited_rows = row_sums > 0
    row_sums_norm = trans_norm[visited_rows].sum(axis=1)
    assert (np.abs(row_sums_norm - 1.0) < 1e-9).all(), (
        f"Transition matrix rows do not sum to 1: {row_sums_norm}"
    )
