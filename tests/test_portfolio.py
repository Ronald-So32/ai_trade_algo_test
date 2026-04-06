"""Tests for RiskParityAllocator and VolatilityTargeter."""
import pytest
import numpy as np
import pandas as pd

from qrt.portfolio.risk_parity import RiskParityAllocator
from qrt.portfolio.vol_targeting import VolatilityTargeter


# ---------------------------------------------------------------------------
# RiskParityAllocator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def allocator():
    return RiskParityAllocator(min_weight=0.0, max_weight=1.0)


def test_risk_parity_weights_positive(allocator, sample_returns):
    """All risk-parity weights must be strictly greater than zero."""
    weights = allocator.allocate(sample_returns, method="naive")

    assert (weights > 0).all(), (
        f"Some risk parity weights are non-positive: {weights[weights <= 0]}"
    )


def test_risk_parity_weights_sum(allocator, sample_returns):
    """Risk-parity weights must sum to exactly 1.0."""
    for method in ("naive", "covariance"):
        weights = allocator.allocate(sample_returns, method=method)
        assert abs(weights.sum() - 1.0) < 1e-6, (
            f"Weights ({method}) sum to {weights.sum():.8f}, expected 1.0"
        )


def test_risk_parity_naive_inverse_vol(allocator, sample_returns):
    """
    Naive risk parity weights should be inversely proportional to
    each asset's realised volatility (before normalisation).
    """
    vols = sample_returns.std(ddof=1).values
    inv_vol = 1.0 / vols
    expected = inv_vol / inv_vol.sum()

    weights = allocator.naive_risk_parity(vols)
    np.testing.assert_allclose(weights, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# VolatilityTargeter tests
# ---------------------------------------------------------------------------

@pytest.fixture
def targeter():
    return VolatilityTargeter()


@pytest.fixture
def portfolio_returns(sample_returns):
    """Equal-weight portfolio return series."""
    return sample_returns.mean(axis=1)


def test_vol_targeting_scales_down_in_high_vol(targeter, portfolio_returns):
    """
    When current realised vol > target vol, the scaling factor must be < 1.
    We construct a high-vol period by multiplying returns by 10 and set a
    low target.
    """
    # Amplify returns so realised vol >> target
    high_vol_returns = portfolio_returns * 10.0
    target = 0.01  # 1% annualised — much lower than amplified vol

    scaling = targeter.compute_scaling(
        returns=high_vol_returns,
        target_vol=target,
        lookback=21,
        max_leverage=2.0,
    )

    # Drop early NaN-filled periods
    valid_scaling = scaling.dropna()
    # After warm-up, scaling should be well below 1
    late_scaling = valid_scaling.iloc[30:]  # skip warm-up
    if late_scaling.empty:
        pytest.skip("Not enough data after warm-up period")

    assert late_scaling.mean() < 1.0, (
        f"Expected scaling < 1 in high-vol regime, got mean={late_scaling.mean():.4f}"
    )


def test_vol_targeting_scales_up_in_low_vol(targeter, portfolio_returns):
    """
    When realised vol < target vol, the scaling factor must be > 1
    (up to max_leverage).
    """
    # Suppress returns so realised vol << target
    low_vol_returns = portfolio_returns * 0.01
    target = 0.20  # 20% annualised — much higher than suppressed vol

    scaling = targeter.compute_scaling(
        returns=low_vol_returns,
        target_vol=target,
        lookback=21,
        max_leverage=5.0,  # allow leverage > 1
    )

    valid_scaling = scaling.dropna()
    late_scaling = valid_scaling.iloc[30:]
    if late_scaling.empty:
        pytest.skip("Not enough data after warm-up period")

    assert late_scaling.mean() > 1.0, (
        f"Expected scaling > 1 in low-vol regime, got mean={late_scaling.mean():.4f}"
    )


def test_vol_targeting_respects_max_leverage(targeter, portfolio_returns):
    """Scaling factor must never exceed max_leverage."""
    low_vol_returns = portfolio_returns * 0.001
    max_leverage = 1.5

    scaling = targeter.compute_scaling(
        returns=low_vol_returns,
        target_vol=0.30,
        lookback=21,
        max_leverage=max_leverage,
    )

    assert scaling.max() <= max_leverage + 1e-9, (
        f"Scaling exceeded max_leverage={max_leverage}: max={scaling.max():.6f}"
    )
