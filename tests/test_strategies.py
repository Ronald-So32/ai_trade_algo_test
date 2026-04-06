"""Tests for strategy signal generation and weight computation."""
import pytest
import numpy as np
import pandas as pd

from qrt.strategies.cross_sectional_momentum import CrossSectionalMomentum
from qrt.strategies.mean_reversion import MeanReversion
from qrt.strategies.distance_pairs import DistancePairs


# ---------------------------------------------------------------------------
# test_strategy_signals_shape
# ---------------------------------------------------------------------------

def test_strategy_signals_shape(sample_prices, sample_returns):
    """Signals DataFrame must have the same shape as input prices."""
    strategy = CrossSectionalMomentum(lookback=63, skip_days=5, decile=0.2)
    signals = strategy.generate_signals(sample_prices, sample_returns)

    assert signals.shape == sample_prices.shape, (
        f"Expected signals shape {sample_prices.shape}, got {signals.shape}"
    )
    assert list(signals.columns) == list(sample_prices.columns), (
        "Signal columns do not match price columns"
    )
    assert list(signals.index) == list(sample_prices.index), (
        "Signal index does not match price index"
    )


# ---------------------------------------------------------------------------
# test_strategy_signals_range
# ---------------------------------------------------------------------------

def test_strategy_signals_range(sample_prices, sample_returns):
    """All signal values must lie within [-1, 1]."""
    strategy = CrossSectionalMomentum(lookback=63, skip_days=5, decile=0.2)
    signals = strategy.generate_signals(sample_prices, sample_returns)

    assert signals.min().min() >= -1.0 - 1e-9, (
        f"Signal below -1: {signals.min().min()}"
    )
    assert signals.max().max() <= 1.0 + 1e-9, (
        f"Signal above +1: {signals.max().max()}"
    )


# ---------------------------------------------------------------------------
# test_weights_sum
# ---------------------------------------------------------------------------

def test_weights_sum(sample_prices, sample_returns):
    """
    Long-short weights should approximately sum to ~0 for a market-neutral
    strategy.  We allow a small tolerance since equal-weighting with
    integer long/short counts may not be exactly balanced.
    """
    strategy = CrossSectionalMomentum(lookback=63, skip_days=5, decile=0.2)
    signals = strategy.generate_signals(sample_prices, sample_returns)
    weights = strategy.compute_weights(signals)

    # Rows with at least one position
    active_rows = weights[weights.abs().sum(axis=1) > 0]
    if active_rows.empty:
        pytest.skip("No active signal rows to check weight sum")

    row_sums = active_rows.sum(axis=1)
    # Market-neutral: net weight should be close to 0
    assert row_sums.abs().mean() < 0.15, (
        f"Mean |net weight| = {row_sums.abs().mean():.4f} — strategy not market-neutral"
    )


# ---------------------------------------------------------------------------
# test_momentum_positive_autocorrelation
# ---------------------------------------------------------------------------

def test_momentum_positive_autocorrelation(sample_prices, sample_returns):
    """
    Momentum signal should be positively correlated with past (trailing)
    returns: assets ranked as 'buy' (signal=+1) should have had higher
    recent returns on average than 'sell' assets (signal=-1).
    """
    strategy = CrossSectionalMomentum(lookback=63, skip_days=5, decile=0.2)
    signals = strategy.generate_signals(sample_prices, sample_returns)

    # Use trailing 63-day return as the reference past return
    trailing_ret = sample_prices.pct_change(63).dropna(how="all")

    corr_values = []
    for date in signals.index:
        if date not in trailing_ret.index:
            continue
        sig_row = signals.loc[date]
        ret_row = trailing_ret.loc[date]
        active = sig_row[sig_row != 0]
        if len(active) < 4:
            continue
        common = active.index.intersection(ret_row.dropna().index)
        if len(common) < 4:
            continue
        c = np.corrcoef(active[common].values, ret_row[common].values)[0, 1]
        if not np.isnan(c):
            corr_values.append(c)

    assert len(corr_values) > 0, "No dates had enough active signals to compute correlation"
    mean_corr = np.mean(corr_values)
    assert mean_corr > 0, (
        f"Momentum signal should be positively correlated with past returns, "
        f"got mean correlation = {mean_corr:.4f}"
    )


# ---------------------------------------------------------------------------
# test_mean_reversion_negative_autocorrelation
# ---------------------------------------------------------------------------

def test_mean_reversion_negative_autocorrelation(sample_prices, sample_returns):
    """
    Mean reversion signal should be negatively correlated with recent short-term
    returns: assets with high recent returns get a negative (short) signal.
    """
    strategy = MeanReversion(lookback=30, entry_threshold=1.0, exit_threshold=0.3, max_holding=5)
    signals = strategy.generate_signals(sample_prices, sample_returns)

    # 5-day return as the "recent" return
    recent_ret = sample_prices.pct_change(5).dropna(how="all")

    corr_values = []
    for date in signals.index:
        if date not in recent_ret.index:
            continue
        sig_row = signals.loc[date]
        ret_row = recent_ret.loc[date]
        active = sig_row[sig_row != 0]
        if len(active) < 3:
            continue
        common = active.index.intersection(ret_row.dropna().index)
        if len(common) < 3:
            continue
        c = np.corrcoef(active[common].values, ret_row[common].values)[0, 1]
        if not np.isnan(c):
            corr_values.append(c)

    if len(corr_values) == 0:
        pytest.skip("No active mean-reversion signals generated — thresholds may be too tight")

    mean_corr = np.mean(corr_values)
    assert mean_corr < 0, (
        f"Mean reversion signal should be negatively correlated with recent returns, "
        f"got mean correlation = {mean_corr:.4f}"
    )


# ---------------------------------------------------------------------------
# test_pairs_hedged
# ---------------------------------------------------------------------------

def test_pairs_hedged(sample_prices, sample_returns):
    """
    Pairs strategy weight vector should be approximately dollar-neutral:
    long and short legs should be of similar size.
    """
    # Use a short formation period to allow signals within 500-day window
    strategy = DistancePairs(
        formation_period=100,
        n_pairs=3,
        entry_z=1.5,
        exit_z=0.5,
        max_holding=20,
        rebalance_freq=100,
    )
    signals = strategy.generate_signals(sample_prices, sample_returns)
    weights = strategy.compute_weights(signals)

    active_rows = weights[(weights != 0).any(axis=1)]
    if active_rows.empty:
        pytest.skip("No pair trades triggered — skipping neutrality check")

    row_sums = active_rows.sum(axis=1)
    # Dollar neutrality: net weight close to 0 on average
    assert row_sums.abs().mean() < 0.15, (
        f"Pairs strategy net weight not near zero: mean |net| = {row_sums.abs().mean():.4f}"
    )


# ---------------------------------------------------------------------------
# test_vol_managed_leverage_cap
# ---------------------------------------------------------------------------

def test_vol_managed_leverage_cap(sample_prices, sample_returns):
    """
    After applying volatility-managed weights, gross leverage should not
    exceed the strategy's target_gross parameter.
    """
    strategy = MeanReversion(
        lookback=30,
        entry_threshold=1.0,
        exit_threshold=0.3,
        max_holding=5,
        target_gross=1.0,
        vol_scale=True,
    )
    signals = strategy.generate_signals(sample_prices, sample_returns)
    weights = strategy.compute_weights(signals, returns=sample_returns)

    gross = weights.abs().sum(axis=1)
    active_gross = gross[gross > 0]

    if active_gross.empty:
        pytest.skip("No active weights to check leverage cap")

    # Gross leverage should not materially exceed target (allow floating point slack)
    target = strategy.params["target_gross"]
    assert active_gross.max() <= target + 1e-6, (
        f"Gross leverage {active_gross.max():.4f} exceeds target_gross={target}"
    )
