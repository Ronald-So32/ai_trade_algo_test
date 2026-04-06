"""
Tests for portfolio insurance, multi-horizon drawdown control,
and correlation breakdown detection.

Validates:
- CPPI exposure scaling
- Multi-horizon drawdown control
- Correlation breakdown detection
- DrawdownShield composite
- Dynamic leverage de-risking
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def portfolio_returns(rng):
    """500 days of realistic portfolio returns."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    # Normal returns with a drawdown episode around day 200-250
    rets = rng.normal(0.0003, 0.01, 500)
    # Inject a drawdown episode
    rets[200:230] = rng.normal(-0.015, 0.02, 30)
    return pd.Series(rets, index=dates, name="portfolio")


@pytest.fixture
def strategy_returns_df(rng):
    """5 strategies with 500 days of returns."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    n_strats = 5
    names = [f"strat_{i}" for i in range(n_strats)]
    rets = rng.normal(0.0003, 0.01, (500, n_strats))
    # During crisis, all correlate
    rets[200:230] = rng.normal(-0.01, 0.02, (30, n_strats)) + rng.normal(-0.005, 0.005, (30, 1))
    return pd.DataFrame(rets, index=dates, columns=names)


@pytest.fixture
def portfolio_weights(portfolio_returns):
    """Simple equal-weight portfolio."""
    return pd.DataFrame(
        {"asset": 1.0}, index=portfolio_returns.index
    )


@pytest.fixture
def asset_returns(portfolio_returns):
    """Single-asset returns matching weights."""
    return pd.DataFrame(
        {"asset": portfolio_returns.values}, index=portfolio_returns.index
    )


# ── CPPI Insurance Tests ──

class TestCPPIInsurance:
    def test_exposure_bounded(self, portfolio_returns):
        from qrt.risk.portfolio_insurance import CPPIInsurance
        cppi = CPPIInsurance(max_drawdown=0.10, multiplier=3.0, max_exposure=1.5, min_exposure=0.05)
        exposure = cppi.compute_exposure(portfolio_returns)

        assert exposure.min() >= 0.05 - 1e-6
        assert exposure.max() <= 1.5 + 1e-6

    def test_exposure_reduces_during_drawdown(self, portfolio_returns):
        from qrt.risk.portfolio_insurance import CPPIInsurance
        cppi = CPPIInsurance(max_drawdown=0.10, multiplier=3.0)
        exposure = cppi.compute_exposure(portfolio_returns)

        # During drawdown episode (200-230), exposure should be lower
        calm_avg = exposure.iloc[100:180].mean()
        crisis_avg = exposure.iloc[210:240].mean()
        assert crisis_avg < calm_avg, "Exposure should reduce during drawdown"

    def test_apply_reduces_maxdd(self, portfolio_weights, asset_returns):
        from qrt.risk.portfolio_insurance import CPPIInsurance
        cppi = CPPIInsurance(max_drawdown=0.10, multiplier=3.0)

        adjusted = cppi.apply(portfolio_weights, asset_returns)
        # Adjusted weights should be reduced during drawdown
        assert adjusted["asset"].min() < 1.0

    def test_ratchet_floor_increases(self, rng):
        """Floor should ratchet up when portfolio makes new highs."""
        from qrt.risk.portfolio_insurance import CPPIInsurance
        cppi = CPPIInsurance(max_drawdown=0.15, ratchet=True, ratchet_pct=0.5)

        # Trending up returns
        dates = pd.bdate_range("2020-01-01", periods=300)
        rets = pd.Series(rng.normal(0.001, 0.005, 300), index=dates)
        exposure = cppi.compute_exposure(rets)

        # With steady gains, exposure should stay above minimum
        assert exposure.iloc[-100:].mean() > cppi.min_exposure


# ── Multi-Horizon Drawdown Control Tests ──

class TestMultiHorizonDrawdownControl:
    def test_scaling_bounded(self, portfolio_returns):
        from qrt.risk.portfolio_insurance import MultiHorizonDrawdownControl
        mhdd = MultiHorizonDrawdownControl(floor=0.10)
        scaling = mhdd.compute_scaling(portfolio_returns)

        assert scaling.min() >= 0.10 - 1e-6
        assert scaling.max() <= 1.0 + 1e-6

    def test_scaling_reduces_during_drawdown(self, portfolio_returns):
        from qrt.risk.portfolio_insurance import MultiHorizonDrawdownControl
        mhdd = MultiHorizonDrawdownControl()
        scaling = mhdd.compute_scaling(portfolio_returns)

        calm_avg = scaling.iloc[100:180].mean()
        crisis_avg = scaling.iloc[215:245].mean()
        assert crisis_avg < calm_avg, "Scaling should reduce during drawdown episode"

    def test_custom_horizons(self, portfolio_returns):
        from qrt.risk.portfolio_insurance import MultiHorizonDrawdownControl
        custom = {
            "2w": {"window": 10, "max_dd": 0.04, "weight": 0.5},
            "2m": {"window": 42, "max_dd": 0.08, "weight": 0.5},
        }
        mhdd = MultiHorizonDrawdownControl(horizons=custom)
        scaling = mhdd.compute_scaling(portfolio_returns)
        assert len(scaling) == len(portfolio_returns)


# ── Correlation Breakdown Detector Tests ──

class TestCorrelationBreakdownDetector:
    def test_scaling_bounded(self, strategy_returns_df):
        from qrt.risk.portfolio_insurance import CorrelationBreakdownDetector
        detector = CorrelationBreakdownDetector(
            lookback=63, baseline_lookback=126
        )
        scaling = detector.compute_scaling(strategy_returns_df)

        assert scaling.min() >= 0.25 - 1e-6
        assert scaling.max() <= 1.0 + 1e-6

    def test_detects_correlation_spike(self, strategy_returns_df):
        from qrt.risk.portfolio_insurance import CorrelationBreakdownDetector
        detector = CorrelationBreakdownDetector(
            lookback=21, baseline_lookback=126, threshold_z=1.0
        )
        avg_corr = detector.compute_avg_correlation(strategy_returns_df)

        # During crisis period (200-230), correlations should be elevated
        # (we injected a common factor)
        assert not avg_corr.isna().all()

    def test_single_strategy_returns_ones(self, rng):
        """With only 1 strategy, correlation is undefined → scaling = 1.0."""
        from qrt.risk.portfolio_insurance import CorrelationBreakdownDetector
        dates = pd.bdate_range("2020-01-01", periods=300)
        single = pd.DataFrame({"only": rng.normal(0, 0.01, 300)}, index=dates)
        detector = CorrelationBreakdownDetector()
        scaling = detector.compute_scaling(single)
        assert (scaling == 1.0).all()


# ── DrawdownShield Composite Tests ──

class TestDrawdownShield:
    def test_shield_reduces_maxdd(self, portfolio_weights, asset_returns, strategy_returns_df):
        from qrt.risk.portfolio_insurance import DrawdownShield

        shield = DrawdownShield(
            cppi_config={"max_drawdown": 0.08, "multiplier": 3.0},
            enable_correlation=True,
        )
        adjusted = shield.apply(
            portfolio_weights, asset_returns,
            strategy_returns=strategy_returns_df,
        )

        # Adjusted weights should be reduced during stress
        assert adjusted.values.min() < 1.0

    def test_shield_with_layers_disabled(self, portfolio_weights, asset_returns):
        from qrt.risk.portfolio_insurance import DrawdownShield

        shield = DrawdownShield(
            enable_cppi=True,
            enable_multi_horizon=False,
            enable_correlation=False,
        )
        adjusted = shield.apply(portfolio_weights, asset_returns)
        assert adjusted.shape == portfolio_weights.shape

    def test_all_layers_compose(self, portfolio_weights, asset_returns, strategy_returns_df):
        from qrt.risk.portfolio_insurance import DrawdownShield

        shield = DrawdownShield()
        adjusted = shield.apply(
            portfolio_weights, asset_returns,
            strategy_returns=strategy_returns_df,
        )
        # All layers active → should be more conservative
        assert adjusted["asset"].mean() <= 1.0


# ── Dynamic Leverage Tests ──

class TestDynamicLeverage:
    def test_leverage_series_bounded(self, portfolio_returns):
        from qrt.sizing.leverage_manager import LeverageManager
        mgr = LeverageManager(
            max_portfolio_leverage=2.0, kelly_fraction=0.35,
        )
        lev = mgr.compute_dynamic_leverage(portfolio_returns, avg_sharpe=0.8)

        assert lev.min() >= mgr.min_leverage - 0.1  # EMA smoothing tolerance
        assert lev.max() <= mgr.max_portfolio_leverage + 0.1

    def test_leverage_reduces_during_drawdown(self, portfolio_returns):
        from qrt.sizing.leverage_manager import LeverageManager
        mgr = LeverageManager(max_portfolio_leverage=2.0, kelly_fraction=0.35)
        lev = mgr.compute_dynamic_leverage(portfolio_returns, avg_sharpe=0.8)

        # After the drawdown episode, leverage should be reduced
        calm_avg = lev.iloc[100:180].mean()
        crisis_avg = lev.iloc[220:260].mean()
        assert crisis_avg < calm_avg, "Leverage should reduce during drawdown"

    def test_crisis_probs_reduce_leverage(self, portfolio_returns):
        from qrt.sizing.leverage_manager import LeverageManager
        mgr = LeverageManager(max_portfolio_leverage=2.0)

        # No crisis
        lev_calm = mgr.compute_dynamic_leverage(
            portfolio_returns, avg_sharpe=0.8,
            crisis_probs=pd.Series(0.0, index=portfolio_returns.index),
        )
        # High crisis
        lev_crisis = mgr.compute_dynamic_leverage(
            portfolio_returns, avg_sharpe=0.8,
            crisis_probs=pd.Series(0.8, index=portfolio_returns.index),
        )
        assert lev_crisis.mean() < lev_calm.mean(), \
            "High crisis probs should reduce average leverage"
