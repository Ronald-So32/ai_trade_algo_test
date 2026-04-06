"""
Tests for drawdown-constrained leverage optimizer.

Validates:
- Grid search finds feasible leverage within DD target
- Higher DD target → higher optimal leverage
- Vol drag computation
- Analytical leverage estimate
- Integration with DrawdownShield
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def base_returns(rng):
    """Realistic base portfolio returns: ~10% CAGR, ~10% vol."""
    dates = pd.bdate_range("2015-01-01", periods=2000)
    daily_mu = np.log(1.10) / 252
    daily_vol = 0.10 / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_vol, 2000)
    # Add drawdown episode
    rets[500:530] = rng.normal(-0.01, 0.02, 30)
    return pd.Series(rets, index=dates, name="portfolio")


class TestLeverageOptimizer:
    def test_optimizer_finds_feasible_leverage(self, base_returns):
        from qrt.sizing.leverage_optimizer import LeverageOptimizer
        opt = LeverageOptimizer(max_dd_target=0.15, leverage_range=(1.0, 6.0), n_grid=20)
        # Use shield — without it, any leverage > 1x will exceed DD target
        result = opt.optimize_with_dynamic_shield(
            base_returns,
            cppi_config={"max_drawdown": 0.15, "multiplier": 3.0},
        )

        assert result.optimal_leverage >= 1.0
        assert abs(result.expected_maxdd) <= 0.15 + 0.02  # tolerance for edge cases

    def test_higher_dd_target_allows_more_leverage(self, base_returns):
        from qrt.sizing.leverage_optimizer import LeverageOptimizer

        opt_tight = LeverageOptimizer(max_dd_target=0.05, n_grid=15, annual_financing_rate=0.0)
        opt_loose = LeverageOptimizer(max_dd_target=0.20, n_grid=15, annual_financing_rate=0.0)

        result_tight = opt_tight.optimize_with_dynamic_shield(
            base_returns, cppi_config={"max_drawdown": 0.05, "multiplier": 3.0},
        )
        result_loose = opt_loose.optimize_with_dynamic_shield(
            base_returns, cppi_config={"max_drawdown": 0.20, "multiplier": 3.0},
        )

        assert result_loose.optimal_leverage >= result_tight.optimal_leverage

    def test_grids_have_correct_length(self, base_returns):
        from qrt.sizing.leverage_optimizer import LeverageOptimizer
        n = 25
        opt = LeverageOptimizer(n_grid=n)
        result = opt.optimize(base_returns)

        assert len(result.leverage_grid) == n
        assert len(result.cagr_grid) == n
        assert len(result.maxdd_grid) == n
        assert len(result.calmar_grid) == n

    def test_optimize_with_shield(self, base_returns):
        from qrt.sizing.leverage_optimizer import LeverageOptimizer
        opt = LeverageOptimizer(max_dd_target=0.12, n_grid=15)
        result = opt.optimize_with_dynamic_shield(
            base_returns,
            cppi_config={"max_drawdown": 0.12, "multiplier": 3.0},
        )

        assert result.optimal_leverage >= 1.0
        assert result.expected_calmar > 0


class TestVolDrag:
    def test_vol_drag_formula(self):
        from qrt.sizing.leverage_optimizer import compute_vol_drag
        # At 2x leverage with 15% vol: drag = 2*1/2 * 0.15^2 = 0.0225
        drag = compute_vol_drag(2.0, 0.15)
        assert abs(drag - 0.0225) < 1e-6

    def test_vol_drag_increases_with_leverage(self):
        from qrt.sizing.leverage_optimizer import compute_vol_drag
        drag_2x = compute_vol_drag(2.0, 0.15)
        drag_5x = compute_vol_drag(5.0, 0.15)
        assert drag_5x > drag_2x

    def test_no_drag_at_1x(self):
        from qrt.sizing.leverage_optimizer import compute_vol_drag
        drag = compute_vol_drag(1.0, 0.15)
        assert abs(drag) < 1e-10


class TestAnalyticalLeverage:
    def test_analytical_estimate_positive(self):
        from qrt.sizing.leverage_optimizer import analytical_optimal_leverage
        lev = analytical_optimal_leverage(sharpe=1.5, annual_vol=0.10, max_dd=0.12)
        assert lev >= 1.0

    def test_higher_sharpe_more_leverage(self):
        from qrt.sizing.leverage_optimizer import analytical_optimal_leverage
        lev_low = analytical_optimal_leverage(sharpe=0.5, annual_vol=0.10, max_dd=0.12)
        lev_high = analytical_optimal_leverage(sharpe=2.0, annual_vol=0.10, max_dd=0.12)
        assert lev_high >= lev_low

    def test_higher_vol_less_leverage(self):
        from qrt.sizing.leverage_optimizer import analytical_optimal_leverage
        lev_low_vol = analytical_optimal_leverage(sharpe=1.5, annual_vol=0.08, max_dd=0.12)
        lev_high_vol = analytical_optimal_leverage(sharpe=1.5, annual_vol=0.25, max_dd=0.12)
        assert lev_low_vol >= lev_high_vol
