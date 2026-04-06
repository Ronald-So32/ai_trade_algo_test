"""
Tests for leverage-specific stress testing module.

Validates:
- Volatility shock scenarios
- Drawdown stress scenarios
- Estimation error analysis
- Correlation shock testing
- Risk score computation
- Full stress test integration
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def base_returns(rng):
    """Realistic daily returns: ~10% CAGR, ~10% vol."""
    dates = pd.bdate_range("2015-01-01", periods=2000)
    daily_mu = np.log(1.10) / 252
    daily_vol = 0.10 / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_vol, 2000)
    # Add drawdown episode
    rets[500:530] = rng.normal(-0.01, 0.02, 30)
    return pd.Series(rets, index=dates, name="portfolio")


@pytest.fixture
def multi_strategy_returns(rng):
    """5 strategies with realistic returns."""
    dates = pd.bdate_range("2015-01-01", periods=2000)
    strategies = {}
    for i in range(5):
        mu = 0.0002 * (i + 1)
        rets = rng.normal(mu, 0.01, 2000)
        strategies[f"strat_{i}"] = pd.Series(rets, index=dates)
    return strategies


class TestLeverageStressTester:
    def test_base_metrics(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=3.0)
        result = tester.run_full_stress_test(base_returns)

        assert result.base_metrics["leverage"] == 3.0
        assert result.base_metrics["vol"] > 0
        assert result.base_metrics["vol_drag"] > 0

    def test_vol_shock_produces_scenarios(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=5.0)
        result = tester.run_full_stress_test(base_returns)

        assert len(result.vol_shock_results) == 5
        # Higher vol → worse MaxDD
        normal = [s for s in result.vol_shock_results if s["vol_multiplier"] == 1.0][0]
        severe = [s for s in result.vol_shock_results if s["vol_multiplier"] == 3.0][0]
        assert severe["maxdd"] < normal["maxdd"]  # more negative

    def test_drawdown_scenarios(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=5.0)
        result = tester.run_full_stress_test(base_returns)

        assert len(result.drawdown_scenarios) >= 3
        for s in result.drawdown_scenarios:
            assert s["maxdd"] <= 0

    def test_estimation_error(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=5.0)
        result = tester.run_full_stress_test(base_returns)

        ee = result.estimation_error
        assert ee["sharpe_se"] > 0
        assert ee["sharpe_ci_low"] < ee["sharpe_ci_high"]
        assert ee["error_amplification"] == 5.0

    def test_correlation_shock(self, base_returns, multi_strategy_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=3.0)
        result = tester.run_full_stress_test(
            base_returns, strategy_returns=multi_strategy_returns,
        )

        assert len(result.correlation_shock_results) >= 2

    def test_risk_score_bounded(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=5.0)
        result = tester.run_full_stress_test(base_returns)

        assert 0 <= result.overall_risk_score <= 10

    def test_higher_leverage_worse_risk_score(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        low = LeverageStressTester(leverage=2.0).run_full_stress_test(base_returns)
        high = LeverageStressTester(leverage=8.0).run_full_stress_test(base_returns)
        assert high.overall_risk_score <= low.overall_risk_score

    def test_recommendations_generated(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=5.0)
        result = tester.run_full_stress_test(base_returns)
        assert len(result.recommendations) > 0

    def test_markdown_report(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=3.0)
        result = tester.run_full_stress_test(base_returns)
        md = result.to_markdown()
        assert "Leverage Stress Test Report" in md
        assert "Volatility Shock" in md

    def test_regime_analysis(self, base_returns, rng):
        from qrt.validation.leverage_stress import LeverageStressTester
        # Create regime labels
        labels = pd.Series(
            rng.choice(["calm", "stress"], size=len(base_returns)),
            index=base_returns.index,
        )
        tester = LeverageStressTester(leverage=3.0)
        result = tester.run_full_stress_test(
            base_returns, regime_labels=labels,
        )
        assert len(result.regime_analysis) >= 1

    def test_no_leverage(self, base_returns):
        from qrt.validation.leverage_stress import LeverageStressTester
        tester = LeverageStressTester(leverage=1.0)
        result = tester.run_full_stress_test(base_returns)
        assert result.base_metrics["vol_drag"] == 0.0  # no drag at 1x
