"""
Tests for overfitting detection and statistical validation module.

Validates:
- Deflated Sharpe Ratio computation
- Probabilistic Sharpe Ratio
- Minimum Backtest Length
- Multiple hypothesis testing correction
- White's Reality Check
- IS vs OOS degradation tracking
- Leverage risk haircut
- Full OverfittingTestSuite integration
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def strategy_returns(rng):
    """Generate 5 strategies with 1000 days of returns."""
    dates = pd.bdate_range("2018-01-01", periods=1000)
    strategies = {}
    for i in range(5):
        mu = 0.0003 * (i + 1)  # increasing expected return
        rets = rng.normal(mu, 0.01, 1000)
        strategies[f"strat_{i}"] = pd.Series(rets, index=dates)
    return strategies


@pytest.fixture
def random_strategy_returns(rng):
    """Generate strategies with zero expected return (null hypothesis)."""
    dates = pd.bdate_range("2018-01-01", periods=1000)
    strategies = {}
    for i in range(10):
        rets = rng.normal(0, 0.01, 1000)
        strategies[f"random_{i}"] = pd.Series(rets, index=dates)
    return strategies


# ── Deflated Sharpe Ratio Tests ──

class TestDeflatedSharpe:
    def test_dsr_positive_for_strong_strategy(self):
        from qrt.validation.overfitting_tests import deflated_sharpe_ratio
        # Strong Sharpe with few trials → high DSR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=5, n_observations=2520,
        )
        assert dsr > 0.5

    def test_dsr_low_for_weak_strategy_many_trials(self):
        from qrt.validation.overfitting_tests import deflated_sharpe_ratio
        # Weak Sharpe with many trials → low DSR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.5, n_trials=100, n_observations=500,
        )
        assert dsr < 0.5

    def test_dsr_increases_with_observations(self):
        from qrt.validation.overfitting_tests import deflated_sharpe_ratio
        # Use moderate Sharpe and few trials so DSR is in a non-saturated range
        dsr_short = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=3, n_observations=100,
        )
        dsr_long = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=3, n_observations=5000,
        )
        assert dsr_long > dsr_short

    def test_dsr_decreases_with_more_trials(self):
        from qrt.validation.overfitting_tests import deflated_sharpe_ratio
        dsr_few = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=5, n_observations=2520,
        )
        dsr_many = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=100, n_observations=2520,
        )
        assert dsr_few > dsr_many

    def test_dsr_handles_edge_cases(self):
        from qrt.validation.overfitting_tests import deflated_sharpe_ratio
        assert deflated_sharpe_ratio(0, 1, 0) == 0.0
        assert deflated_sharpe_ratio(1.0, 0, 100) == 0.0


# ── Probabilistic Sharpe Ratio Tests ──

class TestProbabilisticSharpe:
    def test_psr_high_for_good_sharpe(self):
        from qrt.validation.overfitting_tests import probabilistic_sharpe_ratio
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=2.0, benchmark_sharpe=0.0,
            n_observations=2520,
        )
        assert psr > 0.95

    def test_psr_low_for_marginal_sharpe(self):
        from qrt.validation.overfitting_tests import probabilistic_sharpe_ratio
        # Very weak Sharpe with short sample → not significant
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=0.1, benchmark_sharpe=0.0,
            n_observations=50,
        )
        assert psr < 0.95

    def test_psr_returns_zero_for_insufficient_data(self):
        from qrt.validation.overfitting_tests import probabilistic_sharpe_ratio
        assert probabilistic_sharpe_ratio(1.0, 0.0, 1) == 0.0


# ── Minimum Backtest Length Tests ──

class TestMinimumBacktestLength:
    def test_btl_finite_for_positive_sharpe(self):
        from qrt.validation.overfitting_tests import minimum_backtest_length
        # Use high Sharpe that exceeds expected max null with few trials
        btl = minimum_backtest_length(observed_sharpe=3.0, n_trials=5)
        assert 0 < btl < 100  # should be a reasonable number of years

    def test_btl_infinite_for_zero_sharpe(self):
        from qrt.validation.overfitting_tests import minimum_backtest_length
        btl = minimum_backtest_length(observed_sharpe=0.0)
        assert btl == float("inf")

    def test_btl_increases_with_more_trials(self):
        from qrt.validation.overfitting_tests import minimum_backtest_length
        # Use Sharpe high enough to be finite even with many trials
        btl_few = minimum_backtest_length(observed_sharpe=3.0, n_trials=2)
        btl_many = minimum_backtest_length(observed_sharpe=3.0, n_trials=10)
        assert btl_many > btl_few

    def test_btl_decreases_with_higher_sharpe(self):
        from qrt.validation.overfitting_tests import minimum_backtest_length
        btl_low = minimum_backtest_length(observed_sharpe=0.8, n_trials=10)
        btl_high = minimum_backtest_length(observed_sharpe=2.0, n_trials=10)
        assert btl_high < btl_low


# ── Multiple Testing Correction Tests ──

class TestMultipleTesting:
    def test_correction_identifies_strong_strategies(self, strategy_returns):
        from qrt.validation.overfitting_tests import multiple_testing_correction
        sharpe_dict = {}
        for name, rets in strategy_returns.items():
            sharpe_dict[name] = float(rets.mean() / rets.std() * np.sqrt(252))

        result = multiple_testing_correction(sharpe_dict, n_observations=1000)

        assert result["n_strategies"] == 5
        assert result["bonferroni_threshold"] > 0
        assert result["expected_max_null_sharpe"] > 0

    def test_random_strategies_mostly_rejected(self, random_strategy_returns):
        from qrt.validation.overfitting_tests import multiple_testing_correction
        sharpe_dict = {}
        for name, rets in random_strategy_returns.items():
            sharpe_dict[name] = float(rets.mean() / rets.std() * np.sqrt(252))

        result = multiple_testing_correction(sharpe_dict, n_observations=1000)

        # Most random strategies should not survive
        assert result["n_surviving_bonferroni"] <= 2  # at most 1-2 by chance

    def test_empty_input(self):
        from qrt.validation.overfitting_tests import multiple_testing_correction
        result = multiple_testing_correction({}, n_observations=100)
        assert result["n_strategies"] == 0


# ── White's Reality Check Tests ──

class TestWhitesRealityCheck:
    def test_reality_check_returns_valid_structure(self, strategy_returns):
        from qrt.validation.overfitting_tests import whites_reality_check
        result = whites_reality_check(
            strategy_returns, n_bootstrap=200, block_size=5,
        )
        assert "p_value" in result
        assert "best_strategy" in result
        assert 0 <= result["p_value"] <= 1

    def test_strong_strategy_has_low_pvalue(self, rng):
        from qrt.validation.overfitting_tests import whites_reality_check
        dates = pd.bdate_range("2018-01-01", periods=1000)
        # One clearly profitable strategy, rest random
        strategies = {
            "strong": pd.Series(rng.normal(0.002, 0.01, 1000), index=dates),
        }
        for i in range(4):
            strategies[f"weak_{i}"] = pd.Series(rng.normal(0, 0.01, 1000), index=dates)

        result = whites_reality_check(strategies, n_bootstrap=500)
        assert result["best_strategy"] == "strong"
        assert result["p_value"] < 0.10  # should be significant


# ── IS vs OOS Degradation Tests ──

class TestISOOSDegradation:
    def test_degradation_computed(self, strategy_returns):
        from qrt.validation.overfitting_tests import compute_is_oos_degradation
        result = compute_is_oos_degradation(strategy_returns)
        assert len(result) == 5  # all strategies
        for name, comp in result.items():
            assert "is_sharpe" in comp
            assert "oos_sharpe" in comp
            assert "degradation_pct" in comp

    def test_overfit_strategy_flagged(self, rng):
        from qrt.validation.overfitting_tests import compute_is_oos_degradation
        dates = pd.bdate_range("2018-01-01", periods=1000)
        # Trending strategy: great IS, poor OOS
        is_rets = rng.normal(0.002, 0.01, 600)
        oos_rets = rng.normal(-0.001, 0.015, 400)
        rets = pd.Series(np.concatenate([is_rets, oos_rets]), index=dates)

        result = compute_is_oos_degradation({"overfit": rets})
        assert result["overfit"]["degradation_pct"] > 50


# ── Leverage Risk Haircut Tests ──

class TestLeverageHaircut:
    def test_haircut_reduces_sharpe(self):
        from qrt.validation.overfitting_tests import leverage_risk_haircut
        result = leverage_risk_haircut(
            base_sharpe=2.0, leverage=5.0, base_vol=0.10,
            base_maxdd=-0.05, n_strategies=18, n_observations=2520,
        )
        assert result["haircut_sharpe"] < result["raw_sharpe"]
        assert result["vol_drag"] > 0
        assert 0 <= result["safety_score"] <= 10

    def test_higher_leverage_worse_safety(self):
        from qrt.validation.overfitting_tests import leverage_risk_haircut
        r_low = leverage_risk_haircut(
            base_sharpe=1.5, leverage=2.0, base_vol=0.10,
            base_maxdd=-0.05, n_strategies=18,
        )
        r_high = leverage_risk_haircut(
            base_sharpe=1.5, leverage=8.0, base_vol=0.10,
            base_maxdd=-0.05, n_strategies=18,
        )
        assert r_high["safety_score"] <= r_low["safety_score"]
        assert abs(r_high["expected_oos_maxdd"]) > abs(r_low["expected_oos_maxdd"])


# ── Full Test Suite Integration ──

class TestOverfittingTestSuite:
    def test_suite_runs_all_tests(self, strategy_returns):
        from qrt.validation.overfitting_tests import OverfittingTestSuite
        portfolio = pd.concat(strategy_returns.values(), axis=1).mean(axis=1)
        suite = OverfittingTestSuite(n_strategies_tested=5, bootstrap_n=200)
        report = suite.run_all(
            strategy_returns=strategy_returns,
            portfolio_returns=portfolio,
            leverage=3.0,
        )

        assert report.overall_confidence in [
            "HIGH", "MODERATE", "LOW", "VERY LOW — LIKELY OVERFIT", "UNKNOWN"
        ]
        assert len(report.strategy_tests) == 5
        assert report.multiple_testing
        assert report.is_vs_oos
        assert report.reality_check

    def test_suite_generates_markdown(self, strategy_returns):
        from qrt.validation.overfitting_tests import OverfittingTestSuite
        portfolio = pd.concat(strategy_returns.values(), axis=1).mean(axis=1)
        suite = OverfittingTestSuite(n_strategies_tested=5, bootstrap_n=100)
        report = suite.run_all(strategy_returns=strategy_returns,
                               portfolio_returns=portfolio)
        md = report.to_markdown()
        assert "Backtest Overfitting Diagnostic Report" in md
        assert "Strategy-Level" in md
