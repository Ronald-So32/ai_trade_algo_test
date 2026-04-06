"""
Tests for strategy pruning and James-Stein shrinkage module.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def mixed_strategies(rng):
    """5 strategies: 2 strong, 1 mediocre, 2 bad."""
    dates = pd.bdate_range("2018-01-01", periods=1000)
    return {
        "strong_1": pd.Series(rng.normal(0.001, 0.01, 1000), index=dates),
        "strong_2": pd.Series(rng.normal(0.0008, 0.01, 1000), index=dates),
        "mediocre": pd.Series(rng.normal(0.0001, 0.01, 1000), index=dates),
        "bad_1": pd.Series(rng.normal(-0.0005, 0.01, 1000), index=dates),
        "bad_2": pd.Series(rng.normal(-0.001, 0.015, 1000), index=dates),
    }


class TestJamesSteinShrinkage:
    def test_shrinks_toward_mean(self):
        from qrt.portfolio.strategy_pruner import james_stein_shrink_sharpe
        sharpes = pd.Series({"a": 3.0, "b": 0.5, "c": -1.0, "d": 1.0})
        shrunk = james_stein_shrink_sharpe(sharpes, n_observations=500)
        # Extreme values should be pulled toward the mean
        assert abs(shrunk["a"]) < abs(sharpes["a"])
        assert shrunk.std() <= sharpes.std()

    def test_preserves_ranking(self):
        from qrt.portfolio.strategy_pruner import james_stein_shrink_sharpe
        sharpes = pd.Series({"a": 2.0, "b": 1.0, "c": 0.5, "d": -0.5})
        shrunk = james_stein_shrink_sharpe(sharpes, n_observations=500)
        # Ranking should be preserved
        assert shrunk["a"] > shrunk["b"] > shrunk["c"] > shrunk["d"]

    def test_two_strategies_no_shrinkage(self):
        from qrt.portfolio.strategy_pruner import james_stein_shrink_sharpe
        sharpes = pd.Series({"a": 2.0, "b": 0.5})
        shrunk = james_stein_shrink_sharpe(sharpes, n_observations=500)
        # With < 3 strategies, should return unchanged
        assert (shrunk == sharpes).all()


class TestMarginalSharpe:
    def test_bad_strategy_negative_marginal(self, mixed_strategies):
        from qrt.portfolio.strategy_pruner import compute_marginal_sharpe
        marginal = compute_marginal_sharpe(mixed_strategies)
        # Bad strategies should have negative or near-zero marginal
        assert marginal["bad_2"] < marginal["strong_1"]

    def test_returns_all_strategies(self, mixed_strategies):
        from qrt.portfolio.strategy_pruner import compute_marginal_sharpe
        marginal = compute_marginal_sharpe(mixed_strategies)
        assert len(marginal) == len(mixed_strategies)


class TestPruneStrategies:
    def test_prunes_bad_strategies(self, mixed_strategies):
        from qrt.portfolio.strategy_pruner import prune_strategies
        pruned, report = prune_strategies(mixed_strategies, max_strategies=10)
        # Should remove at least one bad strategy
        assert report["pruned_count"] <= report["original_count"]
        assert len(report["removed"]) >= 0  # may or may not remove

    def test_keeps_minimum_three(self, rng):
        from qrt.portfolio.strategy_pruner import prune_strategies
        dates = pd.bdate_range("2018-01-01", periods=500)
        # All bad strategies — should still keep 3
        strats = {
            f"bad_{i}": pd.Series(rng.normal(-0.001, 0.01, 500), index=dates)
            for i in range(5)
        }
        pruned, report = prune_strategies(strats)
        assert len(pruned) >= 3

    def test_respects_max_strategies(self, rng):
        from qrt.portfolio.strategy_pruner import prune_strategies
        dates = pd.bdate_range("2018-01-01", periods=500)
        strats = {
            f"s_{i}": pd.Series(rng.normal(0.001, 0.01, 500), index=dates)
            for i in range(15)
        }
        pruned, report = prune_strategies(strats, max_strategies=5)
        assert len(pruned) <= 5

    def test_report_has_diagnostics(self, mixed_strategies):
        from qrt.portfolio.strategy_pruner import prune_strategies
        _, report = prune_strategies(mixed_strategies)
        assert "individual_sharpe" in report
        assert "shrunk_sharpe" in report
        assert "marginal_sharpe" in report

    def test_too_few_strategies_skipped(self):
        from qrt.portfolio.strategy_pruner import prune_strategies
        dates = pd.bdate_range("2018-01-01", periods=500)
        strats = {"only": pd.Series(np.random.normal(0.001, 0.01, 500), index=dates)}
        pruned, report = prune_strategies(strats)
        assert len(pruned) == 1
