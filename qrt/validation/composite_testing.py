"""
Fundamental Composite Testing Framework
=========================================
Test combinations of fundamental factors (value, quality, profitability, etc.)
using walk-forward validation to determine the best configuration.

Supported combinations:
  - Individual factors
  - Pairwise combinations
  - Full composite

Each combination is evaluated with proper walk-forward testing.
"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .benchmark import compute_metrics, drawdown_analysis

logger = logging.getLogger(__name__)


class FundamentalCompositeTester:
    """
    Test fundamental factor combinations with walk-forward validation.

    Parameters
    ----------
    train_years : int
        Number of years for training window (default 3).
    test_months : int
        Number of months for test window (default 6).
    rebalance_freq : int
        Rebalance frequency in trading days (default 21).
    """

    # Map strategy names to fundamental factor categories
    FACTOR_MAPPING = {
        # Value factors
        "carry": "value",
        "mean_reversion": "value",
        # Quality/Profitability
        "pead": "quality",
        # Momentum
        "time_series_momentum": "momentum",
        "cross_sectional_momentum": "momentum",
        "factor_momentum": "momentum",
        "residual_momentum": "momentum",
        # Low-risk
        "low_risk_bab": "low_risk",
        # Statistical
        "pca_stat_arb": "statistical",
        "distance_pairs": "statistical",
        "kalman_pairs": "statistical",
        # Volatility
        "volatility_breakout": "volatility",
        "vol_managed": "volatility",
    }

    def __init__(
        self,
        train_years: int = 3,
        test_months: int = 6,
        rebalance_freq: int = 21,
    ) -> None:
        self.train_years = train_years
        self.test_months = test_months
        self.rebalance_freq = rebalance_freq

    def run_composite_test(
        self,
        strategy_returns: dict[str, pd.Series],
        combinations: list[tuple[str, ...]] | None = None,
    ) -> pd.DataFrame:
        """
        Test all specified factor combinations.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Strategy return series keyed by name.
        combinations : list of tuples, optional
            Specific combinations to test. If None, generates all singles,
            pairs, and the full composite.

        Returns
        -------
        pd.DataFrame
            Results indexed by combination name with performance metrics
            for both in-sample and out-of-sample periods.
        """
        if combinations is None:
            combinations = self._generate_combinations(list(strategy_returns.keys()))

        results = []
        for combo in combinations:
            # Filter to available strategies
            available = [s for s in combo if s in strategy_returns]
            if not available:
                continue

            combo_name = " + ".join(available)
            logger.info(f"Testing combination: {combo_name}")

            # Equal-weight blend of strategies in the combination
            combo_returns = self._blend_returns(
                {s: strategy_returns[s] for s in available}
            )

            # Walk-forward evaluation
            is_metrics, oos_metrics = self._walk_forward_evaluate(combo_returns)

            result = {
                "combination": combo_name,
                "n_strategies": len(available),
                "strategies": available,
            }
            for k, v in is_metrics.items():
                result[f"is_{k}"] = v
            for k, v in oos_metrics.items():
                result[f"oos_{k}"] = v

            results.append(result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("oos_sharpe", ascending=False)
        return df

    def _generate_combinations(self, strategy_names: list[str]) -> list[tuple[str, ...]]:
        """Generate all meaningful factor combinations."""
        combos = []

        # Singles
        for name in strategy_names:
            combos.append((name,))

        # Pairs (limit to cross-category pairs for efficiency)
        categories = {}
        for name in strategy_names:
            cat = self.FACTOR_MAPPING.get(name, "other")
            categories.setdefault(cat, []).append(name)

        cat_names = list(categories.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i + 1:]:
                for s1 in categories[cat1][:2]:  # limit per category
                    for s2 in categories[cat2][:2]:
                        combos.append((s1, s2))

        # Triples: one from each of the main categories
        main_cats = ["value", "momentum", "low_risk"]
        triple_candidates = []
        for cat in main_cats:
            if cat in categories:
                triple_candidates.append(categories[cat][:1])

        if len(triple_candidates) >= 3:
            for combo in itertools.product(*triple_candidates):
                combos.append(combo)

        # Full composite
        combos.append(tuple(strategy_names))

        return combos

    def _blend_returns(self, strategy_returns: dict[str, pd.Series]) -> pd.Series:
        """Equal-weight blend of strategy returns."""
        df = pd.DataFrame(strategy_returns)
        return df.mean(axis=1)

    def _walk_forward_evaluate(
        self,
        returns: pd.Series,
    ) -> tuple[dict, dict]:
        """
        Simple walk-forward split: first portion is in-sample, rest is OOS.

        Returns (in_sample_metrics, oos_metrics).
        """
        n = len(returns)
        train_days = self.train_years * 252
        test_days = self.test_months * 21

        if n < train_days + test_days:
            # Not enough data for proper walk-forward; use 70/30 split
            split = int(n * 0.7)
        else:
            split = train_days

        is_returns = returns.iloc[:split]
        oos_returns = returns.iloc[split:]

        is_metrics = compute_metrics(is_returns)
        oos_metrics = compute_metrics(oos_returns) if len(oos_returns) > 20 else {
            k: 0.0 for k in is_metrics
        }

        return is_metrics, oos_metrics

    def rank_combinations(
        self,
        results: pd.DataFrame,
        primary_metric: str = "oos_sharpe",
    ) -> pd.DataFrame:
        """Rank combinations by a primary metric."""
        if results.empty:
            return results
        return results.sort_values(primary_metric, ascending=False).reset_index(drop=True)

    def category_analysis(
        self,
        strategy_returns: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Analyze performance by factor category."""
        rows = []
        categories = {}
        for name, ret in strategy_returns.items():
            cat = self.FACTOR_MAPPING.get(name, "other")
            categories.setdefault(cat, []).append(name)

        for cat, names in categories.items():
            cat_returns = self._blend_returns(
                {n: strategy_returns[n] for n in names if n in strategy_returns}
            )
            metrics = compute_metrics(cat_returns)
            metrics["category"] = cat
            metrics["n_strategies"] = len(names)
            metrics["strategies"] = ", ".join(names)
            rows.append(metrics)

        return pd.DataFrame(rows).set_index("category")
