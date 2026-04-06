"""
Portfolio optimiser: combine multiple strategy return streams into a
single, volatility-targeted portfolio.

Workflow
--------
1. Receive a dict of per-strategy daily return series.
2. Estimate a covariance matrix from those returns.
3. Apply risk parity to get cross-strategy weights.
4. Build a combined return series from those weights.
5. Apply volatility targeting to scale aggregate exposure.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from .risk_parity import RiskParityAllocator, AllocationMethod
from .vol_targeting import VolatilityTargeter

logger = logging.getLogger(__name__)

CombineMethod = Literal["risk_parity", "equal_weight"]


class PortfolioOptimizer:
    """
    Combine multiple strategy return streams into a single portfolio.

    Parameters
    ----------
    min_weight : float
        Minimum weight for any single strategy (default 0.0).
    max_weight : float
        Maximum weight for any single strategy (default 1.0).
    annualisation_factor : int
        Periods per year used in volatility calculations (default 252).
    """

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        annualisation_factor: int = 252,
    ) -> None:
        self._allocator = RiskParityAllocator(
            min_weight=min_weight, max_weight=max_weight
        )
        self._vol_targeter = VolatilityTargeter(
            annualisation_factor=annualisation_factor
        )
        self.min_weight = min_weight
        self.max_weight = max_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine_strategies(
        self,
        strategy_returns: Dict[str, pd.Series],
        method: CombineMethod = "risk_parity",
        allocation_method: AllocationMethod = "covariance",
        min_history: int = 63,
        crisis_prob: float = 0.0,
    ) -> pd.Series:
        """
        Combine strategies into a single daily return series.

        The static weights are estimated once from the full history
        supplied (intended for the in-sample training window).

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Mapping of strategy name to daily return series.
        method : {"risk_parity", "equal_weight"}
            Weighting scheme used to combine strategies.
        allocation_method : {"naive", "covariance"}
            Sub-method passed to :class:`RiskParityAllocator` when
            ``method="risk_parity"``.
        min_history : int
            Minimum number of overlapping observations required to
            estimate weights (default 63).

        Returns
        -------
        combined_returns : pd.Series
            Daily combined portfolio return series.
        """
        if not strategy_returns:
            raise ValueError("strategy_returns dict is empty.")

        # Build aligned returns DataFrame (inner join on dates)
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all")
        if returns_df.shape[0] < min_history:
            logger.warning(
                "Only %d overlapping observations — fewer than "
                "min_history=%d.  Proceeding with equal weights.",
                returns_df.shape[0],
                min_history,
            )
            method = "equal_weight"

        if method == "equal_weight" or returns_df.shape[1] == 1:
            weights = pd.Series(
                np.ones(returns_df.shape[1]) / returns_df.shape[1],
                index=returns_df.columns,
                name="weight",
            )
        elif method == "risk_parity":
            weights = self._allocator.allocate(
                returns_df, method=allocation_method, crisis_prob=crisis_prob,
            )
        else:
            raise ValueError(
                f"Unknown combine method '{method}'. "
                "Choose 'risk_parity' or 'equal_weight'."
            )

        logger.info(
            "Strategy weights (%s / %s):\n%s",
            method,
            allocation_method,
            weights.to_string(),
        )

        combined = (returns_df * weights).sum(axis=1)
        combined.name = "combined_returns"
        return combined

    def apply_vol_target(
        self,
        combined_weights: pd.Series,
        portfolio_returns: pd.Series,
        target_vol: float = 0.10,
        lookback: int = 63,
        max_leverage: float = 2.0,
    ) -> pd.Series:
        """
        Scale a combined weight (or return) series to hit a vol target.

        Typically called after :meth:`combine_strategies`.  The scaling
        is computed from ``portfolio_returns`` (the in-sample history)
        and then applied forward to ``combined_weights``.

        Parameters
        ----------
        combined_weights : pd.Series
            Daily combined strategy weights or allocations.
        portfolio_returns : pd.Series
            Return series used to estimate realised volatility.
        target_vol : float
            Annualised volatility target (default 0.10 = 10 %).
        lookback : int
            Rolling window for realised vol estimation.
        max_leverage : float
            Maximum scaling factor (leverage cap).

        Returns
        -------
        scaled_weights : pd.Series
            Vol-targeted version of ``combined_weights``.
        """
        scaling = self._vol_targeter.compute_scaling(
            portfolio_returns,
            target_vol=target_vol,
            lookback=lookback,
            max_leverage=max_leverage,
        )

        # Align on common dates
        common = combined_weights.index.intersection(scaling.index)
        if common.empty:
            raise ValueError(
                "combined_weights and portfolio_returns share no common dates."
            )

        scaled = combined_weights.loc[common].mul(
            scaling.reindex(common).fillna(1.0)
        )
        scaled.name = "scaled_weights"
        return scaled

    def build_portfolio(
        self,
        strategy_returns: Dict[str, pd.Series],
        combine_method: CombineMethod = "risk_parity",
        allocation_method: AllocationMethod = "covariance",
        target_vol: float = 0.10,
        lookback: int = 63,
        max_leverage: float = 2.0,
        min_history: int = 63,
        crisis_prob: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Full pipeline: combine strategies then apply volatility targeting.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Per-strategy daily return series.
        combine_method : str
            How to combine strategies ("risk_parity" or "equal_weight").
        allocation_method : str
            Sub-method for risk parity ("naive" or "covariance").
        target_vol : float
            Annualised volatility target.
        lookback : int
            Lookback window for realised vol.
        max_leverage : float
            Leverage cap.
        min_history : int
            Minimum history length to use risk parity (else equal weight).

        Returns
        -------
        result : dict with keys:
            - ``"combined_returns"``: raw combined return series.
            - ``"scaled_returns"``: vol-targeted return series.
            - ``"scaling_factors"``: applied scaling factors.
        """
        combined = self.combine_strategies(
            strategy_returns,
            method=combine_method,
            allocation_method=allocation_method,
            min_history=min_history,
            crisis_prob=crisis_prob,
        )

        scaling = self._vol_targeter.compute_scaling(
            combined,
            target_vol=target_vol,
            lookback=lookback,
            max_leverage=max_leverage,
        )

        scaled = combined.mul(scaling.reindex(combined.index).fillna(1.0))
        scaled.name = "scaled_returns"

        return {
            "combined_returns": combined,
            "scaled_returns": scaled,
            "scaling_factors": scaling.reindex(combined.index).fillna(1.0),
        }
