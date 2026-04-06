"""
Enhanced Portfolio Allocation
==============================
Integrates advanced risk management techniques into the portfolio allocation
pipeline, providing a unified interface that combines:

1. Multiple allocation methods (CVaR, downside RP, max diversification)
2. Systemic risk overlays (turbulence, absorption ratio)
3. Adaptive position management (regime-aware stops)
4. Time-varying strategy weighting with risk-budget constraints

The core idea: allocation decisions should be driven by *tail risk*
and *diversification quality*, not just variance and Sharpe ratios.

References
----------
- Rockafellar & Uryasev (2000) — CVaR optimization
- Sortino & van der Meer (1991) — Downside risk
- Choueifaty & Coignard (2008) — Maximum diversification
- Kritzman & Li (2010) — Turbulence index
- Kritzman et al. (2011) — Absorption ratio
- Kaminski & Lo (2014) — Adaptive stop-losses
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from qrt.risk.advanced_risk import (
    TurbulenceIndex,
    AbsorptionRatio,
    AdaptiveStopLoss,
    DownsideRiskParity,
    CVaROptimizer,
    MaxDiversification,
    CompositeRiskOverlay,
)
from qrt.risk.drawdown_risk import compute_drawdown_series, compute_cvar, compute_cdar

logger = logging.getLogger(__name__)


class EnhancedAllocator:
    """
    Multi-layer portfolio allocator that combines allocation optimisation
    with risk overlays for robust loss reduction.

    Pipeline:
    1. Choose base allocation (CVaR, downside RP, max diversification, or blend)
    2. Apply systemic risk overlay (turbulence + absorption ratio)
    3. Apply adaptive stops per strategy
    4. Apply continuous drawdown scaling
    5. Enforce leverage and position constraints

    Parameters
    ----------
    allocation_method : str
        Base allocation: "cvar", "downside_rp", "max_div", or "blend".
        "blend" takes the average of all three (most robust).
    apply_systemic_overlay : bool
        Apply turbulence + absorption ratio overlay (default True).
    apply_adaptive_stops : bool
        Apply regime-aware adaptive stops (default True).
    rebalance_freq : int
        Rebalance allocation every N days (default 21).
    min_weight : float
        Minimum per-strategy weight (default 0.02).
    max_weight : float
        Maximum per-strategy weight (default 0.40).
    max_leverage : float
        Maximum gross leverage (default 2.0).
    target_vol : float
        Target portfolio volatility (default 0.10).
    """

    def __init__(
        self,
        allocation_method: str = "blend",
        apply_systemic_overlay: bool = True,
        apply_adaptive_stops: bool = True,
        rebalance_freq: int = 21,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        max_leverage: float = 2.0,
        target_vol: float = 0.10,
    ) -> None:
        self.allocation_method = allocation_method
        self.apply_systemic_overlay = apply_systemic_overlay
        self.apply_adaptive_stops = apply_adaptive_stops
        self.rebalance_freq = rebalance_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_leverage = max_leverage
        self.target_vol = target_vol

        # Sub-components
        self.cvar_optimizer = CVaROptimizer(
            min_weight=min_weight, max_weight=max_weight,
        )
        self.downside_rp = DownsideRiskParity(
            min_weight=min_weight, max_weight=max_weight,
        )
        self.max_div = MaxDiversification(
            min_weight=min_weight, max_weight=max_weight,
        )
        self.risk_overlay = CompositeRiskOverlay(combination="min")
        self.adaptive_stops = AdaptiveStopLoss(
            vol_lookback=21, stop_multiplier=2.0, cooldown_days=10,
        )

    def compute_allocation(
        self,
        strategy_returns: dict[str, pd.Series],
        regime_labels: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Compute time-varying enhanced allocation weights.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Per-strategy daily return series.
        regime_labels : pd.Series, optional
            Regime labels per date.

        Returns
        -------
        weights_df : pd.DataFrame
            Time-varying strategy weights (dates x strategies).
        diagnostics : dict
            Detailed diagnostics for analysis.
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
        names = list(returns_df.columns)
        n_strats = len(names)
        dates = returns_df.index
        min_history = 252

        diagnostics = {
            "allocation_method": self.allocation_method,
            "n_strategies": n_strats,
        }

        # -- Step 1: Compute base allocation weights (rolling) --
        weights_arr = np.full((len(dates), n_strats), 1.0 / n_strats)
        last_weights = np.ones(n_strats) / n_strats

        allocation_log = []

        for t in range(len(dates)):
            if t < min_history or t % self.rebalance_freq != 0:
                weights_arr[t] = last_weights
                continue

            window = returns_df.iloc[max(0, t - min_history) : t]

            try:
                w = self._compute_base_weights(window)
                weights_arr[t] = w
                last_weights = w
                allocation_log.append({
                    "date": dates[t],
                    "method": self.allocation_method,
                    "weights": dict(zip(names, w)),
                })
            except Exception as e:
                logger.warning(
                    "Allocation failed at %s: %s. Using previous weights.",
                    dates[t], e,
                )
                weights_arr[t] = last_weights

        weights_df = pd.DataFrame(weights_arr, index=dates, columns=names)

        # -- Step 2: Apply systemic risk overlay --
        if self.apply_systemic_overlay and n_strats >= 2:
            overlay_scaling = self.risk_overlay.compute_scaling(returns_df)
            weights_df = weights_df.mul(overlay_scaling, axis=0)

            # Renormalise to maintain full investment
            row_sums = weights_df.sum(axis=1).clip(lower=1e-8)
            weights_df = weights_df.div(row_sums, axis=0)

            diagnostics["overlay_scaling_mean"] = float(overlay_scaling.mean())
            diagnostics["overlay_days_reduced"] = int((overlay_scaling < 1.0).sum())

        # -- Step 3: Vol targeting --
        portfolio_returns = (returns_df * weights_df.shift(1)).sum(axis=1)
        vol_scale = self._compute_vol_scaling(portfolio_returns)
        weights_df = weights_df.mul(vol_scale, axis=0)

        diagnostics["vol_scale_mean"] = float(vol_scale.mean())

        # -- Step 4: Enforce leverage constraint --
        gross = weights_df.abs().sum(axis=1)
        over_leveraged = gross > self.max_leverage
        if over_leveraged.any():
            scale_down = self.max_leverage / gross[over_leveraged]
            weights_df.loc[over_leveraged] = weights_df.loc[over_leveraged].mul(
                scale_down, axis=0
            )

        logger.info(
            "Enhanced allocation: method=%s, %d strategies, "
            "overlay=%s, adaptive_stops=%s",
            self.allocation_method, n_strats,
            self.apply_systemic_overlay, self.apply_adaptive_stops,
        )

        return weights_df, diagnostics

    def _compute_base_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Compute base allocation weights using the specified method."""
        if self.allocation_method == "cvar":
            w = self.cvar_optimizer.allocate(returns_window)
        elif self.allocation_method == "downside_rp":
            w = self.downside_rp.allocate(returns_window)
        elif self.allocation_method == "max_div":
            w = self.max_div.allocate(returns_window)
        elif self.allocation_method == "blend":
            # Blend of all three — most robust
            w_cvar = self.cvar_optimizer.allocate(returns_window).values
            w_drp = self.downside_rp.allocate(returns_window).values
            w_mdiv = self.max_div.allocate(returns_window).values
            w_blend = (w_cvar + w_drp + w_mdiv) / 3.0
            w_blend = np.clip(w_blend, self.min_weight, self.max_weight)
            w_blend = w_blend / w_blend.sum()
            return w_blend
        else:
            raise ValueError(f"Unknown allocation method: {self.allocation_method}")

        return w.values

    def _compute_vol_scaling(self, portfolio_returns: pd.Series) -> pd.Series:
        """Asymmetric volatility targeting — more aggressive on downside."""
        vol = portfolio_returns.rolling(63, min_periods=21).std() * np.sqrt(252)
        vol = vol.clip(lower=1e-6)

        scaling = self.target_vol / vol
        scaling = scaling.clip(upper=self.max_leverage)

        # Asymmetric: on large negative days, reduce further
        z = (portfolio_returns - portfolio_returns.rolling(63, min_periods=21).mean()) / \
            portfolio_returns.rolling(63, min_periods=21).std().clip(lower=1e-8)
        extreme_neg = z < -2.0
        scaling[extreme_neg] *= 0.7

        return scaling.fillna(1.0)

    def apply_strategy_stops(
        self,
        strategy_weights: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Apply adaptive stops to individual strategy weight matrices.

        Parameters
        ----------
        strategy_weights : dict[str, pd.DataFrame]
            Per-strategy weight DataFrames (dates x assets).
        returns : pd.DataFrame
            Asset returns.

        Returns
        -------
        dict[str, pd.DataFrame]
            Weights with adaptive stops applied.
        """
        if not self.apply_adaptive_stops:
            return strategy_weights

        result = {}
        for name, weights in strategy_weights.items():
            result[name] = self.adaptive_stops.apply_adaptive_stops(weights, returns)

        return result


# ---------------------------------------------------------------------------
# Allocation Comparison Framework
# ---------------------------------------------------------------------------

class AllocationComparator:
    """
    Compare different allocation methods head-to-head on the same strategy
    returns, computing risk/return metrics for each.

    This enables rigorous evaluation of which allocation method best
    reduces risk while maintaining returns.
    """

    METHODS = ["equal_weight", "cvar", "downside_rp", "max_div", "blend"]

    def __init__(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.50,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compare(
        self,
        strategy_returns: pd.DataFrame,
        min_history: int = 252,
    ) -> pd.DataFrame:
        """
        Run all allocation methods and compare metrics.

        Parameters
        ----------
        strategy_returns : pd.DataFrame
            Strategy returns (dates x strategies).
        min_history : int
            Minimum history before computing allocation.

        Returns
        -------
        pd.DataFrame
            Comparison table with metrics per method.
        """
        from qrt.risk.enhanced_metrics import compute_full_metrics

        results = []

        for method in self.METHODS:
            try:
                if method == "equal_weight":
                    combined = strategy_returns.mean(axis=1)
                else:
                    # Use second half for evaluation, first half for fitting
                    T = len(strategy_returns)
                    train = strategy_returns.iloc[:min_history]
                    test = strategy_returns.iloc[min_history:]

                    if len(test) < 63:
                        logger.warning("Insufficient test data for %s", method)
                        continue

                    if method == "cvar":
                        alloc = CVaROptimizer(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        )
                    elif method == "downside_rp":
                        alloc = DownsideRiskParity(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        )
                    elif method == "max_div":
                        alloc = MaxDiversification(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        )
                    elif method == "blend":
                        # Average of all three
                        w_c = CVaROptimizer(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        ).allocate(train)
                        w_d = DownsideRiskParity(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        ).allocate(train)
                        w_m = MaxDiversification(
                            min_weight=self.min_weight, max_weight=self.max_weight,
                        ).allocate(train)
                        w = (w_c + w_d + w_m) / 3
                        w = w / w.sum()
                        combined = (test * w).sum(axis=1)
                        metrics = compute_full_metrics(combined, name=method)
                        results.append(metrics)
                        continue
                    else:
                        continue

                    w = alloc.allocate(train)
                    combined = (test * w).sum(axis=1)

                metrics = compute_full_metrics(combined, name=method)
                results.append(metrics)

            except Exception as e:
                logger.warning("Allocation method %s failed: %s", method, e)
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index("name")
