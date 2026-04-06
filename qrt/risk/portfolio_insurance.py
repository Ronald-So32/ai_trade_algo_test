"""
CPPI Portfolio Insurance & Multi-Horizon Drawdown Control
==========================================================
Implements dynamic portfolio insurance mechanisms to cap maximum drawdown
while preserving upside participation.

Mechanisms:

1. **CPPI (Constant Proportion Portfolio Insurance)** — dynamically adjusts
   risky asset exposure based on the "cushion" (portfolio value - floor).
   When the portfolio approaches the floor, exposure drops toward zero.
   References:
     - Black & Jones (1987), "Simplifying Portfolio Insurance"
     - Perold & Sharpe (1988), "Dynamic Strategies for Asset Allocation"
     - Grossman & Zhou (1993), "Optimal Investment Strategies for Controlling
       Drawdowns" — proved TIPP is optimal under drawdown constraints

2. **Multi-Horizon Drawdown Control** — monitors drawdowns at multiple
   time horizons (1-week, 1-month, 3-month) with independent limits.
   Prevents both fast crashes and slow bleeds from exceeding tolerances.
   Reference:
     - Chekhlov, Uryasev & Zabarankin (2005), "Drawdown Measure in
       Portfolio Optimization" — CDaR family spans max DD to avg DD
     - Boyd et al. (2017), "Multi-Period Trading via Convex Optimization"

3. **Correlation Breakdown Detector** — monitors rolling pairwise
   correlation among strategies. When correlations spike (diversification
   breakdown), exposure is reduced preemptively.
   References:
     - Kritzman & Li (2010), "Skulls, Financial Turbulence, and Risk
       Management" — turbulence detects correlation shifts
     - Silvapulle & Granger (2001), "Large Returns, Conditional Correlation
       and Portfolio Diversification" — asymmetric correlation in crashes
     - Longin & Solnik (2001), "Extreme Correlation of International Equity
       Markets" — correlations increase in bear markets
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CPPI Portfolio Insurance (Black & Jones 1987, Grossman & Zhou 1993)
# ---------------------------------------------------------------------------

class CPPIInsurance:
    """
    Constant Proportion Portfolio Insurance with drawdown floor.

    The CPPI mechanism works by maintaining:
        exposure_t = multiplier × cushion_t
        cushion_t  = portfolio_value_t - floor_t

    where floor_t is a time-varying floor that ratchets up with portfolio
    highs (TIPP variant per Grossman & Zhou 1993).

    When the portfolio drops toward the floor, exposure decreases
    automatically.  When far above the floor, exposure can exceed 1.0
    (leveraged) up to max_exposure.

    Parameters
    ----------
    max_drawdown : float
        Maximum tolerable drawdown from peak (default 0.15 = 15%).
    multiplier : float
        CPPI multiplier — higher = more aggressive (default 3.0).
        m=3 means exposure = 3× the cushion above the floor.
    max_exposure : float
        Maximum exposure multiplier (default 1.5 = 150%).
    min_exposure : float
        Minimum exposure even at floor (default 0.05 = 5%).
    ratchet : bool
        If True, floor ratchets up with new highs (TIPP). Default True.
    ratchet_pct : float
        Fraction of new highs added to floor (default 0.8).
        Higher = more aggressive floor ratcheting.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        multiplier: float = 3.0,
        max_exposure: float = 1.5,
        min_exposure: float = 0.05,
        ratchet: bool = True,
        ratchet_pct: float = 0.80,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.multiplier = multiplier
        self.max_exposure = max_exposure
        self.min_exposure = min_exposure
        self.ratchet = ratchet
        self.ratchet_pct = ratchet_pct

    def compute_exposure(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute CPPI exposure multipliers from a return series.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.

        Returns
        -------
        pd.Series
            Exposure multipliers in [min_exposure, max_exposure].
        """
        cum = (1 + returns).cumprod()
        n = len(cum)

        exposure = pd.Series(1.0, index=returns.index)
        floor_val = cum.iloc[0] * (1 - self.max_drawdown)
        hwm = cum.iloc[0]

        for t in range(1, n):
            port_val = cum.iloc[t]

            # Update HWM and ratchet floor
            if port_val > hwm:
                hwm = port_val
                if self.ratchet:
                    new_floor = hwm * (1 - self.max_drawdown)
                    # Only ratchet up, never down
                    floor_val = max(floor_val,
                                    floor_val + self.ratchet_pct * (new_floor - floor_val))

            # Cushion: how far above the floor
            cushion = (port_val - floor_val) / port_val if port_val > 0 else 0.0
            cushion = max(cushion, 0.0)

            # CPPI exposure
            exp = self.multiplier * cushion
            exp = np.clip(exp, self.min_exposure, self.max_exposure)

            exposure.iloc[t] = exp

        return exposure

    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply CPPI insurance to portfolio weights.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates × assets).
        returns : pd.DataFrame
            Asset returns aligned with weights.

        Returns
        -------
        pd.DataFrame
            CPPI-adjusted weights.
        """
        # Compute portfolio returns from current weights
        port_returns = (weights.shift(1) * returns).sum(axis=1)

        # Get CPPI exposure multipliers
        exposure = self.compute_exposure(port_returns)

        # Apply exposure scaling
        adjusted = weights.mul(exposure, axis=0)

        # Log diagnostics
        avg_exp = exposure.mean()
        min_exp = exposure.min()
        pct_reduced = (exposure < 0.95).mean() * 100

        logger.info(
            "CPPI insurance: avg_exposure=%.2f, min_exposure=%.2f, "
            "reduced on %.1f%% of days, max_dd_target=%.1f%%",
            avg_exp, min_exp, pct_reduced, self.max_drawdown * 100,
        )

        return adjusted


# ---------------------------------------------------------------------------
# 2. Multi-Horizon Drawdown Control (Chekhlov et al. 2005, Boyd et al. 2017)
# ---------------------------------------------------------------------------

class MultiHorizonDrawdownControl:
    """
    Monitor and control drawdowns at multiple time horizons simultaneously.

    Instead of a single max_dd threshold, this enforces independent limits
    at short (1 week), medium (1 month), and long (3 month) horizons.

    This catches both:
    - Fast crashes (caught by short horizon)
    - Slow bleeds (caught by medium/long horizons before they accumulate)

    Parameters
    ----------
    horizons : dict[str, dict]
        Configuration per horizon. Keys are horizon names, values contain:
        - window: int (lookback in trading days)
        - max_dd: float (maximum drawdown for this horizon)
        - weight: float (importance weight for combining signals)
    """

    DEFAULT_HORIZONS = {
        "1w": {"window": 5, "max_dd": 0.03, "weight": 0.3},
        "1m": {"window": 21, "max_dd": 0.07, "weight": 0.4},
        "3m": {"window": 63, "max_dd": 0.12, "weight": 0.3},
    }

    def __init__(
        self,
        horizons: Optional[dict] = None,
        floor: float = 0.10,
    ) -> None:
        self.horizons = horizons or self.DEFAULT_HORIZONS
        self.floor = floor

    def compute_scaling(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute exposure scaling based on multi-horizon drawdown monitoring.

        For each horizon, if the rolling drawdown exceeds the threshold,
        scaling is reduced proportionally.  The final scaling is the
        weighted average across horizons.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0].
        """
        scaling_components = {}

        for name, cfg in self.horizons.items():
            window = cfg["window"]
            max_dd = cfg["max_dd"]
            weight = cfg["weight"]

            # Rolling drawdown for this horizon
            rolling_cum = returns.rolling(window, min_periods=max(1, window // 2)).sum()
            rolling_dd = rolling_cum.clip(upper=0)  # only negative values

            # Scale: 1.0 when no dd, linearly to floor at max_dd
            horizon_scale = pd.Series(1.0, index=returns.index)
            breached = rolling_dd.abs() > max_dd * 0.5  # start at 50% of limit
            if breached.any():
                progress = ((rolling_dd.abs()[breached] - max_dd * 0.5) / (max_dd * 0.5)).clip(0, 1)
                horizon_scale[breached] = 1.0 - progress * (1.0 - self.floor)

            scaling_components[name] = (horizon_scale, weight)

        # Weighted combination
        total_weight = sum(w for _, w in scaling_components.values())
        combined = pd.Series(0.0, index=returns.index)
        for name, (scale, weight) in scaling_components.items():
            combined += scale * (weight / total_weight)

        combined = combined.clip(lower=self.floor, upper=1.0)

        n_reduced = (combined < 0.95).sum()
        logger.info(
            "Multi-horizon DD control: %d/%d days reduced (%.1f%%), "
            "horizons=%s",
            n_reduced, len(combined), 100 * n_reduced / max(1, len(combined)),
            list(self.horizons.keys()),
        )

        return combined

    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply multi-horizon drawdown scaling to weights."""
        port_returns = (weights.shift(1) * returns).sum(axis=1)
        scaling = self.compute_scaling(port_returns)
        return weights.mul(scaling, axis=0)


# ---------------------------------------------------------------------------
# 3. Correlation Breakdown Detector (Longin & Solnik 2001, Silvapulle 2001)
# ---------------------------------------------------------------------------

class CorrelationBreakdownDetector:
    """
    Detect correlation spikes that signal diversification breakdown.

    During crises, cross-asset correlations spike toward 1.0, destroying
    diversification exactly when it's needed most (Longin & Solnik 2001).

    This detector monitors rolling average pairwise correlation among
    strategies and reduces exposure when it exceeds historical norms.

    Parameters
    ----------
    lookback : int
        Rolling window for correlation estimation (default 63).
    baseline_lookback : int
        Window for establishing baseline correlation (default 252).
    threshold_z : float
        Z-score of correlation above which scaling begins (default 1.5).
    critical_z : float
        Z-score at which exposure hits floor (default 3.0).
    floor : float
        Minimum exposure (default 0.25).
    """

    def __init__(
        self,
        lookback: int = 63,
        baseline_lookback: int = 252,
        threshold_z: float = 1.5,
        critical_z: float = 3.0,
        floor: float = 0.25,
    ) -> None:
        self.lookback = lookback
        self.baseline_lookback = baseline_lookback
        self.threshold_z = threshold_z
        self.critical_z = critical_z
        self.floor = floor

    def compute_avg_correlation(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute rolling average pairwise correlation.

        Parameters
        ----------
        returns : pd.DataFrame
            Strategy or asset returns (dates × N).

        Returns
        -------
        pd.Series
            Average pairwise correlation per date.
        """
        n_cols = returns.shape[1]
        if n_cols < 2:
            return pd.Series(0.0, index=returns.index)

        # Use rolling correlation of all pairs
        avg_corr = pd.Series(np.nan, index=returns.index)

        for t in range(self.lookback, len(returns)):
            window = returns.iloc[t - self.lookback: t]
            corr_mat = window.corr().values

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            pairwise = corr_mat[mask]
            pairwise = pairwise[~np.isnan(pairwise)]

            if len(pairwise) > 0:
                avg_corr.iloc[t] = pairwise.mean()

        return avg_corr

    def compute_scaling(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute exposure scaling based on correlation dynamics.

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0].
        """
        avg_corr = self.compute_avg_correlation(returns)

        scaling = pd.Series(1.0, index=returns.index)

        for t in range(self.baseline_lookback + self.lookback, len(returns)):
            # Baseline: expanding or rolling mean/std of avg correlation
            baseline = avg_corr.iloc[self.lookback: t]
            baseline = baseline.dropna()

            if len(baseline) < 30:
                continue

            mu = baseline.mean()
            sigma = baseline.std()
            if sigma < 1e-6:
                continue

            current = avg_corr.iloc[t]
            if np.isnan(current):
                continue

            z = (current - mu) / sigma

            if z <= self.threshold_z:
                scaling.iloc[t] = 1.0
            elif z >= self.critical_z:
                scaling.iloc[t] = self.floor
            else:
                progress = (z - self.threshold_z) / (self.critical_z - self.threshold_z)
                scaling.iloc[t] = 1.0 - progress * (1.0 - self.floor)

        n_reduced = (scaling < 0.95).sum()
        logger.info(
            "Correlation breakdown detector: %d/%d days reduced (%.1f%%)",
            n_reduced, len(scaling), 100 * n_reduced / max(1, len(scaling)),
        )

        return scaling

    def apply(
        self,
        weights: pd.DataFrame,
        strategy_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply correlation-based scaling to weights."""
        scaling = self.compute_scaling(strategy_returns)
        return weights.mul(scaling.reindex(weights.index).fillna(1.0), axis=0)


# ---------------------------------------------------------------------------
# 4. Combined Drawdown Shield (wraps all three mechanisms)
# ---------------------------------------------------------------------------

class DrawdownShield:
    """
    Composite drawdown protection combining CPPI, multi-horizon control,
    and correlation breakdown detection.

    The final exposure is the product of all three scaling factors,
    ensuring the most conservative action when multiple risk signals fire.

    Parameters
    ----------
    cppi_config : dict, optional
        kwargs for CPPIInsurance.
    multi_horizon_config : dict, optional
        kwargs for MultiHorizonDrawdownControl.
    correlation_config : dict, optional
        kwargs for CorrelationBreakdownDetector.
    enable_cppi : bool
        Enable CPPI insurance layer (default True).
    enable_multi_horizon : bool
        Enable multi-horizon drawdown control (default True).
    enable_correlation : bool
        Enable correlation breakdown detection (default True).
    """

    def __init__(
        self,
        cppi_config: Optional[dict] = None,
        multi_horizon_config: Optional[dict] = None,
        correlation_config: Optional[dict] = None,
        enable_cppi: bool = True,
        enable_multi_horizon: bool = True,
        enable_correlation: bool = True,
    ) -> None:
        self.enable_cppi = enable_cppi
        self.enable_multi_horizon = enable_multi_horizon
        self.enable_correlation = enable_correlation

        if enable_cppi:
            self.cppi = CPPIInsurance(**(cppi_config or {}))
        if enable_multi_horizon:
            self.multi_horizon = MultiHorizonDrawdownControl(
                **(multi_horizon_config or {})
            )
        if enable_correlation:
            self.corr_detector = CorrelationBreakdownDetector(
                **(correlation_config or {})
            )

    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        strategy_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply all enabled drawdown protection layers.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates × assets).
        returns : pd.DataFrame
            Asset returns.
        strategy_returns : pd.DataFrame, optional
            Strategy-level returns for correlation monitoring.
            If None, uses asset-level returns.

        Returns
        -------
        pd.DataFrame
            Protected weights.
        """
        adjusted = weights.copy()

        # Layer 1: CPPI portfolio insurance
        if self.enable_cppi:
            adjusted = self.cppi.apply(adjusted, returns)

        # Layer 2: Multi-horizon drawdown control
        if self.enable_multi_horizon:
            adjusted = self.multi_horizon.apply(adjusted, returns)

        # Layer 3: Correlation breakdown detection
        if self.enable_correlation:
            corr_input = strategy_returns if strategy_returns is not None else returns
            # Use subset for speed if many columns
            if corr_input.shape[1] > 20:
                corr_input = corr_input.iloc[:, :20]
            adjusted = self.corr_detector.apply(adjusted, corr_input)

        # Final safety check: compute actual drawdown after all layers
        port_ret = (adjusted.shift(1) * returns).sum(axis=1)
        cum = (1 + port_ret).cumprod()
        actual_dd = (cum / cum.cummax() - 1).min()

        logger.info(
            "DrawdownShield: actual MaxDD after protection = %.2f%%",
            actual_dd * 100,
        )

        return adjusted
