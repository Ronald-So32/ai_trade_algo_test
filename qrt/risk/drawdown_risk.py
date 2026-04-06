"""
Drawdown Risk Management
=========================
Replaces binary circuit-breaker stop-outs with continuous, drawdown-aware
risk scaling based on CDaR (Conditional Drawdown-at-Risk).

Key insight from the research: binary stop-out rules can reduce expected
return under random-walk-like dynamics and lock in drawdowns by missing
recoveries.  A continuous drawdown-aware scaler reduces exposure
*proportionally* as drawdown deepens, avoiding the whipsaw behaviour of
hard on/off switches.

References
----------
- Chekhlov, Uryasev & Zabarankin (2005), "Drawdown Measure in Portfolio
  Optimization" — defines CDaR as the mean of the worst tail of drawdowns.
- Goldberg & Mahmoud (2017), "Drawdown: From Practice to Theory and Back
  Again" — practical drawdown risk frameworks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core drawdown computations
# ---------------------------------------------------------------------------

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute running drawdown from a return series.

    Returns
    -------
    pd.Series
        Drawdown values (negative numbers, e.g. -0.15 = 15% drawdown).
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return dd


def compute_cdar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Conditional Drawdown-at-Risk: mean of the worst (1-alpha) tail of
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    alpha : float
        Confidence level (default 0.95 → mean of worst 5% drawdowns).

    Returns
    -------
    float
        CDaR value (positive number representing loss magnitude).
    """
    dd = compute_drawdown_series(returns)
    dd_abs = dd.abs()
    threshold = dd_abs.quantile(alpha)
    tail = dd_abs[dd_abs >= threshold]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (Expected Shortfall): mean of the worst
    (1-alpha) tail of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    alpha : float
        Confidence level (default 0.95 → mean of worst 5% daily losses).

    Returns
    -------
    float
        CVaR value (positive number representing loss magnitude).
    """
    threshold = returns.quantile(1 - alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Continuous drawdown-aware scaling
# ---------------------------------------------------------------------------

class ContinuousDrawdownScaler:
    """
    Continuous drawdown-aware risk scaling — replaces binary circuit breaker.

    Instead of a hard on/off switch at a drawdown threshold, this scaler
    reduces exposure *proportionally* as drawdown deepens, following a
    smooth sigmoid-like ramp:

        scale(dd) = 1.0                              if |dd| < soft_start
        scale(dd) = 1 - (|dd| - soft_start) / ramp   if soft_start ≤ |dd| < hard_limit
        scale(dd) = floor                             if |dd| ≥ hard_limit

    This avoids the "whipsaw" problem where binary stop-outs lock in
    losses and miss recoveries.

    Parameters
    ----------
    max_dd : float
        Maximum tolerable drawdown (e.g. 0.20 = 20%).
    soft_start_pct : float
        Fraction of max_dd at which scaling begins (default 0.50 → at 10%
        DD if max_dd=0.20).
    floor : float
        Minimum exposure scaling factor (default 0.05 → never fully zero).
    recovery_speed : float
        How quickly exposure ramps back up after drawdown eases.
        1.0 = immediate, <1.0 = slower recovery (default 0.5).
    """

    def __init__(
        self,
        max_dd: float = 0.20,
        soft_start_pct: float = 0.50,
        floor: float = 0.05,
        recovery_speed: float = 0.5,
    ) -> None:
        self.max_dd = max_dd
        self.soft_start = max_dd * soft_start_pct
        self.hard_limit = max_dd * 0.95  # start scaling well before limit
        self.floor = floor
        self.recovery_speed = recovery_speed

    def compute_scaling(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute continuous drawdown-aware scaling factors.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates x assets).
        returns : pd.DataFrame
            Asset returns aligned with weights.

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0], same index as weights.
        """
        # Compute strategy returns from current weights
        strat_returns = (weights.shift(1) * returns).sum(axis=1)
        dd = compute_drawdown_series(strat_returns)
        dd_abs = dd.abs()

        ramp = self.hard_limit - self.soft_start
        if ramp <= 0:
            ramp = 1e-8

        # Compute raw scaling: linear ramp from 1.0 to floor
        raw_scale = pd.Series(1.0, index=weights.index)

        # Region where scaling kicks in
        scaling_zone = dd_abs > self.soft_start
        if scaling_zone.any():
            progress = ((dd_abs[scaling_zone] - self.soft_start) / ramp).clip(0, 1)
            raw_scale[scaling_zone] = 1.0 - progress * (1.0 - self.floor)

        raw_scale = raw_scale.clip(lower=self.floor, upper=1.0)

        # Smooth recovery: use EMA to prevent rapid oscillation
        # When drawdown eases, scale recovers at recovery_speed rate
        smoothed = raw_scale.copy()
        for i in range(1, len(smoothed)):
            prev = smoothed.iloc[i - 1]
            curr = raw_scale.iloc[i]
            if curr > prev:
                # Recovery phase: smooth upward
                smoothed.iloc[i] = prev + self.recovery_speed * (curr - prev)
            else:
                # Drawdown deepening: respond immediately
                smoothed.iloc[i] = curr

        smoothed.name = "dd_scaling"
        return smoothed

    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        n_passes: int = 3,
    ) -> pd.DataFrame:
        """
        Apply continuous drawdown scaling to weights, iterating to converge.

        Parameters
        ----------
        weights : pd.DataFrame
            Raw portfolio weights.
        returns : pd.DataFrame
            Asset returns.
        n_passes : int
            Number of scaling passes (default 3).

        Returns
        -------
        pd.DataFrame
            Adjusted weights with drawdown protection.
        """
        adjusted = weights.copy()

        for _pass in range(n_passes):
            scaling = self.compute_scaling(adjusted, returns)
            adjusted = adjusted.mul(scaling, axis=0)

            # Check if we're within target
            strat_returns = (adjusted.shift(1) * returns).sum(axis=1)
            actual_dd = compute_drawdown_series(strat_returns).min()
            if abs(actual_dd) <= self.max_dd:
                break

        return adjusted


# ---------------------------------------------------------------------------
# CDaR-constrained weight adjustment (portfolio level)
# ---------------------------------------------------------------------------

class CDaRRiskBudget:
    """
    CDaR-aware risk budgeting at the portfolio level.

    Uses rolling CDaR estimates to scale strategy allocations, directing
    risk budget away from strategies with high tail drawdown risk.

    Parameters
    ----------
    cdar_alpha : float
        CDaR confidence level (default 0.95).
    lookback : int
        Rolling window for CDaR estimation (default 252).
    max_cdar : float
        Maximum acceptable CDaR per strategy (default 0.15).
    rebalance_freq : int
        Rebalance CDaR weights every N days (default 21).
    """

    def __init__(
        self,
        cdar_alpha: float = 0.95,
        lookback: int = 252,
        max_cdar: float = 0.15,
        rebalance_freq: int = 21,
    ) -> None:
        self.cdar_alpha = cdar_alpha
        self.lookback = lookback
        self.max_cdar = max_cdar
        self.rebalance_freq = rebalance_freq

    def compute_cdar_weights(
        self,
        strategy_returns: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compute time-varying CDaR-aware strategy weights.

        Strategies with higher rolling CDaR get lower weight, inversely
        proportional to their tail drawdown risk.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Per-strategy daily return series.

        Returns
        -------
        pd.DataFrame
            CDaR-adjusted weights (dates x strategies).
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
        names = list(returns_df.columns)
        n_strats = len(names)
        dates = returns_df.index

        weights = np.full((len(dates), n_strats), 1.0 / n_strats)
        last_weights = np.ones(n_strats) / n_strats

        for t in range(len(dates)):
            if t < self.lookback or t % self.rebalance_freq != 0:
                weights[t] = last_weights
                continue

            # Compute rolling CDaR for each strategy
            window = returns_df.iloc[t - self.lookback: t]
            cdars = []
            for col in names:
                cdar_val = compute_cdar(window[col], alpha=self.cdar_alpha)
                cdars.append(max(cdar_val, 1e-8))

            cdars = np.array(cdars)

            # Inverse CDaR weighting: lower CDaR → higher weight
            inv_cdar = 1.0 / cdars
            w = inv_cdar / inv_cdar.sum()

            # Penalize strategies exceeding max CDaR
            for i, c in enumerate(cdars):
                if c > self.max_cdar:
                    penalty = self.max_cdar / c
                    w[i] *= penalty

            # Renormalize
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(n_strats) / n_strats

            weights[t] = w
            last_weights = w

        weights_df = pd.DataFrame(weights, index=dates, columns=names)

        logger.info(
            "CDaR risk budget: %d strategies, lookback=%d, "
            "mean CDaR weights: %s",
            n_strats,
            self.lookback,
            {n: f"{weights_df[n].mean():.3f}" for n in names},
        )

        return weights_df
