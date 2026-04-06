"""
Drawdown-Constrained Leverage Optimizer
========================================
Finds the optimal leverage that maximises CAGR subject to a maximum
drawdown constraint, accounting for volatility drag.

Academic basis:
  - Kelly (1956): "A New Interpretation of Information Rate" — optimal growth
  - Chan (2010): "How do you limit drawdown using Kelly formula?" — subaccount
    method: set aside D% of equity, apply Kelly to the sub-account
  - Busseti, Ryu & Boyd (2016): "Risk-Constrained Kelly Gambling" — maximise
    log-growth subject to drawdown/CVaR constraints via convex optimisation
  - Grossman & Zhou (1993): "Optimal Investment Strategies for Controlling
    Drawdowns" — proved TIPP is optimal under DD constraints
  - Avellaneda & Zhang (2010): "Path-Dependence of Leveraged ETF Returns" —
    volatility drag formula: E[r_L] ≈ L·μ - L(L-1)/2 · σ²

Key insight: naive leverage scaling (L × return) ignores volatility drag.
At high leverage, the drag term L(L-1)/2 × σ² dominates, causing net
returns to *decrease* beyond the optimal leverage point.

The optimizer uses historical returns to:
1. Simulate leveraged + DrawdownShield-protected returns at various L
2. Account for vol drag, drawdown compounding, and shield de-risking
3. Find the L that maximises CAGR subject to MaxDD ≤ target

Usage:
    optimizer = LeverageOptimizer(max_dd_target=0.12)
    result = optimizer.optimize(base_returns, shield=shield)
    # result.optimal_leverage, result.expected_cagr, result.expected_maxdd
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeverageOptimizationResult:
    """Result of drawdown-constrained leverage optimization."""
    optimal_leverage: float
    expected_cagr: float
    expected_maxdd: float
    expected_sharpe: float
    expected_calmar: float
    vol_drag_pct: float          # Annual vol drag at optimal leverage
    financing_cost_pct: float    # Annual financing cost at optimal leverage
    leverage_grid: list[float]   # All tested leverage levels
    cagr_grid: list[float]       # CAGR at each level
    maxdd_grid: list[float]      # MaxDD at each level
    calmar_grid: list[float]     # Calmar at each level


class LeverageOptimizer:
    """
    Find optimal leverage subject to maximum drawdown constraint.

    Uses empirical simulation: applies leverage to actual historical
    returns (with DrawdownShield protection), measuring realised
    CAGR and MaxDD at each leverage level.

    This is more accurate than analytical formulas because it captures:
    - Non-normal return distributions (fat tails, skew)
    - Path-dependent drawdown compounding
    - DrawdownShield's dynamic de-risking interaction with leverage
    - Actual vol drag under real market conditions
    - Realistic financing costs (IBKR rates)

    Parameters
    ----------
    max_dd_target : float
        Maximum acceptable drawdown (default 0.12 = 12%).
    leverage_range : tuple[float, float]
        Range of leverage to search (default (1.0, 10.0)).
    n_grid : int
        Number of leverage levels to test (default 50).
    safety_margin : float
        Fraction of max_dd_target to use as buffer (default 0.85).
        Actual target = max_dd_target × safety_margin.
        Accounts for estimation uncertainty.
    annual_financing_rate : float
        Annual cost of leverage financing (default 0.048 = 4.8%).
        Based on IBKR Pro blended margin rate for $100k-$500k accounts.
        Alternatives: futures ~4.5%, box spreads ~4.0%, margin ~4.8-5.1%.
        Applied as daily deduction: (L-1) × rate / 252 per day.
    """

    def __init__(
        self,
        max_dd_target: float = 0.12,
        leverage_range: tuple[float, float] = (1.0, 10.0),
        n_grid: int = 50,
        safety_margin: float = 0.85,
        annual_financing_rate: float = 0.048,
    ) -> None:
        self.max_dd_target = max_dd_target
        self.leverage_range = leverage_range
        self.n_grid = n_grid
        self.safety_margin = safety_margin
        self.annual_financing_rate = annual_financing_rate

    def _simulate_leveraged(
        self,
        base_returns: pd.Series,
        leverage: float,
        shield=None,
        strategy_returns: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Simulate leveraged portfolio with optional DrawdownShield.

        Accounts for:
        - Volatility drag per Avellaneda & Zhang (2010): implicitly captured
          by compounding leveraged daily returns
        - Financing costs: (L-1) × annual_rate / 252 deducted daily
          Per IBKR Pro rates: ~4.8% for $100k-$500k account balance
        """
        # Apply leverage to daily returns and subtract financing costs
        # Cost = (L-1) × annual_rate / 252 — only borrowed portion incurs cost
        daily_financing_cost = (leverage - 1) * self.annual_financing_rate / 252
        lev_returns = base_returns * leverage - daily_financing_cost

        # Apply DrawdownShield if provided
        if shield is not None:
            pseudo_weights = pd.DataFrame(
                {"portfolio": 1.0}, index=lev_returns.index
            )
            pseudo_returns = pd.DataFrame(
                {"portfolio": lev_returns.values}, index=lev_returns.index
            )
            shielded_weights = shield.apply(
                pseudo_weights, pseudo_returns,
                strategy_returns=strategy_returns,
            )
            lev_returns = lev_returns * shielded_weights["portfolio"]

        # Compute metrics
        cum = (1 + lev_returns).cumprod()
        total_days = len(cum)

        if total_days < 2 or cum.iloc[-1] <= 0:
            return {
                "cagr": -1.0, "maxdd": -1.0, "sharpe": 0.0,
                "calmar": 0.0, "vol_drag": 0.0,
            }

        cagr = cum.iloc[-1] ** (252 / total_days) - 1
        dd = (cum / cum.cummax() - 1)
        maxdd = dd.min()
        vol = lev_returns.std() * np.sqrt(252)
        sharpe = (lev_returns.mean() / lev_returns.std() * np.sqrt(252)
                  if lev_returns.std() > 0 else 0.0)
        calmar = cagr / abs(maxdd) if maxdd != 0 else 0.0

        # Theoretical vol drag
        base_vol = base_returns.std() * np.sqrt(252)
        vol_drag = leverage * (leverage - 1) / 2 * base_vol ** 2
        financing_cost = (leverage - 1) * self.annual_financing_rate

        return {
            "cagr": float(cagr),
            "maxdd": float(maxdd),
            "sharpe": float(sharpe),
            "calmar": float(calmar),
            "vol_drag": float(vol_drag),
            "financing_cost": float(financing_cost),
        }

    def optimize(
        self,
        base_returns: pd.Series,
        shield=None,
        strategy_returns: Optional[pd.DataFrame] = None,
    ) -> LeverageOptimizationResult:
        """
        Find optimal leverage via grid search over historical returns.

        Parameters
        ----------
        base_returns : pd.Series
            Unlevered (or base-levered) daily portfolio returns.
        shield : DrawdownShield, optional
            If provided, applies dynamic drawdown protection at each
            leverage level.  This is critical — without the shield,
            high leverage levels will produce unrealistic MaxDD.
        strategy_returns : pd.DataFrame, optional
            Strategy-level returns for correlation monitoring in shield.

        Returns
        -------
        LeverageOptimizationResult
            Optimal leverage and diagnostic grids.
        """
        effective_target = self.max_dd_target * self.safety_margin

        leverage_grid = np.linspace(
            self.leverage_range[0], self.leverage_range[1], self.n_grid
        ).tolist()

        cagr_grid = []
        maxdd_grid = []
        calmar_grid = []

        for lev in leverage_grid:
            result = self._simulate_leveraged(
                base_returns, lev,
                shield=shield,
                strategy_returns=strategy_returns,
            )
            cagr_grid.append(result["cagr"])
            maxdd_grid.append(result["maxdd"])
            calmar_grid.append(result["calmar"])

        # Find optimal: max CAGR where |MaxDD| ≤ target
        best_idx = -1
        best_cagr = -np.inf

        for i, (cagr, maxdd) in enumerate(zip(cagr_grid, maxdd_grid)):
            if abs(maxdd) <= effective_target and cagr > best_cagr:
                best_cagr = cagr
                best_idx = i

        # If no leverage meets the DD constraint, use the one with
        # best Calmar ratio (return per unit drawdown)
        if best_idx < 0:
            logger.warning(
                "No leverage level meets MaxDD target %.1f%%. "
                "Selecting by best Calmar ratio instead.",
                effective_target * 100,
            )
            best_idx = int(np.argmax(calmar_grid))

        optimal_lev = leverage_grid[best_idx]

        # Re-simulate at optimal for final metrics
        final = self._simulate_leveraged(
            base_returns, optimal_lev,
            shield=shield,
            strategy_returns=strategy_returns,
        )

        result = LeverageOptimizationResult(
            optimal_leverage=round(optimal_lev, 2),
            expected_cagr=final["cagr"],
            expected_maxdd=final["maxdd"],
            expected_sharpe=final["sharpe"],
            expected_calmar=final["calmar"],
            vol_drag_pct=final["vol_drag"],
            financing_cost_pct=final["financing_cost"],
            leverage_grid=leverage_grid,
            cagr_grid=cagr_grid,
            maxdd_grid=maxdd_grid,
            calmar_grid=calmar_grid,
        )

        logger.info(
            "Leverage optimizer: optimal=%.1fx, CAGR=%.2f%% (net of %.2f%% "
            "financing), MaxDD=%.2f%%, Calmar=%.2f, vol_drag=%.2f%%",
            result.optimal_leverage,
            result.expected_cagr * 100,
            result.financing_cost_pct * 100,
            result.expected_maxdd * 100,
            result.expected_calmar,
            result.vol_drag_pct * 100,
        )

        return result

    def optimize_with_dynamic_shield(
        self,
        base_returns: pd.Series,
        strategy_returns: Optional[pd.DataFrame] = None,
        cppi_config: Optional[dict] = None,
        multi_horizon_config: Optional[dict] = None,
        correlation_config: Optional[dict] = None,
    ) -> LeverageOptimizationResult:
        """
        Convenience method: create shield with given configs and optimize.

        The shield's max_drawdown parameter is set to match the optimizer's
        target, ensuring consistent drawdown control at all leverage levels.
        """
        from qrt.risk.portfolio_insurance import DrawdownShield

        # Scale shield's max_dd with leverage target
        if cppi_config is None:
            cppi_config = {}
        cppi_config.setdefault("max_drawdown", self.max_dd_target)
        cppi_config.setdefault("multiplier", 4.0)
        cppi_config.setdefault("max_exposure", 1.3)
        cppi_config.setdefault("min_exposure", 0.10)
        cppi_config.setdefault("ratchet_pct", 0.80)

        shield = DrawdownShield(
            cppi_config=cppi_config,
            multi_horizon_config=multi_horizon_config,
            correlation_config=correlation_config,
            enable_correlation=strategy_returns is not None,
        )

        return self.optimize(
            base_returns,
            shield=shield,
            strategy_returns=strategy_returns,
        )


def compute_vol_drag(leverage: float, annual_vol: float) -> float:
    """
    Compute annualised volatility drag for a given leverage level.

    Per Avellaneda & Zhang (2010):
        drag = L(L-1)/2 × σ²

    Parameters
    ----------
    leverage : float
        Leverage multiplier.
    annual_vol : float
        Annualised portfolio volatility.

    Returns
    -------
    float
        Annual drag as a fraction (e.g. 0.05 = 5% annual drag).
    """
    return leverage * (leverage - 1) / 2 * annual_vol ** 2


def analytical_optimal_leverage(
    sharpe: float,
    annual_vol: float,
    max_dd: float,
    confidence: float = 0.95,
) -> float:
    """
    Analytical estimate of optimal leverage under drawdown constraint.

    Combines Kelly criterion with Chan (2010) subaccount method:
        1. Kelly optimal: L* = μ/σ² = Sharpe/σ
        2. Drawdown-constrained: L_dd ≈ max_dd / (z × σ × √T_dd)

    where z is the confidence z-score and T_dd is typical drawdown duration.

    This is an approximation — use LeverageOptimizer for empirical accuracy.

    Parameters
    ----------
    sharpe : float
        Annualised Sharpe ratio.
    annual_vol : float
        Annualised volatility.
    max_dd : float
        Maximum acceptable drawdown (e.g. 0.12).
    confidence : float
        Confidence level for drawdown estimate (default 0.95).

    Returns
    -------
    float
        Estimated optimal leverage.
    """
    from scipy.stats import norm

    if annual_vol <= 0:
        return 1.0

    # Kelly optimal leverage
    kelly_lev = sharpe / annual_vol

    # Drawdown-constrained leverage (Chan subaccount approach)
    # Expected max DD for a normal process over N periods:
    # E[MaxDD] ≈ z × σ_daily × √(T_dd)
    # where T_dd ≈ 252 (1 year) for a conservative estimate
    z = norm.ppf(confidence)
    daily_vol = annual_vol / np.sqrt(252)
    dd_lev = max_dd / (z * daily_vol * np.sqrt(252))

    # Take the more conservative
    optimal = min(kelly_lev, dd_lev)

    # Apply half-Kelly for safety (MacLean, Thorp & Ziemba 2011)
    optimal *= 0.5

    return max(1.0, optimal)
