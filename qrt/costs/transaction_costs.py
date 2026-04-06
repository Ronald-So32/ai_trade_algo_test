"""
Transaction Cost Model
========================
A realistic, multi-component transaction cost model suitable for
equity long/short strategy backtesting.

Cost components
---------------
commission
    Flat per-trade brokerage fee expressed in basis points of trade value.
    Representative range: 1–5 bps for institutional equity brokers.

spread
    Half-spread cost (assumed to be paid on both entry and exit) expressed
    in basis points.  Uses a volume-adjusted formula:

        spread_cost = base_spread_bps × (1 + spread_volume_scaling × ADV_ratio)

    where ADV_ratio = |trade_value| / (price × volume).  Larger trades
    relative to average daily volume widen the effective spread.

slippage
    Market-impact slippage approximated by the square-root law:

        slippage_cost = slippage_bps × sqrt(|trade_value| / (price × volume))

    This captures the empirical observation that price impact grows
    sub-linearly with order size.

turnover_penalty
    Daily portfolio-level turnover penalty expressed as a fraction of
    two-way turnover (Σ|Δw_i|).  Applied in ``compute_cost_drag`` to
    penalise high-frequency rebalancing without requiring per-trade
    volume data.

All bps parameters are in basis points (1 bps = 0.0001 = 0.01 %).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BPS: float = 1e-4          # 1 basis point as a decimal fraction
_EPSILON: float = 1e-10


class TransactionCostModel:
    """
    Multi-component transaction cost model with temporary/permanent impact
    separation and regime-aware spread widening.

    Cost components:
    - Commission: flat brokerage fee
    - Spread: half-spread with volume/regime scaling
    - Temporary impact: short-term price displacement (sqrt law)
    - Permanent impact: lasting information content of the trade
    - Turnover penalty: portfolio-level rebalancing cost

    Parameters
    ----------
    commission_bps : float, default 2.0
        Flat brokerage commission in basis points of trade value.
    spread_bps : float, default 1.5
        Base half-spread cost in basis points.
    slippage_bps : float, default 3.0
        Market-impact slippage coefficient in basis points (scales with
        sqrt of participation rate).  Split into temporary + permanent.
    turnover_penalty_bps : float, default 5.0
        Daily portfolio-level turnover penalty expressed in basis points
        of two-way turnover.  Set to 0 to disable.
    spread_volume_scaling : float, default 0.5
        Sensitivity of the spread cost to the participation rate
        (|trade_value| / ADV).  Higher values increase the spread cost
        for large trades relative to daily volume.
    min_commission : float, default 0.0
        Minimum commission per trade in currency units (e.g. USD).  Set
        to a positive value to model per-ticket minimum charges.
    permanent_impact_frac : float, default 0.3
        Fraction of total slippage attributed to permanent impact
        (remainder is temporary).  Almgren-Chriss style decomposition.
    regime_spread_mult : dict, optional
        Multiplier for spreads per regime label.  E.g., {"crisis": 3.0}
        means spreads triple in crisis.  Default widens spreads in
        high-vol and crisis regimes.
    borrow_cost_bps : float, default 0.0
        Annual borrow cost for short positions in basis points.

    Attributes
    ----------
    total_cost_computed_ : float
        Cumulative total cost computed across all ``estimate_cost`` calls.
    """

    # Default regime spread multipliers (liquidity worsens in stress)
    DEFAULT_REGIME_SPREAD_MULT: dict[str, float] = {
        "low_vol": 0.8,
        "medium_vol": 1.0,
        "high_vol": 1.5,
        "crisis": 3.0,
    }

    def __init__(
        self,
        commission_bps: float = 2.0,
        spread_bps: float = 1.5,
        slippage_bps: float = 3.0,
        turnover_penalty_bps: float = 5.0,
        spread_volume_scaling: float = 0.5,
        min_commission: float = 0.0,
        permanent_impact_frac: float = 0.3,
        regime_spread_mult: dict[str, float] | None = None,
        borrow_cost_bps: float = 0.0,
    ) -> None:
        if commission_bps < 0:
            raise ValueError("commission_bps must be non-negative.")
        if spread_bps < 0:
            raise ValueError("spread_bps must be non-negative.")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative.")
        if turnover_penalty_bps < 0:
            raise ValueError("turnover_penalty_bps must be non-negative.")
        if spread_volume_scaling < 0:
            raise ValueError("spread_volume_scaling must be non-negative.")

        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.turnover_penalty_bps = turnover_penalty_bps
        self.spread_volume_scaling = spread_volume_scaling
        self.min_commission = min_commission
        self.permanent_impact_frac = permanent_impact_frac
        self.regime_spread_mult = regime_spread_mult or self.DEFAULT_REGIME_SPREAD_MULT
        self.borrow_cost_bps = borrow_cost_bps

        self.total_cost_computed_: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        trade_value: float,
        price: float,
        volume: float,
    ) -> float:
        """
        Estimate the total transaction cost for a single trade.

        Parameters
        ----------
        trade_value : float
            Signed notional value of the trade (positive = buy, negative = sell).
            The absolute value is used in all cost calculations.
        price : float
            Execution price per share / unit.
        volume : float
            Average daily volume (ADV) of the instrument in the same units as
            ``trade_value / price`` (i.e., number of shares / units per day).
            Must be > 0.

        Returns
        -------
        float
            Total estimated transaction cost as a currency amount (same units
            as ``trade_value``).  Always non-negative.
        """
        breakdown = self.cost_breakdown(trade_value, price, volume)
        total = sum(breakdown.values())
        self.total_cost_computed_ += total
        return total

    def cost_breakdown(
        self,
        trade_value: float,
        price: float,
        volume: float,
    ) -> dict[str, float]:
        """
        Detailed cost breakdown for a single trade.

        Parameters
        ----------
        trade_value : float
            Signed notional value of the trade.
        price : float
            Execution price per unit.
        volume : float
            Average daily volume in units per day.

        Returns
        -------
        dict with keys:
            commission  – brokerage fee
            spread      – half-spread cost
            slippage    – market-impact cost
        All values are non-negative currency amounts.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}.")
        if volume <= 0:
            raise ValueError(f"volume must be positive, got {volume}.")

        abs_value = abs(float(trade_value))

        # ---- ADV ratio (participation rate) ----
        adv_value = price * volume          # daily dollar volume
        adv_ratio = abs_value / max(adv_value, _EPSILON)

        # ---- Commission ----
        commission = max(
            abs_value * self.commission_bps * _BPS,
            self.min_commission,
        )

        # ---- Spread ----
        effective_spread_bps = self.spread_bps * (
            1.0 + self.spread_volume_scaling * adv_ratio
        )
        spread_cost = abs_value * effective_spread_bps * _BPS

        # ---- Market Impact (Almgren-Chriss style: temporary + permanent) ----
        participation = min(adv_ratio, 1.0)  # cap at 100 % of ADV
        total_impact = abs_value * self.slippage_bps * _BPS * np.sqrt(participation)
        temporary_impact = total_impact * (1 - self.permanent_impact_frac)
        permanent_impact = total_impact * self.permanent_impact_frac

        return {
            "commission": float(commission),
            "spread": float(spread_cost),
            "temporary_impact": float(temporary_impact),
            "permanent_impact": float(permanent_impact),
        }

    def compute_cost_drag(
        self,
        weights_history: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute the daily cost drag (as a return fraction) from rebalancing.

        For each day, the cost drag is the sum of per-asset transaction costs
        incurred by the change in weights, expressed as a fraction of
        beginning-of-day NAV.

        The method handles missing volume data gracefully: when ``volumes``
        is ``None`` (or has missing entries for specific assets), it falls
        back to using a synthetic ADV computed as price × 1 (one share per
        day), so costs are reported purely on a bps basis without volume
        scaling.

        Parameters
        ----------
        weights_history : pd.DataFrame
            Target portfolio weights (rows = dates, columns = assets).
            Weights represent fractions of NAV (e.g. 0.05 for 5 %).
        prices : pd.DataFrame
            Asset prices aligned to ``weights_history``.
        volumes : pd.DataFrame, optional
            Average daily volumes (shares / units) aligned to
            ``weights_history``.  When ``None``, volume scaling is disabled
            (equivalent to ``volumes = 1 / price`` per asset, so ADV ratio = 1
            for all trades).

        Returns
        -------
        pd.Series
            Daily cost drag as a fraction of NAV (negative number representing
            a cost).  Index matches ``weights_history.index``.
        """
        if not isinstance(weights_history, pd.DataFrame):
            raise TypeError("weights_history must be a pd.DataFrame.")
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pd.DataFrame.")

        weights_history = weights_history.reindex(columns=prices.columns)
        weights_history = weights_history.fillna(0.0)

        # Trade (weight change) between consecutive dates
        dw = weights_history.diff().fillna(0.0)

        # Align prices and volumes
        prices_aligned = prices.reindex_like(weights_history).ffill().bfill()

        if volumes is not None:
            volumes_aligned = volumes.reindex_like(weights_history).ffill().bfill()
            # Fill remaining NaNs with a fallback of 1.0 (ADV=1 unit/day)
            volumes_aligned = volumes_aligned.fillna(1.0)
        else:
            # No volume data: use 1/price so adv_value = price*1/price = 1,
            # effectively setting ADV ratio = |trade_value| (raw bps costs)
            volumes_aligned = (1.0 / prices_aligned.replace(0, np.nan)).fillna(1.0)

        daily_drag: list[float] = []

        for date in weights_history.index:
            day_cost = 0.0

            # Regime-aware spread multiplier
            spread_mult = 1.0
            if regime_labels is not None and date in regime_labels.index:
                regime = regime_labels.loc[date]
                if isinstance(regime, str):
                    spread_mult = self.regime_spread_mult.get(regime, 1.0)

            for asset in weights_history.columns:
                delta_w = float(dw.loc[date, asset])
                if abs(delta_w) < _EPSILON:
                    continue
                p = float(prices_aligned.loc[date, asset])
                v = float(volumes_aligned.loc[date, asset])
                if p <= 0 or v <= 0:
                    continue
                # Trade value = |delta_weight| × NAV; we work in weight space
                # so trade_value = |Δw|, NAV-normalised
                trade_val = abs(delta_w)

                # Temporarily adjust spread for regime
                original_spread = self.spread_bps
                self.spread_bps = original_spread * spread_mult
                breakdown = self.cost_breakdown(
                    trade_value=trade_val, price=p, volume=v
                )
                self.spread_bps = original_spread

                day_cost += sum(breakdown.values())

                # Borrow cost for short positions (annualized, paid daily)
                w = float(weights_history.loc[date, asset])
                if w < 0 and self.borrow_cost_bps > 0:
                    day_cost += abs(w) * self.borrow_cost_bps * _BPS / 252.0

            # Add turnover penalty (bps of two-way turnover)
            two_way_turnover = dw.loc[date].abs().sum()
            turnover_cost = (
                two_way_turnover * self.turnover_penalty_bps * _BPS
            )
            day_cost += float(turnover_cost)

            daily_drag.append(-day_cost)  # negative = cost to returns

        return pd.Series(daily_drag, index=weights_history.index, name="cost_drag")

    def summary(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
    ) -> dict[str, float]:
        """
        Summarise strategy performance gross vs. net of transaction costs.

        Parameters
        ----------
        gross_returns : pd.Series
            Daily gross strategy returns (before costs).
        net_returns : pd.Series
            Daily net strategy returns (after subtracting cost drag).

        Returns
        -------
        dict with keys:
            gross_total_return   – cumulative gross return
            net_total_return     – cumulative net return
            gross_annual_return  – annualised gross return
            net_annual_return    – annualised net return
            gross_sharpe         – annualised Sharpe ratio (gross)
            net_sharpe           – annualised Sharpe ratio (net)
            avg_daily_cost_drag  – mean daily cost drag
            total_cost_drag      – sum of all cost drag
            cost_drag_bps        – average daily cost drag in bps
            trading_days         – number of trading days in sample
        """
        if not isinstance(gross_returns, pd.Series):
            raise TypeError("gross_returns must be a pd.Series.")
        if not isinstance(net_returns, pd.Series):
            raise TypeError("net_returns must be a pd.Series.")

        gross_returns = gross_returns.astype(float).dropna()
        net_returns = net_returns.astype(float).reindex(gross_returns.index).dropna()

        n = len(gross_returns)
        if n == 0:
            return {
                "gross_total_return": 0.0,
                "net_total_return": 0.0,
                "gross_annual_return": 0.0,
                "net_annual_return": 0.0,
                "gross_sharpe": 0.0,
                "net_sharpe": 0.0,
                "avg_daily_cost_drag": 0.0,
                "total_cost_drag": 0.0,
                "cost_drag_bps": 0.0,
                "trading_days": 0,
            }

        ann_factor = 252.0
        periods_per_year = ann_factor

        # Cumulative returns
        gross_total = float((1.0 + gross_returns).prod() - 1.0)
        net_total = float((1.0 + net_returns).prod() - 1.0)

        # Annualised returns (geometric)
        gross_annual = float((1.0 + gross_total) ** (periods_per_year / n) - 1.0)
        net_annual = float((1.0 + net_total) ** (periods_per_year / n) - 1.0)

        # Sharpe ratios (assumes risk-free = 0)
        def _sharpe(r: pd.Series) -> float:
            std = float(r.std(ddof=1))
            if std < _EPSILON:
                return 0.0
            return float(r.mean() / std * np.sqrt(periods_per_year))

        gross_sharpe = _sharpe(gross_returns)
        net_sharpe = _sharpe(net_returns)

        # Cost drag statistics
        cost_drag = net_returns - gross_returns
        avg_daily_drag = float(cost_drag.mean())
        total_drag = float(cost_drag.sum())
        drag_bps = avg_daily_drag / _BPS  # convert to bps

        return {
            "gross_total_return": gross_total,
            "net_total_return": net_total,
            "gross_annual_return": gross_annual,
            "net_annual_return": net_annual,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "avg_daily_cost_drag": avg_daily_drag,
            "total_cost_drag": total_drag,
            "cost_drag_bps": drag_bps,
            "trading_days": n,
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TransactionCostModel("
            f"commission_bps={self.commission_bps}, "
            f"spread_bps={self.spread_bps}, "
            f"slippage_bps={self.slippage_bps}, "
            f"turnover_penalty_bps={self.turnover_penalty_bps})"
        )
