"""
Portfolio state tracker.

Maintains positions, cash, and derived exposure metrics.
All prices are assumed to be in the same currency denomination.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Fill:
    """Represents a single executed order fill."""
    security_id: str
    shares: float          # signed: positive = buy, negative = sell/short
    executed_price: float
    commission: float
    slippage: float
    spread_cost: float

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage + self.spread_cost

    @property
    def cash_impact(self) -> float:
        """Cash consumed by this fill (negative = cash out for buy)."""
        return -(self.shares * self.executed_price) - self.total_cost


class PortfolioState:
    """
    Tracks portfolio state through the backtest.

    Positions are stored as share counts (signed). PnL attribution uses
    average cost basis for long positions; short positions carry a credit.

    Parameters
    ----------
    initial_capital : starting cash in base currency
    max_leverage    : hard cap; raise if violated during _enforce_limits
    """

    def __init__(
        self,
        initial_capital: float,
        max_leverage: float = 2.0,
    ) -> None:
        self.initial_capital: float = initial_capital
        self.max_leverage: float = max_leverage

        # Core state
        self.positions: Dict[str, float] = {}   # security_id -> signed shares
        self.cost_basis: Dict[str, float] = {}  # security_id -> avg cost per share
        self.cash: float = initial_capital

        # Running realised PnL by security
        self.realised_pnl: Dict[str, float] = {}

        # History
        self._snapshots: List[dict] = []

    # ------------------------------------------------------------------
    # Derived metrics (properties)
    # ------------------------------------------------------------------

    def _long_market_value(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for sec, shares in self.positions.items():
            if shares > 0:
                total += shares * prices.get(sec, 0.0)
        return total

    def _short_market_value(self, prices: Dict[str, float]) -> float:
        """Returns a positive number representing gross short exposure."""
        total = 0.0
        for sec, shares in self.positions.items():
            if shares < 0:
                total += abs(shares) * prices.get(sec, 0.0)
        return total

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        """Gross dollar exposure (long + abs(short))."""
        return self._long_market_value(prices) + self._short_market_value(prices)

    def net_exposure(self, prices: Dict[str, float]) -> float:
        """Net dollar exposure (long - abs(short))."""
        return self._long_market_value(prices) - self._short_market_value(prices)

    def leverage(self, prices: Dict[str, float]) -> float:
        """Gross leverage ratio = gross_exposure / portfolio_value."""
        pv = self.mark_to_market(prices)
        return self.gross_exposure(prices) / pv if pv != 0 else 0.0

    # ------------------------------------------------------------------
    # Core state transitions
    # ------------------------------------------------------------------

    def update(
        self,
        fills: List[Fill],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply a list of fills, update positions and cash.

        Returns a dict of realised PnL per security for this batch.
        """
        batch_realised: Dict[str, float] = {}

        for fill in fills:
            sid = fill.security_id
            old_shares = self.positions.get(sid, 0.0)
            old_basis = self.cost_basis.get(sid, 0.0)
            new_shares_delta = fill.shares

            # Realised PnL: only when reducing or flipping a position
            realised = 0.0
            if old_shares != 0.0:
                # How many shares are being closed out?
                if (old_shares > 0 and new_shares_delta < 0) or (
                    old_shares < 0 and new_shares_delta > 0
                ):
                    closing_shares = min(
                        abs(new_shares_delta), abs(old_shares)
                    ) * np.sign(old_shares)
                    realised = closing_shares * (fill.executed_price - old_basis)

            # Update average cost basis for the remaining / new position
            new_shares = old_shares + new_shares_delta
            if new_shares == 0.0:
                new_basis = 0.0
            elif (old_shares >= 0 and new_shares_delta > 0) or (
                old_shares <= 0 and new_shares_delta < 0
            ):
                # Adding to existing direction: weighted average
                if old_shares + new_shares_delta != 0:
                    new_basis = (
                        old_shares * old_basis + new_shares_delta * fill.executed_price
                    ) / (old_shares + new_shares_delta)
                else:
                    new_basis = fill.executed_price
            else:
                # Partial close or flip: basis of remaining is unchanged (or reset for flip)
                if abs(new_shares_delta) > abs(old_shares):
                    # Position flipped; new basis is the fill price
                    new_basis = fill.executed_price
                else:
                    new_basis = old_basis

            # Persist
            if abs(new_shares) < 1e-10:
                self.positions.pop(sid, None)
                self.cost_basis.pop(sid, None)
            else:
                self.positions[sid] = new_shares
                self.cost_basis[sid] = new_basis

            self.cash += fill.cash_impact

            # Track realised PnL
            self.realised_pnl[sid] = self.realised_pnl.get(sid, 0.0) + realised
            batch_realised[sid] = batch_realised.get(sid, 0.0) + realised

        return batch_realised

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """
        Return total portfolio value (cash + market value of all positions).
        """
        equity = sum(
            shares * prices.get(sid, 0.0)
            for sid, shares in self.positions.items()
        )
        return self.cash + equity

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Return a dict of {security_id: weight} where weight = market_value / portfolio_value.
        Weights sum to net_exposure / portfolio_value (not necessarily 1).
        """
        pv = self.mark_to_market(prices)
        if pv == 0:
            return {}
        return {
            sid: (shares * prices.get(sid, 0.0)) / pv
            for sid, shares in self.positions.items()
        }

    def unrealised_pnl(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Return unrealised PnL per security."""
        result: Dict[str, float] = {}
        for sid, shares in self.positions.items():
            basis = self.cost_basis.get(sid, 0.0)
            result[sid] = shares * (prices.get(sid, 0.0) - basis)
        return result

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, prices: Dict[str, float], timestamp: Optional[pd.Timestamp] = None) -> dict:
        """
        Return a fully self-contained dictionary describing current state.

        Suitable for time-series storage — all values are scalars or
        JSON-serialisable primitives (dicts of float).
        """
        pv = self.mark_to_market(prices)
        snap: dict = {
            "timestamp": timestamp,
            "portfolio_value": pv,
            "cash": self.cash,
            "gross_exposure": self.gross_exposure(prices),
            "net_exposure": self.net_exposure(prices),
            "leverage": self.leverage(prices),
            "long_market_value": self._long_market_value(prices),
            "short_market_value": self._short_market_value(prices),
            "num_positions": len(self.positions),
            "positions": deepcopy(self.positions),
            "weights": self.get_weights(prices),
            "unrealised_pnl": self.unrealised_pnl(prices),
            "total_realised_pnl": sum(self.realised_pnl.values()),
        }
        self._snapshots.append(snap)
        return snap

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        """Return all snapshots as a DataFrame indexed by timestamp."""
        if not self._snapshots:
            return pd.DataFrame()
        records = []
        for s in self._snapshots:
            row = {k: v for k, v in s.items() if k not in ("positions", "weights", "unrealised_pnl")}
            records.append(row)
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def check_leverage(self, prices: Dict[str, float]) -> Tuple[bool, float]:
        """Return (within_limit, current_leverage)."""
        lev = self.leverage(prices)
        return (lev <= self.max_leverage + 1e-9), lev

    def scale_to_leverage_limit(
        self, target_weights: Dict[str, float], prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Scale target weights down proportionally so gross leverage ≤ max_leverage.
        Returns adjusted weights.
        """
        gross = sum(abs(w) for w in target_weights.values())
        if gross <= self.max_leverage + 1e-9:
            return target_weights
        scale = self.max_leverage / gross
        return {k: v * scale for k, v in target_weights.items()}
