"""
Trade Ledger - records every fill with full audit trail.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    trade_id: str
    timestamp: pd.Timestamp
    security_id: str
    strategy: str
    order_type: str          # "BUY" | "SELL" | "SHORT" | "COVER"
    signal_strength: float   # raw signal value driving the trade
    executed_price: float
    position_size: float     # shares / contracts (signed: + long, - short)
    notional: float          # abs(position_size * executed_price)
    commission: float        # dollar cost
    slippage: float          # dollar cost
    spread_cost: float       # dollar cost
    total_cost: float        # commission + slippage + spread_cost
    pnl: float               # realised PnL on this fill (0 for opening trades)
    side: str                # "OPEN_LONG" | "OPEN_SHORT" | "CLOSE_LONG" | "CLOSE_SHORT" | "REBALANCE"


class TradeLedger:
    """
    Append-only record of every executed trade.

    Tracks full cost breakdown and realised PnL so that attribution,
    turnover and cost analysis can be performed post-backtest.
    """

    def __init__(self) -> None:
        self._records: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append_trade(
        self,
        timestamp: pd.Timestamp,
        security_id: str,
        strategy: str,
        order_type: str,
        signal_strength: float,
        executed_price: float,
        position_size: float,
        commission: float,
        slippage: float,
        spread_cost: float = 0.0,
        pnl: float = 0.0,
        side: str = "REBALANCE",
        trade_id: Optional[str] = None,
    ) -> str:
        """
        Record a single fill.

        Parameters
        ----------
        timestamp       : bar timestamp for the fill
        security_id     : ticker / asset identifier
        strategy        : name of the strategy generating the order
        order_type      : "BUY" | "SELL" | "SHORT" | "COVER"
        signal_strength : raw alpha signal value
        executed_price  : average fill price (post-slippage)
        position_size   : signed shares traded (+buy, -sell)
        commission      : dollar commission charged
        slippage        : dollar slippage cost
        spread_cost     : dollar half-spread cost
        pnl             : realised PnL recognised on this fill
        side            : semantic label for the trade leg
        trade_id        : optional deterministic id; auto-generated if None

        Returns
        -------
        trade_id string
        """
        if trade_id is None:
            trade_id = str(uuid.uuid4())

        notional = abs(position_size * executed_price)
        total_cost = commission + slippage + spread_cost

        record = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            security_id=security_id,
            strategy=strategy,
            order_type=order_type,
            signal_strength=signal_strength,
            executed_price=executed_price,
            position_size=position_size,
            notional=notional,
            commission=commission,
            slippage=slippage,
            spread_cost=spread_cost,
            total_cost=total_cost,
            pnl=pnl,
            side=side,
        )
        self._records.append(record)
        return trade_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all trades as a tidy DataFrame sorted by timestamp."""
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "trade_id", "timestamp", "security_id", "strategy",
                    "order_type", "signal_strength", "executed_price",
                    "position_size", "notional", "commission", "slippage",
                    "spread_cost", "total_cost", "pnl", "side",
                ]
            )
        df = pd.DataFrame([asdict(r) for r in self._records])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a dictionary of aggregate trade statistics.

        Keys
        ----
        total_trades          : number of fills
        total_notional        : sum of abs notional traded
        total_commission      : total commission paid
        total_slippage        : total slippage paid
        total_spread_cost     : total spread cost paid
        total_cost            : all-in transaction cost
        total_realised_pnl    : sum of realised PnL on fills
        net_pnl               : total_realised_pnl - total_cost
        avg_cost_bps          : average cost as fraction of notional (bps)
        win_rate              : fraction of trades with pnl > 0
        avg_pnl_per_trade     : mean PnL per trade
        largest_win           : max single-trade PnL
        largest_loss          : min single-trade PnL
        unique_securities     : number of distinct tickers traded
        """
        if not self._records:
            return {
                "total_trades": 0,
                "total_notional": 0.0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_spread_cost": 0.0,
                "total_cost": 0.0,
                "total_realised_pnl": 0.0,
                "net_pnl": 0.0,
                "avg_cost_bps": 0.0,
                "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "unique_securities": 0,
            }

        df = self.to_dataframe()

        total_notional = df["notional"].sum()
        total_commission = df["commission"].sum()
        total_slippage = df["slippage"].sum()
        total_spread_cost = df["spread_cost"].sum()
        total_cost = df["total_cost"].sum()
        total_realised_pnl = df["pnl"].sum()

        avg_cost_bps = (total_cost / total_notional * 10_000) if total_notional > 0 else 0.0
        winning = df[df["pnl"] > 0]
        win_rate = len(winning) / len(df) if len(df) > 0 else 0.0

        return {
            "total_trades": len(df),
            "total_notional": round(total_notional, 2),
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "total_spread_cost": round(total_spread_cost, 2),
            "total_cost": round(total_cost, 2),
            "total_realised_pnl": round(total_realised_pnl, 2),
            "net_pnl": round(total_realised_pnl - total_cost, 2),
            "avg_cost_bps": round(avg_cost_bps, 4),
            "win_rate": round(win_rate, 4),
            "avg_pnl_per_trade": round(df["pnl"].mean(), 4),
            "largest_win": round(df["pnl"].max(), 4),
            "largest_loss": round(df["pnl"].min(), 4),
            "unique_securities": int(df["security_id"].nunique()),
        }

    def cost_breakdown_by_security(self) -> pd.DataFrame:
        """Return per-security aggregated cost and PnL breakdown."""
        if not self._records:
            return pd.DataFrame()
        df = self.to_dataframe()
        return (
            df.groupby("security_id")
            .agg(
                trades=("trade_id", "count"),
                total_notional=("notional", "sum"),
                total_commission=("commission", "sum"),
                total_slippage=("slippage", "sum"),
                total_cost=("total_cost", "sum"),
                total_pnl=("pnl", "sum"),
            )
            .assign(
                cost_bps=lambda x: x["total_cost"] / x["total_notional"].replace(0, np.nan) * 10_000
            )
            .sort_values("total_notional", ascending=False)
        )

    def daily_turnover(self, portfolio_values: Optional[pd.Series] = None) -> pd.Series:
        """
        Compute daily notional turnover.

        If portfolio_values is supplied, returns turnover as a fraction
        of portfolio value; otherwise returns raw dollar notional.
        """
        if not self._records:
            return pd.Series(dtype=float)
        df = self.to_dataframe()
        daily = df.groupby(df["timestamp"].dt.normalize())["notional"].sum()
        if portfolio_values is not None:
            daily = daily / portfolio_values.reindex(daily.index).ffill()
        return daily
