"""
BacktestResult - post-backtest analytics and reporting.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .trade_ledger import TradeLedger


class BacktestResult:
    """
    Container for all backtest outputs with a rich analytics API.

    Parameters
    ----------
    portfolio_values    : pd.Series  (DatetimeIndex -> float)
    returns             : pd.Series  (DatetimeIndex -> float)  daily returns
    weights_history     : pd.DataFrame (DatetimeIndex x securities)
    trade_ledger        : TradeLedger instance
    portfolio_snapshots : list of snapshot dicts from PortfolioState.snapshot()
    ann_factor          : trading days per year (default 252)
    """

    def __init__(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        weights_history: pd.DataFrame,
        trade_ledger: TradeLedger,
        portfolio_snapshots: List[dict],
        ann_factor: int = 252,
    ) -> None:
        self.portfolio_values: pd.Series = portfolio_values.copy()
        self.returns: pd.Series = returns.copy()
        self.weights_history: pd.DataFrame = weights_history.copy()
        self.trade_ledger: TradeLedger = trade_ledger
        self.portfolio_snapshots: List[dict] = portfolio_snapshots
        self.ann_factor: int = ann_factor

        # Normalise index
        self.portfolio_values.index = pd.to_datetime(self.portfolio_values.index)
        self.returns.index = pd.to_datetime(self.returns.index)

    # ------------------------------------------------------------------
    # Core curves
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> pd.Series:
        """Portfolio value through time."""
        return self.portfolio_values

    @property
    def drawdown_curve(self) -> pd.Series:
        """
        Drawdown series: fraction below the running high-water mark.
        Values are <= 0, e.g. -0.15 means 15% below peak.
        """
        hwm = self.portfolio_values.cummax()
        dd = self.portfolio_values / hwm - 1.0
        dd.name = "drawdown"
        return dd

    # ------------------------------------------------------------------
    # Return statistics (properties)
    # ------------------------------------------------------------------

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        years = len(self.returns) / self.ann_factor
        if years == 0:
            return 0.0
        total_return = self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]
        return float(total_return ** (1.0 / years) - 1.0)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio (assumes zero risk-free rate)."""
        r = self.returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(self.ann_factor))

    @property
    def sortino(self) -> float:
        """Annualised Sortino ratio (downside deviation denominator)."""
        r = self.returns.dropna()
        downside = r[r < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std == 0:
            return 0.0
        return float(r.mean() / downside_std * np.sqrt(self.ann_factor))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown (negative value, e.g. -0.25 = -25%)."""
        dd = self.drawdown_curve
        return float(dd.min()) if len(dd) > 0 else 0.0

    @property
    def calmar(self) -> float:
        """Calmar ratio = CAGR / |max_drawdown|."""
        mdd = abs(self.max_drawdown)
        return float(self.cagr / mdd) if mdd > 1e-10 else 0.0

    @property
    def volatility(self) -> float:
        """Annualised daily return volatility."""
        r = self.returns.dropna()
        return float(r.std() * np.sqrt(self.ann_factor)) if len(r) > 1 else 0.0

    @property
    def total_return(self) -> float:
        """Total return over the full backtest period."""
        if len(self.portfolio_values) < 2:
            return 0.0
        return float(self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0] - 1.0)

    @property
    def total_turnover(self) -> float:
        """
        Total one-way turnover: sum of abs weight changes across all days.
        Expressed as a multiple of AUM (e.g. 2.0 = 200% one-way turnover).
        """
        if self.weights_history.empty:
            return 0.0
        diffs = self.weights_history.fillna(0.0).diff().abs()
        return float(diffs.sum().sum() / 2.0)  # one-way

    @property
    def avg_daily_turnover(self) -> float:
        """Average daily one-way turnover as a fraction of AUM."""
        n = max(len(self.weights_history) - 1, 1)
        return self.total_turnover / n

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """
        Rolling Sharpe ratio (annualised) over `window` trading days.
        """
        r = self.returns.dropna()
        roll_mean = r.rolling(window).mean()
        roll_std = r.rolling(window).std()
        rs = roll_mean / roll_std * np.sqrt(self.ann_factor)
        rs.name = f"rolling_sharpe_{window}d"
        return rs

    def rolling_volatility(self, window: int = 63) -> pd.Series:
        """Rolling annualised volatility."""
        r = self.returns.dropna()
        rv = r.rolling(window).std() * np.sqrt(self.ann_factor)
        rv.name = f"rolling_vol_{window}d"
        return rv

    def rolling_drawdown(self) -> pd.Series:
        """Full drawdown time series (same as drawdown_curve property)."""
        return self.drawdown_curve

    # ------------------------------------------------------------------
    # Periodic returns
    # ------------------------------------------------------------------

    def monthly_returns(self) -> pd.DataFrame:
        """
        Monthly return grid (rows = year, columns = month 1-12).
        Each cell is the compounded return for that calendar month.
        """
        r = self.returns.dropna()
        monthly = (1 + r).resample("ME").prod() - 1
        monthly.index = monthly.index.to_period("M")
        df = monthly.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month
        grid = df.pivot(index="year", columns="month", values="return")
        grid.columns.name = "month"
        return grid

    def annual_returns(self) -> pd.Series:
        """Compounded calendar-year returns."""
        r = self.returns.dropna()
        annual = (1 + r).resample("YE").prod() - 1
        annual.index = annual.index.year
        annual.name = "annual_return"
        return annual

    # ------------------------------------------------------------------
    # Snapshot utilities
    # ------------------------------------------------------------------

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio_snapshots list into a tidy DataFrame."""
        if not self.portfolio_snapshots:
            return pd.DataFrame()
        rows = []
        for snap in self.portfolio_snapshots:
            row = {k: v for k, v in snap.items() if k not in ("positions", "weights", "unrealised_pnl")}
            rows.append(row)
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    def exposure_history(self) -> pd.DataFrame:
        """Time series of gross/net exposure and leverage from snapshots."""
        df = self.snapshots_to_dataframe()
        cols = [c for c in ["gross_exposure", "net_exposure", "leverage", "long_market_value",
                            "short_market_value", "num_positions"] if c in df.columns]
        return df[cols] if cols else pd.DataFrame()

    # ------------------------------------------------------------------
    # Comprehensive summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a flat dictionary of all key performance metrics.

        Suitable for display, JSON serialisation, or comparison tables.
        """
        trade_summary = self.trade_ledger.summary()

        dd = self.drawdown_curve
        # Drawdown duration: max consecutive days in drawdown
        in_dd = (dd < -1e-8).astype(int)
        max_dd_duration = int(
            in_dd.groupby((in_dd == 0).cumsum()).sum().max()
        ) if in_dd.sum() > 0 else 0

        return {
            # Return metrics
            "total_return": round(self.total_return, 6),
            "cagr": round(self.cagr, 6),
            "volatility_ann": round(self.volatility, 6),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            # Drawdown
            "max_drawdown": round(self.max_drawdown, 6),
            "max_dd_duration_days": max_dd_duration,
            # Turnover / costs
            "total_turnover": round(self.total_turnover, 4),
            "avg_daily_turnover": round(self.avg_daily_turnover, 6),
            "total_commission": round(trade_summary.get("total_commission", 0.0), 2),
            "total_slippage": round(trade_summary.get("total_slippage", 0.0), 2),
            "total_transaction_cost": round(trade_summary.get("total_cost", 0.0), 2),
            "avg_cost_bps": round(trade_summary.get("avg_cost_bps", 0.0), 4),
            # Trade stats
            "total_trades": trade_summary.get("total_trades", 0),
            "win_rate": round(trade_summary.get("win_rate", 0.0), 4),
            "total_notional_traded": round(trade_summary.get("total_notional", 0.0), 2),
            # Portfolio
            "start_capital": float(self.portfolio_values.iloc[0]) if len(self.portfolio_values) else 0.0,
            "end_capital": float(self.portfolio_values.iloc[-1]) if len(self.portfolio_values) else 0.0,
            "backtest_days": len(self.returns),
            "backtest_years": round(len(self.returns) / self.ann_factor, 4),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BacktestResult("
            f"CAGR={s['cagr']:.2%}, "
            f"Sharpe={s['sharpe']:.2f}, "
            f"MaxDD={s['max_drawdown']:.2%}, "
            f"Sortino={s['sortino']:.2f}, "
            f"Calmar={s['calmar']:.2f}, "
            f"Trades={s['total_trades']})"
        )
