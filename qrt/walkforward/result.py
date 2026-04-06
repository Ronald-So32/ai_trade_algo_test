"""
Walk-forward result container and performance analytics.

Aggregates per-window out-of-sample results produced by
:class:`~qrt.walkforward.walk_forward.WalkForwardTester` into a coherent
set of summary statistics, per-window metrics, and a stitched equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualisation
TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class WindowRecord:
    """Metadata and results for a single walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    oos_returns: pd.Series          # out-of-sample daily returns
    oos_positions: Optional[pd.DataFrame] = None  # optional position snapshot


class WalkForwardResult:
    """
    Aggregate out-of-sample results from a walk-forward backtest.

    Parameters
    ----------
    windows : list[WindowRecord]
        Ordered list of per-window records produced during the walk-forward
        run.  Out-of-sample periods must not overlap.
    risk_free_rate : float
        Annualised risk-free rate used in Sharpe / Sortino calculations
        (default 0.0).
    """

    def __init__(
        self,
        windows: List[WindowRecord],
        risk_free_rate: float = 0.0,
    ) -> None:
        if not windows:
            raise ValueError("windows list is empty.")
        self._windows = windows
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Aggregate performance summary across all out-of-sample windows.

        Returns
        -------
        summary : dict
            Keys: ``n_windows``, ``total_oos_days``, ``cagr``, ``sharpe``,
            ``sortino``, ``max_drawdown``, ``avg_turnover``,
            ``oos_start``, ``oos_end``, ``win_rate``.
        """
        equity = self.combined_equity_curve()
        oos_returns = equity.pct_change().dropna()
        wm = self.window_metrics()

        n_days = len(oos_returns)
        cagr = self._cagr(equity)
        sharpe = self._sharpe(oos_returns)
        sortino = self._sortino(oos_returns)
        max_dd = self._max_drawdown(equity)
        avg_turnover = wm["turnover"].mean() if "turnover" in wm.columns else np.nan
        win_rate = (wm["cagr"] > 0).mean() if len(wm) > 0 else np.nan

        return {
            "n_windows": len(self._windows),
            "total_oos_days": n_days,
            "oos_start": oos_returns.index[0] if n_days > 0 else None,
            "oos_end": oos_returns.index[-1] if n_days > 0 else None,
            "cagr": round(cagr, 6),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 6),
            "avg_turnover": round(float(avg_turnover), 6) if not np.isnan(avg_turnover) else None,
            "win_rate": round(float(win_rate), 4) if not np.isnan(win_rate) else None,
        }

    def window_metrics(self) -> pd.DataFrame:
        """
        Per-window performance metrics.

        Returns
        -------
        metrics : pd.DataFrame
            One row per window; columns: ``train_start``, ``train_end``,
            ``test_start``, ``test_end``, ``n_days``, ``cagr``, ``sharpe``,
            ``sortino``, ``max_drawdown``, ``turnover``.
        """
        records = []
        for w in self._windows:
            r = w.oos_returns.fillna(0.0)
            equity = (1.0 + r).cumprod()
            equity = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), equity])

            turn = self._compute_turnover(w.oos_positions)

            records.append(
                {
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "n_days": len(r),
                    "cagr": self._cagr(equity),
                    "sharpe": self._sharpe(r),
                    "sortino": self._sortino(r),
                    "max_drawdown": self._max_drawdown(equity),
                    "turnover": turn,
                }
            )

        return pd.DataFrame(records)

    def combined_equity_curve(self) -> pd.Series:
        """
        Build a stitched out-of-sample equity curve.

        Each window's returns are appended sequentially; the start NAV of
        each window is scaled to continue from where the previous window
        ended, producing a single compounded equity curve starting at 1.0.

        Returns
        -------
        equity : pd.Series
            Cumulative equity indexed by date, starting at 1.0.
        """
        all_returns: List[pd.Series] = []
        for w in self._windows:
            all_returns.append(w.oos_returns.fillna(0.0))

        combined_returns = pd.concat(all_returns).sort_index()

        # Guard against duplicate dates across windows (should not occur
        # in a correctly configured walk-forward, but be safe)
        if combined_returns.index.duplicated().any():
            logger.warning(
                "Duplicate dates found in combined OOS returns — "
                "keeping first occurrence."
            )
            combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]

        equity = (1.0 + combined_returns).cumprod()
        # Prepend a 1.0 base NAV one business day before the first return
        base_date = equity.index[0] - pd.tseries.offsets.BDay(1)
        base = pd.Series([1.0], index=[base_date])
        equity = pd.concat([base, equity])
        equity.name = "equity_curve"
        return equity

    # ------------------------------------------------------------------
    # Private performance metric helpers
    # ------------------------------------------------------------------

    def _cagr(self, equity: pd.Series) -> float:
        """Annualised compound growth rate from an equity curve."""
        if len(equity) < 2:
            return 0.0
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        if n_years <= 0:
            return 0.0
        end_val = float(equity.iloc[-1])
        start_val = float(equity.iloc[0])
        if start_val <= 0 or end_val <= 0:
            return 0.0
        return float((end_val / start_val) ** (1.0 / n_years) - 1.0)

    def _sharpe(self, returns: pd.Series) -> float:
        """Annualised Sharpe ratio."""
        excess = returns - self._daily_rf
        mu = excess.mean()
        sigma = excess.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            return 0.0
        return float(mu / sigma * np.sqrt(TRADING_DAYS_PER_YEAR))

    def _sortino(self, returns: pd.Series) -> float:
        """Annualised Sortino ratio (downside deviation denominator)."""
        excess = returns - self._daily_rf
        mu = excess.mean()
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        downside_vol = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_vol == 0:
            return np.inf
        return float(mu * TRADING_DAYS_PER_YEAR / downside_vol)

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown of an equity curve (negative value)."""
        roll_max = equity.cummax()
        drawdowns = equity / roll_max - 1.0
        return float(drawdowns.min())

    @staticmethod
    def _compute_turnover(positions: Optional[pd.DataFrame]) -> float:
        """
        Average daily one-way turnover from a positions DataFrame.

        Returns NaN when no position data is available.
        """
        if positions is None or positions.empty:
            return np.nan
        diffs = positions.diff().abs().sum(axis=1)
        return float(diffs.mean())
