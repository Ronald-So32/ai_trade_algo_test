"""
Walk-forward backtesting engine.

Implements a rolling in-sample / out-of-sample framework:

    for each window:
        1. Fit / train the strategy on [train_start, train_end]
        2. Freeze the learned parameters
        3. Generate signals on [test_start, test_end]
        4. Collect out-of-sample returns and record metrics

The strategy object must implement a :class:`StrategyProtocol`-compatible
interface (``fit`` + ``predict``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from .result import WalkForwardResult, WindowRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy protocol — any object with these two methods is compatible
# ---------------------------------------------------------------------------

class StrategyProtocol(Protocol):
    """
    Minimal interface that a strategy must satisfy to work with
    :class:`WalkForwardTester`.
    """

    def fit(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Train / calibrate the strategy on in-sample data."""
        ...

    def predict(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs: Any,
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate out-of-sample signals.

        Returns
        -------
        oos_returns : pd.Series
            Daily strategy returns over the test window.
        positions : pd.DataFrame or None
            Optional daily position snapshot (used for turnover).
        """
        ...


# ---------------------------------------------------------------------------
# Window descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    """Date boundaries for a single walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Window(train=[{self.train_start.date()}→{self.train_end.date()}], "
            f"test=[{self.test_start.date()}→{self.test_end.date()}])"
        )


# ---------------------------------------------------------------------------
# Walk-forward tester
# ---------------------------------------------------------------------------

class WalkForwardTester:
    """
    Rolling walk-forward backtesting engine.

    The tester slices the full date range into overlapping train/test
    windows, calls ``strategy.fit()`` on the in-sample slice and
    ``strategy.predict()`` on the out-of-sample slice, then stitches
    the out-of-sample results into a :class:`~qrt.walkforward.result.WalkForwardResult`.

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free rate for performance metric calculations.
    verbose : bool
        If True, log progress at INFO level for each window.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def generate_windows(
        self,
        dates: pd.DatetimeIndex,
        train_years: int = 3,
        test_months: int = 6,
    ) -> List[Window]:
        """
        Generate a list of non-overlapping test windows.

        The first window trains on the first ``train_years`` of data;
        subsequent windows roll forward by ``test_months`` each step.
        The training window expands (or rolls) to always end immediately
        before the test window begins.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Full sorted date range available for the backtest.
        train_years : int
            Length of the in-sample training period in years (default 3).
        test_months : int
            Length of each out-of-sample test period in months (default 6).

        Returns
        -------
        windows : list[Window]
            Ordered list of :class:`Window` objects.  Empty if the date
            range is insufficient to form even one window.
        """
        if train_years <= 0:
            raise ValueError("train_years must be positive.")
        if test_months <= 0:
            raise ValueError("test_months must be positive.")
        if dates.empty:
            return []

        dates = dates.sort_values().unique()
        start = dates[0]
        total_end = dates[-1]

        train_delta = pd.DateOffset(years=train_years)
        test_delta = pd.DateOffset(months=test_months)

        windows: List[Window] = []
        test_start = start + train_delta

        while test_start <= total_end:
            # Snap to actual available dates
            train_start_snap = dates[dates >= start][0]
            train_end_date = test_start - pd.Timedelta(days=1)
            train_end_idx = dates[dates <= train_end_date]
            if train_end_idx.empty:
                break
            train_end_snap = train_end_idx[-1]

            test_start_idx = dates[dates >= test_start]
            if test_start_idx.empty:
                break
            test_start_snap = test_start_idx[0]

            test_end_date = test_start + test_delta - pd.Timedelta(days=1)
            test_end_idx = dates[dates <= test_end_date]
            if test_end_idx.empty:
                break
            test_end_snap = test_end_idx[-1]

            # Ensure at least one test day exists
            if test_start_snap > test_end_snap:
                test_start = test_start + test_delta
                continue

            windows.append(
                Window(
                    train_start=train_start_snap,
                    train_end=train_end_snap,
                    test_start=test_start_snap,
                    test_end=test_end_snap,
                )
            )

            if self.verbose:
                logger.info("Generated %s", windows[-1])

            test_start = test_start + test_delta

        if not windows:
            logger.warning(
                "No walk-forward windows generated.  Date range "
                "%s → %s with train_years=%d, test_months=%d may be "
                "too short.",
                start.date(),
                total_end.date(),
                train_years,
                test_months,
            )

        return windows

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        strategy: StrategyProtocol,
        train_years: int = 3,
        test_months: int = 6,
        expanding_window: bool = False,
        **kwargs: Any,
    ) -> WalkForwardResult:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or adjusted-close price DataFrame (dates × assets).
        returns : pd.DataFrame
            Pre-computed daily return DataFrame (dates × assets).
            Must share the same index as *prices*.
        strategy : StrategyProtocol
            Strategy object implementing ``fit(prices, returns, **kwargs)``
            and ``predict(prices, returns, **kwargs) -> (returns, positions)``.
        train_years : int
            Length of each training window in years.
        test_months : int
            Length of each test window in months.
        expanding_window : bool
            If True, the training window expands from the fixed start date
            rather than rolling forward (default False — rolling window).
        **kwargs
            Additional keyword arguments forwarded to ``strategy.fit``
            and ``strategy.predict``.

        Returns
        -------
        result : WalkForwardResult
            Aggregated out-of-sample results across all windows.
        """
        if prices.empty or returns.empty:
            raise ValueError("prices and returns must not be empty.")

        dates = returns.index.intersection(prices.index)
        if dates.empty:
            raise ValueError("prices and returns share no common dates.")

        windows = self.generate_windows(
            pd.DatetimeIndex(dates),
            train_years=train_years,
            test_months=test_months,
        )
        if not windows:
            raise ValueError(
                "No walk-forward windows could be generated from the "
                "available data.  Consider reducing train_years or "
                "test_months."
            )

        window_records: List[WindowRecord] = []
        origin_train_start = windows[0].train_start

        for i, w in enumerate(windows):
            effective_train_start = origin_train_start if expanding_window else w.train_start

            # Slice in-sample data
            is_prices = prices.loc[effective_train_start : w.train_end]
            is_returns = returns.loc[effective_train_start : w.train_end]

            # Slice out-of-sample data
            oos_prices = prices.loc[w.test_start : w.test_end]
            oos_returns_raw = returns.loc[w.test_start : w.test_end]

            if self.verbose:
                logger.info(
                    "[Window %d/%d] Training %s→%s (%d days), "
                    "Testing %s→%s (%d days).",
                    i + 1,
                    len(windows),
                    effective_train_start.date(),
                    w.train_end.date(),
                    len(is_returns),
                    w.test_start.date(),
                    w.test_end.date(),
                    len(oos_returns_raw),
                )

            # --- 1. Fit on in-sample ---
            try:
                strategy.fit(is_prices, is_returns, **kwargs)
            except Exception as exc:
                logger.error(
                    "[Window %d] strategy.fit() failed: %s.  Skipping window.",
                    i + 1,
                    exc,
                    exc_info=True,
                )
                continue

            # --- 2. Predict on out-of-sample ---
            try:
                result = strategy.predict(oos_prices, oos_returns_raw, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    oos_strat_returns, oos_positions = result
                else:
                    # Strategy returned only a Series (no position data)
                    oos_strat_returns = result
                    oos_positions = None
            except Exception as exc:
                logger.error(
                    "[Window %d] strategy.predict() failed: %s.  Skipping window.",
                    i + 1,
                    exc,
                    exc_info=True,
                )
                continue

            # --- 3. Validate outputs ---
            if not isinstance(oos_strat_returns, pd.Series):
                logger.error(
                    "[Window %d] strategy.predict() must return a pd.Series "
                    "as the first element.  Got %s.  Skipping.",
                    i + 1,
                    type(oos_strat_returns),
                )
                continue

            if oos_strat_returns.empty:
                logger.warning("[Window %d] OOS returns are empty.  Skipping.", i + 1)
                continue

            # --- 4. Record ---
            window_records.append(
                WindowRecord(
                    train_start=effective_train_start,
                    train_end=w.train_end,
                    test_start=w.test_start,
                    test_end=w.test_end,
                    oos_returns=oos_strat_returns,
                    oos_positions=oos_positions,
                )
            )

        if not window_records:
            raise RuntimeError(
                "All walk-forward windows failed or produced no results."
            )

        logger.info(
            "Walk-forward complete: %d/%d windows produced results.",
            len(window_records),
            len(windows),
        )

        return WalkForwardResult(
            windows=window_records,
            risk_free_rate=self.risk_free_rate,
        )
