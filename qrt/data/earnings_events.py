"""
Earnings Event Data Ingestion Module
=====================================
Downloads and normalizes earnings event data for the PEAD strategy.

Supports multiple data sources with adapter pattern:
  - Yahoo Finance earnings dates (free, primary)
  - SEC EDGAR CompanyFacts API (free, supplementary)
  - Synthetic fallback from price/return data

Enforces point-in-time correctness:
  - After-close announcements -> tradable next trading day
  - Before-open announcements -> tradable same day
  - Unknown timing -> conservative next trading day default

Output:
  data/parquet/earnings_events.parquet
  data/parquet/earnings_surprises.parquet
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event timing rules
# ---------------------------------------------------------------------------

def resolve_tradable_date(
    announcement_date: pd.Timestamp,
    report_time: str,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp:
    """Determine the first date a signal from this event is tradable.

    Parameters
    ----------
    announcement_date : pd.Timestamp
        Calendar date of the earnings announcement.
    report_time : str
        One of "BMO" (before market open), "AMC" (after market close),
        or "unknown".
    trading_dates : pd.DatetimeIndex
        Sorted index of valid trading dates.

    Returns
    -------
    pd.Timestamp
        First date the signal can be used to trade.
    """
    if report_time == "BMO":
        # Before market open: tradable same day
        idx = trading_dates.searchsorted(announcement_date, side="left")
        if idx < len(trading_dates) and trading_dates[idx] == announcement_date:
            return trading_dates[idx]
        # If announcement_date is not a trading day, use next
        idx = min(idx, len(trading_dates) - 1)
        return trading_dates[idx]
    else:
        # AMC or unknown: tradable next trading day
        idx = trading_dates.searchsorted(announcement_date, side="right")
        idx = min(idx, len(trading_dates) - 1)
        return trading_dates[idx]


# ---------------------------------------------------------------------------
# Yahoo Finance adapter
# ---------------------------------------------------------------------------

class YahooEarningsAdapter:
    """Fetch earnings dates and EPS data from Yahoo Finance."""

    @staticmethod
    def fetch(tickers: list[str]) -> pd.DataFrame:
        """Download earnings calendar for each ticker.

        Returns DataFrame with columns:
            ticker, announcement_date, actual_eps, consensus_eps,
            earnings_surprise, earnings_surprise_pct, report_time
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed; cannot fetch earnings")
            return pd.DataFrame()

        rows = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Get earnings dates
                try:
                    cal = stock.get_earnings_dates(limit=60)
                except Exception:
                    cal = None

                if cal is None or cal.empty:
                    continue

                for date_idx, row in cal.iterrows():
                    ann_date = pd.Timestamp(date_idx)
                    if pd.isna(ann_date):
                        continue

                    actual = row.get("Reported EPS", np.nan)
                    estimate = row.get("EPS Estimate", np.nan)

                    if pd.notna(actual) and pd.notna(estimate) and estimate != 0:
                        surprise = actual - estimate
                        surprise_pct = surprise / abs(estimate)
                    else:
                        surprise = np.nan
                        surprise_pct = np.nan

                    rows.append({
                        "ticker": ticker,
                        "announcement_date": ann_date.normalize(),
                        "actual_eps": actual,
                        "consensus_eps": estimate,
                        "earnings_surprise": surprise,
                        "earnings_surprise_pct": surprise_pct,
                        "report_time": "unknown",
                    })
            except Exception as e:
                logger.debug(f"Failed to fetch earnings for {ticker}: {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["announcement_date"] = pd.to_datetime(df["announcement_date"])
        return df


# ---------------------------------------------------------------------------
# Synthetic earnings adapter (fallback)
# ---------------------------------------------------------------------------

class SyntheticEarningsAdapter:
    """Generate approximate earnings events from price data.

    Places earnings events roughly every 63 trading days per asset.
    Surprise is estimated from abnormal returns in a 3-day window
    around the synthetic event date.
    """

    @staticmethod
    def generate(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        frequency: int = 63,
    ) -> pd.DataFrame:
        """Create synthetic earnings events.

        Parameters
        ----------
        prices : pd.DataFrame
            Dates x assets close prices.
        returns : pd.DataFrame
            Dates x assets daily returns.
        frequency : int
            Approximate trading days between earnings (default 63 = quarterly).

        Returns
        -------
        pd.DataFrame with earnings event columns.
        """
        trading_dates = prices.index
        market_return = returns.mean(axis=1)

        rows = []
        for col in prices.columns:
            asset_ret = returns[col].dropna()
            if len(asset_ret) < frequency * 2:
                continue

            # Place events every `frequency` days starting from offset
            event_indices = list(range(frequency, len(trading_dates), frequency))

            for idx in event_indices:
                ann_date = trading_dates[idx]

                # Compute abnormal return in [-1, +1] day window around event
                window_start = max(0, idx - 1)
                window_end = min(len(trading_dates) - 1, idx + 1)
                abnormal_rets = (
                    returns[col].iloc[window_start:window_end + 1]
                    - market_return.iloc[window_start:window_end + 1]
                )
                cum_abnormal = abnormal_rets.sum()

                # Use abnormal return as surprise proxy
                # Scale to look like EPS surprise percentage
                surprise_pct = cum_abnormal * 10  # amplify for ranking purposes

                rows.append({
                    "ticker": col,
                    "announcement_date": ann_date,
                    "actual_eps": np.nan,
                    "consensus_eps": np.nan,
                    "earnings_surprise": np.nan,
                    "earnings_surprise_pct": surprise_pct,
                    "report_time": "unknown",
                })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main ingestion class
# ---------------------------------------------------------------------------

class EarningsDataManager:
    """Orchestrate earnings data acquisition, validation, and storage.

    Parameters
    ----------
    data_dir : str or Path
        Directory for parquet output (default "data/parquet").
    """

    def __init__(self, data_dir: str | Path = "data/parquet") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._events: Optional[pd.DataFrame] = None

    def fetch_yahoo(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch earnings from Yahoo Finance."""
        logger.info(f"Fetching Yahoo Finance earnings for {len(tickers)} tickers")
        df = YahooEarningsAdapter.fetch(tickers)
        logger.info(f"Fetched {len(df)} earnings events")
        return df

    def generate_synthetic(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate synthetic earnings events from price data."""
        logger.info("Generating synthetic earnings events")
        df = SyntheticEarningsAdapter.generate(prices, returns)
        logger.info(f"Generated {len(df)} synthetic earnings events")
        return df

    def fetch_or_generate(
        self,
        tickers: list[str],
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        prefer_real: bool = True,
    ) -> pd.DataFrame:
        """Try real data first, fall back to synthetic.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to fetch.
        prices, returns : pd.DataFrame
            Price/return data for synthetic fallback.
        prefer_real : bool
            If True, attempt Yahoo first.

        Returns
        -------
        pd.DataFrame of earnings events.
        """
        df = pd.DataFrame()

        if prefer_real:
            try:
                df = self.fetch_yahoo(tickers)
            except Exception as e:
                logger.warning(f"Yahoo earnings fetch failed: {e}")

        if df.empty:
            df = self.generate_synthetic(prices, returns)

        if not df.empty:
            df = self.validate(df, prices.index)

        self._events = df
        return df

    def resolve_tradable_dates(
        self,
        events: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Add tradable_date column to events DataFrame."""
        if events.empty:
            events["tradable_date"] = pd.Series(dtype="datetime64[ns]")
            return events

        tradable = events.apply(
            lambda row: resolve_tradable_date(
                row["announcement_date"],
                row.get("report_time", "unknown"),
                trading_dates,
            ),
            axis=1,
        )
        events = events.copy()
        events["tradable_date"] = tradable
        return events

    def validate(
        self,
        events: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Run validation checks on earnings data.

        Checks:
        - No future-dated announcements
        - No duplicate events (same ticker + date)
        - Tradable date correctly derived
        """
        n_before = len(events)

        # Remove future-dated
        today = pd.Timestamp.today().normalize()
        events = events[events["announcement_date"] <= today].copy()

        # Remove duplicates
        events = events.drop_duplicates(
            subset=["ticker", "announcement_date"], keep="first"
        )

        # Resolve tradable dates
        events = self.resolve_tradable_dates(events, trading_dates)

        # Filter to events within our trading date range
        if len(trading_dates) > 0:
            events = events[
                (events["tradable_date"] >= trading_dates[0])
                & (events["tradable_date"] <= trading_dates[-1])
            ]

        n_after = len(events)
        if n_before != n_after:
            logger.info(
                f"Validation: {n_before} -> {n_after} events "
                f"(removed {n_before - n_after})"
            )

        return events.reset_index(drop=True)

    def save(self, events: pd.DataFrame) -> None:
        """Persist events to parquet."""
        path = self.data_dir / "earnings_events.parquet"
        events.to_parquet(path, index=False)
        logger.info(f"Saved {len(events)} events to {path}")

    def load(self) -> pd.DataFrame:
        """Load events from parquet."""
        path = self.data_dir / "earnings_events.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
