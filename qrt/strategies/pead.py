"""
Post-Earnings Announcement Drift (PEAD) Strategy
==================================================
Go long securities with positive earnings surprises, short those with negative
surprises.  Signals are held for a configurable window (default 20 trading
days) after each announcement, lagged one day to avoid look-ahead bias.

When explicit earnings event data is not supplied, the strategy synthesizes
approximate quarterly announcement dates from the price history and estimates
surprise magnitude from abnormal returns around those dates.

Supports optional sector-neutral ranking: when sector metadata is provided,
securities are ranked within each sector rather than across the full universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class PEAD(Strategy):
    """
    Post-Earnings Announcement Drift strategy.

    Parameters
    ----------
    holding_period : int
        Number of trading days to hold the signal after each earnings event
        (default 20).
    long_pct : float
        Fraction of the universe (or sector) to go long (default 0.20).
    short_pct : float
        Fraction of the universe (or sector) to go short (default 0.20).
    target_gross : float
        Target gross exposure; long + |short| are normalised to this value
        (default 1.0).
    min_abs_surprise : float
        Minimum absolute earnings-surprise percentage required to generate a
        signal.  Events with ``|surprise| < min_abs_surprise`` are ignored
        (default 0.01, i.e. 1 %).
    rebalance_freq : int
        How often (in trading days) the strategy scans for new events and
        refreshes signals (default 1 = daily).
    sector_neutral : bool
        If True **and** sector information is available via *kwargs*, rank
        securities within each sector rather than across the whole universe
        (default True).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Ball & Brown (1968), Bernard & Thomas (1989) — post-earnings announcement drift"
        ),
        "historical_evidence": (
            "One of the oldest documented anomalies; robust across markets and time periods"
        ),
        "implementation_risks": (
            "Requires accurate event timing (BMO/AMC), earnings surprise measurement error, "
            "crowding around earnings"
        ),
        "realistic_expectations": (
            "Well-evidenced research premium; implementation quality "
            "(timing, surprise accuracy) is critical"
        ),
    }

    def __init__(
        self,
        holding_period: int = 20,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        target_gross: float = 1.0,
        min_abs_surprise: float = 0.01,
        rebalance_freq: int = 1,
        sector_neutral: bool = True,
    ) -> None:
        params = dict(
            holding_period=holding_period,
            long_pct=long_pct,
            short_pct=short_pct,
            target_gross=target_gross,
            min_abs_surprise=min_abs_surprise,
            rebalance_freq=rebalance_freq,
            sector_neutral=sector_neutral,
        )
        super().__init__(name="PEAD", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesize_earnings_events(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        quarterly_spacing: int = 63,
        window: int = 3,
    ) -> pd.DataFrame:
        """Estimate approximate earnings dates and surprise from price data.

        Heuristic: place earnings dates every *quarterly_spacing* trading days
        for each security, then measure abnormal return (asset return minus
        cross-sectional mean) in a short window around the estimated date.
        The abnormal return serves as a proxy for earnings surprise.

        Parameters
        ----------
        prices : pd.DataFrame
            Price matrix (dates x securities).
        returns : pd.DataFrame
            Return matrix aligned with *prices*.
        quarterly_spacing : int
            Approximate number of trading days between earnings events.
        window : int
            Half-window (in days) around the estimated date used to compute
            the abnormal return.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``announcement_date``, ``earnings_surprise_pct``.
        """
        cross_mean = returns.mean(axis=1)
        abnormal = returns.sub(cross_mean, axis=0)

        rows: list[dict] = []
        dates = prices.index
        n_dates = len(dates)

        for col in prices.columns:
            col_abn = abnormal[col]
            # Place synthetic events every quarterly_spacing days
            idx = quarterly_spacing
            while idx < n_dates:
                start = max(0, idx - window)
                end = min(n_dates, idx + window + 1)
                abn_slice = col_abn.iloc[start:end].dropna()
                if len(abn_slice) == 0:
                    idx += quarterly_spacing
                    continue
                surprise = float(abn_slice.sum())
                rows.append(
                    {
                        "ticker": col,
                        "announcement_date": dates[idx],
                        "earnings_surprise_pct": surprise,
                    }
                )
                idx += quarterly_spacing

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "announcement_date", "earnings_surprise_pct"]
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _normalise_events(events: pd.DataFrame) -> pd.DataFrame:
        """Normalise an earnings-events DataFrame to a canonical schema.

        Accepts ``security_id`` or ``ticker`` for the security column and
        coerces ``announcement_date`` to datetime.  Rows with NaN surprise
        are dropped.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``announcement_date``, ``earnings_surprise_pct``.
        """
        df = events.copy()

        # Resolve security identifier column
        if "ticker" not in df.columns and "security_id" in df.columns:
            df = df.rename(columns={"security_id": "ticker"})

        # Compute earnings_surprise_pct from raw EPS fields if needed
        if "earnings_surprise_pct" not in df.columns:
            if {"actual_eps", "consensus_eps"}.issubset(df.columns):
                denom = df["consensus_eps"].replace(0, np.nan).abs()
                df["earnings_surprise_pct"] = (
                    (df["actual_eps"] - df["consensus_eps"]) / denom
                )
            else:
                raise ValueError(
                    "earnings_events must contain 'earnings_surprise_pct' or "
                    "both 'actual_eps' and 'consensus_eps'."
                )

        df["announcement_date"] = pd.to_datetime(df["announcement_date"])
        df = df.dropna(subset=["earnings_surprise_pct", "ticker", "announcement_date"])

        return df[["ticker", "announcement_date", "earnings_surprise_pct"]]

    def _rank_and_assign(
        self,
        surprises: pd.Series,
        sectors: pd.Series | None,
    ) -> pd.Series:
        """Rank securities by surprise and assign +1 (long) / -1 (short) / 0.

        Parameters
        ----------
        surprises : pd.Series
            Indexed by ticker, values are earnings surprise percentages.
        sectors : pd.Series or None
            Indexed by ticker, values are sector strings.  If provided and
            ``self.params['sector_neutral']`` is True, ranking is done within
            each sector.

        Returns
        -------
        pd.Series
            Signal direction per ticker: +1, -1, or 0.
        """
        long_pct: float = self.params["long_pct"]
        short_pct: float = self.params["short_pct"]
        min_abs: float = self.params["min_abs_surprise"]

        # Filter out negligible surprises
        valid = surprises[surprises.abs() >= min_abs].dropna()
        if valid.empty:
            return pd.Series(0.0, index=surprises.index)

        use_sectors = (
            self.params["sector_neutral"]
            and sectors is not None
            and not sectors.empty
        )

        direction = pd.Series(0.0, index=surprises.index)

        if use_sectors:
            # Rank within each sector independently
            sector_groups = sectors.reindex(valid.index).dropna()
            for _sector, members in sector_groups.groupby(sector_groups):
                group_surprises = valid.reindex(members.index).dropna()
                self._assign_long_short(group_surprises, direction, long_pct, short_pct)
        else:
            self._assign_long_short(valid, direction, long_pct, short_pct)

        return direction

    @staticmethod
    def _assign_long_short(
        surprises: pd.Series,
        direction: pd.Series,
        long_pct: float,
        short_pct: float,
    ) -> None:
        """In-place assignment of +1 / -1 into *direction* based on rank."""
        n = len(surprises)
        if n == 0:
            return
        n_long = max(1, int(np.floor(n * long_pct)))
        n_short = max(1, int(np.floor(n * short_pct)))

        ranked = surprises.rank(ascending=True, method="first")
        long_tickers = ranked.nlargest(n_long).index
        short_tickers = ranked.nsmallest(n_short).index

        direction.loc[long_tickers] = 1.0
        direction.loc[short_tickers] = -1.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute PEAD signals from earnings events.

        For each trading day the method checks which securities had an
        earnings event within the trailing *holding_period* window.  Those
        securities receive a signal (+1 long or -1 short) determined by their
        earnings-surprise rank.  Signals are **lagged by one day**: an event
        on day *t* first appears in the signal on day *t + 1*.

        Parameters
        ----------
        prices : pd.DataFrame
            Price matrix (dates x securities).
        returns : pd.DataFrame
            Return matrix aligned with *prices*.
        **kwargs
            earnings_events : pd.DataFrame, optional
                Explicit earnings data with columns ``ticker`` (or
                ``security_id``), ``announcement_date``, and either
                ``earnings_surprise_pct`` or ``actual_eps`` + ``consensus_eps``.
            sectors : dict, optional
                Mapping of ticker -> sector string, used for sector-neutral
                ranking.

        Returns
        -------
        pd.DataFrame
            Signals in {-1, 0, +1}, same shape as *prices*.
        """
        holding: int = self.params["holding_period"]
        rebalance_freq: int = self.params["rebalance_freq"]

        # --- Obtain and normalise earnings events -------------------------
        raw_events = kwargs.get("earnings_events", None)
        if raw_events is not None and not raw_events.empty:
            events = self._normalise_events(raw_events)
        else:
            events = self._synthesize_earnings_events(prices, returns)

        # Keep only events whose tickers exist in the price universe
        valid_tickers = set(prices.columns)
        events = events[events["ticker"].isin(valid_tickers)].copy()

        if events.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Sector metadata (optional) -----------------------------------
        sectors_raw = kwargs.get("sectors", None)
        sectors_series: pd.Series | None = None
        if sectors_raw is not None:
            if isinstance(sectors_raw, dict):
                sectors_series = pd.Series(sectors_raw, name="sector")
            elif isinstance(sectors_raw, pd.Series):
                sectors_series = sectors_raw
            elif isinstance(sectors_raw, pd.DataFrame):
                # Assume first column maps ticker -> sector
                col = sectors_raw.columns[0]
                sectors_series = sectors_raw.set_index(sectors_raw.columns[0])[
                    sectors_raw.columns[1]
                ] if len(sectors_raw.columns) >= 2 else None

        # --- Build per-day signal matrix ----------------------------------
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        dates = prices.index
        date_set = set(dates)

        # Pre-index events by announcement date for fast lookup
        events = events.sort_values("announcement_date")

        for day_idx in range(len(dates)):
            if day_idx % rebalance_freq != 0:
                continue

            current_date = dates[day_idx]

            # Collect events whose *tradable date* (announcement + 1 lag day)
            # falls within the holding window ending on current_date.
            # Tradable date = announcement_date + 1 business day.
            # We want events where:
            #   current_date - holding + 1 <= tradable_date <= current_date
            # i.e. announcement_date between (current_date - holding) and
            #      (current_date - 1) inclusive (the lag shifts by one day).
            window_start = current_date - pd.tseries.offsets.BDay(holding)
            window_end = current_date - pd.tseries.offsets.BDay(1)

            mask = (
                (events["announcement_date"] >= window_start)
                & (events["announcement_date"] <= window_end)
            )
            active_events = events.loc[mask]

            if active_events.empty:
                continue

            # For each ticker keep the most recent event (closest announcement)
            latest = (
                active_events
                .sort_values("announcement_date")
                .drop_duplicates(subset="ticker", keep="last")
                .set_index("ticker")
            )

            surprise = latest["earnings_surprise_pct"]
            direction = self._rank_and_assign(surprise, sectors_series)

            for ticker, sig in direction.items():
                if sig != 0.0 and ticker in signals.columns:
                    signals.loc[current_date, ticker] = sig

        # Forward-fill signals up to rebalance_freq - 1 days so that
        # inter-scan days keep the position (only when rebalance_freq > 1).
        if rebalance_freq > 1:
            signals = signals.replace(0.0, np.nan).ffill(limit=rebalance_freq - 1).fillna(0.0)

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Equal-weight within long and short legs, normalised to target gross.

        Each leg (long / short) receives equal weight per constituent.  The
        total gross exposure (sum of absolute weights) is scaled to
        ``target_gross``.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals`` (values in {-1, 0, +1}).
        returns : pd.DataFrame, optional
            Not used directly but accepted for interface consistency.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        target_gross: float = self.params["target_gross"]

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            row = signals.loc[date]
            long_mask = row > 0
            short_mask = row < 0
            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long == 0 and n_short == 0:
                continue

            # Equal-weight within each leg
            if n_long > 0:
                weights.loc[date, long_mask] = 1.0 / n_long
            if n_short > 0:
                weights.loc[date, short_mask] = -1.0 / n_short

            # Normalise to target_gross
            gross = weights.loc[date].abs().sum()
            if gross > 0:
                weights.loc[date] = weights.loc[date] / gross * target_gross

        return weights

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """End-to-end: signals -> weights -> backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
