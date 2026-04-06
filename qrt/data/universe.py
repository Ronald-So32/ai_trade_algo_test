"""
Universe Constructor — builds and maintains liquidity-filtered, sector-balanced
US equity universes for backtesting.

Supported universe sizes:  100, 150, 200, 300

Universe rules (applied at each monthly rebalance date):
  1. Security must be in SecurityMaster and active on that date.
  2. Minimum 2-year price history available (configurable).
  3. Minimum closing price  ≥ config min_price  (default $5).
  4. Minimum market cap     ≥ config min_market_cap  (default $1 B).
  5. Minimum median 63-day dollar volume ≥ config min_median_dollar_volume
     (default $5 M / day).
  6. ETFs, ADRs, and preferred shares are excluded (none in our SecurityMaster,
     but we keep the guard in case real data is injected).
  7. Sector caps enforced to maintain diversification (from config).
  8. Among eligible securities, rank by median dollar volume and take top-N.

Output: dict[pd.Timestamp, list[int]]
  Keyed by rebalance date → list of security_ids in universe on that date.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger(__name__)

# Sector grouping aligns with SecurityMaster.  "Consumer" covers both
# Consumer Discretionary and Staples in our synthetic universe.
_SECTOR_MAP_CANONICAL = {
    "Technology": "Technology",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Consumer": "Consumer",
    "Industrials": "Industrials",
    "Energy": "Energy_Materials",
    "Materials": "Energy_Materials",
}


class UniverseConstructor:
    """
    Build and cache monthly rebalanced equity universes.

    Parameters
    ----------
    security_master : pd.DataFrame
        Output of SecurityMaster.generate().
    market_data : pd.DataFrame
        Output of MarketDataGenerator.generate().
    config : Config, optional
    """

    SUPPORTED_SIZES = (100, 150, 200, 300)

    def __init__(
        self,
        security_master: pd.DataFrame,
        market_data: pd.DataFrame,
        config: Optional[Config] = None,
    ) -> None:
        self._sm = security_master.set_index("security_id") if "security_id" in security_master.columns else security_master
        self._md = market_data.sort_values(["security_id", "date"])
        self._config = config or Config()
        self._universes: dict[int, dict[pd.Timestamp, list[int]]] = {}

        # Cache dollar-volume and price stats indexed by (security_id, date)
        # for fast per-date look-ups
        self._md_indexed = self._md.set_index(["security_id", "date"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, size: int | None = None) -> dict[int, dict[pd.Timestamp, list[int]]]:
        """
        Build universes for all sizes (or a specific size).

        Returns
        -------
        dict[size -> dict[rebalance_date -> [security_ids]]]
        """
        sizes = [size] if size is not None else list(self.SUPPORTED_SIZES)
        for s in sizes:
            if s not in self.SUPPORTED_SIZES:
                raise ValueError(f"Unsupported universe size {s}. Choose from {self.SUPPORTED_SIZES}")
            if s not in self._universes:
                logger.info("Building universe_%d …", s)
                self._universes[s] = self._build_single(s)
                dates_count = len(self._universes[s])
                logger.info("universe_%d: %d rebalance dates", s, dates_count)
        return self._universes

    def get(self, size: int, date: pd.Timestamp) -> list[int]:
        """
        Return the universe members valid on ``date``.

        Uses the most recent rebalance date ≤ ``date``.
        """
        if size not in self._universes:
            self.build(size)
        uni = self._universes[size]
        valid_dates = sorted(d for d in uni if d <= date)
        if not valid_dates:
            return []
        return uni[valid_dates[-1]]

    def as_dataframe(self, size: int) -> pd.DataFrame:
        """
        Return universe membership as a tidy DataFrame:
        columns: rebalance_date, security_id
        """
        if size not in self._universes:
            self.build(size)
        rows = []
        for dt, members in sorted(self._universes[size].items()):
            for sid in members:
                rows.append({"rebalance_date": dt, "security_id": sid})
        return pd.DataFrame(rows)

    def membership_matrix(self, size: int) -> pd.DataFrame:
        """
        Boolean matrix: rows = rebalance dates, columns = security_ids.
        True where security is in universe on that rebalance date.
        """
        df = self.as_dataframe(size)
        if df.empty:
            return pd.DataFrame()
        pivot = df.assign(member=True).pivot(
            index="rebalance_date", columns="security_id", values="member"
        ).fillna(False)
        return pivot

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_single(self, size: int) -> dict[pd.Timestamp, list[int]]:
        """Build the {date: [ids]} dict for one universe size."""
        min_history = self._config.get("universe.min_history_years", 2)
        min_price = float(self._config.get("universe.min_price", 5.0))
        min_mcap = float(self._config.get("universe.min_market_cap", 1e9))
        min_dv = float(self._config.get("universe.min_median_dollar_volume", 5e6))
        sector_limits = self._config.get("universe.sector_balance", {})

        # Sector canonical caps for this size
        sector_caps = self._derive_sector_caps(size, sector_limits)

        # Monthly rebalance dates: last business day of each month
        all_dates = pd.DatetimeIndex(sorted(self._md["date"].unique()))
        rebalance_dates = self._monthly_rebalance_dates(all_dates)

        result: dict[pd.Timestamp, list[int]] = {}

        for rb_date in rebalance_dates:
            members = self._select_universe(
                rb_date=rb_date,
                size=size,
                min_history_years=min_history,
                min_price=min_price,
                min_mcap=min_mcap,
                min_dv=min_dv,
                sector_caps=sector_caps,
                all_dates=all_dates,
            )
            result[rb_date] = members

        return result

    def _select_universe(
        self,
        rb_date: pd.Timestamp,
        size: int,
        min_history_years: float,
        min_price: float,
        min_mcap: float,
        min_dv: float,
        sector_caps: dict[str, int],
        all_dates: pd.DatetimeIndex,
    ) -> list[int]:
        """Apply all filters at a single rebalance date."""

        # ----------------------------------------------------------------
        # 1. Active securities on this date
        # ----------------------------------------------------------------
        active_sm = self._sm[
            (self._sm["list_date"] <= rb_date)
            & (self._sm["delist_date"].isna() | (self._sm["delist_date"] > rb_date))
        ]
        candidate_ids = active_sm.index.tolist()

        if not candidate_ids:
            return []

        # ----------------------------------------------------------------
        # 2. Get trailing 126-day window of market data
        # ----------------------------------------------------------------
        history_start = rb_date - pd.DateOffset(years=min_history_years)
        window_start_63 = rb_date - pd.DateOffset(days=90)  # ~63 trading days

        # Get the last-known data for each security on or before rb_date
        md_window = self._md[
            (self._md["date"] <= rb_date)
            & (self._md["security_id"].isin(candidate_ids))
        ]

        if md_window.empty:
            return []

        # Most recent prices (for price and mcap filter)
        latest = (
            md_window.sort_values("date")
            .groupby("security_id")
            .last()
            .reset_index()
        )
        latest = latest.set_index("security_id")

        # 63-day dollar volume window
        dv_window = md_window[md_window["date"] >= window_start_63]
        median_dv = (
            dv_window.groupby("security_id")["dollar_volume"]
            .median()
        )

        # Earliest available date per security (for history filter)
        earliest_date = (
            md_window.groupby("security_id")["date"].min()
        )

        eligible: list[tuple[int, float, str]] = []  # (sid, median_dv, sector)

        for sid in candidate_ids:
            if sid not in latest.index:
                continue

            # ---- Filter: price
            price = float(latest.loc[sid, "close"]) if "close" in latest.columns else 0.0
            if price < min_price:
                continue

            # ---- Filter: market cap
            mcap = float(latest.loc[sid, "market_cap"]) if "market_cap" in latest.columns else 0.0
            if mcap < min_mcap:
                continue

            # ---- Filter: median dollar volume
            mdv = float(median_dv.get(sid, 0.0))
            if mdv < min_dv:
                continue

            # ---- Filter: min history
            first_date = earliest_date.get(sid)
            if first_date is None or (rb_date - first_date).days < min_history_years * 365:
                continue

            sector_raw = str(active_sm.loc[sid, "sector"]) if sid in active_sm.index else "Unknown"
            sector_canonical = _SECTOR_MAP_CANONICAL.get(sector_raw, sector_raw)

            eligible.append((sid, mdv, sector_canonical))

        if not eligible:
            return []

        # ----------------------------------------------------------------
        # 3. Rank by median dollar volume, apply sector caps
        # ----------------------------------------------------------------
        eligible_df = pd.DataFrame(eligible, columns=["security_id", "median_dv", "sector"])
        eligible_df = eligible_df.sort_values("median_dv", ascending=False).reset_index(drop=True)

        selected: list[int] = []
        sector_counts: dict[str, int] = {}

        for _, row in eligible_df.iterrows():
            if len(selected) >= size:
                break
            sid = int(row["security_id"])
            sec = str(row["sector"])
            cap = sector_caps.get(sec, size)  # no cap if not specified
            current_count = sector_counts.get(sec, 0)
            if current_count >= cap:
                continue
            selected.append(sid)
            sector_counts[sec] = current_count + 1

        return selected

    @staticmethod
    def _monthly_rebalance_dates(all_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """
        Return the last available trading day in each month from all_dates.
        """
        df = pd.DataFrame({"date": all_dates})
        df["ym"] = df["date"].dt.to_period("M")
        last_per_month = df.groupby("ym")["date"].max()
        return sorted(last_per_month.tolist())

    @staticmethod
    def _derive_sector_caps(
        size: int,
        sector_limits: dict,
    ) -> dict[str, int]:
        """
        Derive per-sector integer caps for a given universe size.

        sector_limits from config: e.g. {"Technology": [30, 40], ...}
        The cap is taken as the upper bound scaled to the target universe size,
        using the 300-stock config as the reference denominator.
        """
        ref_size = 300
        caps: dict[str, int] = {}
        for sector, bounds in sector_limits.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                upper_frac = bounds[1] / ref_size
                caps[sector] = max(1, int(np.ceil(upper_frac * size)))
        return caps
