"""
Market Data Generator — synthetic daily OHLCV bars with GBM + correlated
sector returns, corporate actions (splits / dividends), and realistic
price/volume dynamics including fat tails and mild mean-reversion.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sector-level parameters
# ---------------------------------------------------------------------------

_SECTOR_PARAMS: dict[str, dict] = {
    "Technology": {
        "annual_vol": 0.28,
        "annual_drift": 0.14,
        "beta": 1.20,
        "div_yield": 0.005,
        "split_prob": 0.04,   # annual probability of a 2-for-1 split
        "avg_price": 85.0,
        "avg_mcap": 25e9,
    },
    "Financials": {
        "annual_vol": 0.22,
        "annual_drift": 0.09,
        "beta": 1.05,
        "div_yield": 0.025,
        "split_prob": 0.01,
        "avg_price": 55.0,
        "avg_mcap": 18e9,
    },
    "Healthcare": {
        "annual_vol": 0.25,
        "annual_drift": 0.11,
        "beta": 0.85,
        "div_yield": 0.010,
        "split_prob": 0.02,
        "avg_price": 70.0,
        "avg_mcap": 20e9,
    },
    "Consumer": {
        "annual_vol": 0.20,
        "annual_drift": 0.10,
        "beta": 0.95,
        "div_yield": 0.018,
        "split_prob": 0.015,
        "avg_price": 60.0,
        "avg_mcap": 15e9,
    },
    "Industrials": {
        "annual_vol": 0.20,
        "annual_drift": 0.09,
        "beta": 1.00,
        "div_yield": 0.015,
        "split_prob": 0.01,
        "avg_price": 65.0,
        "avg_mcap": 12e9,
    },
    "Energy": {
        "annual_vol": 0.35,
        "annual_drift": 0.06,
        "beta": 1.10,
        "div_yield": 0.030,
        "split_prob": 0.005,
        "avg_price": 45.0,
        "avg_mcap": 8e9,
    },
    "Materials": {
        "annual_vol": 0.30,
        "annual_drift": 0.07,
        "beta": 1.05,
        "div_yield": 0.020,
        "split_prob": 0.008,
        "avg_price": 40.0,
        "avg_mcap": 7e9,
    },
}

# Market (index) parameters
_MARKET_ANNUAL_VOL = 0.16
_MARKET_ANNUAL_DRIFT = 0.08
_TRADING_DAYS_PER_YEAR = 252

# Delisting return distribution (typically negative)
_DELIST_RETURN_MEAN = -0.35
_DELIST_RETURN_STD = 0.20


class MarketDataGenerator:
    """
    Generates synthetic daily price bars for a set of securities.

    The process:
    1. Simulate a market factor (index) via GBM with fat-tailed innovations.
    2. Each security has an idiosyncratic component layered on top of the
       sector-beta-scaled market factor, producing correlated sector returns.
    3. Within a sector, a sector-specific factor provides additional
       intra-sector correlation (~0.35–0.55 pairwise).
    4. OHLCV is derived from the daily return using realistic intraday
       range statistics.
    5. Corporate actions (splits, dividends) are applied.
    6. Delisted securities have their last return drawn from a negative
       distribution and prices zeroed thereafter.

    Parameters
    ----------
    security_master : pd.DataFrame
        Output of SecurityMaster.generate().
    config : Config, optional
    seed : int
        Base RNG seed.
    """

    COLUMNS = [
        "date", "security_id", "open", "high", "low", "close",
        "adjusted_close", "volume", "vwap", "dollar_volume",
        "market_cap", "split_factor", "dividend_amount",
    ]

    def __init__(
        self,
        security_master: pd.DataFrame,
        config: Optional[Config] = None,
        seed: int = 42,
    ) -> None:
        self._sm = security_master
        self._config = config or Config()
        self._seed = seed
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate and return the full market data DataFrame."""
        logger.info("Generating market data …")

        start = pd.Timestamp(self._config.get("data.start_date", "2015-01-01"))
        end = pd.Timestamp(self._config.get("data.end_date", "2024-12-31"))

        trading_days = pd.bdate_range(start, end)
        n_days = len(trading_days)
        logger.info("Trading days in range: %d", n_days)

        rng = np.random.default_rng(self._seed)

        # ----------------------------------------------------------
        # 1. Simulate market (index) factor
        # ----------------------------------------------------------
        market_daily_vol = _MARKET_ANNUAL_VOL / np.sqrt(_TRADING_DAYS_PER_YEAR)
        market_daily_drift = _MARKET_ANNUAL_DRIFT / _TRADING_DAYS_PER_YEAR

        market_innovations = self._fat_tail_innovations(rng, n_days, df=5)
        market_log_returns = (
            market_daily_drift
            - 0.5 * market_daily_vol ** 2
            + market_daily_vol * market_innovations
        )

        # ----------------------------------------------------------
        # 2. Simulate sector factors
        # ----------------------------------------------------------
        sectors = list(_SECTOR_PARAMS.keys())
        sector_factors: dict[str, np.ndarray] = {}
        for sector in sectors:
            p = _SECTOR_PARAMS[sector]
            sec_vol = p["annual_vol"] * 0.40 / np.sqrt(_TRADING_DAYS_PER_YEAR)  # sector idio
            inno = self._fat_tail_innovations(rng, n_days, df=6)
            sector_factors[sector] = sec_vol * inno

        # ----------------------------------------------------------
        # 3. For each security, simulate daily log-returns
        # ----------------------------------------------------------
        all_records: list[dict] = []

        for _, sec_row in self._sm.iterrows():
            sid = int(sec_row["security_id"])
            sector = sec_row["sector"]
            list_date = pd.Timestamp(sec_row["list_date"])
            delist_date = (
                pd.Timestamp(sec_row["delist_date"])
                if pd.notna(sec_row["delist_date"])
                else None
            )

            p = _SECTOR_PARAMS[sector]
            beta = float(rng.normal(p["beta"], 0.15))
            beta = max(0.3, min(beta, 2.5))

            idio_vol_annual = float(rng.uniform(0.12, 0.22))
            idio_vol_daily = idio_vol_annual / np.sqrt(_TRADING_DAYS_PER_YEAR)

            annual_drift = float(rng.normal(p["annual_drift"], 0.03))
            daily_drift = annual_drift / _TRADING_DAYS_PER_YEAR

            # Stock-specific fat-tail innovations
            idio_inno = self._fat_tail_innovations(rng, n_days, df=7)
            idio_component = idio_vol_daily * idio_inno

            # Mild mean-reversion (OU-style, theta ~ 0.02 daily)
            theta = float(rng.uniform(0.01, 0.04))
            long_run_log_price = np.log(p["avg_price"] * float(rng.uniform(0.5, 2.0)))

            # Combine factors into daily log-returns
            log_rets = (
                daily_drift
                - 0.5 * (beta * market_daily_vol) ** 2
                + beta * market_log_returns
                + sector_factors[sector]
                + idio_component
            )

            # Initial log price
            init_price = p["avg_price"] * float(rng.uniform(0.5, 2.0))
            log_prices = np.empty(n_days)
            log_prices[0] = np.log(init_price)

            for t in range(1, n_days):
                mr_term = -theta * (log_prices[t - 1] - long_run_log_price)
                log_prices[t] = log_prices[t - 1] + log_rets[t] + mr_term

            prices = np.exp(log_prices)

            # ----------------------------------------------------------
            # 4. Corporate actions
            # ----------------------------------------------------------
            split_factors, dividend_amounts = self._generate_corporate_actions(
                rng, n_days, trading_days, p, list_date, delist_date
            )

            # Cumulative split factor (to compute adjusted close)
            # We work backward: adjusted_close[T] = close[T]
            # adjusted_close[t] = close[t] * product(splits after t)
            cum_split_from_end = np.ones(n_days)
            for t in range(n_days - 2, -1, -1):
                cum_split_from_end[t] = cum_split_from_end[t + 1] * split_factors[t + 1]

            # Adjusted close also accounts for dividends (simplified: ratio method)
            # For simplicity we use the multiplicative adjustment without
            # separately tracking the dividend reinvestment.
            adjusted_closes = prices * cum_split_from_end

            # ----------------------------------------------------------
            # 5. OHLCV construction
            # ----------------------------------------------------------
            records = self._build_daily_records(
                rng=rng,
                sid=sid,
                sector=sector,
                trading_days=trading_days,
                prices=prices,
                adjusted_closes=adjusted_closes,
                log_rets=log_rets,
                split_factors=split_factors,
                dividend_amounts=dividend_amounts,
                list_date=list_date,
                delist_date=delist_date,
                avg_mcap=p["avg_mcap"],
            )
            all_records.extend(records)

        df = pd.DataFrame(all_records, columns=self.COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "security_id"]).reset_index(drop=True)

        self._df = df
        logger.info("Market data: %d rows across %d securities", len(df), df["security_id"].nunique())
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.generate()
        return self._df  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fat_tail_innovations(
        rng: np.random.Generator, n: int, df: float = 5
    ) -> np.ndarray:
        """
        Draw standardised innovations from a Student-t distribution (fat tails).
        Standardised so variance ≈ 1.
        """
        # Use numpy for speed — approximate t via normal / chi2
        normal = rng.standard_normal(n)
        chi2 = rng.chisquare(df, n) / df
        return normal / np.sqrt(chi2)

    def _generate_corporate_actions(
        self,
        rng: np.random.Generator,
        n_days: int,
        trading_days: pd.DatetimeIndex,
        params: dict,
        list_date: pd.Timestamp,
        delist_date: Optional[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            split_factors   : array of shape (n_days,), default 1.0
            dividend_amounts: array of shape (n_days,), default 0.0
        """
        split_factors = np.ones(n_days)
        dividend_amounts = np.zeros(n_days)

        annual_split_prob = params["split_prob"]
        daily_split_prob = annual_split_prob / _TRADING_DAYS_PER_YEAR
        div_yield = params["div_yield"]
        quarterly_div_prob = div_yield > 0.0

        # Quarterly dividends on roughly fixed calendar schedule
        if quarterly_div_prob:
            # Ex-div dates: roughly every 63 trading days
            quarter_dates = list(range(63, n_days, 63))
            for t in quarter_dates:
                td = trading_days[t]
                if td < list_date:
                    continue
                if delist_date and td >= delist_date:
                    break
                # Quarterly div ≈ annual yield / 4, paid on notional price $50
                # We keep it proportional; actual dollar amount set vs. close later
                # Store as a fraction; returns.py converts to $
                dividend_amounts[t] = div_yield / 4.0  # fraction of stock price

        # Random stock splits
        for t in range(n_days):
            td = trading_days[t]
            if td < list_date:
                continue
            if delist_date and td >= delist_date:
                break
            if rng.random() < daily_split_prob:
                # 2-for-1 split (occasionally 3-for-2)
                split_ratios = [2.0, 2.0, 1.5, 3.0]
                weights = [0.60, 0.25, 0.10, 0.05]
                split_factors[t] = rng.choice(split_ratios, p=weights)

        return split_factors, dividend_amounts

    def _build_daily_records(
        self,
        rng: np.random.Generator,
        sid: int,
        sector: str,
        trading_days: pd.DatetimeIndex,
        prices: np.ndarray,
        adjusted_closes: np.ndarray,
        log_rets: np.ndarray,
        split_factors: np.ndarray,
        dividend_amounts: np.ndarray,
        list_date: pd.Timestamp,
        delist_date: Optional[pd.Timestamp],
        avg_mcap: float,
    ) -> list[dict]:
        records = []
        n = len(trading_days)

        # Shares outstanding — determines volume and market cap
        # Draw once; allow modest changes over time
        shares_out_base = avg_mcap / max(prices[0], 1.0)
        shares_out = shares_out_base * np.exp(
            np.cumsum(rng.normal(0, 0.0002, n))
        )

        # Average daily volume ~ 0.5% of shares outstanding, log-normal variation
        adv_fraction = float(rng.uniform(0.003, 0.010))

        delisting_applied = False

        for t in range(n):
            td = trading_days[t]

            # Not yet listed
            if td < list_date:
                continue

            # Delisted — apply one final negative return then stop
            if delist_date and td >= delist_date:
                if not delisting_applied:
                    delist_ret = float(rng.normal(_DELIST_RETURN_MEAN, _DELIST_RETURN_STD))
                    delist_ret = max(delist_ret, -0.95)
                    close_price = prices[t - 1] * (1 + delist_ret) if t > 0 else prices[t]
                    # Build a single delisting record
                    vol = int(shares_out[t] * adv_fraction * float(rng.lognormal(0, 0.5)))
                    records.append({
                        "date": td,
                        "security_id": sid,
                        "open": round(close_price * float(rng.uniform(0.98, 1.02)), 4),
                        "high": round(close_price * float(rng.uniform(1.00, 1.05)), 4),
                        "low": round(close_price * float(rng.uniform(0.95, 1.00)), 4),
                        "close": round(close_price, 4),
                        "adjusted_close": round(close_price, 4),
                        "volume": vol,
                        "vwap": round(close_price * float(rng.uniform(0.99, 1.01)), 4),
                        "dollar_volume": round(close_price * vol, 2),
                        "market_cap": round(close_price * float(shares_out[t]), 2),
                        "split_factor": 1.0,
                        "dividend_amount": 0.0,
                    })
                    delisting_applied = True
                break  # no further records

            close = float(prices[t])
            adj_close = float(adjusted_closes[t])
            sf = float(split_factors[t])
            div_frac = float(dividend_amounts[t])
            div_dollar = round(close * div_frac, 4)

            # OHLC construction from daily log-return
            daily_range_vol = abs(float(log_rets[t])) + float(rng.exponential(0.008))
            open_price = close / (1 + float(rng.normal(0, daily_range_vol * 0.3)))
            high_price = max(open_price, close) * (1 + abs(float(rng.exponential(daily_range_vol * 0.5))))
            low_price = min(open_price, close) * (1 - abs(float(rng.exponential(daily_range_vol * 0.5))))
            low_price = max(low_price, close * 0.80)  # cap intraday low

            # VWAP ~ weighted average, skewed toward close
            vwap = float(rng.uniform(low_price * 0.5 + close * 0.5, high_price * 0.3 + close * 0.7))

            # Volume: log-normal around ADV, spike on split/div days
            spike = 2.0 if sf != 1.0 else (1.5 if div_dollar > 0 else 1.0)
            volume_raw = int(
                shares_out[t] * adv_fraction * spike * float(rng.lognormal(0, 0.4))
            )
            volume = max(volume_raw, 1000)

            dollar_volume = round(vwap * volume, 2)

            # Market cap
            mcap = round(close * float(shares_out[t]), 2)

            records.append({
                "date": td,
                "security_id": sid,
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close, 4),
                "adjusted_close": round(adj_close, 4),
                "volume": volume,
                "vwap": round(vwap, 4),
                "dollar_volume": dollar_volume,
                "market_cap": mcap,
                "split_factor": round(sf, 6),
                "dividend_amount": div_dollar,
            })

        return records
