"""
Returns Calculator — computes all return series from raw market data.

Return types
------------
ret_raw          : (close[t] - close[t-1]) / close[t-1]  — unadjusted
ret_adj          : based on adjusted_close (accounts for splits & dividends)
ret_ex_div       : price-only return net of dividend effect
ret_incl_delist  : ret_raw but the final (delisting) return is preserved;
                   NaN after delisting rather than zero
log_ret          : log(adjusted_close[t] / adjusted_close[t-1])

All returns are point-in-time correct:
  - Only data available up to date t is used.
  - No look-ahead in split / dividend adjustments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger(__name__)


class ReturnsCalculator:
    """
    Compute return series from the market data DataFrame.

    Parameters
    ----------
    market_data : pd.DataFrame
        Output of MarketDataGenerator.generate().
    security_master : pd.DataFrame
        Output of SecurityMaster.generate(); used for delist_date look-up.
    config : Config, optional
    """

    RETURN_COLUMNS = [
        "date", "security_id",
        "ret_raw", "ret_adj", "ret_ex_div", "ret_incl_delist", "log_ret",
        "adjusted_close", "close",
    ]

    def __init__(
        self,
        market_data: pd.DataFrame,
        security_master: pd.DataFrame,
        config: Optional[Config] = None,
    ) -> None:
        self._md = market_data.copy()
        self._sm = security_master
        self._config = config or Config()
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self) -> pd.DataFrame:
        """
        Compute all return series and return a tidy DataFrame.

        One row per (date, security_id).  The first available date for each
        security has NaN returns (no prior close).
        """
        logger.info("Calculating returns …")

        md = self._md.sort_values(["security_id", "date"]).copy()

        # ------------------------------------------------------------------
        # Build delist date lookup
        # ------------------------------------------------------------------
        delist_lookup: dict[int, pd.Timestamp] = {}
        for _, row in self._sm.iterrows():
            if pd.notna(row["delist_date"]):
                delist_lookup[int(row["security_id"])] = pd.Timestamp(row["delist_date"])

        # ------------------------------------------------------------------
        # Compute returns per security
        # ------------------------------------------------------------------
        result_frames: list[pd.DataFrame] = []

        for sid, grp in md.groupby("security_id", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            sid = int(sid)

            close = grp["close"].values.astype(float)
            adj_close = grp["adjusted_close"].values.astype(float)
            split_factor = grp["split_factor"].values.astype(float)
            div_amount = grp["dividend_amount"].values.astype(float)
            dates = grp["date"].values

            n = len(grp)
            ret_raw = np.empty(n)
            ret_adj = np.empty(n)
            ret_ex_div = np.empty(n)
            ret_incl_delist = np.empty(n)
            log_ret = np.empty(n)

            ret_raw[0] = np.nan
            ret_adj[0] = np.nan
            ret_ex_div[0] = np.nan
            ret_incl_delist[0] = np.nan
            log_ret[0] = np.nan

            delist_date = delist_lookup.get(sid)

            for t in range(1, n):
                prev_close = close[t - 1]
                curr_close = close[t]
                prev_adj = adj_close[t - 1]
                curr_adj = adj_close[t]

                # --------------------
                # Raw return
                # --------------------
                if prev_close > 0:
                    ret_raw[t] = (curr_close - prev_close) / prev_close
                else:
                    ret_raw[t] = np.nan

                # --------------------
                # Adjusted return (split + dividend corrected)
                # --------------------
                if prev_adj > 0:
                    ret_adj[t] = (curr_adj - prev_adj) / prev_adj
                else:
                    ret_adj[t] = np.nan

                # --------------------
                # Ex-dividend return: remove the dividend contribution
                # Div contribution to return: div_amount[t] / prev_close
                # --------------------
                if prev_close > 0:
                    div_contribution = div_amount[t] / prev_close
                    ret_ex_div[t] = ret_raw[t] - div_contribution
                else:
                    ret_ex_div[t] = np.nan

                # --------------------
                # Including delisting: same as raw but after delist_date → NaN
                # (the last row IS the delisting return, kept intact)
                # --------------------
                curr_date = pd.Timestamp(dates[t])
                if delist_date and curr_date > delist_date:
                    ret_incl_delist[t] = np.nan
                else:
                    ret_incl_delist[t] = ret_raw[t]

                # --------------------
                # Log return on adjusted close
                # --------------------
                if prev_adj > 0 and curr_adj > 0:
                    log_ret[t] = np.log(curr_adj / prev_adj)
                else:
                    log_ret[t] = np.nan

            result_frames.append(pd.DataFrame({
                "date": dates,
                "security_id": sid,
                "ret_raw": ret_raw,
                "ret_adj": ret_adj,
                "ret_ex_div": ret_ex_div,
                "ret_incl_delist": ret_incl_delist,
                "log_ret": log_ret,
                "adjusted_close": adj_close,
                "close": close,
            }))

        df = pd.concat(result_frames, ignore_index=True)
        df = df.sort_values(["date", "security_id"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        self._df = df
        logger.info(
            "Returns computed: %d rows, %d securities",
            len(df),
            df["security_id"].nunique(),
        )
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.calculate()
        return self._df  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def pivot(
        self,
        return_type: str = "ret_adj",
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return a wide DataFrame: rows = dates, columns = security_ids.

        Parameters
        ----------
        return_type : one of {'ret_raw','ret_adj','ret_ex_div',
                               'ret_incl_delist','log_ret'}
        fill_value  : value to fill NaN (e.g. 0.0); default leaves NaN.
        """
        valid = {"ret_raw", "ret_adj", "ret_ex_div", "ret_incl_delist", "log_ret"}
        if return_type not in valid:
            raise ValueError(f"return_type must be one of {valid}")

        wide = self.df.pivot(index="date", columns="security_id", values=return_type)
        if fill_value is not None:
            wide = wide.fillna(fill_value)
        return wide

    def as_of(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """All return data up to and including ``as_of_date``."""
        return self.df[self.df["date"] <= as_of_date].copy()

    def trailing_returns(
        self,
        as_of_date: pd.Timestamp,
        lookback_days: int,
        return_type: str = "ret_adj",
    ) -> pd.Series:
        """
        Compound return over the trailing ``lookback_days`` trading days,
        ending on ``as_of_date``.  Returns a Series indexed by security_id.
        """
        window = self.as_of(as_of_date).sort_values("date")
        window = window[window["date"] > window["date"].max() - pd.Timedelta(days=lookback_days * 1.5)]
        result: dict[int, float] = {}
        for sid, grp in window.groupby("security_id"):
            grp = grp.sort_values("date").tail(lookback_days)
            rets = grp[return_type].dropna().values
            if len(rets) == 0:
                result[int(sid)] = np.nan
            else:
                result[int(sid)] = float(np.prod(1.0 + rets) - 1.0)
        return pd.Series(result, name=f"trailing_{lookback_days}d_{return_type}")

    def realized_volatility(
        self,
        as_of_date: pd.Timestamp,
        lookback_days: int = 63,
        annualize: bool = True,
        return_type: str = "log_ret",
    ) -> pd.Series:
        """
        Realized volatility (std of log-returns) over ``lookback_days``.
        """
        window = self.as_of(as_of_date).sort_values("date")
        cutoff = window["date"].max() - pd.Timedelta(days=lookback_days * 1.5)
        window = window[window["date"] > cutoff]
        result: dict[int, float] = {}
        factor = np.sqrt(252) if annualize else 1.0
        for sid, grp in window.groupby("security_id"):
            rets = grp.sort_values("date").tail(lookback_days)[return_type].dropna().values
            if len(rets) < 10:
                result[int(sid)] = np.nan
            else:
                result[int(sid)] = float(np.std(rets, ddof=1) * factor)
        return pd.Series(result, name=f"realized_vol_{lookback_days}d")
