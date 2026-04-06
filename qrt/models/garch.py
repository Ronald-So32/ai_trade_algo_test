"""
GARCH Volatility Forecasting
==============================
Implements GARCH(1,1) and GJR-GARCH(1,1) for volatility forecasting,
used to improve position sizing, leverage decisions, and risk management.

Academic basis:
  - Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"
  - Engle (2002): "Dynamic Conditional Correlation"
  - Glosten, Jagannathan & Runkle (1993): GJR-GARCH for asymmetric volatility
  - Hansen & Lunde (2005): Comparison of volatility models

Key insight: GARCH volatility forecasts are more responsive than rolling-window
estimators, particularly around regime transitions.  This allows the portfolio
to adjust leverage and position sizes faster during vol spikes.

Usage:
    forecaster = GARCHForecaster()
    vol_forecast = forecaster.forecast_volatility(returns_series, horizon=1)
    vol_df = forecaster.rolling_forecast(returns_df, train_window=504)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import arch; fallback to EWMA if unavailable
try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    logger.warning("arch package not installed. GARCH will fall back to EWMA.")


GARCHType = Literal["garch", "gjr-garch", "egarch"]


class GARCHForecaster:
    """
    GARCH-family volatility forecaster.

    Parameters
    ----------
    model_type : str
        "garch" (Bollerslev 1986), "gjr-garch" (GJR 1993), or "egarch".
    p : int
        GARCH lag order (default 1).
    q : int
        ARCH lag order (default 1).
    dist : str
        Error distribution: "normal", "t", "skewt" (default "t" for fat tails).
    rescale : bool
        Whether to rescale returns for numerical stability (default True).
    """

    def __init__(
        self,
        model_type: GARCHType = "gjr-garch",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        rescale: bool = True,
    ) -> None:
        self.model_type = model_type
        self.p = p
        self.q = q
        self.dist = dist
        self.rescale = rescale

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1,
    ) -> float:
        """
        Fit GARCH and forecast volatility for the next `horizon` periods.

        Parameters
        ----------
        returns : pd.Series
            Historical return series.
        horizon : int
            Forecast horizon in periods (default 1 = next day).

        Returns
        -------
        float
            Annualized volatility forecast.
        """
        if not _ARCH_AVAILABLE:
            return self._ewma_forecast(returns)

        clean = returns.dropna()
        if len(clean) < 50:
            return float(clean.std() * np.sqrt(252))

        try:
            # Scale to percentage returns for numerical stability
            y = clean * 100 if self.rescale else clean

            vol_model = "GARCH" if self.model_type == "garch" else "GARCH"
            o = 1 if self.model_type == "gjr-garch" else 0

            am = arch_model(
                y,
                vol=vol_model,
                p=self.p,
                o=o,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)

            # Forecast
            fc = res.forecast(horizon=horizon)
            # variance is in (percentage)^2 if rescaled
            var_forecast = float(fc.variance.iloc[-1].iloc[-1])

            if self.rescale:
                var_forecast /= 1e4  # Convert back from percentage

            # Annualize
            ann_vol = np.sqrt(var_forecast * 252)
            return max(ann_vol, 1e-6)

        except Exception as e:
            logger.debug("GARCH fit failed (%s), using EWMA fallback", e)
            return self._ewma_forecast(returns)

    def rolling_forecast(
        self,
        returns: pd.DataFrame,
        train_window: int = 504,
        refit_freq: int = 21,
    ) -> pd.DataFrame:
        """
        Walk-forward GARCH volatility forecast for each asset.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (dates x assets).
        train_window : int
            Training window in days (default 504 = 2 years).
        refit_freq : int
            Days between refitting the model (default 21 = monthly).

        Returns
        -------
        pd.DataFrame
            Annualized GARCH volatility forecasts (dates x assets).
        """
        dates = returns.index
        n_dates = len(dates)
        cols = returns.columns
        result = pd.DataFrame(np.nan, index=dates, columns=cols)

        if not _ARCH_AVAILABLE:
            # EWMA fallback for all assets
            return self._ewma_rolling(returns, span=63)

        for col in cols:
            series = returns[col].dropna()
            last_fit_vol = None
            last_fit_t = -refit_freq

            for t in range(train_window, n_dates):
                date = dates[t]
                if date not in series.index:
                    continue

                if t - last_fit_t >= refit_freq:
                    # Refit GARCH
                    train_data = series.loc[series.index <= date].iloc[-train_window:]
                    if len(train_data) >= 50:
                        last_fit_vol = self.forecast_volatility(train_data, horizon=1)
                        last_fit_t = t

                if last_fit_vol is not None:
                    result.loc[date, col] = last_fit_vol

        # Forward-fill gaps
        result = result.ffill()

        # Fill any remaining NaN with rolling std
        rolling_vol = returns.rolling(63, min_periods=21).std() * np.sqrt(252)
        result = result.fillna(rolling_vol).fillna(returns.std() * np.sqrt(252))

        return result

    def forecast_portfolio_vol(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> float:
        """
        Forecast portfolio-level volatility using GARCH on portfolio returns.
        """
        port_ret = (returns * weights).sum(axis=1)
        return self.forecast_volatility(port_ret, horizon=1)

    @staticmethod
    def _ewma_forecast(returns: pd.Series, span: int = 63) -> float:
        """EWMA volatility as fallback."""
        ewma_var = returns.ewm(span=span).var().iloc[-1]
        return float(np.sqrt(ewma_var * 252))

    @staticmethod
    def _ewma_rolling(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame:
        """EWMA rolling volatility for all assets."""
        return returns.ewm(span=span).std() * np.sqrt(252)
