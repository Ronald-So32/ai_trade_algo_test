"""
signal_generator.py
-------------------
Systematically generate candidate alpha signals from combinations of price,
return, and volume features.

Signal families
~~~~~~~~~~~~~~~
- Price-based        : multi-lookback returns, price ratios, high-low range
- Momentum           : standard and accelerated momentum, momentum reversal
- Volatility         : vol breakout, vol mean-reversion, vol-of-vol
- Mean-reversion     : rolling z-scores, Ornstein-Uhlenbeck mean-reversion speed
- Cross-sectional    : percentile rank of returns, rank of vol, rank of volume changes
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOMENTUM_LOOKBACKS: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
_ZSCORE_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
_VOL_LOOKBACKS: tuple[int, ...] = (10, 21, 63)
_SHORT_LOOKBACKS: tuple[int, ...] = (5, 10, 21)
_LONG_LOOKBACKS: tuple[int, ...] = (63, 126, 252)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _rolling_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=max(2, window // 2)).std()


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = _rolling_mean(df, window)
    sigma = _rolling_std(df, window).replace(0, np.nan)
    return (df - mu) / sigma


def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each row to cross-sectional percentile ranks in [0, 1]."""
    return df.rank(axis=1, pct=True)


def _winsorize(df: pd.DataFrame, limits: float = 0.01) -> pd.DataFrame:
    """Clip values at the given quantile on each side."""
    lower = df.quantile(limits, axis=1)
    upper = df.quantile(1.0 - limits, axis=1)
    return df.clip(lower=lower, upper=upper, axis=0)


# ---------------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Systematically generate candidate alpha signals.

    Parameters
    ----------
    winsorize_signals : bool
        If True, each generated signal is winsorized at the 1st / 99th
        cross-sectional percentile before being returned.  Default True.
    zscore_normalize : bool
        If True, each signal is cross-sectionally z-scored before being
        returned.  Default True.

    Notes
    -----
    All returned DataFrames share the same (dates x securities) shape as
    the ``returns`` argument passed to :meth:`generate_candidates`.
    """

    def __init__(
        self,
        winsorize_signals: bool = True,
        zscore_normalize: bool = True,
    ) -> None:
        self.winsorize_signals = winsorize_signals
        self.zscore_normalize = zscore_normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate the full library of candidate alpha signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices, shape (dates, securities).
        returns : pd.DataFrame
            Daily log or simple returns, shape (dates, securities).
        volumes : pd.DataFrame
            Share / dollar volumes, shape (dates, securities).
        high : pd.DataFrame, optional
            Daily high prices.  Required for high-low range signals.
        low : pd.DataFrame, optional
            Daily low prices.  Required for high-low range signals.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping ``signal_name -> signal_values`` where every DataFrame
            has the same index and columns as ``returns``.
        """
        logger.info("Starting candidate signal generation ...")

        signals: dict[str, pd.DataFrame] = {}

        signals.update(self._price_based_signals(prices, returns, high, low))
        signals.update(self._momentum_signals(returns))
        signals.update(self._volatility_signals(returns))
        signals.update(self._mean_reversion_signals(returns))
        signals.update(self._cross_sectional_signals(returns, volumes))

        # Post-processing: align to returns index/columns, winsorize, normalise
        clean: dict[str, pd.DataFrame] = {}
        for name, sig in signals.items():
            try:
                sig = sig.reindex(index=returns.index, columns=returns.columns)
                if self.winsorize_signals:
                    sig = _winsorize(sig)
                if self.zscore_normalize:
                    mu = sig.mean(axis=1)
                    sd = sig.std(axis=1).replace(0, np.nan)
                    sig = sig.subtract(mu, axis=0).divide(sd, axis=0)
                clean[name] = sig
            except Exception as exc:  # noqa: BLE001
                logger.warning("Signal '%s' dropped during post-processing: %s", name, exc)

        logger.info("Generated %d candidate signals.", len(clean))
        return clean

    # ------------------------------------------------------------------
    # Price-based signals
    # ------------------------------------------------------------------

    def _price_based_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        high: Optional[pd.DataFrame],
        low: Optional[pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}

        # Multi-lookback total returns (forward-fill safe; skip-1-day lag applied later)
        for lb in (5, 10, 21, 63, 126, 252):
            signals[f"price_return_{lb}d"] = prices.pct_change(lb)

        # Price ratio: close relative to rolling max (proximity to 52-week high)
        for lb in (63, 126, 252):
            rolling_max = prices.rolling(window=lb, min_periods=lb // 2).max()
            signals[f"price_vs_rolling_max_{lb}d"] = prices / rolling_max.replace(0, np.nan) - 1.0

        # Price ratio: close / rolling min
        for lb in (63, 252):
            rolling_min = prices.rolling(window=lb, min_periods=lb // 2).min()
            signals[f"price_vs_rolling_min_{lb}d"] = prices / rolling_min.replace(0, np.nan) - 1.0

        # High-low range signals (require high / low data)
        if high is not None and low is not None:
            # Normalised intraday range
            hl_range = (high - low) / prices.replace(0, np.nan)
            signals["hl_range_pct"] = hl_range

            for lb in _SHORT_LOOKBACKS:
                signals[f"hl_range_zscore_{lb}d"] = _rolling_zscore(hl_range, lb)

            # Close relative to day's range  (0 = at low, 1 = at high)
            denom = (high - low).replace(0, np.nan)
            signals["close_position_in_range"] = (prices - low) / denom

        return signals

    # ------------------------------------------------------------------
    # Momentum signals
    # ------------------------------------------------------------------

    def _momentum_signals(self, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}

        # Standard momentum: cumulative return over lookback (skip most recent day)
        for lb in _MOMENTUM_LOOKBACKS:
            cumret = returns.rolling(window=lb, min_periods=lb // 2).sum()
            signals[f"momentum_{lb}d"] = cumret

        # Momentum acceleration: short-term momentum minus long-term momentum
        # (captures whether momentum is speeding up)
        accel_pairs = [
            (5, 21),
            (10, 63),
            (21, 126),
            (63, 252),
        ]
        for short, long in accel_pairs:
            mom_short = returns.rolling(window=short, min_periods=short // 2).sum()
            mom_long = returns.rolling(window=long, min_periods=long // 2).sum()
            signals[f"momentum_accel_{short}vs{long}d"] = mom_short - mom_long

        # Momentum reversal: negative of very short-term return (mean-reversion proxy)
        for lb in (1, 3, 5):
            short_ret = returns.rolling(window=lb, min_periods=1).sum()
            signals[f"momentum_reversal_{lb}d"] = -short_ret

        # Momentum consistency: fraction of positive days over lookback
        for lb in _MOMENTUM_LOOKBACKS:
            pos_frac = (returns > 0).rolling(window=lb, min_periods=lb // 2).mean()
            signals[f"momentum_consistency_{lb}d"] = pos_frac

        # Risk-adjusted momentum: momentum normalised by realised vol
        for lb in _MOMENTUM_LOOKBACKS:
            cumret = returns.rolling(window=lb, min_periods=lb // 2).sum()
            vol = returns.rolling(window=lb, min_periods=lb // 2).std().replace(0, np.nan)
            signals[f"risk_adj_momentum_{lb}d"] = cumret / vol

        return signals

    # ------------------------------------------------------------------
    # Volatility signals
    # ------------------------------------------------------------------

    def _volatility_signals(self, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}

        for lb in _VOL_LOOKBACKS:
            realised_vol = _rolling_std(returns, lb)

            # Volatility level (annualised)
            signals[f"vol_level_{lb}d"] = realised_vol * np.sqrt(252)

            # Volatility z-score over a longer window (breakout / mean-reversion)
            long_lb = lb * 4
            long_vol = _rolling_std(returns, long_lb)
            long_vol_of_vol = _rolling_std(
                realised_vol, long_lb
            ).replace(0, np.nan)
            signals[f"vol_zscore_{lb}d"] = (
                (realised_vol - long_vol) / long_vol_of_vol
            )

            # Vol breakout: current vol vs rolling max vol
            vol_max = realised_vol.rolling(window=long_lb, min_periods=lb).max()
            signals[f"vol_breakout_{lb}d"] = realised_vol / vol_max.replace(0, np.nan) - 1.0

            # Vol mean reversion: current vol vs rolling min vol
            vol_min = realised_vol.rolling(window=long_lb, min_periods=lb).min()
            signals[f"vol_mean_rev_{lb}d"] = -1.0 * (
                realised_vol / vol_min.replace(0, np.nan) - 1.0
            )

            # Volatility of volatility (second-order vol)
            vol_of_vol = _rolling_std(realised_vol, lb)
            signals[f"vol_of_vol_{lb}d"] = vol_of_vol

        # Vol term structure: short-vol vs long-vol ratio
        for short, long in [(10, 63), (21, 126), (10, 252)]:
            short_vol = _rolling_std(returns, short).replace(0, np.nan)
            long_vol = _rolling_std(returns, long).replace(0, np.nan)
            signals[f"vol_term_structure_{short}vs{long}d"] = short_vol / long_vol - 1.0

        return signals

    # ------------------------------------------------------------------
    # Mean-reversion signals
    # ------------------------------------------------------------------

    def _mean_reversion_signals(self, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}

        # Rolling z-score of returns at multiple horizons
        for lb in _ZSCORE_LOOKBACKS:
            signals[f"return_zscore_{lb}d"] = _rolling_zscore(returns, lb)

        # Cumulative return z-score (price mean-reversion)
        for lb in _ZSCORE_LOOKBACKS:
            cumret = returns.rolling(window=lb, min_periods=lb // 2).sum()
            signals[f"cumret_zscore_{lb}d"] = _rolling_zscore(cumret, lb)

        # Ornstein-Uhlenbeck mean-reversion speed proxy
        # Estimated via AR(1) regression slope on rolling window.
        # Speed = -ln(|ar1_coef|) — higher value means faster mean-reversion.
        for lb in (63, 126, 252):
            signals[f"ou_speed_{lb}d"] = self._ou_speed(returns, lb)

        # Deviation from rolling mean normalised by vol
        for lb in (21, 63, 126):
            mu = _rolling_mean(returns, lb)
            sigma = _rolling_std(returns, lb).replace(0, np.nan)
            deviation = (returns - mu) / sigma
            signals[f"return_deviation_{lb}d"] = -deviation  # negative -> mean reversion

        return signals

    @staticmethod
    def _ou_speed(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Estimate Ornstein-Uhlenbeck mean-reversion speed via rolling AR(1).

        For each asset and each date, fits AR(1) to the ``window``-length
        return series and returns ``-ln(|ar1_coef|)`` (speed parameter).
        Values are set to NaN when the window is not yet full or the fit
        is degenerate.
        """
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        arr = returns.values
        n_dates, n_assets = arr.shape

        for t in range(window, n_dates):
            speeds = np.full(n_assets, np.nan)
            for j in range(n_assets):
                y = arr[t - window : t, j]
                if np.isnan(y).sum() > window * 0.3:
                    continue
                y_clean = y[~np.isnan(y)]
                if len(y_clean) < 10:
                    continue
                y_lag = y_clean[:-1]
                y_cur = y_clean[1:]
                if np.std(y_lag) < 1e-12:
                    continue
                # OLS estimate of AR(1) coefficient
                ar1 = np.corrcoef(y_lag, y_cur)[0, 1]
                if abs(ar1) < 1e-6 or abs(ar1) >= 1.0:
                    continue
                speeds[j] = -np.log(abs(ar1))
            result.iloc[t] = speeds

        return result

    # ------------------------------------------------------------------
    # Cross-sectional signals
    # ------------------------------------------------------------------

    def _cross_sectional_signals(
        self,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        signals: dict[str, pd.DataFrame] = {}

        # Percentile rank of returns at multiple horizons
        for lb in (5, 21, 63, 126):
            cumret = returns.rolling(window=lb, min_periods=lb // 2).sum()
            signals[f"cs_rank_return_{lb}d"] = _cross_sectional_rank(cumret)

        # Cross-sectional rank of realised volatility
        for lb in _VOL_LOOKBACKS:
            vol = _rolling_std(returns, lb)
            signals[f"cs_rank_vol_{lb}d"] = _cross_sectional_rank(vol)

        # Low-vol rank (inverse of vol rank — long low-vol names)
        for lb in _VOL_LOOKBACKS:
            vol = _rolling_std(returns, lb)
            signals[f"cs_rank_low_vol_{lb}d"] = 1.0 - _cross_sectional_rank(vol)

        # Volume change: log change in rolling volume vs longer baseline
        log_vol = np.log(volumes.replace(0, np.nan))
        for short, long in [(5, 21), (10, 63), (21, 126)]:
            short_avg_vol = log_vol.rolling(window=short, min_periods=short // 2).mean()
            long_avg_vol = log_vol.rolling(window=long, min_periods=long // 2).mean()
            vol_change = short_avg_vol - long_avg_vol
            signals[f"cs_rank_vol_change_{short}vs{long}d"] = _cross_sectional_rank(vol_change)

        # Volume-weighted return rank
        for lb in (5, 21):
            # Dollar volume as weight proxy
            norm_vol = volumes / volumes.rolling(window=lb, min_periods=lb // 2).mean().replace(0, np.nan)
            vw_ret = (
                returns.multiply(norm_vol)
                .rolling(window=lb, min_periods=lb // 2)
                .sum()
            )
            signals[f"cs_rank_vw_return_{lb}d"] = _cross_sectional_rank(vw_ret)

        # Turnover rank: recent volume relative to historical (liquidity signal)
        for lb in (21, 63):
            recent_vol = log_vol.rolling(window=5, min_periods=3).mean()
            hist_vol = log_vol.rolling(window=lb, min_periods=lb // 2).mean()
            turnover_signal = recent_vol - hist_vol
            signals[f"cs_rank_turnover_{lb}d"] = _cross_sectional_rank(turnover_signal)

        return signals
