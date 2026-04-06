"""
Meta-Model Feature Engineering
================================
Constructs a rich, look-ahead-free feature matrix from raw market
inputs for the ML meta-model.

All lags and rolling windows are applied *before* the current
observation is visible to the model, i.e. a feature computed at time
``t`` uses only data available at ``t-1`` or earlier (controlled by
``lag=1`` on the rolling windows).

Classes
-------
MetaFeatureEngineer
    Stateful transformer that fits normalisation statistics on training
    data and applies them consistently to unseen data.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default rolling windows used to compute summary statistics
_DEFAULT_WINDOWS: list[int] = [5, 21, 63]


class MetaFeatureEngineer:
    """
    Rolling, lagged and standardised feature engineer for the ML
    meta-model.

    The engineer constructs the following feature groups from the raw
    inputs passed to :meth:`transform`:

    * **Signal features** — each strategy's signal at lag-1, plus
      rolling mean and standard deviation over short / medium / long
      windows.
    * **Volatility features** — realised vol at lag-1 and its rolling
      z-score relative to its own recent history.
    * **Correlation features** — average pairwise correlation at lag-1
      and its rolling trend.
    * **Drawdown features** — current drawdown depth at lag-1, rolling
      minimum drawdown (worst recent drawdown).
    * **Regime features** — one column per regime probability at lag-1.

    After construction all features are standardised (zero mean, unit
    variance) using statistics estimated from the *training* call to
    :meth:`transform` (``fit=True``).

    Parameters
    ----------
    windows : list[int], default [5, 21, 63]
        Rolling window lengths (in trading days) used for summary
        statistics.
    lag : int, default 1
        Number of periods to shift all features forward to prevent
        look-ahead bias.  Must be >= 1.
    min_periods_ratio : float, default 0.5
        A rolling window of length ``w`` requires at least
        ``ceil(w * min_periods_ratio)`` non-NaN observations; otherwise
        the feature is NaN.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted scaler; available after the first call to
        ``transform(..., fit=True)``.
    feature_names_ : list[str] or None
        Ordered list of feature column names produced by the engineer.
    is_fitted_ : bool
        Whether the scaler has been fitted.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        lag: int = 1,
        min_periods_ratio: float = 0.5,
    ) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1 to prevent look-ahead bias; got {lag}.")
        if not 0.0 < min_periods_ratio <= 1.0:
            raise ValueError(
                f"min_periods_ratio must be in (0, 1]; got {min_periods_ratio}."
            )

        self.windows: list[int] = sorted(windows or _DEFAULT_WINDOWS)
        self.lag = lag
        self.min_periods_ratio = min_periods_ratio

        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        raw_features: pd.DataFrame,
        lookback: int = 63,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Build an engineered feature matrix from pre-assembled raw inputs.

        The *raw_features* DataFrame is expected to contain columns
        produced by :meth:`~qrt.ml_meta.meta_model.MetaModel.build_features`,
        i.e. signal columns (prefixed ``signal_``), a ``volatility``
        column, a ``correlation`` column, a ``drawdown`` column and
        optional ``regime_`` columns.

        Parameters
        ----------
        raw_features : pd.DataFrame
            Raw input frame with a DatetimeIndex, one row per period.
        lookback : int, default 63
            Longest rolling window used when computing lag-based
            z-scores.  Features are NaN for the first ``lookback``
            periods; callers should drop those rows before fitting
            downstream models.
        fit : bool, default False
            When ``True`` the StandardScaler is (re-)fitted on the
            supplied data before transforming.  Set to ``True`` for the
            training set and ``False`` for all subsequent out-of-sample
            calls.

        Returns
        -------
        pd.DataFrame
            Engineered feature matrix, same index as *raw_features*,
            columns fully standardised (when ``fit=True`` or the scaler
            is already fitted).
        """
        if raw_features.empty:
            raise ValueError("raw_features is empty.")

        frames: list[pd.DataFrame] = []

        # ---- Signal features ------------------------------------------
        signal_cols = [c for c in raw_features.columns if c.startswith("signal_")]
        if signal_cols:
            frames.append(self._signal_features(raw_features[signal_cols]))

        # ---- Volatility features --------------------------------------
        if "volatility" in raw_features.columns:
            frames.append(
                self._scalar_features(raw_features["volatility"], prefix="vol")
            )

        # ---- Correlation features -------------------------------------
        if "correlation" in raw_features.columns:
            frames.append(
                self._scalar_features(raw_features["correlation"], prefix="corr")
            )

        # ---- Drawdown features ----------------------------------------
        if "drawdown" in raw_features.columns:
            frames.append(
                self._drawdown_features(raw_features["drawdown"])
            )

        # ---- Regime features ------------------------------------------
        regime_cols = [c for c in raw_features.columns if c.startswith("regime_")]
        if regime_cols:
            frames.append(self._regime_features(raw_features[regime_cols]))

        if not frames:
            raise ValueError(
                "No recognised columns found in raw_features. "
                "Expected columns prefixed with 'signal_', 'regime_' "
                "or named 'volatility', 'correlation', 'drawdown'."
            )

        engineered = pd.concat(frames, axis=1)

        # Apply lag to shift all features forward by `self.lag` periods
        # so that at time t the model sees only data from t-lag onwards.
        engineered = engineered.shift(self.lag)

        # Standardise
        engineered = self._standardise(engineered, fit=fit)

        self.feature_names_ = list(engineered.columns)

        logger.info(
            "MetaFeatureEngineer.transform — produced %d features "
            "over %d periods (lag=%d, fit=%s).",
            engineered.shape[1],
            engineered.shape[0],
            self.lag,
            fit,
        )
        return engineered

    # ------------------------------------------------------------------
    # Internal feature builders
    # ------------------------------------------------------------------

    def _signal_features(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        For each strategy signal compute the raw value plus rolling
        mean and standard deviation at each configured window.
        """
        parts: list[pd.Series] = []
        for col in signals.columns:
            s = signals[col].astype(float)
            parts.append(s.rename(f"{col}_raw"))
            for w in self.windows:
                mp = max(1, int(np.ceil(w * self.min_periods_ratio)))
                parts.append(
                    s.rolling(w, min_periods=mp).mean().rename(f"{col}_mean_{w}d")
                )
                parts.append(
                    s.rolling(w, min_periods=mp).std().rename(f"{col}_std_{w}d")
                )
                # Signal z-score (momentum relative to recent average)
                roll_mean = s.rolling(w, min_periods=mp).mean()
                roll_std = s.rolling(w, min_periods=mp).std().replace(0, np.nan)
                z = ((s - roll_mean) / roll_std).rename(f"{col}_zscore_{w}d")
                parts.append(z)
        return pd.concat(parts, axis=1)

    def _scalar_features(self, series: pd.Series, prefix: str) -> pd.DataFrame:
        """
        Build level, rolling mean, rolling z-score and trend features
        for a scalar time series (volatility, correlation).
        """
        s = series.astype(float)
        parts: list[pd.Series] = [s.rename(f"{prefix}_raw")]
        for w in self.windows:
            mp = max(1, int(np.ceil(w * self.min_periods_ratio)))
            roll_mean = s.rolling(w, min_periods=mp).mean()
            roll_std = s.rolling(w, min_periods=mp).std().replace(0, np.nan)
            parts.append(roll_mean.rename(f"{prefix}_mean_{w}d"))
            z = ((s - roll_mean) / roll_std).rename(f"{prefix}_zscore_{w}d")
            parts.append(z)

        # Linear trend: slope of the series over the longest window
        longest = self.windows[-1]
        mp_long = max(1, int(np.ceil(longest * self.min_periods_ratio)))
        parts.append(
            self._rolling_slope(s, window=longest, min_periods=mp_long).rename(
                f"{prefix}_trend_{longest}d"
            )
        )
        return pd.concat(parts, axis=1)

    def _drawdown_features(self, drawdown: pd.Series) -> pd.DataFrame:
        """
        Drawdown depth, rolling minimum (worst recent drawdown) and
        recovery speed (change in drawdown over short window).
        """
        s = drawdown.astype(float)
        parts: list[pd.Series] = [s.rename("drawdown_raw")]

        for w in self.windows:
            mp = max(1, int(np.ceil(w * self.min_periods_ratio)))
            parts.append(
                s.rolling(w, min_periods=mp).min().rename(f"drawdown_min_{w}d")
            )
            parts.append(
                s.rolling(w, min_periods=mp).mean().rename(f"drawdown_mean_{w}d")
            )

        # Recovery speed — positive values indicate drawdown is recovering
        short_w = self.windows[0]
        parts.append(s.diff(short_w).rename(f"drawdown_recovery_{short_w}d"))
        return pd.concat(parts, axis=1)

    def _regime_features(self, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Regime probability columns plus rolling mean of each probability
        (regime persistence measure).
        """
        parts: list[pd.Series] = []
        for col in regimes.columns:
            s = regimes[col].astype(float)
            parts.append(s.rename(f"regime_prob_{col}"))
            for w in self.windows[:2]:  # short and medium windows
                mp = max(1, int(np.ceil(w * self.min_periods_ratio)))
                parts.append(
                    s.rolling(w, min_periods=mp)
                    .mean()
                    .rename(f"regime_prob_{col}_mean_{w}d")
                )
        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    def _standardise(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Standardise all columns using :class:`sklearn.preprocessing.StandardScaler`.

        NaN positions are preserved: scaling is applied only to finite
        values; NaN cells remain NaN in the output.
        """
        if fit or self.scaler_ is None:
            self.scaler_ = StandardScaler()
            # Fit on rows that have at least one non-NaN value
            fit_data = df.dropna(how="all")
            if fit_data.empty:
                logger.warning(
                    "All rows are NaN after feature construction; "
                    "StandardScaler fitted on empty data."
                )
                self.is_fitted_ = False
                return df
            self.scaler_.fit(fit_data.fillna(fit_data.mean()))
            self.is_fitted_ = True

        if not self.is_fitted_:
            return df

        # Apply scaler while preserving NaN mask
        nan_mask = df.isna()
        filled = df.fillna(df.mean())
        scaled_values = self.scaler_.transform(filled)
        scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        scaled_df[nan_mask] = np.nan
        return scaled_df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_slope(
        series: pd.Series,
        window: int,
        min_periods: int,
    ) -> pd.Series:
        """
        Compute the OLS slope of ``series`` over a rolling window using
        a simple analytic formula to avoid per-row regression overhead.

        The slope is the covariance of the time index with the series
        values, divided by the variance of the index.  The function
        receives the raw values array for each window from pandas (its
        length equals the actual window size, which may be smaller than
        ``window`` near the start of the series).
        """
        def _slope_fn(arr: np.ndarray) -> float:
            valid_mask = ~np.isnan(arr)
            n_valid = valid_mask.sum()
            if n_valid < min_periods:
                return np.nan
            # Build integer positions for the valid observations only
            positions = np.where(valid_mask)[0].astype(float)
            y_local = arr[valid_mask]
            xm = positions.mean()
            ym = y_local.mean()
            var = np.var(positions, ddof=0)
            if var == 0:
                return np.nan
            cov = np.mean((positions - xm) * (y_local - ym))
            return float(cov / var)

        return series.rolling(window, min_periods=min_periods).apply(
            _slope_fn, raw=True
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"MetaFeatureEngineer("
            f"windows={self.windows}, "
            f"lag={self.lag}, "
            f"min_periods_ratio={self.min_periods_ratio}, "
            f"status={status})"
        )
