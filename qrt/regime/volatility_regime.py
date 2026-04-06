"""
Volatility Regime Classifier
=============================
Classifies market conditions into four volatility regimes using rolling
realised volatility and its historical percentile distribution.

Regimes
-------
0 – low_vol    : realised vol < 25th percentile of in-sample distribution
1 – medium_vol : 25th <= vol < 60th percentile
2 – high_vol   : 60th <= vol < 90th percentile
3 – crisis     : vol >= 90th percentile

The classifier stores the empirical percentile breakpoints fitted on the
training sample and applies them out-of-sample without look-ahead bias.
Regime probabilities are derived from a smooth Gaussian kernel density
estimate over the four regime intervals.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_LABELS: dict[int, str] = {
    0: "low_vol",
    1: "medium_vol",
    2: "high_vol",
    3: "crisis",
}

# Percentile breakpoints that separate the four regimes
_DEFAULT_PERCENTILE_BREAKS: list[float] = [25.0, 60.0, 90.0]


class VolatilityRegimeClassifier:
    """
    Percentile-based rolling realised volatility regime classifier.

    The classifier uses a configurable rolling window to compute realised
    annualised volatility, then maps each observation to one of four named
    regimes based on the percentile thresholds estimated during ``fit``.

    Soft regime probabilities are estimated for each date using a Gaussian
    kernel centred on the empirical CDF of realised volatility, evaluated
    within each regime interval.

    Parameters
    ----------
    window : int, default 21
        Rolling window (in trading days) used to compute realised volatility.
    annualisation_factor : float, default 252.0
        Factor used to annualise daily realised volatility.
    percentile_breaks : list[float], default [25, 60, 90]
        Percentile thresholds (in [0, 100]) that delimit the four regimes.
        Must be a strictly increasing list of exactly three values.
    min_periods : int, default 5
        Minimum number of non-NaN observations required to compute a rolling
        volatility estimate; otherwise ``NaN`` is returned for that window.

    Attributes
    ----------
    thresholds_ : np.ndarray of shape (3,)
        Volatility level thresholds fitted from the training data.
    is_fitted_ : bool
        Whether the classifier has been fitted.
    realized_vol_train_ : pd.Series
        The annualised rolling realised volatility computed during ``fit``.
    """

    def __init__(
        self,
        window: int = 21,
        annualisation_factor: float = 252.0,
        percentile_breaks: Optional[list[float]] = None,
        min_periods: int = 5,
    ) -> None:
        if percentile_breaks is None:
            percentile_breaks = _DEFAULT_PERCENTILE_BREAKS

        if len(percentile_breaks) != 3:
            raise ValueError(
                "percentile_breaks must contain exactly three values "
                f"(got {len(percentile_breaks)})."
            )
        if not all(0 < p < 100 for p in percentile_breaks):
            raise ValueError("All percentile_breaks must be in the open interval (0, 100).")
        if not all(percentile_breaks[i] < percentile_breaks[i + 1] for i in range(2)):
            raise ValueError("percentile_breaks must be strictly increasing.")

        self.window = window
        self.annualisation_factor = annualisation_factor
        self.percentile_breaks = percentile_breaks
        self.min_periods = min_periods

        # Fitted attributes
        self.thresholds_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.realized_vol_train_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "VolatilityRegimeClassifier":
        """
        Estimate regime thresholds from a training series of returns.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns or simple returns indexed by date.  Must not be
            all-NaN.

        Returns
        -------
        self
        """
        returns = self._validate_returns(returns, name="returns")

        rv = self._compute_realized_vol(returns)
        valid_rv = rv.dropna()

        if valid_rv.empty:
            raise ValueError(
                "No valid realised volatility values could be computed from "
                "the provided returns.  Check that the series is long enough "
                f"relative to window={self.window}."
            )

        self.thresholds_ = np.percentile(valid_rv.values, self.percentile_breaks)
        self.realized_vol_train_ = rv
        self.is_fitted_ = True

        logger.info(
            "VolatilityRegimeClassifier fitted on %d observations. "
            "Thresholds (ann. vol): low/med=%.4f, med/high=%.4f, high/crisis=%.4f",
            len(valid_rv),
            self.thresholds_[0],
            self.thresholds_[1],
            self.thresholds_[2],
        )
        return self

    def predict(
        self,
        returns: pd.Series,
        return_probabilities: bool = True,
    ) -> pd.DataFrame:
        """
        Assign regime labels (and optionally probabilities) to each date.

        Parameters
        ----------
        returns : pd.Series
            Daily returns to classify.  Can be in-sample or out-of-sample.
        return_probabilities : bool, default True
            When ``True``, the returned DataFrame includes one probability
            column per regime in addition to the ``regime`` and
            ``regime_label`` columns.

        Returns
        -------
        pd.DataFrame
            Index matches ``returns``.  Columns:

            * ``regime``        – integer regime code (0–3).
            * ``regime_label``  – human-readable label string.
            * ``realized_vol``  – annualised rolling realised volatility.
            * ``prob_low_vol``         (if ``return_probabilities=True``)
            * ``prob_medium_vol``      (if ``return_probabilities=True``)
            * ``prob_high_vol``        (if ``return_probabilities=True``)
            * ``prob_crisis``          (if ``return_probabilities=True``)
        """
        self._check_fitted()
        returns = self._validate_returns(returns, name="returns")

        rv = self._compute_realized_vol(returns)
        regimes = rv.apply(self._assign_regime)

        result = pd.DataFrame(
            {
                "regime": regimes.astype("Int64"),
                "regime_label": regimes.map(
                    lambda x: REGIME_LABELS[int(x)] if pd.notna(x) else pd.NA
                ),
                "realized_vol": rv,
            },
            index=returns.index,
        )

        if return_probabilities:
            probs = self._compute_probabilities(rv)
            result = pd.concat([result, probs], axis=1)

        return result

    def predict_latest(self, returns: pd.Series) -> dict:
        """
        Convenience method: classify the most recent date in ``returns``.

        Returns
        -------
        dict with keys: regime, regime_label, realized_vol,
                         prob_low_vol, prob_medium_vol, prob_high_vol, prob_crisis
        """
        result_df = self.predict(returns, return_probabilities=True)
        latest = result_df.iloc[-1]
        return latest.to_dict()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def regime_names(self) -> dict[int, str]:
        """Mapping from integer regime code to human-readable label."""
        return dict(REGIME_LABELS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_realized_vol(self, returns: pd.Series) -> pd.Series:
        """
        Compute rolling annualised realised volatility.

        Uses the sample standard deviation over the rolling window and
        scales by ``sqrt(annualisation_factor)``.
        """
        rv = (
            returns.rolling(window=self.window, min_periods=self.min_periods)
            .std()
            .mul(np.sqrt(self.annualisation_factor))
        )
        return rv.rename("realized_vol")

    def _assign_regime(self, vol_value: float) -> Optional[int]:
        """Map a single realised-volatility scalar to an integer regime code."""
        if pd.isna(vol_value):
            return pd.NA
        t = self.thresholds_
        if vol_value < t[0]:
            return 0
        elif vol_value < t[1]:
            return 1
        elif vol_value < t[2]:
            return 2
        else:
            return 3

    def _compute_probabilities(self, rv: pd.Series) -> pd.DataFrame:
        """
        Derive soft regime probabilities using a Gaussian CDF approximation.

        For each observation the probability of belonging to regime *k* is
        approximated as the area under a Gaussian (centred at the observed
        vol, with bandwidth = IQR/4 of training vol) that falls within the
        regime's vol interval.  Probabilities are normalised to sum to one.

        Parameters
        ----------
        rv : pd.Series
            Annualised realised volatility for each date.

        Returns
        -------
        pd.DataFrame
            Columns: prob_low_vol, prob_medium_vol, prob_high_vol, prob_crisis.
        """
        assert self.thresholds_ is not None  # guaranteed by _check_fitted
        assert self.realized_vol_train_ is not None

        t = self.thresholds_
        boundaries = np.array([-np.inf, t[0], t[1], t[2], np.inf])
        label_cols = [f"prob_{v}" for v in REGIME_LABELS.values()]

        # Bandwidth: quarter of IQR of training vol (robust spread estimator)
        train_valid = self.realized_vol_train_.dropna().values
        q75, q25 = np.percentile(train_valid, [75, 25])
        bw = max((q75 - q25) / 4.0, 1e-6)

        def _row_probs(v: float) -> np.ndarray:
            if np.isnan(v):
                return np.full(4, np.nan)
            raw = np.array(
                [
                    norm.cdf(boundaries[k + 1], loc=v, scale=bw)
                    - norm.cdf(boundaries[k], loc=v, scale=bw)
                    for k in range(4)
                ]
            )
            total = raw.sum()
            return raw / total if total > 0 else np.full(4, 0.25)

        # Handle edge case if rv is empty
        if rv.empty:
            return pd.DataFrame(columns=label_cols, index=rv.index)

        probs_array = np.array([_row_probs(v) for v in rv.values])
        return pd.DataFrame(probs_array, index=rv.index, columns=label_cols)

    @staticmethod
    def _validate_returns(returns: pd.Series, name: str = "returns") -> pd.Series:
        if not isinstance(returns, pd.Series):
            raise TypeError(f"{name} must be a pd.Series, got {type(returns).__name__}.")
        if returns.empty:
            raise ValueError(f"{name} is empty.")
        if returns.isna().all():
            raise ValueError(f"{name} contains only NaN values.")
        return returns.astype(float)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This VolatilityRegimeClassifier instance is not fitted yet. "
                "Call fit() before predict()."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"VolatilityRegimeClassifier("
            f"window={self.window}, "
            f"annualisation_factor={self.annualisation_factor}, "
            f"percentile_breaks={self.percentile_breaks}, "
            f"status={status})"
        )
