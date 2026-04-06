"""
Bayesian Kelly Position Sizer
==============================
Combines a shrinkage estimator for expected returns (James–Stein style
Bayesian shrinkage toward a grand mean prior) with the Kelly criterion to
produce fractional position weights subject to per-asset, per-strategy, and
total leverage constraints.

Mathematical overview
---------------------
1. **Bayesian shrinkage of expected returns**

   The sample mean μ̂ is shrunk toward a prior μ₀ (defaults to zero or a
   user-supplied grand mean):

       μ_bayes = (1 - λ) · μ̂  +  λ · μ₀

   The shrinkage intensity λ ∈ [0, 1] is estimated via the James–Stein
   formula applied to the cross-sectional vector of excess returns:

       λ = min(1,  (N - 2) · σ²_pool  /  (T · ‖μ̂ - μ₀‖²))

   where σ²_pool is the pooled variance across assets and T is the number
   of observations.

2. **Kelly weights**

   For a diagonal approximation (single-asset view):

       f_i  =  μ_bayes_i  /  σ²_i

   Full-covariance version (used when ``use_full_covariance=True``):

       f  =  Σ⁻¹ · μ_bayes

3. **Fractional Kelly**

       w_i  =  fraction · f_i          (default fraction = 0.25)

4. **Constraints** (applied sequentially inside ``size_positions``)

   * ``max_asset_exposure``   – |w_i| ≤ max_asset_exposure  per asset
   * ``max_leverage``         – Σ|w_i| ≤ max_leverage
   * ``max_strategy_exposure`` – Σmax(w_i, 0) ≤ max_strategy_exposure  and
                                  Σmin(w_i, 0) ≥ -max_strategy_exposure
                                  (applied to gross long / gross short separately)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FRACTION: float = 0.25
_DEFAULT_MAX_ASSET_EXPOSURE: float = 0.05      # 5 %
_DEFAULT_MAX_LEVERAGE: float = 2.0
_DEFAULT_MAX_STRATEGY_EXPOSURE: float = 0.30   # 30 %
_EPSILON: float = 1e-10


class BayesianKellySizer:
    """
    Bayesian shrinkage + fractional Kelly position sizer.

    Parameters
    ----------
    fraction : float, default 0.25
        Fractional Kelly multiplier.  A value of 1.0 gives full Kelly;
        0.25 gives quarter Kelly (recommended for live trading).
    max_asset_exposure : float, default 0.05
        Maximum absolute weight per individual asset (as a fraction of NAV).
    max_leverage : float, default 2.0
        Maximum sum of absolute weights (gross leverage).
    max_strategy_exposure : float, default 0.30
        Maximum gross long *or* gross short exposure as a fraction of NAV.
    use_full_covariance : bool, default False
        When ``True``, use the full covariance matrix Σ to compute Kelly
        weights via f = Σ⁻¹μ.  When ``False``, use only the diagonal
        variances (faster and more robust for large universes).
    shrinkage_target : float or None, default None
        Prior mean for all assets.  ``None`` defaults to the grand mean of
        the sample means (cross-sectional average).
    min_history : int, default 21
        Minimum number of return observations required per asset.  Assets
        with fewer observations are excluded.

    Attributes
    ----------
    last_expected_returns_ : pd.Series or None
        Expected returns from the most recent call to
        ``estimate_expected_returns``.
    last_shrinkage_intensity_ : float or None
        Shrinkage intensity λ from the most recent call.
    """

    def __init__(
        self,
        fraction: float = _DEFAULT_FRACTION,
        max_asset_exposure: float = _DEFAULT_MAX_ASSET_EXPOSURE,
        max_leverage: float = _DEFAULT_MAX_LEVERAGE,
        max_strategy_exposure: float = _DEFAULT_MAX_STRATEGY_EXPOSURE,
        use_full_covariance: bool = False,
        shrinkage_target: Optional[float] = None,
        min_history: int = 21,
    ) -> None:
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1].  Got {fraction}.")
        if max_asset_exposure <= 0:
            raise ValueError("max_asset_exposure must be positive.")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        if max_strategy_exposure <= 0:
            raise ValueError("max_strategy_exposure must be positive.")

        self.fraction = fraction
        self.max_asset_exposure = max_asset_exposure
        self.max_leverage = max_leverage
        self.max_strategy_exposure = max_strategy_exposure
        self.use_full_covariance = use_full_covariance
        self.shrinkage_target = shrinkage_target
        self.min_history = min_history

        # State
        self.last_expected_returns_: Optional[pd.Series] = None
        self.last_shrinkage_intensity_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_expected_returns(
        self,
        returns: pd.DataFrame,
        prior_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Bayesian (James–Stein) shrinkage estimate of per-asset expected returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily returns (rows = dates, columns = assets).
        prior_returns : pd.Series, optional
            Per-asset prior mean returns.  When ``None``, the prior is set to
            either ``self.shrinkage_target`` (if set) or the cross-sectional
            grand mean of sample means.

        Returns
        -------
        pd.Series
            Bayesian posterior expected returns for each asset.
        """
        returns = self._validate_returns_df(returns)

        # Drop assets with insufficient history
        valid_counts = returns.notna().sum()
        eligible = valid_counts[valid_counts >= self.min_history].index
        if eligible.empty:
            logger.warning(
                "No assets have at least %d return observations.  "
                "Returning zero expected returns.",
                self.min_history,
            )
            return pd.Series(0.0, index=returns.columns)

        ret = returns[eligible].dropna(how="any")
        if ret.empty:
            ret = returns[eligible]  # use available data even if ragged

        N = len(eligible)              # number of assets
        T = ret.shape[0]               # effective number of observations

        mu_hat = ret.mean(axis=0)      # sample means

        # ---- Prior ----
        if prior_returns is not None:
            mu0 = prior_returns.reindex(eligible).fillna(0.0)
        elif self.shrinkage_target is not None:
            mu0 = pd.Series(self.shrinkage_target, index=eligible)
        else:
            mu0 = pd.Series(mu_hat.mean(), index=eligible)  # grand mean

        # ---- James–Stein shrinkage intensity ----
        excess = mu_hat - mu0
        pooled_var = (
            ret.var(axis=0, ddof=1).mean()
        )  # average per-asset variance
        sq_norm = float((excess**2).sum())

        if N > 2 and sq_norm > _EPSILON:
            lam = min(1.0, ((N - 2) * pooled_var) / (T * sq_norm))
        elif N <= 2:
            lam = 0.0  # shrinkage not defined for N <= 2
            logger.debug("N=%d <= 2; shrinkage intensity set to 0.", N)
        else:
            lam = 1.0  # degenerate case: all means equal prior; shrink fully

        mu_bayes = (1.0 - lam) * mu_hat + lam * mu0

        # Re-index to full universe (excluded assets get zero)
        result = pd.Series(0.0, index=returns.columns)
        result[eligible] = mu_bayes

        self.last_expected_returns_ = result
        self.last_shrinkage_intensity_ = lam

        logger.debug(
            "estimate_expected_returns: N=%d, T=%d, λ=%.4f, "
            "μ_range=[%.6f, %.6f]",
            N, T, lam, result.min(), result.max(),
        )
        return result

    def kelly_weights(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute raw (unconstrained) Kelly weights.

        Parameters
        ----------
        expected_returns : pd.Series
            Per-asset expected returns (e.g., from ``estimate_expected_returns``).
        covariance : pd.DataFrame
            Covariance matrix of asset returns (annualised or daily – must be
            consistent with the scale of ``expected_returns``).

        Returns
        -------
        pd.Series
            Raw Kelly weights *before* fractional scaling and constraints.
            These may exceed 1 in absolute value.
        """
        if not isinstance(expected_returns, pd.Series):
            raise TypeError("expected_returns must be a pd.Series.")
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance must be a pd.DataFrame.")

        assets = expected_returns.index
        mu = expected_returns.values.astype(float)

        if self.use_full_covariance:
            cov_aligned = covariance.reindex(index=assets, columns=assets).fillna(0.0)
            cov_mat = cov_aligned.values.astype(float)
            # Regularise: add small ridge to diagonal to ensure PD
            cov_mat += np.eye(len(assets)) * _EPSILON
            try:
                cov_inv = pinvh(cov_mat)
            except np.linalg.LinAlgError as exc:
                logger.warning(
                    "Covariance matrix pseudo-inverse failed (%s).  "
                    "Falling back to diagonal.",
                    exc,
                )
                diag_var = np.diag(cov_mat).clip(min=_EPSILON)
                kelly = mu / diag_var
            else:
                kelly = cov_inv @ mu
        else:
            diag_var = np.diag(
                covariance.reindex(index=assets, columns=assets).fillna(0.0).values
            ).clip(min=_EPSILON)
            kelly = mu / diag_var

        return pd.Series(kelly, index=assets, name="kelly_weight")

    def size_positions(
        self,
        raw_weights: pd.Series,
        returns_history: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Apply fractional Kelly scaling and portfolio-level constraints.

        Constraint application order
        -----------------------------
        1. Fractional Kelly scaling     : w ← fraction · w
        2. Per-asset cap                : |w_i| ≤ max_asset_exposure
        3. Gross leverage rescaling     : if Σ|w_i| > max_leverage, scale down
        4. Strategy exposure caps       : gross long ≤ max_strategy_exposure,
                                          gross short ≥ -max_strategy_exposure

        Parameters
        ----------
        raw_weights : pd.Series
            Unconstrained Kelly weights (output of ``kelly_weights``).
        returns_history : pd.DataFrame, optional
            Not used in the current implementation.  Reserved for future
            regime-conditional scaling.

        Returns
        -------
        pd.Series
            Constrained, fractional-Kelly-scaled weights.
        """
        if not isinstance(raw_weights, pd.Series):
            raise TypeError("raw_weights must be a pd.Series.")

        w = raw_weights.copy().astype(float).fillna(0.0)

        # 1. Fractional Kelly
        w = w * self.fraction

        # 2. Per-asset exposure cap
        w = w.clip(lower=-self.max_asset_exposure, upper=self.max_asset_exposure)

        # 3. Gross leverage rescaling
        gross_leverage = w.abs().sum()
        if gross_leverage > self.max_leverage:
            w = w * (self.max_leverage / gross_leverage)
            logger.debug(
                "Leverage rescaled: %.4f → %.4f", gross_leverage, self.max_leverage
            )

        # 4. Strategy exposure caps (long and short sides independently)
        w = self._cap_strategy_exposure(w)

        logger.debug(
            "size_positions: gross_leverage=%.4f, net=%.4f, "
            "max_pos=%.4f, min_pos=%.4f",
            w.abs().sum(),
            w.sum(),
            w.max(),
            w.min(),
        )
        return w.rename("sized_weight")

    # ------------------------------------------------------------------
    # Convenience end-to-end method
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        returns: pd.DataFrame,
        prior_returns: Optional[pd.Series] = None,
        covariance: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        End-to-end: estimate returns → compute Kelly → size positions.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns used both for shrinkage and for covariance
            estimation when ``covariance`` is not supplied.
        prior_returns : pd.Series, optional
            Prior expected returns (passed to ``estimate_expected_returns``).
        covariance : pd.DataFrame, optional
            Pre-computed covariance matrix.  If ``None``, estimated from
            ``returns``.

        Returns
        -------
        pd.Series
            Constrained position weights.
        """
        mu = self.estimate_expected_returns(returns, prior_returns=prior_returns)

        if covariance is None:
            covariance = returns.cov()

        raw = self.kelly_weights(mu, covariance)
        return self.size_positions(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cap_strategy_exposure(self, weights: pd.Series) -> pd.Series:
        """
        Rescale long and short legs independently so that neither exceeds
        ``max_strategy_exposure`` in gross notional terms.
        """
        w = weights.copy()

        long_mask = w > 0
        short_mask = w < 0

        gross_long = w[long_mask].sum()
        gross_short = w[short_mask].abs().sum()  # positive number

        if gross_long > self.max_strategy_exposure:
            scale = self.max_strategy_exposure / gross_long
            w[long_mask] *= scale
            logger.debug(
                "Long-side exposure rescaled: %.4f → %.4f",
                gross_long,
                self.max_strategy_exposure,
            )

        if gross_short > self.max_strategy_exposure:
            scale = self.max_strategy_exposure / gross_short
            w[short_mask] *= scale
            logger.debug(
                "Short-side exposure rescaled: %.4f → %.4f",
                gross_short,
                self.max_strategy_exposure,
            )

        return w

    @staticmethod
    def _validate_returns_df(returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError(
                f"returns must be a pd.DataFrame, got {type(returns).__name__}."
            )
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        return returns.astype(float)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianKellySizer("
            f"fraction={self.fraction}, "
            f"max_asset_exposure={self.max_asset_exposure}, "
            f"max_leverage={self.max_leverage}, "
            f"max_strategy_exposure={self.max_strategy_exposure}, "
            f"use_full_covariance={self.use_full_covariance})"
        )
