"""
Risk parity portfolio allocation.

Implements two risk parity approaches:
  - Naive: weights proportional to inverse volatility.
  - Covariance: equalise the marginal contribution to portfolio risk (MRC)
    for every asset by solving a constrained optimisation problem.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Supported allocation methods
AllocationMethod = Literal["naive", "covariance"]


class RiskParityAllocator:
    """
    Risk parity portfolio allocator.

    Naive risk parity assigns weights inversely proportional to each
    asset's realised volatility.  Covariance risk parity equalises
    every asset's marginal contribution to total portfolio risk, which
    requires a full covariance matrix and a numerical optimiser.

    Parameters
    ----------
    min_weight : float
        Lower bound on any single weight (default 0.0 — no short
        positions).
    max_weight : float
        Upper bound on any single weight (default 1.0).
    """

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> None:
        if not 0.0 <= min_weight < max_weight <= 1.0:
            raise ValueError(
                f"Require 0 <= min_weight ({min_weight}) < "
                f"max_weight ({max_weight}) <= 1"
            )
        self.min_weight = min_weight
        self.max_weight = max_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def naive_risk_parity(self, volatilities: np.ndarray) -> np.ndarray:
        """
        Compute naive risk parity weights (1/vol normalised to sum to 1).

        Parameters
        ----------
        volatilities : array-like, shape (n,)
            Annualised (or consistent-frequency) volatilities for each
            asset.  Must be strictly positive.

        Returns
        -------
        weights : np.ndarray, shape (n,)
            Portfolio weights that sum to 1.
        """
        vols = np.asarray(volatilities, dtype=float)
        if vols.ndim != 1:
            raise ValueError("volatilities must be a 1-D array.")
        if np.any(vols <= 0):
            raise ValueError("All volatilities must be strictly positive.")

        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        # Re-normalise after clipping
        total = weights.sum()
        if total == 0:
            raise ValueError("All weights clipped to zero — check bounds.")
        return weights / total

    def covariance_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute covariance-based risk parity weights.

        Minimises the squared deviation of each asset's fractional risk
        contribution from the equal-contribution target (1/n)::

            min  sum_i [ (w_i * (Sigma @ w)_i / (w^T Sigma w) - 1/n)^2 ]
            s.t. sum(w) = 1,  min_weight <= w_i <= max_weight

        Parameters
        ----------
        covariance_matrix : array-like, shape (n, n)
            Positive semi-definite covariance matrix of asset returns.

        Returns
        -------
        weights : np.ndarray, shape (n,)
            Portfolio weights that sum to 1.
        """
        Sigma = np.asarray(covariance_matrix, dtype=float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("covariance_matrix must be a square 2-D array.")
        n = Sigma.shape[0]
        target = 1.0 / n

        # Regularise to ensure positive definiteness
        Sigma = self._regularise(Sigma)

        def _objective(w: np.ndarray) -> float:
            portfolio_variance = float(w @ Sigma @ w)
            if portfolio_variance <= 0:
                return 1e10
            mrc = (Sigma @ w) * w / portfolio_variance  # fractional risk contrib
            return float(np.sum((mrc - target) ** 2))

        def _gradient(w: np.ndarray) -> np.ndarray:
            """Analytical gradient of the objective."""
            Sw = Sigma @ w
            var = float(w @ Sw)
            if var <= 0:
                return np.zeros(n)
            mrc = Sw * w / var
            residuals = mrc - target  # shape (n,)
            # d(mrc_i)/d(w_j) derived via quotient rule
            # mrc_i = w_i (Sw)_i / var
            # d(mrc_i)/d(w_j) = [ delta_{ij}(Sw)_i + w_i Sigma_{ij} ] / var
            #                    - w_i (Sw)_i * 2*(Sw)_j / var^2
            dobj = np.zeros(n)
            for j in range(n):
                d_mrc = (
                    np.diag(Sigma[:, j]) @ w  # w_i * Sigma_{ij}
                    + Sw * (np.arange(n) == j)  # delta_{ij} * (Sw)_i
                ) / var - (Sw * w) * (2.0 * Sw[j]) / var**2
                dobj[j] = 2.0 * float(residuals @ d_mrc)
            return dobj

        # Warm-start from naive risk parity
        vols = np.sqrt(np.diag(Sigma))
        vols = np.where(vols > 0, vols, 1e-8)
        w0 = 1.0 / vols
        w0 /= w0.sum()
        w0 = np.clip(w0, self.min_weight + 1e-6, self.max_weight - 1e-6)

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(self.min_weight, self.max_weight)] * n

        result = minimize(
            _objective,
            w0,
            jac=_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )

        if not result.success:
            logger.warning(
                "covariance_risk_parity optimisation did not converge "
                "(status %d: %s). Falling back to naive risk parity.",
                result.status,
                result.message,
            )
            return self.naive_risk_parity(vols)

        weights = np.clip(result.x, self.min_weight, self.max_weight)
        weights /= weights.sum()
        return weights

    def allocate(
        self,
        returns: pd.DataFrame,
        method: AllocationMethod = "covariance",
        use_shrinkage: bool = True,
        crisis_prob: float = 0.0,
    ) -> pd.Series:
        """
        Compute risk parity weights from a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily (or any consistent frequency) returns; columns are
            assets, rows are observations.
        method : {"naive", "covariance"}
            Which risk parity flavour to use.
        use_shrinkage : bool
            If True and method="covariance", use Ledoit-Wolf shrinkage
            covariance instead of raw sample covariance (default True).
            Shrinkage reduces estimation error and improves realised
            portfolio outcomes per Ledoit & Wolf (2004).
        crisis_prob : float
            Current HMM crisis state probability (0 to 1).  When elevated,
            the covariance matrix is inflated to reflect the empirical
            finding that correlations increase in bear/crisis regimes
            (Ang & Bekaert 2002).  This produces more conservative
            allocations during stress periods.

        Returns
        -------
        weights : pd.Series
            Indexed by asset name, values sum to 1.
        """
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        if returns.shape[1] < 2:
            raise ValueError("Need at least two assets for risk parity.")

        clean = returns.dropna(how="all").ffill().fillna(0.0)

        if method == "naive":
            vols = clean.std(ddof=1).values
            vols = np.where(vols > 0, vols, 1e-8)
            # Regime-adjust volatilities: inflate in crisis
            if crisis_prob > 0.2:
                vol_inflation = 1.0 + crisis_prob * 0.5  # up to 50% vol inflation
                vols = vols * vol_inflation
            raw_weights = self.naive_risk_parity(vols)
        elif method == "covariance":
            if use_shrinkage:
                try:
                    from .shrinkage import ShrinkageEstimator
                    shrink = ShrinkageEstimator(target="constant_correlation")
                    cov = shrink.estimate(clean)
                    logger.info(
                        "Risk parity using shrinkage covariance "
                        "(intensity=%.3f)", shrink.last_intensity_,
                    )
                except Exception as e:
                    logger.warning(
                        "Shrinkage failed (%s), falling back to sample cov", e
                    )
                    cov = clean.cov().values
            else:
                cov = clean.cov().values

            # Regime-conditional covariance scaling (Ang & Bekaert 2002):
            # Inflate off-diagonal elements (correlations) when crisis
            # probability is elevated, reflecting the well-documented
            # increase in cross-asset correlations during stress periods.
            if crisis_prob > 0.2:
                cov = self._regime_adjust_covariance(cov, crisis_prob)
                logger.info(
                    "Applied regime covariance scaling (crisis_prob=%.2f)",
                    crisis_prob,
                )

            raw_weights = self.covariance_risk_parity(cov)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'naive' or 'covariance'.")

        return pd.Series(raw_weights, index=returns.columns, name="weight")

    @staticmethod
    def _regime_adjust_covariance(
        cov: np.ndarray, crisis_prob: float
    ) -> np.ndarray:
        """
        Inflate covariance matrix correlations based on crisis probability.

        Per Ang & Bekaert (2002), cross-asset correlations increase
        significantly in bear/crisis regimes.  This adjustment:
        1. Inflates off-diagonal covariance by up to 40% at full crisis
        2. Inflates diagonal (variance) by up to 30% at full crisis

        The result is more conservative (more equal) risk parity weights
        during stress, which empirically reduces drawdowns.
        """
        n = cov.shape[0]
        # Scale factor: 0 when crisis_prob=0.2, up to max at crisis_prob=1.0
        intensity = min(1.0, max(0.0, (crisis_prob - 0.2) / 0.8))

        # Inflate diagonal (variance) — moderate
        diag_scale = 1.0 + intensity * 0.3
        # Inflate off-diagonal (covariance) — stronger
        offdiag_scale = 1.0 + intensity * 0.4

        adjusted = cov.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    adjusted[i, j] *= diag_scale
                else:
                    adjusted[i, j] *= offdiag_scale

        return adjusted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regularise(Sigma: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Add a small diagonal perturbation to ensure positive definiteness."""
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < epsilon:
            Sigma = Sigma + (epsilon - min_eig) * np.eye(Sigma.shape[0])
        return Sigma
