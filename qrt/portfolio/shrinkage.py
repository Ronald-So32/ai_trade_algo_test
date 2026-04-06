"""
Shrinkage Covariance Estimation
=================================
Implements the Ledoit-Wolf linear shrinkage estimator for covariance matrices.

The sample covariance matrix is unreliable when the number of observations
is not much larger than the number of assets.  Shrinkage pulls the sample
covariance toward a structured target (constant-correlation or identity
scaled), reducing estimation error and improving downstream portfolio
optimization.

References
----------
- Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix"
- Ledoit & Wolf (2012), "Nonlinear Shrinkage Estimation of Large-
  Dimensional Covariance Matrices" (for the nonlinear extension)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ShrinkageTarget = Literal["constant_correlation", "identity", "diagonal"]


def ledoit_wolf_shrinkage(
    returns: np.ndarray,
    target: ShrinkageTarget = "constant_correlation",
) -> tuple[np.ndarray, float]:
    """
    Compute the Ledoit-Wolf shrinkage estimator of the covariance matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Demeaned returns matrix (observations x assets).
    target : str
        Shrinkage target: "constant_correlation", "identity", or "diagonal".

    Returns
    -------
    shrunk_cov : np.ndarray, shape (N, N)
        Shrinkage-estimated covariance matrix.
    shrinkage_intensity : float
        Optimal shrinkage intensity in [0, 1].
    """
    T, N = returns.shape

    if T < 2 or N < 2:
        sample_cov = np.cov(returns, rowvar=False) if T >= 2 else np.eye(N)
        return sample_cov, 0.0

    # Demean
    X = returns - returns.mean(axis=0)
    sample_cov = (X.T @ X) / T

    # Compute shrinkage target
    if target == "constant_correlation":
        F = _constant_correlation_target(sample_cov)
    elif target == "identity":
        trace = np.trace(sample_cov) / N
        F = trace * np.eye(N)
    elif target == "diagonal":
        F = np.diag(np.diag(sample_cov))
    else:
        raise ValueError(f"Unknown target: {target}")

    # Compute optimal shrinkage intensity (Ledoit-Wolf formula)
    delta = sample_cov - F

    # Sum of squared Frobenius norms
    sum_sq_delta = np.sum(delta ** 2)

    # Estimate asymptotic quantities
    # pi: sum of asymptotic variances of scaled sample cov entries
    Y = X ** 2
    phi_mat = (Y.T @ Y) / T - sample_cov ** 2
    pi = np.sum(phi_mat)

    # rho: asymptotic covariance of target and sample entries
    if target == "constant_correlation":
        rho = _compute_rho_cc(X, sample_cov, F, T, N)
    else:
        # For identity/diagonal targets, rho simplification
        rho = np.sum(np.diag(phi_mat))

    # Kappa
    kappa = (pi - rho) / sum_sq_delta if sum_sq_delta > 0 else 0.0

    # Shrinkage intensity: clamp to [0, 1]
    shrinkage = max(0.0, min(1.0, kappa / T))

    # Shrunk covariance
    shrunk = shrinkage * F + (1 - shrinkage) * sample_cov

    # Ensure positive semi-definiteness
    min_eig = np.linalg.eigvalsh(shrunk).min()
    if min_eig < 1e-10:
        shrunk += (1e-10 - min_eig) * np.eye(N)

    logger.debug(
        "Ledoit-Wolf shrinkage: target=%s, intensity=%.4f, N=%d, T=%d",
        target, shrinkage, N, T,
    )

    return shrunk, float(shrinkage)


def _constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """Compute the constant-correlation shrinkage target.

    F_{ij} = sqrt(sigma_ii * sigma_jj) * rbar  for i != j
    F_{ii} = sigma_ii
    """
    N = sample_cov.shape[0]
    vols = np.sqrt(np.diag(sample_cov))
    vols = np.where(vols > 0, vols, 1e-10)

    # Average pairwise correlation
    corr = sample_cov / np.outer(vols, vols)
    np.fill_diagonal(corr, 0.0)
    n_pairs = N * (N - 1)
    rbar = corr.sum() / n_pairs if n_pairs > 0 else 0.0

    F = rbar * np.outer(vols, vols)
    np.fill_diagonal(F, np.diag(sample_cov))

    return F


def _compute_rho_cc(
    X: np.ndarray,
    sample_cov: np.ndarray,
    F: np.ndarray,
    T: int,
    N: int,
) -> float:
    """Compute rho for constant-correlation target (Ledoit-Wolf 2004)."""
    vols = np.sqrt(np.diag(sample_cov))
    vols = np.where(vols > 0, vols, 1e-10)

    # Simplified: use diagonal rho as approximation for speed
    Y = X ** 2
    phi_diag = np.diag((Y.T @ Y) / T - sample_cov ** 2)
    rho = np.sum(phi_diag)

    return float(rho)


class ShrinkageEstimator:
    """
    Wrapper for shrinkage covariance estimation with caching and diagnostics.

    Parameters
    ----------
    target : str
        Shrinkage target (default "constant_correlation").
    min_obs : int
        Minimum observations required (default 63).
    """

    def __init__(
        self,
        target: ShrinkageTarget = "constant_correlation",
        min_obs: int = 63,
    ) -> None:
        self.target = target
        self.min_obs = min_obs
        self.last_intensity_: float = 0.0

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (dates x assets).

        Returns
        -------
        np.ndarray
            Shrinkage-estimated covariance matrix.
        """
        clean = returns.dropna(how="all").ffill().fillna(0.0)
        if len(clean) < self.min_obs:
            logger.warning(
                "ShrinkageEstimator: only %d obs (need %d), using sample cov",
                len(clean), self.min_obs,
            )
            cov = clean.cov().values
            self.last_intensity_ = 0.0
            return cov

        X = clean.values
        shrunk, intensity = ledoit_wolf_shrinkage(X, target=self.target)
        self.last_intensity_ = intensity

        # Apply RMT covariance cleaning if available (Laloux et al. 1999)
        # This removes noise eigenvalues from the covariance matrix,
        # improving portfolio optimization stability
        try:
            from qrt.models.rmt_covariance import RMTCovarianceCleaner
            rmt = RMTCovarianceCleaner(detone=True)
            rmt_cleaned = rmt.clean(clean)
            # Blend: shrinkage target (structured) with RMT-cleaned (denoised)
            # intensity controls how much structured target to mix in
            N = clean.shape[1]
            trace = np.trace(rmt_cleaned) / N
            F_identity = trace * np.eye(N)
            shrunk = intensity * F_identity + (1 - intensity) * rmt_cleaned
            logger.info(
                "Applied RMT cleaning: %d signal eigenvalues preserved (of %d)",
                rmt.last_n_signal_, clean.shape[1],
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug("RMT cleaning failed (%s), using shrinkage only", e)

        return shrunk
