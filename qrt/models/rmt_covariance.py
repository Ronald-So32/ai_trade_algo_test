"""
Random Matrix Theory (RMT) Covariance Cleaning
================================================
Implements Marchenko-Pastur based covariance cleaning to separate signal
from noise in the sample covariance matrix.

Academic basis:
  - Laloux, Cizeau, Bouchaud & Potters (1999): "Noise Dressing of Financial
    Correlation Matrices"
  - Bouchaud & Potters (2009): Financial Applications of Random Matrix Theory
  - Ledoit & Wolf (2020): Analytical Nonlinear Shrinkage
  - Plerou et al. (2002): "Random matrix approach to cross correlations
    in financial data"

Key insight: In a universe of N assets with T observations, the sample
covariance matrix has N(N+1)/2 parameters but only NT data points.
When N/T is not small, most eigenvalues are dominated by noise.  RMT
identifies the noise eigenvalues via the Marchenko-Pastur distribution
and replaces them, preserving only the signal eigenvalues.

This dramatically improves portfolio optimization by reducing estimation
error in the covariance matrix, leading to more stable and better-
performing portfolios.

Usage:
    cleaner = RMTCovarianceCleaner()
    clean_cov = cleaner.clean(returns_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RMTCovarianceCleaner:
    """
    Clean a covariance matrix using Random Matrix Theory.

    Parameters
    ----------
    detone : bool
        If True, remove the market mode (first eigenvalue) before
        cleaning and add it back afterward (default True).
    method : str
        "marchenko-pastur" (default): replace noise eigenvalues with
        their average.
        "targeted-shrinkage": shrink noise eigenvalues toward the
        MP predicted value.
    """

    def __init__(
        self,
        detone: bool = True,
        method: str = "marchenko-pastur",
    ) -> None:
        self.detone = detone
        self.method = method
        self.last_n_signal_: int = 0
        self.last_mp_bound_: float = 0.0

    def clean(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Clean the covariance matrix estimated from returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (T x N).

        Returns
        -------
        np.ndarray
            Cleaned covariance matrix (N x N).
        """
        X = returns.dropna(how="all").ffill().fillna(0.0).values
        T, N = X.shape

        if T < 2 or N < 2:
            return np.cov(X, rowvar=False) if T >= 2 else np.eye(N)

        # Compute correlation matrix (work in correlation space for RMT)
        cov = np.cov(X, rowvar=False)
        vols = np.sqrt(np.diag(cov))
        vols = np.where(vols > 0, vols, 1e-10)
        corr = cov / np.outer(vols, vols)

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marchenko-Pastur bounds
        q = N / T  # ratio
        lambda_plus = (1 + np.sqrt(q)) ** 2
        lambda_minus = (1 - np.sqrt(q)) ** 2

        self.last_mp_bound_ = lambda_plus

        # Identify signal vs noise eigenvalues
        if self.detone:
            # First eigenvalue is the "market mode" — preserve it always
            signal_mask = np.zeros(N, dtype=bool)
            signal_mask[0] = True  # market mode
            # Check remaining eigenvalues against MP upper bound
            for i in range(1, N):
                if eigenvalues[i] > lambda_plus:
                    signal_mask[i] = True
        else:
            signal_mask = eigenvalues > lambda_plus

        n_signal = signal_mask.sum()
        n_noise = N - n_signal
        self.last_n_signal_ = int(n_signal)

        logger.debug(
            "RMT cleaning: N=%d, T=%d, q=%.3f, MP_upper=%.3f, "
            "signal=%d, noise=%d",
            N, T, q, lambda_plus, n_signal, n_noise,
        )

        # Clean: replace noise eigenvalues
        cleaned_eigenvalues = eigenvalues.copy()

        if n_noise > 0:
            noise_eigenvalues = eigenvalues[~signal_mask]

            if self.method == "marchenko-pastur":
                # Replace all noise eigenvalues with their average
                # This preserves the trace while removing noise structure
                avg_noise = noise_eigenvalues.mean() if len(noise_eigenvalues) > 0 else 1.0
                cleaned_eigenvalues[~signal_mask] = avg_noise

            elif self.method == "targeted-shrinkage":
                # Shrink noise eigenvalues toward 1 (the expected value
                # under the null of no signal)
                alpha = 0.5  # shrinkage intensity
                cleaned_eigenvalues[~signal_mask] = (
                    alpha * 1.0 + (1 - alpha) * noise_eigenvalues
                )

        # Ensure all eigenvalues are positive
        cleaned_eigenvalues = np.maximum(cleaned_eigenvalues, 1e-8)

        # Reconstruct correlation matrix
        clean_corr = (
            eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        )

        # Force exact symmetry and unit diagonal
        clean_corr = (clean_corr + clean_corr.T) / 2
        d = np.sqrt(np.diag(clean_corr))
        d = np.where(d > 0, d, 1e-10)
        clean_corr = clean_corr / np.outer(d, d)
        np.fill_diagonal(clean_corr, 1.0)

        # Convert back to covariance matrix
        clean_cov = clean_corr * np.outer(vols, vols)

        return clean_cov

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Clean covariance and return as DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned covariance matrix with column/index labels.
        """
        cov = self.clean(returns)
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
