"""
Ornstein-Uhlenbeck (OU) Process for Mean-Reversion Trading
============================================================
Calibrates an OU process to spread/residual series and derives
optimal entry/exit thresholds that maximise expected profit per trade.

Academic basis:
  - Uhlenbeck & Ornstein (1930): Original OU process definition
  - Leung & Li (2015): "Optimal Mean Reversion Trading with Transaction
    Costs and Stop-Loss Exit"
  - Bertram (2010): "Analytic Solutions for Optimal Statistical Arbitrage
    Trading"
  - Endres & Stübinger (2019): OU-based pairs trading with regime switching

The OU process:  dX_t = κ(μ - X_t)dt + σ dW_t

  - κ (kappa): mean-reversion speed (higher = faster reversion)
  - μ (mu): long-run mean
  - σ (sigma): volatility of the process
  - Half-life: ln(2)/κ — time to revert halfway to the mean

Key insight: The OU model provides analytically optimal entry/exit
thresholds that account for mean-reversion speed, unlike ad-hoc z-score
thresholds.  This improves both hit rate and profit per trade.

Usage:
    ou = OUCalibrator()
    params = ou.calibrate(spread_series)
    thresholds = ou.optimal_thresholds(params, cost=0.001)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    """Calibrated Ornstein-Uhlenbeck parameters."""
    kappa: float     # mean-reversion speed
    mu: float        # long-run mean
    sigma: float     # process volatility
    half_life: float  # ln(2)/kappa — days to half-revert
    r_squared: float  # goodness of fit


@dataclass
class OUThresholds:
    """Optimal OU-based entry/exit thresholds."""
    entry_long: float   # buy when spread falls below this
    entry_short: float  # sell when spread rises above this
    exit_long: float    # close long when spread rises to this
    exit_short: float   # close short when spread falls to this
    stop_loss: float    # stop-loss distance from entry


class OUCalibrator:
    """
    Calibrate an Ornstein-Uhlenbeck process and derive optimal
    trading thresholds.

    Parameters
    ----------
    min_kappa : float
        Minimum mean-reversion speed to consider valid (default 0.01).
        If kappa < min_kappa, the series is not mean-reverting.
    max_half_life : float
        Maximum half-life in days (default 126 ~ 6 months).
    """

    def __init__(
        self,
        min_kappa: float = 0.01,
        max_half_life: float = 126.0,
    ) -> None:
        self.min_kappa = min_kappa
        self.max_half_life = max_half_life

    def calibrate(self, series: pd.Series, dt: float = 1.0) -> Optional[OUParams]:
        """
        Calibrate OU process parameters via maximum likelihood (OLS on
        discrete AR(1) representation).

        The discrete-time AR(1) model:
            X_{t+1} - X_t = a + b * X_t + ε_t

        Maps to OU parameters:
            κ = -b/dt,  μ = -a/b,  σ² = Var(ε) * 2κ / (1 - exp(-2κdt))

        Parameters
        ----------
        series : pd.Series
            Spread or residual series to calibrate.
        dt : float
            Time step (default 1.0 for daily).

        Returns
        -------
        OUParams or None
            Calibrated parameters, or None if series is not mean-reverting.
        """
        x = series.dropna().values
        if len(x) < 30:
            return None

        # AR(1) regression: ΔX = a + b*X_{t-1} + ε
        dx = np.diff(x)
        x_lag = x[:-1]

        # OLS: [a, b] = (X'X)^{-1} X'y
        n = len(dx)
        X_mat = np.column_stack([np.ones(n), x_lag])
        try:
            beta = np.linalg.lstsq(X_mat, dx, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        a, b = beta[0], beta[1]

        # b must be negative for mean-reversion
        if b >= 0:
            logger.debug("Series is not mean-reverting (b=%.4f >= 0)", b)
            return None

        kappa = -b / dt
        if kappa < self.min_kappa:
            return None

        mu = -a / b
        half_life = np.log(2) / kappa

        if half_life > self.max_half_life:
            logger.debug("Half-life too long: %.1f days", half_life)
            return None

        # Process volatility
        residuals = dx - X_mat @ beta
        var_residuals = np.var(residuals)
        sigma_sq = var_residuals * 2 * kappa / (1 - np.exp(-2 * kappa * dt))
        sigma = np.sqrt(max(sigma_sq, 1e-12))

        # R-squared of the AR(1) fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((dx - dx.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        params = OUParams(
            kappa=kappa,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
            r_squared=max(0, r_sq),
        )

        logger.debug(
            "OU calibration: κ=%.4f, μ=%.4f, σ=%.4f, half_life=%.1f days, R²=%.3f",
            kappa, mu, sigma, half_life, r_sq,
        )

        return params

    def optimal_thresholds(
        self,
        params: OUParams,
        cost: float = 0.001,
        risk_mult: float = 3.0,
    ) -> OUThresholds:
        """
        Compute optimal entry/exit thresholds per Bertram (2010) /
        Leung & Li (2015).

        The optimal entry threshold balances:
          - Higher threshold → more profit per trade but fewer trades
          - Lower threshold → more trades but less profit per trade
          - Transaction costs reduce optimal threshold

        For an OU process, the optimal entry is approximately:
            d* ≈ σ/sqrt(2κ) * sqrt(2*log(σ/(c*sqrt(2πκ))))

        where c is the round-trip cost.

        Parameters
        ----------
        params : OUParams
            Calibrated OU parameters.
        cost : float
            Round-trip transaction cost as fraction (default 0.001 = 10bps).
        risk_mult : float
            Stop-loss multiplier on equilibrium std (default 3.0).

        Returns
        -------
        OUThresholds
            Optimal entry, exit, and stop-loss levels.
        """
        kappa = params.kappa
        mu = params.mu
        sigma = params.sigma

        # Equilibrium standard deviation of the OU process
        eq_std = sigma / np.sqrt(2 * kappa)

        # Optimal entry distance (Bertram 2010 approximation)
        arg = sigma / (cost * np.sqrt(2 * np.pi * kappa)) if cost > 0 else 100.0
        if arg > 1:
            d_star = eq_std * np.sqrt(2 * np.log(arg))
        else:
            # Cost too high relative to signal — use conservative threshold
            d_star = 2.0 * eq_std

        # Clamp d_star to reasonable range (1-4 equilibrium stds)
        d_star = np.clip(d_star, 1.0 * eq_std, 4.0 * eq_std)

        # Exit is at the mean (or slightly before to lock in profits)
        exit_offset = 0.25 * eq_std  # small buffer

        return OUThresholds(
            entry_long=mu - d_star,
            entry_short=mu + d_star,
            exit_long=mu - exit_offset,
            exit_short=mu + exit_offset,
            stop_loss=risk_mult * eq_std,
        )

    def score_mean_reversion(self, series: pd.Series) -> float:
        """
        Score how mean-reverting a series is (0 to 1).

        Combines:
          - Half-life (shorter is better for trading)
          - R² of the OU fit (higher is better)
          - Kappa significance

        Returns 0.0 if the series is not mean-reverting.
        """
        params = self.calibrate(series)
        if params is None:
            return 0.0

        # Half-life score: best around 5-20 days for daily trading
        hl = params.half_life
        if hl < 2:
            hl_score = 0.3  # too fast, might be noise
        elif hl <= 20:
            hl_score = 1.0
        elif hl <= 60:
            hl_score = 0.7
        elif hl <= 126:
            hl_score = 0.3
        else:
            hl_score = 0.0

        # R² score
        r2_score = min(1.0, params.r_squared * 5)  # amplify small R²

        # Kappa score (normalise)
        kappa_score = min(1.0, params.kappa / 0.2)

        return 0.4 * hl_score + 0.4 * r2_score + 0.2 * kappa_score

    def rolling_calibration(
        self,
        series: pd.Series,
        window: int = 252,
        step: int = 21,
    ) -> pd.DataFrame:
        """
        Rolling OU calibration to track time-varying mean-reversion.

        Returns DataFrame with columns: kappa, mu, sigma, half_life, r_squared.
        """
        dates = series.index
        n = len(dates)
        results = []

        for t in range(window, n, step):
            window_data = series.iloc[t - window:t]
            params = self.calibrate(window_data)
            results.append({
                "date": dates[t],
                "kappa": params.kappa if params else np.nan,
                "mu": params.mu if params else np.nan,
                "sigma": params.sigma if params else np.nan,
                "half_life": params.half_life if params else np.nan,
                "r_squared": params.r_squared if params else np.nan,
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).set_index("date")
        return df.reindex(series.index).ffill()
