"""
Volatility targeting for portfolio exposure management.

Scales a portfolio's notional exposure so that the realised volatility
tracks a specified annualised target.  Leverage is capped to avoid
excessive concentration in low-volatility regimes.

Implements two vol estimators:
  - Rolling window (original): simple rolling std
  - EWMA (Harvey et al. 2018): exponentially weighted variance with
    configurable halflife, providing faster reaction to vol changes.
    Formula: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

Also supports Moreira & Muir (2017) vol-managed leverage:
  L_t = L_base × (σ_baseline / σ_t)  capped to [L_min, L_max]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading days per year used for annualisation
TRADING_DAYS_PER_YEAR: int = 252


def ewma_volatility(
    returns: pd.Series,
    halflife: int = 20,
    annualise: bool = True,
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
    min_periods: int = 10,
) -> pd.Series:
    """
    EWMA volatility estimator per Harvey et al. (2018).

    Uses exponentially weighted variance:
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    where λ = exp(-ln2 / halflife) ≈ 0.966 for halflife=20.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    halflife : int
        Halflife in days for the exponential decay (default 20 per
        Harvey et al. 2018).  λ = exp(-ln2/halflife).
    annualise : bool
        If True, multiply by √252 to get annualised vol.
    annualisation_factor : int
        Trading days per year.
    min_periods : int
        Minimum observations before producing a value.

    Returns
    -------
    pd.Series
        EWMA volatility estimate, same index as returns.
    """
    returns = returns.fillna(0.0)

    ewma_var = returns.ewm(halflife=halflife, min_periods=min_periods).var()
    ewma_vol = np.sqrt(ewma_var)

    if annualise:
        ewma_vol = ewma_vol * np.sqrt(annualisation_factor)

    ewma_vol.name = "ewma_vol"
    return ewma_vol


def vol_managed_leverage(
    returns: pd.Series,
    base_leverage: float,
    halflife: int = 20,
    max_leverage: float = 10.0,
    min_leverage: float = 0.5,
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Moreira & Muir (2017) vol-managed dynamic leverage.

    Scales leverage inversely with realised volatility:
        L_t = L_base × (σ_baseline / σ_t)

    where σ_baseline is the full-sample median EWMA vol (so average
    leverage ≈ L_base) and σ_t is the EWMA vol estimate at time t.

    In calm periods σ_t < σ_baseline → leverage increases.
    In turbulent periods σ_t > σ_baseline → leverage decreases.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns (pre-leverage).
    base_leverage : float
        Target average leverage (from optimizer).
    halflife : int
        EWMA halflife for vol estimation (default 20 days).
    max_leverage : float
        Hard cap on dynamic leverage.
    min_leverage : float
        Floor on dynamic leverage.
    annualisation_factor : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Daily dynamic leverage factors, same index as returns.
    """
    vol = ewma_volatility(
        returns,
        halflife=halflife,
        annualise=True,
        annualisation_factor=annualisation_factor,
    )

    # ── LOOK-AHEAD BIAS FIX ──
    # Original used vol.median() over the FULL sample — that's future-peeking.
    # Per Cederburg et al. (2020) critique of Moreira & Muir: the normalisation
    # constant must be causal.  We use an EXPANDING window median so that on
    # day t, baseline_vol_t = median(σ_1, …, σ_t).  This is strictly causal
    # and converges to the full-sample median as t → ∞.
    # Barroso & Santa-Clara (2015) used a similar expanding-window approach
    # for their risk-managed momentum strategy.
    baseline_vol = vol.expanding(min_periods=1).median()

    # If all baseline values are bad, fall back to constant leverage
    if baseline_vol.iloc[-1] <= 0 or np.isnan(baseline_vol.iloc[-1]):
        logger.warning("Baseline vol is zero/NaN — returning constant leverage.")
        return pd.Series(base_leverage, index=returns.index, name="dynamic_leverage")

    # Moreira-Muir scaling: L_t = L_base × (σ_baseline_t / σ_t)
    # Now both σ_baseline_t and σ_t use only data available at time t
    dynamic_lev = base_leverage * (baseline_vol / vol)

    # Clip to bounds and fill early NaNs with base leverage
    dynamic_lev = dynamic_lev.clip(lower=min_leverage, upper=max_leverage).fillna(
        base_leverage
    )
    dynamic_lev.name = "dynamic_leverage"

    logger.info(
        "Vol-managed leverage (causal): base=%.1fx, mean=%.2fx, "
        "min=%.2fx, max=%.2fx, final_baseline_vol=%.2f%%",
        base_leverage,
        dynamic_lev.mean(),
        dynamic_lev.min(),
        dynamic_lev.max(),
        baseline_vol.iloc[-1] * 100,
    )
    return dynamic_lev


class VolatilityTargeter:
    """
    Volatility targeting: scale portfolio exposure to hit a vol target.

    The scaling factor on day *t* is::

        scaling_t = target_vol / realised_vol_t

    where ``realised_vol_t`` is the annualised rolling standard deviation
    of portfolio returns over a trailing ``lookback``-day window.  The
    resulting scaler is clipped to ``[0, max_leverage]``.

    Parameters
    ----------
    annualisation_factor : int
        Number of periods per year (default 252 for daily data).
    """

    def __init__(self, annualisation_factor: int = TRADING_DAYS_PER_YEAR) -> None:
        if annualisation_factor <= 0:
            raise ValueError("annualisation_factor must be positive.")
        self.annualisation_factor = annualisation_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_scaling(
        self,
        returns: pd.Series,
        target_vol: float,
        lookback: int = 63,
        max_leverage: float = 2.0,
    ) -> pd.Series:
        """
        Compute daily exposure scaling factors.

        Parameters
        ----------
        returns : pd.Series
            Time series of portfolio (or instrument) returns.
        target_vol : float
            Annualised volatility target expressed as a decimal
            (e.g. 0.10 for 10 %).
        lookback : int
            Rolling window length in periods used to estimate realised
            volatility (default 63 ≈ 3 months of daily data).
        max_leverage : float
            Maximum allowed scaling factor (default 2.0).

        Returns
        -------
        scaling : pd.Series
            Daily scaling factors, same index as *returns*.
            Periods with insufficient history are assigned a scaling of
            1.0 (neutral).
        """
        if target_vol <= 0:
            raise ValueError("target_vol must be strictly positive.")
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be strictly positive.")

        returns = returns.copy().fillna(0.0)

        # Annualised rolling volatility
        realized_vol: pd.Series = (
            returns.rolling(window=lookback, min_periods=max(2, lookback // 2))
            .std(ddof=1)
            .mul(np.sqrt(self.annualisation_factor))
        )

        # Avoid division by zero — replace non-positive vol with NaN so
        # we fall back to the neutral scaling below
        realized_vol = realized_vol.where(realized_vol > 0)

        raw_scaling: pd.Series = target_vol / realized_vol

        # Clip to [0, max_leverage] and fill early NaNs with neutral (1.0)
        scaling: pd.Series = (
            raw_scaling.clip(lower=0.0, upper=max_leverage).fillna(1.0)
        )
        scaling.name = "scaling"

        logger.debug(
            "compute_scaling: target_vol=%.4f, lookback=%d, "
            "mean_scaling=%.4f, max_scaling=%.4f",
            target_vol,
            lookback,
            scaling.mean(),
            scaling.max(),
        )
        return scaling

    def apply_scaling(
        self,
        weights: pd.DataFrame,
        scaling_factors: pd.Series,
    ) -> pd.DataFrame:
        """
        Multiply portfolio weights by daily scaling factors.

        Parameters
        ----------
        weights : pd.DataFrame
            DataFrame of portfolio weights; index is dates, columns are
            assets.  May contain NaNs for periods before a strategy
            starts producing signals.
        scaling_factors : pd.Series
            Daily scaling factors indexed by date (output of
            :meth:`compute_scaling`).

        Returns
        -------
        scaled_weights : pd.DataFrame
            Element-wise product of ``weights`` and ``scaling_factors``,
            aligned on the common date index.
        """
        if weights.empty:
            raise ValueError("weights DataFrame is empty.")

        # Align on shared dates
        common_idx = weights.index.intersection(scaling_factors.index)
        if common_idx.empty:
            raise ValueError(
                "weights and scaling_factors share no common dates."
            )

        aligned_weights = weights.loc[common_idx].copy()
        aligned_scaling = scaling_factors.reindex(common_idx).fillna(1.0)

        # Broadcast scaling (n_dates,) across columns (n_assets,)
        scaled = aligned_weights.multiply(aligned_scaling, axis="index")
        scaled.columns = weights.columns
        return scaled

    # ------------------------------------------------------------------
    # Convenience: combined helper
    # ------------------------------------------------------------------

    def target_and_scale(
        self,
        weights: pd.DataFrame,
        portfolio_returns: pd.Series,
        target_vol: float = 0.10,
        lookback: int = 63,
        max_leverage: float = 2.0,
    ) -> pd.DataFrame:
        """
        One-step helper: compute scaling factors then apply them.

        Parameters
        ----------
        weights : pd.DataFrame
            Raw portfolio weights (dates × assets).
        portfolio_returns : pd.Series
            Historical portfolio returns used to estimate realised vol.
        target_vol : float
            Annualised vol target (default 10 %).
        lookback : int
            Lookback for realised vol estimation.
        max_leverage : float
            Maximum scaling factor.

        Returns
        -------
        scaled_weights : pd.DataFrame
            Weights after vol-targeting adjustment.
        """
        scaling = self.compute_scaling(
            portfolio_returns,
            target_vol=target_vol,
            lookback=lookback,
            max_leverage=max_leverage,
        )
        return self.apply_scaling(weights, scaling)
