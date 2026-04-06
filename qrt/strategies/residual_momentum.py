"""
Residual Momentum Strategy
============================
Estimate rolling factor exposures via OLS regression against the market
(and optionally sector) returns, strip out systematic effects, then rank
securities by trailing cumulative residual return.  Go long the top
quintile, short the bottom quintile, rebalance monthly.

The key insight is that residual momentum — momentum after removing
market and sector beta — is a stronger, more persistent alpha signal
than raw momentum because it avoids rewarding passive factor exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Ordinary least-squares coefficients: beta = (X'X)^{-1} X'y.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Dependent variable (stock returns).
    X : np.ndarray, shape (T, K)
        Regressors (intercept + market + optional sectors).

    Returns
    -------
    np.ndarray, shape (K,)
        OLS beta coefficients.  Returns zeros if the system is singular.
    """
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        return beta
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def _compute_market_return(
    returns: pd.DataFrame,
    market_caps: pd.DataFrame | None = None,
) -> pd.Series:
    """Equal- or cap-weighted cross-sectional market return."""
    if market_caps is not None:
        # Market-cap weighted
        aligned_caps = market_caps.reindex_like(returns).ffill()
        weights = aligned_caps.div(aligned_caps.sum(axis=1), axis=0)
        mkt = (returns * weights).sum(axis=1)
    else:
        mkt = returns.mean(axis=1)
    return mkt


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class ResidualMomentum(Strategy):
    """
    Residual momentum: rank stocks by cumulative return after removing
    market (and sector) beta exposure, then form a long-short portfolio.

    Parameters
    ----------
    regression_window : int
        Number of trading days for the rolling OLS window (default 252).
    momentum_lookback : int
        Trailing window over which to cumulate residual returns (default 252).
    skip_days : int
        Most-recent days to skip to avoid short-term reversal (default 21).
    long_pct : float
        Fraction of the universe forming the long leg (default 0.20).
    short_pct : float
        Fraction of the universe forming the short leg (default 0.20).
    target_gross : float
        Target gross exposure after normalisation (default 1.0).
    rebalance_freq : int
        Rebalance every *rebalance_freq* trading days (default 21).
    vol_scale : bool
        If True, scale raw signals by inverse realised volatility before
        gross-normalising (default True).
    vol_lookback : int
        Window for realised vol estimation when *vol_scale* is True
        (default 63).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Blitz, Huij & Martens (2011) — \"Residual Momentum\""
        ),
        "historical_evidence": (
            "Residual momentum shows less crash risk than raw momentum; "
            "similar or better Sharpe"
        ),
        "implementation_risks": (
            "Factor model misspecification, residuals may absorb alpha or noise; "
            "higher turnover"
        ),
        "realistic_expectations": (
            "Research-supported premium with improved tail properties vs raw momentum; "
            "factor model choice matters"
        ),
    }

    def __init__(
        self,
        regression_window: int = 252,
        momentum_lookback: int = 252,
        skip_days: int = 21,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        target_gross: float = 1.0,
        rebalance_freq: int = 21,
        vol_scale: bool = True,
        vol_lookback: int = 63,
    ) -> None:
        params = dict(
            regression_window=regression_window,
            momentum_lookback=momentum_lookback,
            skip_days=skip_days,
            long_pct=long_pct,
            short_pct=short_pct,
            target_gross=target_gross,
            rebalance_freq=rebalance_freq,
            vol_scale=vol_scale,
            vol_lookback=vol_lookback,
        )
        super().__init__(name="ResidualMomentum", params=params)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate residual momentum signals.

        For each rebalance date *t*:
          1. Use returns over [t - regression_window : t] to run OLS of each
             stock's return on the market return (plus an intercept and any
             sector dummies supplied via *kwargs*).
          2. Compute daily residual returns = actual - predicted.
          3. Cumulate residuals over [t - momentum_lookback : t - skip_days].
          4. Rank by cumulative residual return.
          5. Long the top *long_pct* fraction, short the bottom *short_pct*.

        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices (dates x securities).
        returns : pd.DataFrame
            Asset returns (dates x securities), aligned with *prices*.
        **kwargs
            market_caps : pd.DataFrame, optional
                Market capitalisations for cap-weighting the market return.
            sector_returns : pd.DataFrame, optional
                Sector return time-series (dates x sectors) to include as
                additional regressors in the OLS.

        Returns
        -------
        pd.DataFrame
            Signal matrix, same shape as *prices*.  Values in {-1, 0, +1}
            indicating short, neutral, and long membership.
        """
        reg_win: int = self.params["regression_window"]
        mom_lb: int = self.params["momentum_lookback"]
        skip: int = self.params["skip_days"]
        long_pct: float = self.params["long_pct"]
        short_pct: float = self.params["short_pct"]
        rebal: int = self.params["rebalance_freq"]

        market_caps = kwargs.get("market_caps", None)
        sector_returns = kwargs.get("sector_returns", None)

        dates = prices.index
        n_dates = len(dates)
        assets = prices.columns

        signals = pd.DataFrame(0.0, index=dates, columns=assets)

        # Minimum rows needed before the first signal can fire
        min_history = max(reg_win, mom_lb + skip)
        if n_dates <= min_history:
            return signals

        # Pre-compute market return series
        mkt_ret = _compute_market_return(returns, market_caps)

        # Determine rebalance dates (every rebal days starting from min_history)
        rebalance_indices = list(range(min_history, n_dates, rebal))

        for t in rebalance_indices:
            # ---- Step 1: rolling OLS to estimate betas ----
            reg_start = max(0, t - reg_win)
            ret_window = returns.iloc[reg_start:t]  # (reg_win x n_assets)
            mkt_window = mkt_ret.iloc[reg_start:t].values  # (reg_win,)

            n_obs = len(ret_window)
            if n_obs < max(30, reg_win // 4):
                continue

            # Build regressor matrix: intercept + market [+ sectors]
            X_parts = [np.ones((n_obs, 1)), mkt_window.reshape(-1, 1)]
            if sector_returns is not None:
                sec_window = sector_returns.iloc[reg_start:t]
                sec_vals = sec_window.values
                if sec_vals.shape[0] == n_obs:
                    X_parts.append(sec_vals)
            X = np.hstack(X_parts)  # (n_obs, K)

            # ---- Step 2: compute residual returns for each asset ----
            # We need residuals over the full momentum_lookback window,
            # which may be wider than the regression window.  Re-use the
            # betas estimated over [reg_start:t] to compute residuals over
            # the momentum window.
            mom_start = max(0, t - mom_lb)
            mom_end = max(0, t - skip)
            if mom_end <= mom_start:
                continue

            ret_mom = returns.iloc[mom_start:mom_end]
            mkt_mom = mkt_ret.iloc[mom_start:mom_end].values
            n_mom = len(ret_mom)

            X_mom_parts = [np.ones((n_mom, 1)), mkt_mom.reshape(-1, 1)]
            if sector_returns is not None:
                sec_mom = sector_returns.iloc[mom_start:mom_end]
                sec_vals_mom = sec_mom.values
                if sec_vals_mom.shape[0] == n_mom:
                    X_mom_parts.append(sec_vals_mom)
            X_mom = np.hstack(X_mom_parts)

            cum_resid = {}
            valid_assets = []

            for col in assets:
                y_reg = ret_window[col].values
                # Skip assets with insufficient non-NaN data
                valid_mask_reg = np.isfinite(y_reg)
                if valid_mask_reg.sum() < max(30, reg_win // 4):
                    continue

                # Replace NaN with 0 for regression (conservative)
                y_clean = np.where(valid_mask_reg, y_reg, 0.0)
                beta = _ols_beta(y_clean, X)

                # Residuals over the momentum window
                y_mom = ret_mom[col].values
                valid_mask_mom = np.isfinite(y_mom)
                if valid_mask_mom.sum() < n_mom // 2:
                    continue

                predicted = X_mom @ beta
                residuals = np.where(valid_mask_mom, y_mom - predicted, 0.0)
                cum_resid[col] = float(np.sum(residuals))
                valid_assets.append(col)

            if len(valid_assets) < max(3, int(1 / min(long_pct, short_pct))):
                continue

            # ---- Step 3-5: rank and assign long/short ----
            resid_series = pd.Series(cum_resid)
            ranked = resid_series.rank(ascending=True)
            n_valid = len(valid_assets)
            n_long = max(1, int(np.floor(n_valid * long_pct)))
            n_short = max(1, int(np.floor(n_valid * short_pct)))

            long_assets = ranked.nlargest(n_long).index
            short_assets = ranked.nsmallest(n_short).index

            # Equal weight within each leg at the signal level
            signals.loc[dates[t], long_assets] = 1.0
            signals.loc[dates[t], short_assets] = -1.0

        # Forward-fill signals between rebalance dates
        # (only fill non-zero rows to keep initial zeros as zeros)
        mask = (signals != 0).any(axis=1)
        first_signal = mask.idxmax() if mask.any() else None
        if first_signal is not None:
            signals.loc[first_signal:] = signals.loc[first_signal:].replace(
                0.0, np.nan
            )
            # Re-insert true zeros on rebalance dates for neutral assets
            for t in rebalance_indices:
                if t < len(dates):
                    row = signals.loc[dates[t]]
                    signals.loc[dates[t]] = row.fillna(0.0)
            signals = signals.ffill().fillna(0.0)

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert signals to portfolio weights with optional inverse-vol
        scaling and gross-exposure normalisation.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals`` ({-1, 0, +1} values).
        returns : pd.DataFrame, optional
            Daily returns used for volatility scaling.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        vol_scale: bool = self.params["vol_scale"]
        vol_lb: int = self.params["vol_lookback"]

        raw_weights = signals.copy()

        # Inverse-volatility scaling
        if vol_scale and returns is not None:
            realized_vol = (
                returns.rolling(vol_lb, min_periods=max(1, vol_lb // 2))
                .std()
                .clip(lower=1e-6)
            )
            raw_weights = raw_weights / realized_vol

        raw_weights = raw_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Gross-normalise to target_gross
        gross = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        return weights

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """End-to-end: signals -> weights -> backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
