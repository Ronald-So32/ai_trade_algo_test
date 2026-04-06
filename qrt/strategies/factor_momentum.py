"""
Factor Momentum Strategy
=========================
Compute factor returns (value, size, momentum proxies) from the cross-section
of asset returns.  Apply time-series momentum to the factor returns themselves.
Tilt the portfolio toward assets with high exposure to winning factors.

Factor definitions (cross-sectional):
  - Value proxy   : long bottom quintile by trailing 12m return (contrarian)
  - Size proxy    : long bottom quintile by rolling average price level
  - Momentum proxy: long top quintile by trailing 12m return (skip 1m)

Factor portfolios are long-short, equal-weighted within each leg.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import Strategy


# ---------------------------------------------------------------------------
# Factor return computation helpers
# ---------------------------------------------------------------------------

def _quintile_long_short(
    scores: pd.Series,
    q: float = 0.20,
    long_high: bool = True,
) -> pd.Series:
    """
    Return {+1, -1, 0} signal: top-q → long (+1), bottom-q → short (-1).
    If long_high is False, reverse the direction.
    """
    try:
        n = scores.dropna()
        if len(n) < max(4, int(1 / q)):
            return pd.Series(0.0, index=scores.index)
        k = max(1, int(np.floor(len(n) * q)))
        ranked = n.rank(ascending=True)
        # Build signal on the non-NaN index first, then reindex to full index
        # to avoid "Unalignable boolean Series provided as indexer" errors
        sig_n = pd.Series(0.0, index=n.index)
        if long_high:
            sig_n.loc[ranked >= ranked.max() - k + 1] = 1.0
            sig_n.loc[ranked <= k] = -1.0
        else:
            sig_n.loc[ranked <= k] = 1.0
            sig_n.loc[ranked >= ranked.max() - k + 1] = -1.0
        # Reindex back to the original scores index (NaN entries get 0.0)
        return sig_n.reindex(scores.index, fill_value=0.0)
    except Exception:
        return pd.Series(0.0, index=scores.index)


def _factor_portfolio_return(signals: pd.Series, asset_returns: pd.Series) -> float:
    """Equal-weight long-short factor return for one period."""
    long_ret = asset_returns[signals == 1].mean() if (signals == 1).any() else 0.0
    short_ret = asset_returns[signals == -1].mean() if (signals == -1).any() else 0.0
    return float(long_ret - short_ret)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class FactorMomentum(Strategy):
    """
    Time-series momentum applied to cross-sectional factor returns.

    Parameters
    ----------
    factor_lookback : int
        Lookback for defining factor portfolios (default 126).
    momentum_lookback : int
        Window for factor-return time-series momentum (default 126).
    skip_days : int
        Days skipped at the end when computing factor momentum (default 21).
    vol_lookback : int
        Window for factor-return vol scaling (default 63).
    target_gross : float
        Target gross exposure (default 1.0).
    quantile : float
        Fraction of assets in each factor leg (default 0.20).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Arnott et al. (2019) — \"Factor Momentum Everywhere\"; "
            "Ehsani & Linnainmaa (2022)"
        ),
        "historical_evidence": (
            "Factors themselves exhibit momentum; contributes to explaining "
            "security momentum"
        ),
        "implementation_risks": (
            "Factor timing is notoriously difficult; factor momentum may be "
            "a restatement of asset momentum"
        ),
        "realistic_expectations": (
            "Emerging research area; incremental alpha over standard momentum is debated"
        ),
    }

    FACTORS = ["value", "size", "momentum"]

    def __init__(
        self,
        factor_lookback: int = 126,
        momentum_lookback: int = 126,
        skip_days: int = 21,
        vol_lookback: int = 63,
        target_gross: float = 1.0,
        quantile: float = 0.20,
    ) -> None:
        params = dict(
            factor_lookback=factor_lookback,
            momentum_lookback=momentum_lookback,
            skip_days=skip_days,
            vol_lookback=vol_lookback,
            target_gross=target_gross,
            quantile=quantile,
        )
        super().__init__(name="FactorMomentum", params=params)

    # ------------------------------------------------------------------
    # Factor construction
    # ------------------------------------------------------------------

    def _build_factor_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        t: int,
    ) -> Dict[str, pd.Series]:
        """
        Compute today's factor membership signals (+1 / -1 / 0) for each asset.
        """
        q: float = self.params["quantile"]
        fl: int = self.params["factor_lookback"]
        skip: int = self.params["skip_days"]

        t_lag_skip = max(0, t - skip)
        t_lag_full = max(0, t - fl)

        zero_sig = pd.Series(0.0, index=prices.columns)

        try:
            # Value proxy: low trailing 12m-return assets are "cheap" (long low return)
            trailing_ret = prices.iloc[t_lag_skip] / prices.iloc[t_lag_full] - 1
            trailing_ret = trailing_ret.reindex(prices.columns)
            value_sig = _quintile_long_short(trailing_ret, q=q, long_high=False)
        except Exception:
            trailing_ret = zero_sig
            value_sig = zero_sig.copy()

        try:
            # Size proxy: low average price → small cap proxy (long low price)
            avg_price = prices.iloc[max(0, t - 63): t].mean()
            avg_price = avg_price.reindex(prices.columns)
            size_sig = _quintile_long_short(avg_price, q=q, long_high=False)
        except Exception:
            size_sig = zero_sig.copy()

        try:
            # Momentum proxy: high trailing return, skip 1m
            mom_sig = _quintile_long_short(trailing_ret, q=q, long_high=True)
        except Exception:
            mom_sig = zero_sig.copy()

        return {
            "value": value_sig,
            "size": size_sig,
            "momentum": mom_sig,
        }

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
        1. Compute historical factor returns.
        2. Apply time-series momentum to determine which factors are "winning".
        3. Tilt the portfolio toward assets that load heavily on winning factors.

        Returns
        -------
        pd.DataFrame
            Continuous signals in [-1, +1], same shape as *prices*.
        """
        fl: int = self.params["factor_lookback"]
        ml: int = self.params["momentum_lookback"]
        skip: int = self.params["skip_days"]
        vol_lb: int = self.params["vol_lookback"]
        target_gross: float = self.params["target_gross"]

        dates = prices.index
        n_dates = len(dates)
        min_history = fl + ml + skip

        signals = pd.DataFrame(0.0, index=dates, columns=prices.columns)

        if n_dates < min_history:
            return signals

        # Step 1: compute time series of factor returns
        factor_returns: Dict[str, list[float]] = {f: [] for f in self.FACTORS}
        factor_dates: list[pd.Timestamp] = []

        for t in range(fl, n_dates):
            date = dates[t]
            asset_ret = returns.iloc[t]
            factor_sigs = self._build_factor_signals(prices, returns, date, t)
            for fname in self.FACTORS:
                fr = _factor_portfolio_return(factor_sigs[fname], asset_ret)
                factor_returns[fname].append(fr)
            factor_dates.append(date)

        # Build factor returns DataFrame
        fret_df = pd.DataFrame(factor_returns, index=factor_dates)

        # Step 2: time-series momentum on factors
        # Factor momentum signal = sign of trailing ml-day cumulative factor return
        # (skip last skip_days)
        for t in range(fl + ml + skip, n_dates):
            date = dates[t]
            t_fret = fret_df.index.get_loc(date) if date in fret_df.index else None
            if t_fret is None:
                continue
            t_fret_int = int(t_fret)

            window_end = max(0, t_fret_int - skip)
            window_start = max(0, window_end - ml)
            if window_end - window_start < ml // 2:
                continue

            factor_window = fret_df.iloc[window_start:window_end]
            factor_cum_ret = (1 + factor_window).prod() - 1  # series over FACTORS

            # Factor momentum: sign of cumulative return
            factor_sign = np.sign(factor_cum_ret)

            # Factor vol scaling
            factor_vol = fret_df.iloc[max(0, t_fret_int - vol_lb): t_fret_int].std()
            factor_vol = factor_vol.clip(lower=1e-6)
            factor_tilt = (factor_sign / factor_vol)

            # Normalize factor tilts
            tilt_sum = factor_tilt.abs().sum()
            if tilt_sum > 1e-8:
                factor_tilt = factor_tilt / tilt_sum

            # Step 3: compute today's factor signals and aggregate tilt
            factor_sigs_today = self._build_factor_signals(prices, returns, date, t)

            # Asset signal = sum of (factor_tilt * factor_membership)
            asset_signal = pd.Series(0.0, index=prices.columns)
            for fname in self.FACTORS:
                tilt = float(factor_tilt.get(fname, 0.0))
                asset_signal += tilt * factor_sigs_today[fname]

            # Clip to [-1, +1]
            signals.loc[date] = asset_signal.clip(-1.0, 1.0)

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Gross-normalise signals with optional vol scaling.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        vol_lb: int = self.params["vol_lookback"]

        if returns is not None:
            realized_vol = (
                returns.rolling(vol_lb, min_periods=max(1, vol_lb // 2))
                .std()
                .mul(np.sqrt(252))
                .clip(lower=1e-6)
            )
            raw_weights = signals / realized_vol
        else:
            raw_weights = signals.copy()

        raw_weights = raw_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
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
        """End-to-end: signals → weights → backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
