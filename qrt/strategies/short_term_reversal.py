"""
Short-Term Reversal Strategy
==============================
Exploits the well-documented tendency of recent losers (winners) to
outperform (underperform) over the next 1-4 weeks.

Academic basis:
  - Jegadeesh (1990): "Evidence of Predictable Behavior of Security Returns"
  - Lehmann (1990): "Fads, Martingales, and Market Efficiency"
  - Avramov, Chordia & Goyal (2006): "Liquidity and Autocorrelations in
    Individual Stock Returns" — reversal is stronger in illiquid stocks
  - Nagel (2012): "Evaporating Liquidity" — short-term reversal is a
    liquidity provision premium

Key insight: Short-term reversal (1-5 day) captures liquidity provision
premium. It is distinct from longer-term mean reversion and has different
risk characteristics.  The premium is strongest in high-volume, high-
volatility periods.

Signal: Cross-sectional rank of negative trailing 5-day returns.
         Buy recent losers, sell recent winners.

Usage:
    strat = ShortTermReversal()
    signals = strat.generate_signals(prices, returns)
    weights = strat.compute_weights(signals, returns=returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class ShortTermReversal(Strategy):
    """
    Short-term reversal (weekly) strategy.

    Parameters
    ----------
    lookback : int
        Trailing return window for reversal signal (default 5 = 1 week).
    holding_period : int
        Days to hold positions (default 5 = 1 week).
    long_pct : float
        Fraction of universe to go long (default 0.20 = bottom 20%).
    short_pct : float
        Fraction of universe to go short (default 0.20 = top 20%).
    target_gross : float
        Target gross exposure (default 1.0).
    vol_scale : bool
        If True, scale by inverse volatility (default True).
    vol_lookback : int
        Window for vol estimation (default 21).
    liquidity_filter : bool
        If True, only trade stocks with above-median volume (default True).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Jegadeesh (1990), Lehmann (1990) — short-term return predictability; "
            "Nagel (2012) — liquidity provision premium"
        ),
        "historical_evidence": (
            "1-week reversal premium averages 3-8% annualized; "
            "stronger during high-volatility periods (Nagel 2012)"
        ),
        "implementation_risks": (
            "High turnover (~100%+ per week), transaction costs critical; "
            "premium concentrates in small/illiquid names"
        ),
        "realistic_expectations": (
            "After transaction costs, net premium is 2-5% annually. "
            "Diversifies well with momentum strategies (negative correlation)."
        ),
    }

    def __init__(
        self,
        lookback: int = 5,
        holding_period: int = 5,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        target_gross: float = 1.0,
        vol_scale: bool = True,
        vol_lookback: int = 21,
        liquidity_filter: bool = True,
    ) -> None:
        params = dict(
            lookback=lookback,
            holding_period=holding_period,
            long_pct=long_pct,
            short_pct=short_pct,
            target_gross=target_gross,
            vol_scale=vol_scale,
            vol_lookback=vol_lookback,
            liquidity_filter=liquidity_filter,
        )
        super().__init__(name="ShortTermReversal", params=params)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Cross-sectional reversal signal: rank negative trailing returns.

        Returns
        -------
        pd.DataFrame
            Signals in [-1, +1], same shape as *prices*.
        """
        lookback: int = self.params["lookback"]
        long_pct: float = self.params["long_pct"]
        short_pct: float = self.params["short_pct"]
        holding: int = self.params["holding_period"]

        # Trailing return over lookback period
        trailing_ret = returns.rolling(lookback, min_periods=max(1, lookback // 2)).sum()

        # Cross-sectional rank (0 to 1, where 1 = highest return)
        ranked = trailing_ret.rank(axis=1, pct=True)

        # Reversal: go long the bottom losers, short the top winners
        n_cols = ranked.shape[1]
        long_thresh = long_pct
        short_thresh = 1 - short_pct

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranked <= long_thresh] = 1.0   # buy recent losers
        signals[ranked >= short_thresh] = -1.0  # sell recent winners

        # Holding period: re-enter every `holding` days, hold for `holding` days
        # Simple approach: only update signals every `holding` days
        dates = signals.index
        mask = pd.Series(False, index=dates)
        for i in range(0, len(dates), holding):
            mask.iloc[i] = True

        # Forward-fill signals for the holding period
        signals_held = signals.where(mask).ffill().fillna(0.0)

        # Mask first lookback rows
        signals_held.iloc[:lookback] = 0.0

        return signals_held

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Volatility-scaled, gross-normalised weights.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        vol_scale: bool = self.params["vol_scale"]
        vol_lookback: int = self.params["vol_lookback"]

        if vol_scale and returns is not None:
            realized_vol = (
                returns.rolling(vol_lookback, min_periods=max(1, vol_lookback // 2))
                .std()
                .mul(np.sqrt(252))
                .clip(lower=0.01)
            )
            raw_weights = signals / realized_vol
        else:
            raw_weights = signals.copy()

        raw_weights = raw_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        gross = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        # Regime scaling (reversal is a liquidity premium — may dry up in crisis)
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.5,  # more tolerant — reversal can work in vol
                floor=0.30,
            )

        return weights

    def run(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> dict:
        """End-to-end: signals → weights → backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
