"""
Carry Strategy
===============
Use dividend yield (or a proxy derived from total vs price return) as the
carry signal.  Rank assets by carry and go long high-carry / short low-carry.

Input note
----------
``generate_signals`` accepts an optional ``dividend_yields`` keyword argument
(pd.DataFrame, same shape as prices).  If not provided the strategy infers a
carry proxy as:

    carry_proxy = total_return - price_return

which equals the dividend component when total-return and price-return series
are available.  If neither is available, the carry is approximated by the
rolling average of excess return over a benchmark (mean cross-sectional return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class CarryStrategy(Strategy):
    """
    Cross-sectional carry strategy.

    Parameters
    ----------
    carry_lookback : int
        Rolling window (in days) to smooth noisy carry signals (default 63).
    n_quantile : float
        Fraction of universe in each long/short leg (default 0.25).
    target_gross : float
        Target gross exposure (default 1.0).
    min_assets : int
        Minimum number of valid assets required to trade on a given day
        (default 4).
    rebalance_freq : int
        Number of days between rebalances (default 21).  On non-rebalance
        days the previous weights are carried forward.
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Koijen et al. (2018) — \"Carry\" across asset classes"
        ),
        "historical_evidence": (
            "Persistent premium in FX, rates, commodities; "
            "equity carry via dividends weaker"
        ),
        "implementation_risks": (
            "Carry unwind in risk-off events, dividend cuts, sector concentration"
        ),
        "realistic_expectations": (
            "Research-supported low-Sharpe premium; equity dividend carry is "
            "noisier than FX carry"
        ),
    }

    def __init__(
        self,
        carry_lookback: int = 63,
        n_quantile: float = 0.25,
        target_gross: float = 1.0,
        min_assets: int = 4,
        rebalance_freq: int = 21,
    ) -> None:
        params = dict(
            carry_lookback=carry_lookback,
            n_quantile=n_quantile,
            target_gross=target_gross,
            min_assets=min_assets,
            rebalance_freq=rebalance_freq,
        )
        super().__init__(name="CarryStrategy", params=params)

    # ------------------------------------------------------------------
    # Carry proxy computation
    # ------------------------------------------------------------------

    def _compute_carry(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        dividend_yields: pd.DataFrame | None,
        total_returns: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of carry scores (same shape as prices).

        Priority:
        1. Explicit ``dividend_yields`` (used directly).
        2. ``total_returns`` provided → carry = total_ret - price_ret.
        3. Fallback: rolling mean excess return over cross-sectional average.
        """
        lookback: int = self.params["carry_lookback"]

        if dividend_yields is not None:
            # Smooth over lookback window
            carry = dividend_yields.rolling(lookback, min_periods=1).mean()
            return carry.reindex_like(prices)

        if total_returns is not None:
            # Infer carry component
            dividend_component = total_returns - returns  # excess return from divs
            carry = dividend_component.rolling(lookback, min_periods=1).mean()
            return carry.reindex_like(prices)

        # Fallback: rolling average of excess return vs cross-section mean
        xs_mean = returns.mean(axis=1)
        excess = returns.sub(xs_mean, axis=0)
        carry = excess.rolling(lookback, min_periods=max(1, lookback // 4)).mean()
        return carry

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
        Rank assets by carry and assign +1 (long) / -1 (short) / 0 signals.

        Keyword Args
        ------------
        dividend_yields : pd.DataFrame, optional
            Annualised dividend yield, same shape as prices.
        total_returns : pd.DataFrame, optional
            Total-return index, same shape as prices.

        Returns
        -------
        pd.DataFrame
            Signals in {-1, 0, +1}, same shape as *prices*.
        """
        dividend_yields: pd.DataFrame | None = kwargs.get("dividend_yields", None)
        total_returns: pd.DataFrame | None = kwargs.get("total_returns", None)

        n_quantile: float = self.params["n_quantile"]
        min_assets: int = self.params["min_assets"]
        rebal_freq: int = self.params["rebalance_freq"]

        carry = self._compute_carry(prices, returns, dividend_yields, total_returns)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        last_signal: pd.Series | None = None
        last_rebal: int = -rebal_freq  # force rebalance at first opportunity

        for i, date in enumerate(prices.index):
            # Carry-forward between rebalances
            if i - last_rebal < rebal_freq and last_signal is not None:
                signals.loc[date] = last_signal
                continue

            row = carry.loc[date].dropna()
            if len(row) < min_assets:
                if last_signal is not None:
                    signals.loc[date] = last_signal
                continue

            n_select = max(1, int(np.floor(len(row) * n_quantile)))
            ranked = row.rank(ascending=True)

            new_signal = pd.Series(0.0, index=prices.columns)
            new_signal[ranked >= ranked.max() - n_select + 1] = 1.0   # long high carry
            new_signal[ranked <= n_select] = -1.0                       # short low carry

            signals.loc[date] = new_signal
            last_signal = new_signal.copy()
            last_rebal = i

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Equal-weight within legs, normalised to target gross.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        target_gross: float = self.params["target_gross"]

        # Count active positions per day per side
        long_counts = (signals > 0).sum(axis=1).replace(0, np.nan)
        short_counts = (signals < 0).sum(axis=1).replace(0, np.nan)

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        long_w = (target_gross / 2.0) / long_counts
        short_w = (target_gross / 2.0) / short_counts

        for col in signals.columns:
            is_long = signals[col] > 0
            is_short = signals[col] < 0
            weights.loc[is_long, col] = long_w[is_long]
            weights.loc[is_short, col] = -short_w[is_short]

        return weights.fillna(0.0)

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
        weights = self.compute_weights(signals, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
