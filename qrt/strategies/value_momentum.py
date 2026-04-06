"""
Price-Based Value Strategy (52-Week High Proximity)
====================================================
Academic basis:
  - George & Hwang (2004): "The 52-Week High and Momentum Profits"
    JF 59(5): 2145-2176
  - Stocks near their 52-week high OUTPERFORM (not revert).
  - The 52-week high ratio subsumes much of Jegadeesh-Titman momentum.
  - Unlike book-to-market, requires NO fundamental data.

  - Li, He & Rouwenhorst (2021): "The 52-Week High and Momentum in
    International Stock Markets" — replicated globally across 20 markets.

Signal:
  ratio_i = price_i / 52-week-high_i  (ranges from 0 to 1)
  Go LONG stocks nearest to their 52-week high (ratio close to 1)
  Go SHORT stocks farthest from their 52-week high (ratio close to 0)

This is a MOMENTUM-VALUE HYBRID:
  - It captures momentum (winners stay near highs)
  - But it's anchored to a reference point (the 52-week high), not raw returns
  - This anchoring reduces crash risk vs. pure momentum (George & Hwang 2004)
  - Negatively correlated with raw STR (-0.3 to -0.5), providing diversification

Parameters (all from George & Hwang 2004):
  - Lookback: 252 trading days (52 weeks)
  - Selection: top/bottom quintile (20%)
  - Holding period: monthly rebalance
  - Vol-scaled positions (standard practice)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class FiftyTwoWeekHigh(Strategy):
    """
    52-Week High proximity strategy (George & Hwang 2004).

    Parameters
    ----------
    lookback : int
        Window for computing 52-week high (default 252).
    long_pct : float
        Fraction of universe to go long — nearest to high (default 0.20).
    short_pct : float
        Fraction of universe to go short — farthest from high (default 0.20).
    holding_period : int
        Days to hold positions before rebalancing (default 21 = monthly).
    target_gross : float
        Target gross exposure (default 1.0).
    vol_scale : bool
        Inverse-volatility weighting (default True).
    vol_lookback : int
        Window for vol estimation (default 63).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "George & Hwang (2004) — '52-Week High and Momentum Profits'; "
            "Li, He & Rouwenhorst (2021) — international replication"
        ),
        "historical_evidence": (
            "Subsumes much of Jegadeesh-Titman momentum; "
            "documented across 20+ international markets"
        ),
        "implementation_risks": (
            "Reduced crash risk vs. pure momentum due to anchoring, "
            "but still correlated with market direction"
        ),
        "realistic_expectations": (
            "Similar return to momentum but with lower crash risk; "
            "provides diversification when combined with short-term reversal"
        ),
    }

    def __init__(
        self,
        lookback: int = 252,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        holding_period: int = 21,
        target_gross: float = 1.0,
        vol_scale: bool = True,
        vol_lookback: int = 63,
    ) -> None:
        params = dict(
            lookback=lookback,
            long_pct=long_pct,
            short_pct=short_pct,
            holding_period=holding_period,
            target_gross=target_gross,
            vol_scale=vol_scale,
            vol_lookback=vol_lookback,
        )
        super().__init__(name="FiftyTwoWeekHigh", params=params)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute 52-week high proximity signal.

        ratio_i = close_i / max(close over past 252 days)

        Long stocks with ratio near 1 (near their high).
        Short stocks with ratio near 0 (far from their high).
        """
        lookback = self.params["lookback"]
        long_pct = self.params["long_pct"]
        short_pct = self.params["short_pct"]
        holding = self.params["holding_period"]

        # 52-week high for each stock
        rolling_high = prices.rolling(lookback, min_periods=lookback // 2).max()

        # Ratio: current price / 52-week high (0 to 1)
        ratio = prices / rolling_high

        # Cross-sectional rank each day
        ranked = ratio.rank(axis=1, pct=True)

        # Long: stocks nearest to their 52-week high (top quintile)
        # Short: stocks farthest from high (bottom quintile)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranked >= (1 - long_pct)] = 1.0
        signals[ranked <= short_pct] = -1.0

        # Hold for `holding_period` days (monthly rebalance)
        dates = signals.index
        mask = pd.Series(False, index=dates)
        for i in range(0, len(dates), holding):
            mask.iloc[i] = True

        signals_held = signals.where(mask).ffill().fillna(0.0)
        signals_held.iloc[:lookback] = 0.0

        return signals_held

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Vol-scaled, gross-normalized weights."""
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross = self.params["target_gross"]
        vol_scale = self.params["vol_scale"]
        vol_lookback = self.params["vol_lookback"]

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

        return weights

    def run(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> dict:
        """End-to-end: signals -> weights -> backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
