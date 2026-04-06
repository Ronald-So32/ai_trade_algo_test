"""
Residual Short-Term Reversal Strategy
=======================================
Academic basis:
  - Blitz, Huij, Lansdorp & Verbeek (2013): "Short-Term Residual Reversal"
    Journal of Financial Markets 16(3): 477-504
    — Sorting on Fama-French residuals instead of raw returns doubles the
      Sharpe ratio (1.28 vs 0.62) by removing dynamic factor exposures.

  - Blitz, van der Grient & Honarvar (2023): "Reversing the Trend of
    Short-Term Reversal" — Journal of Portfolio Management
    — Within-industry reversal (t=5.49) delivers 1.5x standard reversal.
    — Counteracts contamination from industry/factor momentum.

  - Da, Liu & Schaumburg (2014): "A Closer Look at the Short-Term Return
    Reversal" — Management Science 60(3): 658-674
    — Separating fundamental from non-fundamental returns yields 4x
      risk-adjusted improvement.

Signal construction:
  1. Compute each stock's trailing 5-day return
  2. Subtract the stock's sector mean return (industry-neutral)
  3. Subtract the market-wide mean return (market-neutral)
  4. Sort on these residual returns cross-sectionally
  5. Long bottom quintile (residual losers), short top quintile (residual winners)

This is equivalent to within-industry, market-neutral reversal — the most
robust documented variant of STR. No additional free parameters vs. standard
STR; the only change is what we sort on (residuals vs. raw returns).

Parameters (all from Jegadeesh 1990 / Blitz et al. 2013):
  - Lookback: 5 trading days (1 week)
  - Selection: top/bottom quintile (20%)
  - Holding period: 5 days (1 week)
  - Vol-scaled positions (standard practice)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class ResidualReversal(Strategy):
    """
    Residual short-term reversal (Blitz et al. 2013, 2023).

    Sorts on market- and industry-neutral residual returns rather than
    raw returns. This strips out factor momentum contamination and
    isolates the pure liquidity provision premium.

    Parameters
    ----------
    lookback : int
        Trailing return window (default 5 = 1 week).
    holding_period : int
        Days to hold positions (default 5 = 1 week).
    long_pct : float
        Fraction of universe to go long (default 0.20).
    short_pct : float
        Fraction of universe to go short (default 0.20).
    target_gross : float
        Target gross exposure (default 1.0).
    vol_scale : bool
        Inverse-volatility weighting (default True).
    vol_lookback : int
        Window for vol estimation (default 21).
    sector_map : dict, optional
        {security_id: sector_name} for industry-neutral residuals.
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Blitz, Huij, Lansdorp & Verbeek (2013) — residual reversal doubles Sharpe; "
            "Blitz, van der Grient & Honarvar (2023) — within-industry reversal, t=5.49"
        ),
        "historical_evidence": (
            "Residual reversal Sharpe 1.28 vs 0.62 for conventional; "
            "profits significant in large-caps post-1990"
        ),
        "implementation_risks": (
            "High turnover (~100%+ per week); requires sector classification; "
            "premium linked to liquidity provision (Nagel 2012)"
        ),
        "realistic_expectations": (
            "Documented 2x improvement over raw reversal; "
            "strips out industry momentum contamination that weakens standard STR"
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
        sector_map: dict | None = None,
    ) -> None:
        params = dict(
            lookback=lookback,
            holding_period=holding_period,
            long_pct=long_pct,
            short_pct=short_pct,
            target_gross=target_gross,
            vol_scale=vol_scale,
            vol_lookback=vol_lookback,
        )
        super().__init__(name="ResidualReversal", params=params)
        self._sector_map = sector_map or {}

    def _compute_residual_returns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute market- and industry-neutral residual returns.

        For each stock on each day:
          residual = stock_return - sector_mean_return

        This simultaneously removes market exposure (since all sectors
        share the market factor) and industry exposure. Equivalent to
        Blitz et al. (2023) within-industry reversal.

        If no sector_map is provided, falls back to market-neutral only
        (subtract cross-sectional mean).
        """
        lookback = self.params["lookback"]

        # Trailing returns over lookback period
        trailing = returns.rolling(lookback, min_periods=max(1, lookback // 2)).sum()

        if not self._sector_map:
            # Market-neutral only: subtract cross-sectional mean
            market_mean = trailing.mean(axis=1)
            residual = trailing.sub(market_mean, axis=0)
            return residual

        # Industry-neutral: subtract sector mean for each stock
        # Build sector groups
        sectors = {}
        for col in trailing.columns:
            sector = self._sector_map.get(col, "unknown")
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(col)

        residual = trailing.copy()
        for sector, cols in sectors.items():
            if len(cols) < 2:
                # Single stock in sector — just subtract market mean
                market_mean = trailing.mean(axis=1)
                residual[cols] = trailing[cols].sub(market_mean, axis=0)
            else:
                # Subtract sector mean from each stock in the sector
                sector_mean = trailing[cols].mean(axis=1)
                residual[cols] = trailing[cols].sub(sector_mean, axis=0)

        return residual

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Residual reversal signal: rank negative residual trailing returns.

        Unlike standard STR which sorts on raw returns, this sorts on
        market- and industry-neutral residuals, per Blitz et al. (2013, 2023).
        """
        long_pct = self.params["long_pct"]
        short_pct = self.params["short_pct"]
        holding = self.params["holding_period"]
        lookback = self.params["lookback"]

        # Compute residual returns (industry-neutral if sector_map provided)
        residual = self._compute_residual_returns(returns)

        # Cross-sectional rank (0 = lowest residual, 1 = highest)
        ranked = residual.rank(axis=1, pct=True)

        # Reversal: long residual losers (bottom quintile),
        #           short residual winners (top quintile)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[ranked <= long_pct] = 1.0       # buy residual losers
        signals[ranked >= (1 - short_pct)] = -1.0  # sell residual winners

        # Holding period: rebalance every `holding` days
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
