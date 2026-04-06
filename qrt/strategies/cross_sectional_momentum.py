"""
Cross-Sectional Momentum Strategy
===================================
Rank assets by trailing return (skipping last month to avoid short-term
reversal), go long the top quintile, short the bottom quintile, z-score
weighted within each leg.  Includes a drawdown circuit breaker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class CrossSectionalMomentum(Strategy):
    """
    Cross-sectional (relative) momentum strategy.

    Parameters
    ----------
    lookback : int
        Total momentum lookback window in days (default 252).
    skip_days : int
        Most-recent days to skip to avoid short-term reversal (default 30).
    decile : float
        Fraction of the universe forming each leg, e.g. 0.20 -> top/bottom
        20 % (default 0.20).
    target_gross : float
        Target gross exposure (default 1.0, i.e. 0.5 long / 0.5 short).
    dd_threshold : float
        Maximum drawdown before circuit breaker triggers (default 0.25).
    dd_reduction : float
        Exposure reduction factor when circuit breaker is active (default 0.50).
    dd_cooldown : int
        Number of days to keep reduced exposure after breaker triggers (default 21).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Jegadeesh & Titman (1993), Carhart (1997) four-factor model"
        ),
        "historical_evidence": (
            "One of the most robust anomalies; documented globally across equity markets"
        ),
        "implementation_risks": (
            "Crash risk (Daniel & Moskowitz 2016), short-leg performance post-2000, crowding"
        ),
        "realistic_expectations": (
            "Historically strong return premium; subject to severe drawdowns during reversals"
        ),
    }

    def __init__(
        self,
        lookback: int = 252,
        skip_days: int = 30,
        decile: float = 0.20,
        target_gross: float = 1.0,
        dd_threshold: float = 0.25,
        dd_reduction: float = 0.50,
        dd_cooldown: int = 21,
    ) -> None:
        params = dict(
            lookback=lookback,
            skip_days=skip_days,
            decile=decile,
            target_gross=target_gross,
            dd_threshold=dd_threshold,
            dd_reduction=dd_reduction,
            dd_cooldown=dd_cooldown,
        )
        super().__init__(name="CrossSectionalMomentum", params=params)

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
        Compute cross-sectional momentum signals with z-score weighting.

        For each date t, the momentum score is:
            mom_score_i = P(t - skip) / P(t - lookback) - 1

        Assets in the top *decile* receive positive z-score-based signal;
        assets in the bottom *decile* receive negative z-score-based signal;
        all others receive 0.

        Returns
        -------
        pd.DataFrame
            Signals weighted by cross-sectional z-score, same shape as *prices*.
        """
        lookback: int = self.params["lookback"]
        skip: int = self.params["skip_days"]
        decile: float = self.params["decile"]

        # Momentum return: from (t - lookback) to (t - skip)
        mom_return: pd.DataFrame = prices.shift(skip) / prices.shift(lookback) - 1

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date in prices.index:
            row = mom_return.loc[date].dropna()
            if len(row) < max(3, int(1 / decile)):  # need enough assets to rank
                continue
            n_select = max(1, int(np.floor(len(row) * decile)))
            ranked = row.rank(ascending=True)

            # Cross-sectional z-score of momentum returns
            cs_mean = row.mean()
            cs_std = row.std()
            if cs_std < 1e-12:
                continue
            z_scores = (row - cs_mean) / cs_std

            # Top quintile: highest ranks -> long
            long_mask = ranked >= (ranked.max() - n_select + 1)
            # Bottom quintile: lowest ranks -> short
            short_mask = ranked <= n_select

            # Z-score weighted within each leg, normalized so each leg sums to 1
            long_assets = long_mask[long_mask].index
            short_assets = short_mask[short_mask].index

            if len(long_assets) > 0:
                long_z = z_scores[long_assets].abs()
                long_z_sum = long_z.sum()
                if long_z_sum > 0:
                    signals.loc[date, long_assets] = long_z / long_z_sum
                else:
                    signals.loc[date, long_assets] = 1.0 / len(long_assets)

            if len(short_assets) > 0:
                short_z = z_scores[short_assets].abs()
                short_z_sum = short_z.sum()
                if short_z_sum > 0:
                    signals.loc[date, short_assets] = -(short_z / short_z_sum)
                else:
                    signals.loc[date, short_assets] = -1.0 / len(short_assets)

        # Mask rows without sufficient history
        min_history = lookback + skip
        signals.iloc[:min_history] = 0.0

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Z-score weighted within long and short legs, then scale to target gross.
        Applies drawdown circuit breaker to reduce exposure after large drawdowns.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals``.
        returns : pd.DataFrame, optional
            Daily returns, used for drawdown circuit breaker calculation.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        dd_threshold: float = self.params["dd_threshold"]
        dd_reduction: float = self.params["dd_reduction"]
        dd_cooldown: int = self.params["dd_cooldown"]

        # Signals already carry z-score weights within each leg (summing to ~1)
        # Scale long leg to target_gross/2, short leg to -target_gross/2
        long_sum = signals.clip(lower=0).sum(axis=1).replace(0, np.nan)
        short_sum = signals.clip(upper=0).abs().sum(axis=1).replace(0, np.nan)

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for col in signals.columns:
            is_long = signals[col] > 0
            is_short = signals[col] < 0
            weights.loc[is_long, col] = (
                signals.loc[is_long, col] / long_sum[is_long] * (target_gross / 2.0)
            )
            weights.loc[is_short, col] = (
                signals.loc[is_short, col] / short_sum[is_short] * (target_gross / 2.0)
            )

        weights = weights.fillna(0.0)

        # Drawdown circuit breaker
        if returns is not None:
            # Compute strategy returns from weights and asset returns
            strat_returns = (weights.shift(1) * returns).sum(axis=1)
            cum_returns = (1 + strat_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max

            # Track when breaker is active: if DD exceeds threshold, stay
            # reduced for dd_cooldown days
            breaker_active = pd.Series(False, index=signals.index)
            days_since_trigger = dd_cooldown + 1  # start inactive

            for i in range(len(drawdown)):
                if drawdown.iloc[i] < -dd_threshold:
                    days_since_trigger = 0
                if days_since_trigger <= dd_cooldown:
                    breaker_active.iloc[i] = True
                days_since_trigger += 1

            # Apply reduction
            scale = pd.Series(1.0, index=signals.index)
            scale[breaker_active] = 1.0 - dd_reduction
            weights = weights.mul(scale, axis=0)

        # HMM momentum crash gate (Daniel, Jagannathan & Kim 2019):
        # Cross-sectional momentum is even more crash-prone than time-series
        # momentum (Daniel & Moskowitz 2016).  Scale exposure by crisis prob.
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.25,  # more aggressive gate for CS momentum
                floor=0.10,       # deeper cut — CS momentum crashes harder
            )

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
