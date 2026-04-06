"""
Time Series Momentum Strategy
==============================
For each asset, signal = blended multi-scale momentum (63d/126d/252d).
Volatility-scale positions: weight = signal / realized_vol.
Normalize to target gross exposure.
Includes trend-strength scaling and vol-of-vol position reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class TimeSeriesMomentum(Strategy):
    """
    Time-series (absolute) momentum strategy.

    Parameters
    ----------
    lookback : int
        Primary trailing window for momentum signal computation (default 126).
    vol_lookback : int
        Trailing window for realized-volatility estimation (default 63).
    target_gross : float
        Target gross exposure to normalize weights to (default 1.0).
    vol_floor : float
        Floor on realized volatility to prevent blow-up (default 0.01).
    multi_scale_weights : tuple[float, float, float]
        Weights for 63d / 126d / 252d momentum signals (default (0.4, 0.4, 0.2)).
    trend_strength_cap : float
        Denominator for trend-strength scaling: signal is scaled by
        min(1, abs(trailing_return) / trend_strength_cap) (default 0.20).
    vov_reduction : float
        Factor to reduce position when vol-of-vol is elevated (default 0.50).
    vov_lookback : int
        Lookback for computing vol-of-vol (default 63).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Moskowitz, Ooi & Pedersen (2012) — \"Time Series Momentum\""
        ),
        "historical_evidence": (
            "Documented across asset classes, 1-12 month lookbacks; "
            "average annual excess return 2-8%"
        ),
        "implementation_risks": (
            "Momentum crashes (sudden reversals), crowding, "
            "transaction costs from rebalancing"
        ),
        "realistic_expectations": (
            "Research-supported premium with significant tail risk; "
            "momentum crashes can be severe"
        ),
    }

    def __init__(
        self,
        lookback: int = 126,
        vol_lookback: int = 63,
        target_gross: float = 1.0,
        vol_floor: float = 0.01,
        multi_scale_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        trend_strength_cap: float = 0.20,
        vov_reduction: float = 0.50,
        vov_lookback: int = 63,
    ) -> None:
        params = dict(
            lookback=lookback,
            vol_lookback=vol_lookback,
            target_gross=target_gross,
            vol_floor=vol_floor,
            multi_scale_weights=multi_scale_weights,
            trend_strength_cap=trend_strength_cap,
            vov_reduction=vov_reduction,
            vov_lookback=vov_lookback,
        )
        super().__init__(name="TimeSeriesMomentum", params=params)

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
        Compute multi-scale blended momentum signal with trend-strength scaling.

        The signal blends 63d, 126d, and 252d trailing returns using configurable
        weights, then scales by trend strength so weak trends produce smaller
        positions.

        Returns
        -------
        pd.DataFrame
            Signal values in [-1, +1], same shape as *prices*.
        """
        msw = self.params["multi_scale_weights"]
        trend_cap: float = self.params["trend_strength_cap"]

        # Trailing returns at three scales
        scales = [63, 126, 252]
        trailing_returns = {}
        for s in scales:
            trailing_returns[s] = prices / prices.shift(s) - 1

        # Blend the sign signals with multi-scale weights
        blended_signal: pd.DataFrame = (
            np.sign(trailing_returns[63]) * msw[0]
            + np.sign(trailing_returns[126]) * msw[1]
            + np.sign(trailing_returns[252]) * msw[2]
        )

        # Trend-strength scaling: use primary lookback return magnitude
        lookback: int = self.params["lookback"]
        primary_return = prices / prices.shift(lookback) - 1
        strength = (primary_return.abs() / trend_cap).clip(upper=1.0)

        signals = blended_signal * strength

        # Mask rows where we don't yet have a full 252-day window
        signals.iloc[:max(scales)] = np.nan

        return signals.fillna(0.0)

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert signals to volatility-scaled, gross-normalised weights.

        Weight_i = signal_i / realized_vol_i  (then rescaled to target gross).
        When vol-of-vol is elevated (above median), positions are reduced.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals``.
        returns : pd.DataFrame, optional
            Daily returns used to estimate realized volatility.  If omitted,
            ``kwargs['returns']`` is tried, then a flat vol of 1 is assumed.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        vol_lookback: int = self.params["vol_lookback"]
        target_gross: float = self.params["target_gross"]
        vol_floor: float = self.params["vol_floor"]
        vov_reduction: float = self.params["vov_reduction"]
        vov_lookback: int = self.params["vov_lookback"]

        if returns is not None:
            # Annualised realized volatility
            realized_vol: pd.DataFrame = (
                returns.rolling(vol_lookback, min_periods=max(1, vol_lookback // 2))
                .std()
                .mul(np.sqrt(252))
                .clip(lower=vol_floor)
            )

            # Vol-of-vol scaling: rolling std of rolling vol
            vol_of_vol: pd.DataFrame = realized_vol.rolling(
                vov_lookback, min_periods=max(1, vov_lookback // 2)
            ).std()
            # Median vol-of-vol across time for each asset
            vov_median: pd.Series = vol_of_vol.median(axis=0)
            # Where vol-of-vol exceeds its median, apply reduction
            vov_scale: pd.DataFrame = pd.DataFrame(
                1.0, index=signals.index, columns=signals.columns
            )
            elevated = vol_of_vol.gt(vov_median, axis=1)
            vov_scale[elevated] = 1.0 - vov_reduction
        else:
            # Fallback: equal vol scaling
            realized_vol = pd.DataFrame(
                np.ones_like(signals.values), index=signals.index, columns=signals.columns
            )
            vov_scale = pd.DataFrame(
                1.0, index=signals.index, columns=signals.columns
            )

        # Raw weights: signal / vol, with vol-of-vol adjustment
        raw_weights: pd.DataFrame = (signals / realized_vol) * vov_scale
        raw_weights = raw_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Normalise to target gross exposure each day
        gross: pd.Series = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights: pd.DataFrame = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        # HMM momentum crash gate (Daniel, Jagannathan & Kim 2019):
        # Scale exposure down as crisis probability rises.  This is the
        # single most impactful HMM application for momentum — documented
        # to roughly double the Sharpe ratio by avoiding crash drawdowns.
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.3,  # start reducing at 30% crisis probability
                floor=0.15,      # minimum 15% exposure in full crisis
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
        """
        End-to-end: signals -> weights -> backtest summary.

        Returns
        -------
        dict with keys 'signals', 'weights', 'summary'.
        """
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
