"""
Volatility Risk Premium (VRP) Strategy
========================================
Captures the variance risk premium: the systematic difference between
implied (expected) and realized volatility.

Academic basis:
  - Carr & Wu (2009): "Variance Risk Premiums" — documented significant
    negative VRP across asset classes
  - Bollerslev, Tauchen & Zhou (2009): "Expected Stock Returns and Variance
    Risk Premia" — VRP predicts aggregate stock returns
  - Todorov (2010): "Variance Risk-Premium Dynamics" — time-varying VRP
  - Moreira & Muir (2017): "Volatility-Managed Portfolios" — scaling
    exposure by inverse volatility captures the VRP

Key insight: The VRP manifests as a premium for bearing volatility risk.
On average, implied vol exceeds realized vol, so systematically selling
volatility (in a controlled way) generates positive returns.

Since we don't have options data, we implement a proxy VRP strategy:
  - GARCH-forecasted vol vs realized vol as a VRP proxy
  - When forecast vol > realized vol: VRP is positive → increase equity
  - When forecast vol < realized vol: VRP is negative → reduce equity

Signal: VRP proxy = (GARCH_forecast_vol - realized_vol) / realized_vol
         Positive VRP → long equity (volatility sellers being compensated)
         Negative VRP → reduce/short equity (vol spike, risk off)

Usage:
    strat = VolRiskPremium()
    signals = strat.generate_signals(prices, returns)
    weights = strat.compute_weights(signals, returns=returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class VolRiskPremium(Strategy):
    """
    Volatility risk premium (VRP) strategy.

    Parameters
    ----------
    realized_window : int
        Window for realized vol calculation (default 21 = 1 month).
    forecast_window : int
        EWMA span for vol forecast (default 63 = 3 months).
    signal_lookback : int
        Window for signal smoothing (default 5).
    target_gross : float
        Target gross exposure (default 1.0).
    vol_lookback : int
        Window for position vol-scaling (default 63).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Carr & Wu (2009) — variance risk premium; "
            "Bollerslev, Tauchen & Zhou (2009) — VRP predicts returns; "
            "Moreira & Muir (2017) — volatility-managed portfolios"
        ),
        "historical_evidence": (
            "VRP averages 2-5% annualized in equities; "
            "Sharpe ratio 0.3-0.7 for VRP-timed equity exposure"
        ),
        "implementation_risks": (
            "Vol explosions (2008, 2020) can cause large losses; "
            "requires robust vol estimation; proxy may be noisy"
        ),
        "realistic_expectations": (
            "Modest standalone returns (2-5%), but excellent diversifier. "
            "Negatively correlated with momentum during vol spikes."
        ),
    }

    def __init__(
        self,
        realized_window: int = 21,
        forecast_window: int = 63,
        signal_lookback: int = 5,
        target_gross: float = 1.0,
        vol_lookback: int = 63,
    ) -> None:
        params = dict(
            realized_window=realized_window,
            forecast_window=forecast_window,
            signal_lookback=signal_lookback,
            target_gross=target_gross,
            vol_lookback=vol_lookback,
        )
        super().__init__(name="VolRiskPremium", params=params)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute VRP proxy signals.

        VRP = (forecast_vol - realized_vol) / realized_vol
          - Positive: implied vol > realized → collect VRP → go long
          - Negative: realized vol spiking → risk off → go flat/short

        Returns
        -------
        pd.DataFrame
            Signals in [-1, +1], same shape as *prices*.
        """
        realized_window: int = self.params["realized_window"]
        forecast_window: int = self.params["forecast_window"]
        signal_lookback: int = self.params["signal_lookback"]

        # Realized vol (short window — recent actual vol)
        realized_vol = (
            returns.rolling(realized_window, min_periods=max(1, realized_window // 2))
            .std()
            .mul(np.sqrt(252))
        )

        # Forecast vol (longer EWMA — smoothed expected vol)
        forecast_vol = returns.ewm(span=forecast_window).std() * np.sqrt(252)

        # VRP proxy
        vrp = (forecast_vol - realized_vol) / realized_vol.clip(lower=0.01)

        # Smooth the signal
        vrp_smooth = vrp.rolling(signal_lookback, min_periods=1).mean()

        # Map to signal: positive VRP → long (up to +1), negative → short (down to -1)
        # Use tanh for smooth bounded mapping
        signals = np.tanh(vrp_smooth * 2)  # scale factor of 2 for sensitivity

        # Mask warmup period
        signals.iloc[:forecast_window] = 0.0

        return signals.fillna(0.0)

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Vol-scaled, gross-normalised weights.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        vol_lookback: int = self.params["vol_lookback"]

        if returns is not None:
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

        # VRP strategy should reduce exposure dramatically in crisis
        # (that's when VRP collapses and vol sellers get crushed)
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.25,  # aggressive reduction
                floor=0.10,
            )

        return weights

    def run(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> dict:
        """End-to-end: signals → weights → backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
