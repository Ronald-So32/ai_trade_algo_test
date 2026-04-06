"""
Leveraged Position Sizing with Risk Controls
==============================================
Implements a risk-controlled leverage system that sizes leveraged trades
based on signal confidence/probability, with strict per-trade risk limits.

Academic basis:
  - Kelly (1956): "A New Interpretation of Information Rate" — optimal leverage
  - Thorp (2006): "The Kelly Criterion in Blackjack, Sports Betting and the
    Stock Market" — practical Kelly implementation
  - MacLean, Thorp & Ziemba (2011): "Good and Bad Properties of the Kelly
    Criterion" — fractional Kelly for risk management
  - Moreira & Muir (2017): "Volatility-Managed Portfolios" — dynamic leverage
    based on volatility
  - Grossman & Zhou (1993): "Optimal Investment Strategies for Controlling
    Drawdowns" — max drawdown constraints with leverage

Design principles:
  1. Max 2% of total balance at risk per trade (absolute risk cap)
  2. Max 1:10 leverage (10x)
  3. Leverage amount chosen by probability/confidence of success
  4. Dynamic leverage reduction in high-vol/crisis regimes
  5. Portfolio-level leverage constraint (max total leverage)

The confidence-to-leverage mapping uses half-Kelly:
  leverage = min(max_lev, 0.5 * (p * b - (1-p)) / (b * risk_per_trade))
where p = probability of success, b = payoff ratio

Usage:
    mgr = LeverageManager(max_risk_per_trade=0.02, max_leverage=10.0)
    sized = mgr.size_trade(signal_strength=0.7, volatility=0.20, price=150.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeveragedPosition:
    """Result of leverage sizing for a single trade."""
    leverage: float         # Actual leverage applied (1x to max_leverage)
    position_size: float    # Fraction of portfolio to allocate
    risk_amount: float      # Dollar risk (should be <= max_risk_per_trade * portfolio)
    stop_loss_pct: float    # Stop-loss distance as fraction
    confidence: float       # Estimated probability of success


class LeverageManager:
    """
    Risk-controlled leverage manager.

    Parameters
    ----------
    max_risk_per_trade : float
        Maximum fraction of portfolio at risk per trade (default 0.02 = 2%).
    max_leverage : float
        Maximum leverage multiplier (default 10.0 = 10x).
    min_leverage : float
        Minimum leverage for any trade (default 1.0 = no leverage).
    kelly_fraction : float
        Fraction of Kelly-optimal leverage to use (default 0.5 = half-Kelly).
        Research shows half-Kelly provides ~75% of full-Kelly growth with
        significantly lower variance and drawdown risk.
    vol_scaling : bool
        If True, reduce leverage in high-vol environments (default True).
    target_vol : float
        Target portfolio volatility used for vol-scaling (default 0.10).
    regime_aware : bool
        If True, further reduce leverage during crisis regimes (default True).
    max_portfolio_leverage : float
        Maximum total portfolio leverage across all positions (default 3.0).
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_leverage: float = 10.0,
        min_leverage: float = 1.0,
        kelly_fraction: float = 0.50,
        vol_scaling: bool = True,
        target_vol: float = 0.10,
        regime_aware: bool = True,
        max_portfolio_leverage: float = 3.0,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.kelly_fraction = kelly_fraction
        self.vol_scaling = vol_scaling
        self.target_vol = target_vol
        self.regime_aware = regime_aware
        self.max_portfolio_leverage = max_portfolio_leverage

    def compute_confidence(
        self,
        signal_strength: float,
        hit_rate: float = 0.55,
        strategy_sharpe: float = 0.5,
    ) -> float:
        """
        Estimate trade success probability from signal characteristics.

        Combines:
          - Signal strength (how extreme the signal is)
          - Historical hit rate (win rate of the strategy)
          - Strategy Sharpe ratio (risk-adjusted quality)

        Returns probability in [0.5, 0.95].
        """
        # Base probability from hit rate
        base_p = np.clip(hit_rate, 0.45, 0.70)

        # Signal strength adjustment: stronger signals → higher confidence
        # Map |signal| from [0, 1] to [0, 0.15] probability boost
        signal_boost = np.clip(abs(signal_strength), 0, 1) * 0.15

        # Sharpe quality adjustment
        sharpe_boost = np.clip(strategy_sharpe / 4.0, 0, 0.10)

        confidence = base_p + signal_boost + sharpe_boost
        return np.clip(confidence, 0.50, 0.95)

    def kelly_leverage(
        self,
        confidence: float,
        payoff_ratio: float = 1.5,
    ) -> float:
        """
        Compute Kelly-optimal leverage given probability and payoff.

        Kelly formula for asymmetric payoff:
            f* = (p * b - (1-p)) / b

        where p = probability of win, b = win/loss ratio.
        We use fractional Kelly for safety.

        Parameters
        ----------
        confidence : float
            Probability of trade success (0.5 to 0.95).
        payoff_ratio : float
            Expected win / expected loss ratio (default 1.5).

        Returns
        -------
        float
            Recommended leverage multiplier.
        """
        p = confidence
        b = payoff_ratio

        # Kelly fraction
        f_star = (p * b - (1 - p)) / b

        if f_star <= 0:
            return 0.0  # Negative edge — don't trade

        # Apply fractional Kelly
        f = self.kelly_fraction * f_star

        # Convert to leverage: f represents fraction of capital to risk
        # Leverage = f / risk_per_trade
        leverage = f / self.max_risk_per_trade

        return np.clip(leverage, self.min_leverage, self.max_leverage)

    def size_trade(
        self,
        signal_strength: float,
        asset_volatility: float,
        current_drawdown: float = 0.0,
        crisis_prob: float = 0.0,
        hit_rate: float = 0.55,
        strategy_sharpe: float = 0.5,
        payoff_ratio: float = 1.5,
    ) -> LeveragedPosition:
        """
        Size a single leveraged trade.

        Parameters
        ----------
        signal_strength : float
            Raw signal value, typically in [-1, 1].
        asset_volatility : float
            Annualized volatility of the asset.
        current_drawdown : float
            Current portfolio drawdown (negative value, e.g. -0.05).
        crisis_prob : float
            HMM crisis probability (0 to 1).
        hit_rate : float
            Historical win rate of the strategy.
        strategy_sharpe : float
            Historical Sharpe ratio of the strategy.
        payoff_ratio : float
            Expected win/loss ratio.

        Returns
        -------
        LeveragedPosition
            Sized position with leverage, risk amount, and stop-loss.
        """
        confidence = self.compute_confidence(
            signal_strength, hit_rate, strategy_sharpe
        )

        # Kelly-optimal leverage
        leverage = self.kelly_leverage(confidence, payoff_ratio)

        if leverage <= 0:
            return LeveragedPosition(
                leverage=0.0, position_size=0.0, risk_amount=0.0,
                stop_loss_pct=0.0, confidence=confidence,
            )

        # Volatility scaling: reduce leverage when vol is high
        if self.vol_scaling and asset_volatility > 0:
            vol_ratio = self.target_vol / max(asset_volatility, 0.01)
            vol_scale = np.clip(vol_ratio, 0.2, 2.0)
            leverage *= vol_scale

        # Regime adjustment: reduce leverage in crisis
        if self.regime_aware and crisis_prob > 0.3:
            regime_scale = 1.0 - 0.7 * min(1.0, (crisis_prob - 0.3) / 0.7)
            leverage *= regime_scale

        # Drawdown adjustment: reduce leverage during drawdowns
        # Per Grossman & Zhou (1993)
        if current_drawdown < -0.05:
            dd_scale = max(0.3, 1.0 + current_drawdown * 3)  # linear reduction
            leverage *= dd_scale

        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)

        # Stop-loss based on volatility (2x daily vol)
        daily_vol = asset_volatility / np.sqrt(252)
        stop_loss_pct = max(2 * daily_vol, 0.005)  # min 0.5% stop

        # Position size: limited by max_risk_per_trade
        # Risk = position_size * leverage * stop_loss_pct = max_risk_per_trade
        position_size = self.max_risk_per_trade / (leverage * stop_loss_pct)
        position_size = min(position_size, 1.0 / leverage)  # can't exceed 100%

        risk_amount = position_size * leverage * stop_loss_pct

        return LeveragedPosition(
            leverage=round(leverage, 2),
            position_size=round(position_size, 6),
            risk_amount=round(risk_amount, 6),
            stop_loss_pct=round(stop_loss_pct, 6),
            confidence=round(confidence, 4),
        )

    def apply_portfolio_leverage(
        self,
        weights: pd.DataFrame,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        crisis_probs: Optional[pd.Series] = None,
        garch_vols: Optional[pd.DataFrame] = None,
        strategy_sharpes: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Apply dynamic leverage to portfolio weights based on signal strength,
        volatility forecast, and regime state.

        This is the main integration point with the strategy pipeline.

        Parameters
        ----------
        weights : pd.DataFrame
            Base strategy weights (dates x assets), normalized to gross=1.
        signals : pd.DataFrame
            Raw signal values (dates x assets).
        returns : pd.DataFrame
            Historical returns for volatility estimation.
        crisis_probs : pd.Series, optional
            HMM crisis probabilities per date.
        garch_vols : pd.DataFrame, optional
            GARCH volatility forecasts (dates x assets).
        strategy_sharpes : dict, optional
            Historical Sharpe ratios per strategy for confidence estimation.

        Returns
        -------
        pd.DataFrame
            Leveraged weights (may have gross exposure > 1).
        """
        leveraged = weights.copy()

        # Compute volatility for each asset
        if garch_vols is not None:
            vol_est = garch_vols
        else:
            vol_est = returns.rolling(63, min_periods=21).std() * np.sqrt(252)

        # Compute rolling drawdown
        port_ret = (weights.shift(1) * returns).sum(axis=1)
        cum_ret = (1 + port_ret).cumprod()
        drawdown = cum_ret / cum_ret.cummax() - 1

        # Apply leverage per-date
        for t in range(len(weights)):
            date = weights.index[t]
            sig_row = signals.iloc[t] if t < len(signals) else pd.Series(0, index=weights.columns)
            vol_row = vol_est.iloc[t] if t < len(vol_est) else pd.Series(0.2, index=weights.columns)
            dd = drawdown.iloc[t] if t < len(drawdown) else 0.0
            cp = float(crisis_probs.iloc[t]) if crisis_probs is not None and t < len(crisis_probs) else 0.0

            # Average signal strength for leverage decision
            avg_signal = abs(sig_row).mean()
            avg_vol = vol_row.mean()
            avg_sharpe = np.mean(list(strategy_sharpes.values())) if strategy_sharpes else 0.5

            # Size the overall portfolio leverage
            trade = self.size_trade(
                signal_strength=avg_signal,
                asset_volatility=avg_vol if avg_vol > 0 else 0.2,
                current_drawdown=dd,
                crisis_prob=cp,
                strategy_sharpe=avg_sharpe,
            )

            leverage_mult = trade.leverage
            # Apply portfolio-level leverage cap
            current_gross = leveraged.iloc[t].abs().sum()
            if current_gross * leverage_mult > self.max_portfolio_leverage:
                leverage_mult = self.max_portfolio_leverage / max(current_gross, 0.01)

            leveraged.iloc[t] = weights.iloc[t] * leverage_mult

        # Enforce max risk per position
        max_pos = self.max_risk_per_trade * self.max_leverage
        leveraged = leveraged.clip(-max_pos, max_pos)

        logger.info(
            "Applied leverage: avg=%.2fx, max=%.2fx, gross_avg=%.2f",
            leveraged.abs().sum(axis=1).mean(),
            leveraged.abs().sum(axis=1).max(),
            leveraged.abs().sum(axis=1).mean(),
        )

        return leveraged

    def compute_dynamic_leverage(
        self,
        returns: pd.Series,
        crisis_probs: Optional[pd.Series] = None,
        avg_sharpe: float = 0.5,
        garch_vol: Optional[float] = None,
    ) -> pd.Series:
        """
        Compute time-varying leverage multiplier for portfolio returns.

        Uses rolling drawdown, regime state, and volatility to dynamically
        adjust leverage — de-leveraging aggressively during drawdowns and
        re-leveraging when conditions improve.

        This implements the Grossman & Zhou (1993) insight: optimal leverage
        under drawdown constraints is a function of the current cushion
        above the maximum loss floor.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        crisis_probs : pd.Series, optional
            HMM crisis probabilities per date.
        avg_sharpe : float
            Average strategy Sharpe ratio.
        garch_vol : float, optional
            Current GARCH vol forecast (if None, uses rolling vol).

        Returns
        -------
        pd.Series
            Leverage multiplier per date.
        """
        cum = (1 + returns).cumprod()
        drawdown = cum / cum.cummax() - 1
        rolling_vol = returns.rolling(63, min_periods=21).std() * np.sqrt(252)

        leverage_series = pd.Series(1.0, index=returns.index)

        for t in range(63, len(returns)):
            dd = float(drawdown.iloc[t])
            vol = float(rolling_vol.iloc[t]) if not np.isnan(rolling_vol.iloc[t]) else 0.15
            cp = float(crisis_probs.iloc[t]) if crisis_probs is not None and t < len(crisis_probs) else 0.0

            trade = self.size_trade(
                signal_strength=0.5,
                asset_volatility=garch_vol if garch_vol is not None else vol,
                current_drawdown=dd,
                crisis_prob=cp,
                strategy_sharpe=avg_sharpe,
            )

            lev = min(trade.leverage, self.max_portfolio_leverage)
            leverage_series.iloc[t] = lev

        # Smooth leverage changes to reduce turnover (EMA with 5-day span)
        leverage_series = leverage_series.ewm(span=5).mean()
        leverage_series = leverage_series.clip(lower=self.min_leverage,
                                               upper=self.max_portfolio_leverage)

        logger.info(
            "Dynamic leverage: avg=%.2fx, min=%.2fx, max=%.2fx",
            leverage_series.mean(),
            leverage_series.min(),
            leverage_series.max(),
        )

        return leverage_series
