"""
Risk-Managed Momentum
======================
Implements volatility-scaled exposure for momentum sleeves, based on the
"Momentum has its moments" (Barroso & Santa-Clara, 2015) finding that
scaling momentum exposure to target constant volatility nearly doubles
Sharpe and materially reduces crash severity.

The key insight: momentum volatility is predictable and crashes coincide
with high-volatility states.  By scaling down exposure when realized
momentum volatility is high, we avoid the worst crash episodes without
sacrificing much average return.

Also implements a "crash-aware gating" overlay based on Daniel & Moskowitz
(2016) "Momentum Crashes" — when the market is in a crash recovery state
(high vol + recent large negative returns), momentum exposure is further
reduced or flipped.

References
----------
- Barroso & Santa-Clara (2015), "Momentum has its moments"
- Daniel & Moskowitz (2016), "Momentum Crashes"
- Moreira & Muir (2017), "Volatility-Managed Portfolios"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MomentumRiskManager:
    """
    Apply volatility-scaled exposure to momentum strategy weights.

    For each momentum sleeve:
        w_t = w_raw_t × min(L_max, σ* / σ_hat_t)

    where σ* is the per-sleeve target vol and σ_hat_t is realized or
    forecast volatility of the sleeve.

    Parameters
    ----------
    target_vol : float
        Per-sleeve annualized volatility target (default 0.10 = 10%).
    vol_lookback : int
        Window for estimating sleeve realized vol (default 63).
    max_leverage : float
        Maximum scaling factor (default 2.0).
    crash_gate : bool
        If True, apply crash-aware gating (default True).
    crash_vol_mult : float
        If realized vol > crash_vol_mult × median vol, enter crash state
        (default 2.0).
    crash_reduction : float
        Exposure multiplier in crash state (default 0.20).
    crash_lookback : int
        Window for crash detection features (default 126).
    """

    # Momentum strategy names that should receive risk management
    MOMENTUM_SLEEVES = {
        "cross_sectional_momentum",
        "time_series_momentum",
        "residual_momentum",
        "factor_momentum",
    }

    def __init__(
        self,
        target_vol: float = 0.10,
        vol_lookback: int = 63,
        max_leverage: float = 2.0,
        crash_gate: bool = True,
        crash_vol_mult: float = 2.0,
        crash_reduction: float = 0.20,
        crash_lookback: int = 126,
    ) -> None:
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.crash_gate = crash_gate
        self.crash_vol_mult = crash_vol_mult
        self.crash_reduction = crash_reduction
        self.crash_lookback = crash_lookback

    def risk_manage_sleeve(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        strategy_name: str,
    ) -> pd.DataFrame:
        """
        Apply risk-managed scaling to a single momentum sleeve.

        Parameters
        ----------
        weights : pd.DataFrame
            Raw portfolio weights (dates x assets) for this sleeve.
        returns : pd.DataFrame
            Asset returns aligned with weights.
        strategy_name : str
            Name of the strategy (for logging).

        Returns
        -------
        pd.DataFrame
            Risk-managed weights.
        """
        if strategy_name not in self.MOMENTUM_SLEEVES:
            return weights

        # Compute sleeve-level returns
        sleeve_returns = (weights.shift(1) * returns).sum(axis=1)

        # Sleeve realized volatility (annualized)
        realized_vol = (
            sleeve_returns.rolling(
                self.vol_lookback, min_periods=max(1, self.vol_lookback // 2)
            )
            .std()
            .mul(np.sqrt(252))
        )

        # Volatility scaling: target_vol / realized_vol
        vol_scale = (self.target_vol / realized_vol.clip(lower=1e-6)).clip(
            upper=self.max_leverage
        )
        vol_scale = vol_scale.fillna(1.0)

        # Apply crash gating if enabled
        if self.crash_gate:
            crash_scale = self._compute_crash_gate(sleeve_returns, realized_vol)
            vol_scale = vol_scale * crash_scale

        # Apply scaling
        scaled_weights = weights.mul(vol_scale, axis=0)

        # Log diagnostics
        original_vol = sleeve_returns.std() * np.sqrt(252) if len(sleeve_returns) > 1 else 0
        scaled_returns = (scaled_weights.shift(1) * returns).sum(axis=1)
        new_vol = scaled_returns.std() * np.sqrt(252) if len(scaled_returns) > 1 else 0

        logger.info(
            "  Risk-managed %s: vol %.2f%% → %.2f%%, "
            "mean scale %.2f, crash days %d",
            strategy_name,
            original_vol * 100,
            new_vol * 100,
            vol_scale.mean(),
            (vol_scale < 0.5).sum() if self.crash_gate else 0,
        )

        return scaled_weights

    def _compute_crash_gate(
        self,
        sleeve_returns: pd.Series,
        realized_vol: pd.Series,
    ) -> pd.Series:
        """
        Compute crash-aware gating signal.

        Crash state is detected when:
        1. Realized vol > crash_vol_mult × rolling median vol, AND
        2. Recent cumulative return over crash_lookback is strongly negative.

        In crash state, exposure is reduced to crash_reduction.
        """
        # Median vol (expanding, to avoid lookahead)
        median_vol = realized_vol.expanding(min_periods=63).median()

        # High-vol indicator
        high_vol = realized_vol > self.crash_vol_mult * median_vol

        # Recent cumulative return
        cum_ret = sleeve_returns.rolling(
            self.crash_lookback, min_periods=max(1, self.crash_lookback // 2)
        ).sum()

        # Crash = high vol AND strongly negative recent returns
        crash_state = high_vol & (cum_ret < -0.10)

        # Transition zone: even if not full crash, elevated vol reduces exposure
        transition = high_vol & ~crash_state
        transition_scale = 0.5

        scale = pd.Series(1.0, index=sleeve_returns.index)
        scale[crash_state] = self.crash_reduction
        scale[transition] = transition_scale

        return scale

    def risk_manage_all(
        self,
        strategy_results: dict,
        returns_wide: pd.DataFrame,
    ) -> dict:
        """
        Apply risk management to all momentum sleeves in strategy_results.

        Parameters
        ----------
        strategy_results : dict
            Dict of {name: {"weights": ..., "returns": ..., ...}}.
        returns_wide : pd.DataFrame
            Full asset return matrix.

        Returns
        -------
        dict
            Updated strategy_results with risk-managed weights/returns.
        """
        updated = {}
        for name, res in strategy_results.items():
            if name in self.MOMENTUM_SLEEVES and "weights" in res:
                new_weights = self.risk_manage_sleeve(
                    res["weights"], returns_wide, name
                )
                new_returns = (new_weights.shift(1) * returns_wide).sum(axis=1)
                updated[name] = {
                    **res,
                    "weights": new_weights,
                    "returns": new_returns,
                }
                # Recompute summary
                strategy = res["strategy"]
                updated[name]["summary"] = strategy.backtest_summary(
                    new_weights, returns_wide
                )
            else:
                updated[name] = res

        return updated
