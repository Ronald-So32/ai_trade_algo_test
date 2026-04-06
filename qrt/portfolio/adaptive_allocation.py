"""
Adaptive Strategy Allocation
=============================
Implements two key adaptive mechanisms:

1. **Regime-Conditional Parameters**: Adjusts strategy parameters based on
   the current volatility regime (low_vol, medium_vol, high_vol, crisis).

2. **Dynamic Strategy Allocation**: Replaces static risk parity with
   time-varying weights based on rolling Sharpe ratios, drawdown state,
   and regime context.

These mechanisms adapt *parameters* and *weights* — never the core strategy
logic — avoiding overfitting while responding to structural market shifts.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Regime-Conditional Parameter Sets
# ──────────────────────────────────────────────────────────────────────────────

# For each strategy, define parameter overrides per regime.
# Only parameters that should change are listed; others keep their defaults.
REGIME_PARAM_OVERRIDES: dict[str, dict[str, dict]] = {
    "time_series_momentum": {
        "low_vol": {"lookback": 252, "multi_scale_weights": (0.2, 0.4, 0.4), "target_gross": 1.2},
        "medium_vol": {"lookback": 126, "multi_scale_weights": (0.4, 0.4, 0.2), "target_gross": 1.0},
        "high_vol": {"lookback": 63, "multi_scale_weights": (0.5, 0.3, 0.2), "target_gross": 0.7},
        "crisis": {"lookback": 63, "multi_scale_weights": (0.6, 0.3, 0.1), "target_gross": 0.3},
    },
    "cross_sectional_momentum": {
        "low_vol": {"lookback": 252, "skip_days": 21},
        "medium_vol": {"lookback": 252, "skip_days": 30},
        "high_vol": {"lookback": 126, "skip_days": 42},
        "crisis": {"lookback": 63, "skip_days": 42},
    },
    "mean_reversion": {
        "low_vol": {"lookback": 120, "entry_threshold": 2.0, "max_holding": 10},
        "medium_vol": {"lookback": 120, "entry_threshold": 1.5, "max_holding": 5},
        "high_vol": {"lookback": 60, "entry_threshold": 1.5, "max_holding": 3},
        "crisis": {"lookback": 60, "entry_threshold": 2.5, "max_holding": 2},
    },
    "volatility_breakout": {
        "low_vol": {"breakout_mult": 2.0, "max_holding": 5},
        "medium_vol": {"breakout_mult": 1.5, "max_holding": 3},
        "high_vol": {"breakout_mult": 1.5, "max_holding": 2},
        "crisis": {"breakout_mult": 2.0, "max_holding": 1},
    },
}


def get_regime_params(
    strategy_name: str,
    regime_label: str,
    base_params: dict,
) -> dict:
    """
    Return strategy parameters adjusted for the current regime.

    Parameters
    ----------
    strategy_name : str
        Registry name of the strategy.
    regime_label : str
        Current regime: "low_vol", "medium_vol", "high_vol", or "crisis".
    base_params : dict
        Default parameter dict for the strategy.

    Returns
    -------
    dict
        Merged parameters with regime overrides applied.
    """
    overrides = REGIME_PARAM_OVERRIDES.get(strategy_name, {})
    regime_overrides = overrides.get(regime_label, {})

    merged = {**base_params, **regime_overrides}
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Strategy Allocator
# ──────────────────────────────────────────────────────────────────────────────

class DynamicStrategyAllocator:
    """
    Time-varying strategy weights based on rolling performance and regime.

    Combines three signals to determine allocation:
    1. Rolling Sharpe ratio (trailing window)
    2. Drawdown penalty (reduce allocation to strategies in drawdown)
    3. Regime tilt (favor/penalize strategies by regime)

    Rebalances monthly to avoid excessive turnover.

    Parameters
    ----------
    sharpe_lookback : int
        Rolling window for Sharpe estimation (default 126 = ~6 months).
    rebalance_freq : int
        Rebalance every N trading days (default 21 = monthly).
    min_weight : float
        Minimum allocation per strategy (default 0.02 = 2%).
    max_weight : float
        Maximum allocation per strategy (default 0.40 = 40%).
    drawdown_penalty_threshold : float
        Strategy drawdown level at which penalty kicks in (default -0.10).
    drawdown_penalty_factor : float
        Multiply weight by this when in drawdown (default 0.5).
    """

    # Regime tilts: which strategy families do well in which regime.
    # Rationale: momentum thrives in trends (bull), suffers in reversals (crisis).
    # Mean-reversion works in range-bound (high_vol) but fails in crisis divergence.
    # Vol-managed and low-risk strategies shine in crisis (flight to quality).
    REGIME_TILTS: dict[str, dict[str, float]] = {
        "low_vol": {
            "time_series_momentum": 1.2,
            "cross_sectional_momentum": 1.2,
            "mean_reversion": 0.8,
            "carry": 1.3,
            "distance_pairs": 1.0,
            "kalman_pairs": 1.0,
            "volatility_breakout": 0.7,
            "factor_momentum": 1.1,
            "pca_stat_arb": 1.0,
            "vol_managed": 0.9,
            "pead": 1.0,
            "residual_momentum": 1.1,
            "low_risk_bab": 0.9,
            "ml_alpha": 1.1,
            "short_term_reversal": 1.0,
            "vol_risk_premium": 1.2,
        },
        "medium_vol": {
            "time_series_momentum": 1.0,
            "cross_sectional_momentum": 1.0,
            "mean_reversion": 1.0,
            "carry": 1.0,
            "distance_pairs": 1.0,
            "kalman_pairs": 1.0,
            "volatility_breakout": 1.0,
            "factor_momentum": 1.0,
            "pca_stat_arb": 1.0,
            "vol_managed": 1.0,
            "pead": 1.0,
            "residual_momentum": 1.0,
            "low_risk_bab": 1.0,
            "ml_alpha": 1.0,
            "short_term_reversal": 1.0,
            "vol_risk_premium": 1.0,
        },
        "high_vol": {
            "time_series_momentum": 1.1,
            "cross_sectional_momentum": 0.8,
            "mean_reversion": 1.2,
            "carry": 0.7,
            "distance_pairs": 1.1,
            "kalman_pairs": 1.1,
            "volatility_breakout": 1.3,
            "factor_momentum": 0.8,
            "pca_stat_arb": 1.2,
            "vol_managed": 1.2,
            "pead": 0.9,
            "residual_momentum": 0.8,
            "low_risk_bab": 1.2,
            "ml_alpha": 0.9,
            "short_term_reversal": 1.3,
            "vol_risk_premium": 0.5,
        },
        "crisis": {
            "time_series_momentum": 1.3,
            "cross_sectional_momentum": 0.5,
            "mean_reversion": 0.5,
            "carry": 0.3,
            "distance_pairs": 0.8,
            "kalman_pairs": 0.8,
            "volatility_breakout": 0.6,
            "factor_momentum": 0.5,
            "pca_stat_arb": 0.8,
            "vol_managed": 1.4,
            "pead": 0.7,
            "residual_momentum": 0.6,
            "low_risk_bab": 1.3,
            "ml_alpha": 0.7,
            "short_term_reversal": 1.4,
            "vol_risk_premium": 0.2,
        },
    }

    def __init__(
        self,
        sharpe_lookback: int = 126,
        rebalance_freq: int = 21,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        drawdown_penalty_threshold: float = -0.10,
        drawdown_penalty_factor: float = 0.50,
    ) -> None:
        self.sharpe_lookback = sharpe_lookback
        self.rebalance_freq = rebalance_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.dd_threshold = drawdown_penalty_threshold
        self.dd_penalty = drawdown_penalty_factor

    def compute_dynamic_weights(
        self,
        strategy_returns: dict[str, pd.Series],
        regime_labels: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute time-varying strategy weights.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Per-strategy daily return series.
        regime_labels : pd.Series, optional
            Regime label per date (e.g., "low_vol", "crisis").

        Returns
        -------
        weights_df : pd.DataFrame
            Daily strategy weights (dates x strategies), sum to 1.
        diagnostics_df : pd.DataFrame
            Rolling Sharpe, drawdown, and regime info for dashboard.
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
        strategy_names = list(returns_df.columns)
        n_strats = len(strategy_names)
        dates = returns_df.index

        if n_strats == 0:
            raise ValueError("No strategy returns provided.")

        # 1. Rolling Sharpe
        rolling_mean = returns_df.rolling(self.sharpe_lookback, min_periods=63).mean()
        rolling_std = returns_df.rolling(self.sharpe_lookback, min_periods=63).std()
        rolling_sharpe = (rolling_mean / rolling_std.clip(lower=1e-8)) * np.sqrt(252)

        # 2. Strategy drawdowns
        cum_returns = (1 + returns_df).cumprod()
        running_max = cum_returns.cummax()
        strategy_dd = (cum_returns - running_max) / running_max

        # 3. Build weight series
        weights_arr = np.full((len(dates), n_strats), 1.0 / n_strats)
        last_weights = np.ones(n_strats) / n_strats

        for t in range(len(dates)):
            # Only rebalance at specified frequency
            if t > 0 and t % self.rebalance_freq != 0:
                weights_arr[t] = last_weights
                continue

            if t < 63:
                # Not enough history — equal weight
                weights_arr[t] = np.ones(n_strats) / n_strats
                last_weights = weights_arr[t]
                continue

            # Base weights from rolling Sharpe with James-Stein shrinkage
            # Shrink individual rolling Sharpe toward cross-strategy mean
            # to reduce estimation error (James & Stein 1961)
            sharpe_t = rolling_sharpe.iloc[t].values
            grand_mean = np.nanmean(sharpe_t)
            dispersion = np.nansum((sharpe_t - grand_mean) ** 2)
            if dispersion > 1e-10 and n_strats >= 3:
                shrink_intensity = max(0.0, 1.0 - (n_strats - 2) / (dispersion * self.sharpe_lookback + 1e-8))
                shrink_intensity = min(shrink_intensity, 1.0)
                sharpe_t = grand_mean + shrink_intensity * (sharpe_t - grand_mean)
            sharpe_pos = np.maximum(sharpe_t, 0.0)

            # If all Sharpe <= 0, use equal weight
            if sharpe_pos.sum() < 1e-8:
                w = np.ones(n_strats) / n_strats
            else:
                w = sharpe_pos / sharpe_pos.sum()

            # Apply drawdown penalty
            dd_t = strategy_dd.iloc[t].values
            for i in range(n_strats):
                if dd_t[i] < self.dd_threshold:
                    w[i] *= self.dd_penalty

            # Apply regime tilt
            if regime_labels is not None:
                date = dates[t]
                if date in regime_labels.index:
                    regime = regime_labels.loc[date]
                    if isinstance(regime, str) and regime in self.REGIME_TILTS:
                        tilts = self.REGIME_TILTS[regime]
                        for i, name in enumerate(strategy_names):
                            w[i] *= tilts.get(name, 1.0)

            # Clip and renormalize
            w = np.clip(w, self.min_weight, self.max_weight)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(n_strats) / n_strats

            weights_arr[t] = w
            last_weights = w

        weights_df = pd.DataFrame(weights_arr, index=dates, columns=strategy_names)

        # Build diagnostics
        diagnostics_df = pd.DataFrame({
            "date": dates,
        })
        for name in strategy_names:
            diagnostics_df[f"{name}_sharpe"] = rolling_sharpe[name].values
            diagnostics_df[f"{name}_dd"] = strategy_dd[name].values
            diagnostics_df[f"{name}_weight"] = weights_df[name].values
        diagnostics_df = diagnostics_df.set_index("date")

        logger.info(
            "Dynamic allocation computed: %d strategies, %d dates, "
            "rebalance every %d days",
            n_strats, len(dates), self.rebalance_freq,
        )

        return weights_df, diagnostics_df

    def apply_dynamic_weights(
        self,
        strategy_returns: dict[str, pd.Series],
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Combine strategy returns using time-varying weights.

        Returns
        -------
        pd.Series
            Combined portfolio daily returns.
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
        common_idx = returns_df.index.intersection(weights_df.index)
        combined = (returns_df.loc[common_idx] * weights_df.loc[common_idx]).sum(axis=1)
        combined.name = "dynamic_combined_returns"
        return combined


# ──────────────────────────────────────────────────────────────────────────────
# Tail Risk Management
# ──────────────────────────────────────────────────────────────────────────────

class TailRiskManager:
    """
    Non-overfitting tail risk controls applied at the portfolio level.

    Implements:
    1. Correlation-aware scaling: reduce exposure when cross-strategy
       correlations spike (diversification breaks down).
    2. Drawdown-speed scaling: cut exposure when drawdown velocity is high.
    3. Left-tail dampening: asymmetric vol targeting that reduces more
       aggressively on negative days.

    Parameters
    ----------
    corr_lookback : int
        Window for rolling pairwise correlation (default 63).
    corr_threshold : float
        Average correlation above which exposure is reduced (default 0.50).
    corr_reduction : float
        Multiply exposure by this when correlations are elevated (default 0.60).
    dd_speed_lookback : int
        Window to measure drawdown velocity (default 10).
    dd_speed_threshold : float
        Daily drawdown velocity threshold (default -0.005).
    dd_speed_reduction : float
        Exposure multiplier when drawdown is fast (default 0.50).
    """

    def __init__(
        self,
        corr_lookback: int = 63,
        corr_threshold: float = 0.50,
        corr_reduction: float = 0.60,
        dd_speed_lookback: int = 10,
        dd_speed_threshold: float = -0.005,
        dd_speed_reduction: float = 0.50,
    ) -> None:
        self.corr_lookback = corr_lookback
        self.corr_threshold = corr_threshold
        self.corr_reduction = corr_reduction
        self.dd_speed_lookback = dd_speed_lookback
        self.dd_speed_threshold = dd_speed_threshold
        self.dd_speed_reduction = dd_speed_reduction

    def compute_scaling(
        self,
        strategy_returns: dict[str, pd.Series],
        portfolio_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute daily exposure scaling factors for tail risk management.

        Returns
        -------
        pd.Series
            Scaling factors in (0, 1], same index as portfolio_returns.
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
        scaling = pd.Series(1.0, index=portfolio_returns.index)

        # 1. Correlation-aware scaling
        if returns_df.shape[1] >= 2:
            rolling_corr = self._rolling_avg_correlation(returns_df)
            corr_mask = rolling_corr > self.corr_threshold
            scaling[corr_mask] *= self.corr_reduction
            n_corr_days = corr_mask.sum()
            if n_corr_days > 0:
                logger.info(
                    "  Tail risk: correlation scaling applied on %d days (%.1f%%)",
                    n_corr_days, 100 * n_corr_days / len(scaling),
                )

        # 2. Drawdown speed scaling
        cum = (1 + portfolio_returns).cumprod()
        dd = cum / cum.cummax() - 1
        dd_speed = dd.diff(self.dd_speed_lookback) / self.dd_speed_lookback
        fast_dd = dd_speed < self.dd_speed_threshold
        scaling[fast_dd] *= self.dd_speed_reduction
        n_speed_days = fast_dd.sum()
        if n_speed_days > 0:
            logger.info(
                "  Tail risk: drawdown-speed scaling on %d days (%.1f%%)",
                n_speed_days, 100 * n_speed_days / len(scaling),
            )

        # 3. Left-tail dampening: on days with large negative returns,
        #    scale down more aggressively
        z_ret = (portfolio_returns - portfolio_returns.rolling(63, min_periods=21).mean()) / \
                portfolio_returns.rolling(63, min_periods=21).std().clip(lower=1e-8)
        extreme_neg = z_ret < -2.0
        scaling[extreme_neg] *= 0.7

        scaling = scaling.clip(lower=0.1, upper=1.0)
        return scaling

    def _rolling_avg_correlation(self, returns_df: pd.DataFrame) -> pd.Series:
        """Compute rolling average pairwise correlation across strategies."""
        n_strats = returns_df.shape[1]
        if n_strats < 2:
            return pd.Series(0.0, index=returns_df.index)

        # Use rolling window correlation between all pairs
        avg_corr = pd.Series(np.nan, index=returns_df.index)
        for t in range(self.corr_lookback, len(returns_df)):
            window = returns_df.iloc[t - self.corr_lookback:t]
            corr_mat = window.corr().values
            # Extract upper triangle (excluding diagonal)
            upper = corr_mat[np.triu_indices(n_strats, k=1)]
            avg_corr.iloc[t] = np.nanmean(upper)

        return avg_corr.ffill().fillna(0.0)
