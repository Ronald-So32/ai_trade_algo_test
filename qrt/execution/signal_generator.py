"""
Live signal generator — SIMPLIFIED VERSION (v4).

Runs 2 literature-grounded strategies on real market data (from Alpaca)
and produces target asset-level portfolio weights for the rebalancer.

Design (per Harvey, Liu & Zhu 2016):
  - 2 strategies with ZERO free parameters (all from published papers)
  - Equal-weight allocation (DeMiguel et al. 2009)
  - Fixed 2x leverage (Alpaca Reg T maximum)
  - No HMM, no stop-loss overlay, no ensemble of allocation methods

Strategies:
  1. Time-Series Momentum (Moskowitz, Ooi & Pedersen 2012)
  2. Residual Short-Term Reversal (Blitz et al. 2013, 2023)

v4 changes:
  - Fixed 2x leverage (replaces vol-targeting — simpler, higher CAGR)
  - Dropped 52-Week High (IS Sharpe 0.055 — dead weight)
  - 2-strategy system: each strategy gets 50% weight
  - ~150 stock universe with 72 GICS sub-industry residuals

Flow:
  1. Fetch latest prices from Alpaca (504 trading days lookback)
  2. Run 2 strategies: TSMOM, Residual Reversal
  3. Equal-weight combine (50/50)
  4. Apply fixed 2x leverage
  5. Output: {symbol: target_weight} for rebalancer
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Strategy parameters — ALL from published papers, ZERO tuning ──

# Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum"
TSMOM_PARAMS = {
    "lookback": 252,
    "vol_lookback": 63,
    "target_gross": 1.0,
    "vol_floor": 0.01,
    "multi_scale_weights": (0.0, 0.0, 1.0),  # Pure 12-month
    "trend_strength_cap": 1.0,
    "vov_reduction": 0.0,
    "vov_lookback": 63,
}

# Blitz et al. (2013) — Residual Short-Term Reversal
# Sorts on industry-neutral residuals instead of raw returns (2x Sharpe improvement)
RESIDUAL_STR_PARAMS = {
    "lookback": 5,
    "holding_period": 5,
    "long_pct": 0.20,
    "short_pct": 0.20,
    "target_gross": 1.0,
    "vol_scale": True,
    "vol_lookback": 21,
}

# Dropped: 52-Week High — IS Sharpe 0.055 over 14 years with 151 stocks.
# Zero contribution, dilutes portfolio by 50% (each working strategy goes 1/2 → 1/3).

SIMPLIFIED_STRATEGIES = {
    "time_series_momentum": TSMOM_PARAMS,
    "residual_reversal": RESIDUAL_STR_PARAMS,
}


class LiveSignalGenerator:
    """Generate portfolio target weights from live market data (simplified v4).

    v4 changes:
      - Fixed 2x leverage (no vol-targeting) — simpler, higher expected CAGR
      - 2 strategies: TSMOM + Residual Reversal (dropped 52-Week High)
      - Within-industry residuals via industry_map (72 GICS sub-industries)
    """

    def __init__(
        self,
        leverage: float = 2.0,
    ):
        self.leverage = leverage

    def generate_weights(
        self,
        prices_wide: pd.DataFrame,
        returns_wide: pd.DataFrame,
        skip_strategies: set[str] | None = None,
        industry_map: dict | None = None,
    ) -> dict[str, float]:
        """
        Run simplified strategy pipeline and return target weights.

        Parameters
        ----------
        prices_wide : DataFrame
            (dates x symbols) adjusted close prices.
        returns_wide : DataFrame
            (dates x symbols) daily returns.
        skip_strategies : set, optional
            Strategies to skip.
        industry_map : dict, optional
            {symbol: sub_industry} for within-industry residual reversal.
            If None, ResidualReversal uses market-neutral fallback.

        Returns
        -------
        dict : {symbol: target_weight} for portfolio rebalance.
        """
        from qrt.strategies import STRATEGY_REGISTRY

        common_cols = prices_wide.columns
        strategy_weights = {}
        strategy_returns = {}

        skip = skip_strategies or set()

        for name, params in SIMPLIFIED_STRATEGIES.items():
            if name in skip:
                logger.info(f"  Skipping: {name}")
                continue

            logger.info(f"  Running: {name}")
            try:
                cls = STRATEGY_REGISTRY[name]
                # Pass industry_map to ResidualReversal for within-industry residuals
                # Per Blitz et al. (2023): within-industry reversal (t=5.49)
                if name == "residual_reversal" and industry_map:
                    strategy = cls(**params, sector_map=industry_map)
                else:
                    strategy = cls(**params)

                signals = strategy.generate_signals(prices_wide, returns_wide)
                weights = strategy.compute_weights(signals, returns=returns_wide)

                strat_ret = (weights.shift(1) * returns_wide).sum(axis=1)
                strategy_weights[name] = weights
                strategy_returns[name] = strat_ret

                sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)
                          if strat_ret.std() > 0 else 0)
                logger.info(f"    Sharpe: {sharpe:.3f}")

            except Exception as e:
                logger.warning(f"    FAILED: {e}")

        if not strategy_weights:
            logger.error("No strategies produced signals")
            return {}

        # ── Equal-weight combination (DeMiguel et al. 2009) ──
        n_strats = len(strategy_weights)

        # ── Fixed 2x leverage ──
        leverage = self.leverage
        logger.info(f"  Leverage: {leverage:.2f}x (fixed)")

        # ── Compute per-asset target weights ──
        combined = pd.Series(0.0, index=common_cols)
        for name, w in strategy_weights.items():
            if len(w) > 0:
                latest = w.iloc[-1].reindex(common_cols, fill_value=0.0)
                combined += latest / n_strats

        # Apply leverage
        combined *= leverage

        # Filter tiny weights
        target = {}
        for sym, weight in combined.items():
            if abs(weight) > 0.001:
                target[sym] = float(weight)

        n_long = sum(1 for v in target.values() if v > 0.001)
        n_short = sum(1 for v in target.values() if v < -0.001)
        gross = sum(abs(v) for v in target.values())
        logger.info(f"  Target: {n_long} long, {n_short} short, gross: {gross:.2%}")

        return target
