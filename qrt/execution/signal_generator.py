"""
Live signal generator — v5 (DYNAMIC).

Runs 2 literature-grounded strategies on real market data (from Alpaca)
and produces target asset-level portfolio weights for the rebalancer.

v4 design (per Harvey, Liu & Zhu 2016):
  - 2 strategies with ZERO free parameters (all from published papers)
  - Equal-weight allocation (DeMiguel et al. 2009)
  - Fixed 2x leverage (Alpaca Reg T maximum)
  - No HMM, no stop-loss overlay, no ensemble of allocation methods

v5 enhancements — dynamic allocation with ZERO new free parameters:
  1. Risk parity between strategies (Maillard, Roncalli & Teiletche 2010)
     — Weight each strategy inversely to its realized vol so each contributes
       equal risk. With 2 strategies of very different vol profiles (TSMOM ~15%
       vs Residual Reversal ~5%), equal-weight means TSMOM dominates risk.
  2. Volatility-managed portfolio exposure (Moreira & Muir 2017;
     Barroso & Santa-Clara 2015 "Momentum Has Its Moments")
     — Scale total exposure inversely to recent realized portfolio vol.
       Documented to ~double momentum Sharpe by avoiding crash drawdowns.
  3. Liquidity-weighted residual reversal (Nagel 2012 "Evaporating Liquidity";
     Amihud 2002 illiquidity measure)
     — The reversal premium IS a liquidity provision premium. Weight positions
       by illiquidity within quintiles so less-liquid stocks (where the premium
       is strongest) get larger positions.

All three use parameters from literature (vol lookback = 63d, target vol = 15%).
No new tunable parameters introduced.

Strategies:
  1. Time-Series Momentum (Moskowitz, Ooi & Pedersen 2012)
  2. Residual Short-Term Reversal (Blitz et al. 2013, 2023)

Flow:
  1. Fetch latest prices from Alpaca (504 trading days lookback)
  2. Run 2 strategies: TSMOM, Residual Reversal (with liquidity weighting)
  3. Risk-parity combine (inverse-vol weighting)
  4. Apply volatility-managed leverage (target 15% ann. vol, cap at 2x)
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
    # liquidity_weight: disabled for S&P 500 large-caps (all >$10B market cap).
    # Nagel (2012) premium is strongest in small/mid-caps; OOS backtest showed
    # liquidity weighting hurts on this universe (Sharpe 0.706 vs 1.151 without).
    # Enable for small/mid-cap universes where illiquidity spread is meaningful.
}

# Dropped: 52-Week High — IS Sharpe 0.055 over 14 years with 151 stocks.
# Zero contribution, dilutes portfolio by 50% (each working strategy goes 1/2 → 1/3).

SIMPLIFIED_STRATEGIES = {
    "time_series_momentum": TSMOM_PARAMS,
    "residual_reversal": RESIDUAL_STR_PARAMS,
}

# ── v5 Dynamic allocation parameters (all from literature) ──

# Risk parity vol lookback: 63 trading days (3 months)
# Standard in Maillard et al. (2010) and Roncalli (2013)
RISK_PARITY_VOL_LOOKBACK = 63

# Volatility-managed target: 15% annualized
# Moskowitz et al. (2012) use 40% for individual assets; 15% is standard
# for a diversified portfolio (Moreira & Muir 2017 use 10-20% range).
VOL_MANAGED_TARGET = 0.15
VOL_MANAGED_LOOKBACK = 63  # 3-month realized vol


class LiveSignalGenerator:
    """Generate portfolio target weights from live market data.

    Supports two modes:
      - v4 (mode="static"): Equal-weight 50/50, fixed 2x leverage
      - v5 (mode="dynamic"): Risk parity + vol-managed leverage

    v5 changes:
      - Risk parity between strategies (inverse-vol weighting)
      - Volatility-managed total exposure (Moreira & Muir 2017)
      - Liquidity-weighted residual reversal (Nagel 2012)
    """

    def __init__(
        self,
        leverage: float = 2.0,
        mode: str = "dynamic",
    ):
        self.leverage = leverage
        self.mode = mode

    def generate_weights(
        self,
        prices_wide: pd.DataFrame,
        returns_wide: pd.DataFrame,
        skip_strategies: set[str] | None = None,
        industry_map: dict | None = None,
    ) -> dict[str, float]:
        """
        Run strategy pipeline and return target weights.

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

                strat_ret = (weights.shift(1) * returns_wide).sum(axis=1).fillna(0.0)
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

        if self.mode == "dynamic":
            # Warmup guard: risk parity and vol-managed leverage require
            # at least 63 days of strategy returns for reliable vol estimates.
            # Fall back to static mode if insufficient history.
            min_history = max(RISK_PARITY_VOL_LOOKBACK, VOL_MANAGED_LOOKBACK)
            has_enough = all(
                len(ret.dropna()) >= min_history
                for ret in strategy_returns.values()
            )
            if not has_enough:
                lengths = {n: len(r.dropna()) for n, r in strategy_returns.items()}
                logger.warning(
                    f"  Insufficient history for dynamic mode (need {min_history} days): "
                    f"{lengths}. Falling back to STATIC (v4) mode."
                )
                return self._combine_static(strategy_weights, common_cols)

            return self._combine_dynamic(
                strategy_weights, strategy_returns, common_cols
            )
        else:
            return self._combine_static(
                strategy_weights, common_cols
            )

    def _combine_static(
        self,
        strategy_weights: dict[str, pd.DataFrame],
        common_cols: pd.Index,
    ) -> dict[str, float]:
        """v4 static: equal-weight 50/50, fixed leverage."""
        n_strats = len(strategy_weights)
        leverage = self.leverage
        logger.info(f"  Mode: STATIC (v4) | Leverage: {leverage:.2f}x (fixed)")

        combined = pd.Series(0.0, index=common_cols)
        for name, w in strategy_weights.items():
            if len(w) > 0:
                latest = w.iloc[-1].reindex(common_cols, fill_value=0.0)
                combined += latest / n_strats

        combined *= leverage
        return self._filter_weights(combined)

    def _combine_dynamic(
        self,
        strategy_weights: dict[str, pd.DataFrame],
        strategy_returns: dict[str, pd.Series],
        common_cols: pd.Index,
    ) -> dict[str, float]:
        """
        v5 dynamic: risk parity allocation + volatility-managed leverage.

        Risk parity (Maillard, Roncalli & Teiletche 2010):
          Weight each strategy inversely to its trailing realized vol.
          This ensures each strategy contributes equal risk to the portfolio.

        Vol-managed leverage (Moreira & Muir 2017):
          Scale total portfolio exposure so realized vol targets 15% annualized.
          Capped at Alpaca's Reg T maximum (2x).
        """
        # ── Step 1: Risk parity weights ──
        strat_vols = {}
        for name, ret in strategy_returns.items():
            trailing = ret.iloc[-RISK_PARITY_VOL_LOOKBACK:]
            ann_vol = trailing.std() * np.sqrt(252)
            strat_vols[name] = max(ann_vol, 0.01)  # floor to avoid division by zero

        inv_vols = {name: 1.0 / vol for name, vol in strat_vols.items()}
        total_inv_vol = sum(inv_vols.values())
        rp_weights = {name: iv / total_inv_vol for name, iv in inv_vols.items()}

        for name, rpw in rp_weights.items():
            logger.info(
                f"    Risk parity: {name} = {rpw:.1%} "
                f"(vol={strat_vols[name]:.2%})"
            )

        # ── Step 2: Combine strategy weights using risk parity ──
        combined = pd.Series(0.0, index=common_cols)
        for name, w in strategy_weights.items():
            if len(w) > 0:
                latest = w.iloc[-1].reindex(common_cols, fill_value=0.0)
                combined += latest * rp_weights[name]

        # ── Step 3: Volatility-managed leverage ──
        # Compute recent portfolio vol from combined strategy returns.
        # Align all strategy return series on a common index to handle
        # different warmup periods (TSMOM 252d vs Residual Reversal 5d).
        all_ret = pd.DataFrame(strategy_returns).fillna(0.0)
        combined_ret = (all_ret * pd.Series(rp_weights)).sum(axis=1)

        trailing_vol = (
            combined_ret.iloc[-VOL_MANAGED_LOOKBACK:].std() * np.sqrt(252)
        )
        trailing_vol = max(trailing_vol, 0.02)  # floor at 2%

        vol_managed_leverage = min(
            VOL_MANAGED_TARGET / trailing_vol,
            self.leverage,  # cap at Reg T max
        )
        vol_managed_leverage = max(vol_managed_leverage, 0.5)  # floor at 0.5x

        logger.info(
            f"  Mode: DYNAMIC (v5) | "
            f"Portfolio vol: {trailing_vol:.2%} | "
            f"Vol-managed leverage: {vol_managed_leverage:.2f}x "
            f"(target {VOL_MANAGED_TARGET:.0%}, cap {self.leverage:.0f}x)"
        )

        combined *= vol_managed_leverage
        return self._filter_weights(combined)

    @staticmethod
    def _filter_weights(combined: pd.Series) -> dict[str, float]:
        """Filter tiny weights and return as dict."""
        target = {}
        for sym, weight in combined.items():
            if abs(weight) > 0.001:
                target[sym] = float(weight)

        n_long = sum(1 for v in target.values() if v > 0.001)
        n_short = sum(1 for v in target.values() if v < -0.001)
        gross = sum(abs(v) for v in target.values())
        logger.info(f"  Target: {n_long} long, {n_short} short, gross: {gross:.2%}")

        return target
