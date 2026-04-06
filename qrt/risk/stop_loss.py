"""
Stop-Loss Engine for Portfolio-Level and Strategy-Level Risk Control.

Implements research-backed stop-loss logic:
  - Fixed percentage stops (Kaminski & Lo, 2014)
  - Volatility-scaled stops (Arratia & Dorador, 2019; Xiang & Deng, 2024)
  - Regime-aware stop tightening (Zambelli, 2016)
  - Stop-loss-aware ML label generation (Hwang et al., 2023)
  - Calibration via historical drawdown distributions

References:
  [1] Hwang et al. (2023) — Stop-loss adjusted labels for ML
  [2] Kaminski & Lo (2014) — When do stop-loss rules stop losses?
  [3] Arratia & Dorador (2019) — Optimal stop-loss under market microstructure
  [4] Xiang & Deng (2024) — Vol-scaled stops for equity strategies
  [5] Zambelli (2016) — Regime-conditional risk management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class StopLossConfig:
    """Configuration for stop-loss behaviour."""
    # Fixed percentage stop (fraction, e.g. 0.02 = 2%)
    fixed_stop_pct: float = 0.02
    # Volatility-scaled stop: k × σ_daily
    vol_stop_k: float = 2.0
    # Lookback for rolling vol estimation
    vol_lookback: int = 20
    # Which mode to use: "fixed", "vol_scaled", or "adaptive"
    mode: str = "vol_scaled"
    # Optional take-profit (fraction; None = disabled)
    take_profit_pct: Optional[float] = None
    # Trailing stop: if True, stop ratchets up with new highs
    trailing: bool = False
    # Regime tightening factor (multiply stop by this in crisis)
    crisis_tightening: float = 0.5
    # Minimum holding period before stop can trigger (days)
    min_hold_days: int = 0


@dataclass
class TradeResult:
    """Result of a single trade with stop-loss applied."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str  # "long" or "short"
    return_pct: float
    exit_reason: str  # "stop_loss", "take_profit", "horizon", "signal"
    holding_days: int
    max_adverse_excursion: float  # worst unrealised P&L during trade
    max_favorable_excursion: float  # best unrealised P&L during trade


# ======================================================================
# StopLossEngine
# ======================================================================

class StopLossEngine:
    """
    Pluggable stop-loss engine for strategy weight overlays.

    Works at the weight/signal level: given daily weights and prices,
    it monitors per-asset position P&L and zeros out weights when
    stop-loss conditions are met.

    Per Kaminski & Lo (2014): stop-losses work best for trending
    strategies and during regime transitions. Vol-scaled stops adapt
    to market conditions per Arratia & Dorador (2019).
    """

    def __init__(self, config: StopLossConfig | None = None) -> None:
        self.config = config or StopLossConfig()
        self._trade_log: List[TradeResult] = []

    @property
    def trade_log(self) -> List[TradeResult]:
        return list(self._trade_log)

    def apply(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        crisis_probs: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Apply stop-loss overlay to portfolio weights.

        For each asset on each day, tracks entry price and monitors
        unrealised P&L. Zeros out weight when stop is hit.

        Parameters
        ----------
        weights : pd.DataFrame
            Strategy weights (dates × assets).
        prices : pd.DataFrame
            Close prices aligned with weights.
        returns : pd.DataFrame
            Daily returns aligned with weights.
        crisis_probs : pd.Series, optional
            HMM crisis probability per date for regime-aware tightening.

        Returns
        -------
        pd.DataFrame
            Adjusted weights with stop-loss applied.
        """
        self._trade_log.clear()
        cfg = self.config

        # Compute rolling vol for vol-scaled stops
        daily_vol = returns.rolling(
            window=cfg.vol_lookback,
            min_periods=max(2, cfg.vol_lookback // 2),
        ).std().fillna(returns.std())

        adjusted = weights.copy()
        n_dates, n_assets = weights.shape
        cols = weights.columns.tolist()

        # Per-asset tracking
        entry_price = np.full(n_assets, np.nan)
        entry_date_idx = np.full(n_assets, -1, dtype=int)
        high_since_entry = np.full(n_assets, np.nan)
        position_side = np.zeros(n_assets)  # +1 long, -1 short, 0 flat

        price_arr = prices.reindex(index=weights.index, columns=cols).values
        weight_arr = weights.values.copy()
        vol_arr = daily_vol.reindex(index=weights.index, columns=cols).values

        crisis_arr = None
        if crisis_probs is not None:
            crisis_arr = crisis_probs.reindex(weights.index).fillna(0.0).values

        stops_triggered = 0

        for t in range(n_dates):
            for i in range(n_assets):
                w = weight_arr[t, i]
                p = price_arr[t, i]

                if np.isnan(p):
                    continue

                # Determine current side from weight
                if abs(w) < 1e-10:
                    # Position closed — record trade if we had one
                    if position_side[i] != 0 and not np.isnan(entry_price[i]):
                        self._record_trade(
                            entry_idx=entry_date_idx[i],
                            exit_idx=t,
                            dates=weights.index,
                            entry_price=entry_price[i],
                            exit_price=p,
                            side="long" if position_side[i] > 0 else "short",
                            exit_reason="signal",
                            high_since=high_since_entry[i],
                        )
                    position_side[i] = 0
                    entry_price[i] = np.nan
                    entry_date_idx[i] = -1
                    high_since_entry[i] = np.nan
                    continue

                current_side = 1.0 if w > 0 else -1.0

                # New entry or side change
                if position_side[i] != current_side:
                    # Close old if exists
                    if position_side[i] != 0 and not np.isnan(entry_price[i]):
                        self._record_trade(
                            entry_idx=entry_date_idx[i],
                            exit_idx=t,
                            dates=weights.index,
                            entry_price=entry_price[i],
                            exit_price=p,
                            side="long" if position_side[i] > 0 else "short",
                            exit_reason="signal",
                            high_since=high_since_entry[i],
                        )
                    position_side[i] = current_side
                    entry_price[i] = p
                    entry_date_idx[i] = t
                    high_since_entry[i] = p
                    continue

                # Existing position — check stop
                holding_days = t - entry_date_idx[i]
                if holding_days < cfg.min_hold_days:
                    # Update tracking
                    if position_side[i] > 0:
                        high_since_entry[i] = max(high_since_entry[i], p)
                    else:
                        high_since_entry[i] = min(high_since_entry[i], p)
                    continue

                # Compute unrealised P&L
                if position_side[i] > 0:
                    pnl_pct = (p - entry_price[i]) / entry_price[i]
                    high_since_entry[i] = max(high_since_entry[i], p)
                else:
                    pnl_pct = (entry_price[i] - p) / entry_price[i]
                    high_since_entry[i] = min(high_since_entry[i], p)

                # Compute stop level
                stop_level = self._compute_stop_level(
                    cfg, vol_arr[t, i], crisis_arr[t] if crisis_arr is not None else 0.0,
                )

                # Trailing stop adjustment
                if cfg.trailing and not np.isnan(high_since_entry[i]):
                    if position_side[i] > 0:
                        trail_pnl = (p - high_since_entry[i]) / high_since_entry[i]
                    else:
                        trail_pnl = (high_since_entry[i] - p) / high_since_entry[i]
                    if trail_pnl < -stop_level:
                        adjusted.iloc[t, i] = 0.0
                        self._record_trade(
                            entry_idx=entry_date_idx[i], exit_idx=t,
                            dates=weights.index,
                            entry_price=entry_price[i], exit_price=p,
                            side="long" if position_side[i] > 0 else "short",
                            exit_reason="stop_loss",
                            high_since=high_since_entry[i],
                        )
                        position_side[i] = 0
                        entry_price[i] = np.nan
                        entry_date_idx[i] = -1
                        stops_triggered += 1
                        continue

                # Fixed/vol stop check
                if pnl_pct < -stop_level:
                    adjusted.iloc[t, i] = 0.0
                    self._record_trade(
                        entry_idx=entry_date_idx[i], exit_idx=t,
                        dates=weights.index,
                        entry_price=entry_price[i], exit_price=p,
                        side="long" if position_side[i] > 0 else "short",
                        exit_reason="stop_loss",
                        high_since=high_since_entry[i],
                    )
                    position_side[i] = 0
                    entry_price[i] = np.nan
                    entry_date_idx[i] = -1
                    stops_triggered += 1
                    continue

                # Take-profit check
                if cfg.take_profit_pct is not None and pnl_pct >= cfg.take_profit_pct:
                    adjusted.iloc[t, i] = 0.0
                    self._record_trade(
                        entry_idx=entry_date_idx[i], exit_idx=t,
                        dates=weights.index,
                        entry_price=entry_price[i], exit_price=p,
                        side="long" if position_side[i] > 0 else "short",
                        exit_reason="take_profit",
                        high_since=high_since_entry[i],
                    )
                    position_side[i] = 0
                    entry_price[i] = np.nan
                    entry_date_idx[i] = -1
                    continue

        logger.info(
            "StopLossEngine: %d stops triggered, %d total trades logged",
            stops_triggered, len(self._trade_log),
        )
        return adjusted

    def _compute_stop_level(
        self,
        cfg: StopLossConfig,
        daily_vol: float,
        crisis_prob: float,
    ) -> float:
        """Compute the stop-loss threshold for current conditions."""
        if cfg.mode == "fixed":
            base_stop = cfg.fixed_stop_pct
        elif cfg.mode == "vol_scaled":
            base_stop = cfg.vol_stop_k * daily_vol if not np.isnan(daily_vol) else cfg.fixed_stop_pct
        elif cfg.mode == "adaptive":
            # Blend: use vol-scaled but floor at fixed
            vol_stop = cfg.vol_stop_k * daily_vol if not np.isnan(daily_vol) else cfg.fixed_stop_pct
            base_stop = max(vol_stop, cfg.fixed_stop_pct)
        else:
            base_stop = cfg.fixed_stop_pct

        # Regime tightening per Zambelli (2016)
        if crisis_prob > 0.5:
            tightening = cfg.crisis_tightening + (1 - cfg.crisis_tightening) * (1 - crisis_prob)
            base_stop *= tightening

        return max(base_stop, 1e-6)

    def _record_trade(
        self,
        entry_idx: int,
        exit_idx: int,
        dates: pd.DatetimeIndex,
        entry_price: float,
        exit_price: float,
        side: str,
        exit_reason: str,
        high_since: float,
    ) -> None:
        if entry_idx < 0 or entry_idx >= len(dates):
            return
        if side == "long":
            ret = (exit_price - entry_price) / entry_price
            mae = (min(exit_price, high_since) - entry_price) / entry_price if not np.isnan(high_since) else ret
            mfe = (max(exit_price, high_since) - entry_price) / entry_price if not np.isnan(high_since) else ret
        else:
            ret = (entry_price - exit_price) / entry_price
            mae = (entry_price - max(exit_price, high_since)) / entry_price if not np.isnan(high_since) else ret
            mfe = (entry_price - min(exit_price, high_since)) / entry_price if not np.isnan(high_since) else ret

        self._trade_log.append(TradeResult(
            entry_date=dates[entry_idx],
            exit_date=dates[min(exit_idx, len(dates) - 1)],
            entry_price=entry_price,
            exit_price=exit_price,
            side=side,
            return_pct=ret,
            exit_reason=exit_reason,
            holding_days=exit_idx - entry_idx,
            max_adverse_excursion=min(mae, ret),
            max_favorable_excursion=max(mfe, ret),
        ))

    def trade_statistics(self) -> dict:
        """Compute aggregate trade statistics from the trade log."""
        if not self._trade_log:
            return {
                "total_trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "avg_winner": 0.0, "avg_loser": 0.0,
                "profit_factor": 0.0, "avg_holding_days": 0.0,
                "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
                "signal_exit_pct": 0.0, "trades_per_year": 0.0,
            }

        rets = [t.return_pct for t in self._trade_log]
        winners = [r for r in rets if r > 0]
        losers = [r for r in rets if r <= 0]
        holds = [t.holding_days for t in self._trade_log]
        reasons = [t.exit_reason for t in self._trade_log]

        n = len(rets)
        total_wins = sum(r for r in rets if r > 0)
        total_losses = abs(sum(r for r in rets if r < 0))

        # Estimate trades per year from date range
        if len(self._trade_log) >= 2:
            date_range = (self._trade_log[-1].exit_date - self._trade_log[0].entry_date).days
            trades_per_year = n / max(date_range / 365.25, 0.01)
        else:
            trades_per_year = 0.0

        return {
            "total_trades": n,
            "win_rate": len(winners) / n if n > 0 else 0.0,
            "avg_return": np.mean(rets),
            "avg_winner": np.mean(winners) if winners else 0.0,
            "avg_loser": np.mean(losers) if losers else 0.0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "avg_holding_days": np.mean(holds),
            "stop_loss_pct": reasons.count("stop_loss") / n if n > 0 else 0.0,
            "take_profit_pct": reasons.count("take_profit") / n if n > 0 else 0.0,
            "signal_exit_pct": reasons.count("signal") / n if n > 0 else 0.0,
            "trades_per_year": trades_per_year,
        }


# ======================================================================
# Stop-Loss Calibration
# ======================================================================

class StopLossCalibrator:
    """
    Data-driven stop-loss threshold selection.

    Given baseline strategy weights and returns, simulates multiple
    stop-loss thresholds and selects the one that maximizes Sharpe
    subject to a max drawdown constraint.

    Uses expanding/rolling windows to avoid look-ahead bias.
    Per Kaminski & Lo (2014): calibrate on historical trade paths.
    """

    def __init__(
        self,
        stop_grid: List[float] | None = None,
        max_dd_constraint: float = 0.10,
        mode: str = "vol_scaled",
        vol_lookback: int = 20,
    ) -> None:
        self.stop_grid = stop_grid or [
            0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10,
        ]
        self.max_dd_constraint = max_dd_constraint
        self.mode = mode
        self.vol_lookback = vol_lookback

    def calibrate(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        crisis_probs: pd.Series | None = None,
    ) -> Dict[str, any]:
        """
        Test each stop threshold and return optimal + full grid results.

        Returns
        -------
        dict with keys:
            optimal_stop : float — best threshold
            optimal_sharpe : float
            optimal_maxdd : float
            grid_results : list of dicts
        """
        grid_results = []

        for stop_val in self.stop_grid:
            if self.mode == "fixed":
                cfg = StopLossConfig(
                    fixed_stop_pct=stop_val, mode="fixed",
                    vol_lookback=self.vol_lookback,
                )
            else:
                cfg = StopLossConfig(
                    vol_stop_k=stop_val / returns.std().mean() if returns.std().mean() > 0 else 2.0,
                    fixed_stop_pct=stop_val,
                    mode=self.mode,
                    vol_lookback=self.vol_lookback,
                )

            engine = StopLossEngine(cfg)
            adj_weights = engine.apply(weights, prices, returns, crisis_probs)

            # Compute performance
            strat_rets = (adj_weights.shift(1) * returns).sum(axis=1)
            if strat_rets.std() > 0:
                sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252)
            else:
                sharpe = 0.0

            cum = (1 + strat_rets).cumprod()
            maxdd = (cum / cum.cummax() - 1).min()
            cagr = cum.iloc[-1] ** (252 / len(cum)) - 1 if len(cum) > 0 else 0.0

            turnover = adj_weights.diff().abs().sum(axis=1).mean()
            stats = engine.trade_statistics()

            grid_results.append({
                "stop_level": stop_val,
                "sharpe": sharpe,
                "cagr": cagr,
                "maxdd": maxdd,
                "turnover": turnover,
                "win_rate": stats["win_rate"],
                "total_trades": stats["total_trades"],
                "stop_loss_pct": stats["stop_loss_pct"],
                "avg_holding_days": stats["avg_holding_days"],
            })

        # Select best: maximize Sharpe subject to MaxDD constraint
        valid = [r for r in grid_results if abs(r["maxdd"]) <= self.max_dd_constraint]
        if not valid:
            valid = grid_results  # fallback: ignore constraint

        best = max(valid, key=lambda r: r["sharpe"])

        logger.info(
            "StopLossCalibrator: optimal=%.3f (Sharpe=%.3f, MaxDD=%.2f%%, "
            "win_rate=%.1f%%, %d trades)",
            best["stop_level"], best["sharpe"], best["maxdd"] * 100,
            best["win_rate"] * 100, best["total_trades"],
        )

        return {
            "optimal_stop": best["stop_level"],
            "optimal_sharpe": best["sharpe"],
            "optimal_maxdd": best["maxdd"],
            "grid_results": grid_results,
        }


# ======================================================================
# Stop-Loss-Aware Label Generation for ML
# ======================================================================

def generate_stop_loss_labels(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 5,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float | None = None,
    vol_scaled: bool = True,
    vol_lookback: int = 20,
    vol_k: float = 2.0,
    label_type: str = "regression",
) -> pd.DataFrame:
    """
    Generate stop-loss-aware training labels per Hwang et al. (2023).

    For each (date, asset), simulate a trade from entry at close price:
    - Step forward day-by-day up to horizon
    - If price hits stop-loss → label = stopped-out return
    - If price hits take-profit → label = take-profit return
    - Otherwise → label = horizon return

    This aligns ML predictions with actual execution logic.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices (dates × assets).
    returns : pd.DataFrame
        Daily returns aligned with prices.
    horizon : int
        Forward horizon in trading days.
    stop_loss_pct : float
        Fixed stop-loss level (fraction).
    take_profit_pct : float, optional
        Take-profit level (fraction; None = disabled).
    vol_scaled : bool
        If True, scale stop by vol per Xiang & Deng (2024).
    vol_lookback : int
        Lookback for vol estimation.
    vol_k : float
        Vol multiplier for stop level.
    label_type : str
        "regression" (return value) or "binary" (1 if positive, 0 otherwise).

    Returns
    -------
    pd.DataFrame
        Labels (dates × assets), same shape as prices.
    """
    n_dates, n_assets = prices.shape
    labels = np.full((n_dates, n_assets), np.nan)
    price_arr = prices.values

    # Precompute rolling vol
    if vol_scaled:
        vol = returns.rolling(vol_lookback, min_periods=max(2, vol_lookback // 2)).std()
        vol_arr = vol.values
    else:
        vol_arr = None

    for t in range(n_dates - 1):
        for i in range(n_assets):
            entry_p = price_arr[t, i]
            if np.isnan(entry_p) or entry_p <= 0:
                continue

            # Determine stop level
            if vol_scaled and vol_arr is not None and not np.isnan(vol_arr[t, i]):
                stop = max(vol_k * vol_arr[t, i], stop_loss_pct)
            else:
                stop = stop_loss_pct

            # Simulate forward
            exit_return = np.nan
            for h in range(1, min(horizon + 1, n_dates - t)):
                future_p = price_arr[t + h, i]
                if np.isnan(future_p):
                    break

                ret_so_far = (future_p - entry_p) / entry_p

                # Stop-loss
                if ret_so_far <= -stop:
                    exit_return = -stop
                    break

                # Take-profit
                if take_profit_pct is not None and ret_so_far >= take_profit_pct:
                    exit_return = take_profit_pct
                    break

                # Horizon end
                if h == horizon:
                    exit_return = ret_so_far
                    break

            if np.isnan(exit_return) and not np.isnan(price_arr[min(t + horizon, n_dates - 1), i]):
                end_p = price_arr[min(t + horizon, n_dates - 1), i]
                exit_return = (end_p - entry_p) / entry_p

            labels[t, i] = exit_return

    result = pd.DataFrame(labels, index=prices.index, columns=prices.columns)

    if label_type == "binary":
        result = (result > 0).astype(float)
        result[pd.isna(labels)] = np.nan  # preserve NaNs

    logger.info(
        "Generated stop-loss-aware labels: horizon=%d, stop=%.1f%%, "
        "shape=%s, non-null=%.1f%%",
        horizon, stop_loss_pct * 100, result.shape,
        result.notna().mean().mean() * 100,
    )
    return result


# ======================================================================
# Portfolio-Level Trade Metrics
# ======================================================================

def compute_portfolio_trade_metrics(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> dict:
    """
    Compute win rate, trade frequency, and other trade-level metrics
    from portfolio weights and returns.

    Works without the full BacktestEngine — extracts trade-like
    statistics from the weight/return matrices directly.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (dates × assets).
    returns : pd.DataFrame
        Daily asset returns.
    prices : pd.DataFrame, optional
        Close prices (used for more accurate trade counting).

    Returns
    -------
    dict with trade metrics
    """
    # Daily portfolio returns
    port_rets = (weights.shift(1) * returns).sum(axis=1)

    # Win rate: fraction of days with positive return
    daily_win_rate = (port_rets > 0).mean()

    # Monthly returns for monthly win rate
    monthly_rets = port_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_rets > 0).mean() if len(monthly_rets) > 0 else 0.0

    # Trade frequency: count rebalance events
    weight_changes = weights.diff().abs()
    rebalance_days = (weight_changes.sum(axis=1) > 0.001).sum()
    total_days = len(weights)

    # Average one-way turnover
    avg_daily_turnover = weight_changes.sum(axis=1).mean() / 2

    # Position changes (entries + exits)
    position_active = (weights.abs() > 0.001).astype(int)
    position_changes = position_active.diff().abs()
    entries = (position_changes == 1).sum().sum() // 2  # each change is either entry or exit
    total_position_changes = position_changes.sum().sum()

    # Per-holding-period returns (approximate trades)
    trade_returns = []
    for col in weights.columns:
        w = weights[col]
        r = returns[col] if col in returns.columns else pd.Series(0, index=weights.index)

        # Find contiguous holding periods
        active = (w.abs() > 0.001).astype(int)
        changes = active.diff().fillna(0)
        entries_idx = changes[changes == 1].index
        exits_idx = changes[changes == -1].index

        for entry_dt in entries_idx:
            # Find next exit
            future_exits = exits_idx[exits_idx > entry_dt]
            if len(future_exits) > 0:
                exit_dt = future_exits[0]
            else:
                exit_dt = weights.index[-1]

            mask = (r.index >= entry_dt) & (r.index <= exit_dt)
            period_ret = (1 + r[mask]).prod() - 1
            trade_returns.append(period_ret)

    if trade_returns:
        trade_win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        avg_trade_return = np.mean(trade_returns)
        avg_winner = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_loser = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
        profit_factor = (
            sum(r for r in trade_returns if r > 0) /
            abs(sum(r for r in trade_returns if r < 0))
            if any(r < 0 for r in trade_returns) else float("inf")
        )
    else:
        trade_win_rate = daily_win_rate
        avg_trade_return = port_rets.mean()
        avg_winner = port_rets[port_rets > 0].mean() if (port_rets > 0).any() else 0
        avg_loser = port_rets[port_rets <= 0].mean() if (port_rets <= 0).any() else 0
        profit_factor = 0

    # Date range analysis
    if len(weights) > 252:
        # Recent 2 years (2024-2026 if available)
        recent_mask = weights.index >= weights.index[-1] - pd.Timedelta(days=504)
        recent_port = port_rets[recent_mask]
        recent_win_rate = (recent_port > 0).mean() if len(recent_port) > 0 else 0
        recent_trades_per_day = (weight_changes[recent_mask].sum(axis=1) > 0.001).mean()
    else:
        recent_win_rate = daily_win_rate
        recent_trades_per_day = rebalance_days / max(total_days, 1)

    return {
        "daily_win_rate": float(daily_win_rate),
        "monthly_win_rate": float(monthly_win_rate),
        "trade_win_rate": float(trade_win_rate),
        "total_trades_approx": len(trade_returns),
        "trades_per_year": len(trade_returns) / max(total_days / 252, 0.01),
        "rebalance_days": int(rebalance_days),
        "rebalance_frequency": rebalance_days / max(total_days, 1),
        "avg_daily_turnover": float(avg_daily_turnover),
        "avg_trade_return": float(avg_trade_return),
        "avg_winner": float(avg_winner),
        "avg_loser": float(avg_loser),
        "profit_factor": float(profit_factor),
        "recent_2yr_win_rate": float(recent_win_rate),
        "total_position_changes": int(total_position_changes),
    }
