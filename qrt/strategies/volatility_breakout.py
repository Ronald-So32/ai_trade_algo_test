"""
Volatility Breakout Strategy
==============================
Signal fires when the daily range (High - Low) exceeds K * ATR(N).
Direction is determined by close vs open (gap direction).
Position size is inversely proportional to recent volatility.

Input note
----------
This strategy requires OHLC data.  ``generate_signals`` accepts:

  - ``prices``  : close prices DataFrame (columns=assets, index=dates)
  - ``returns`` : daily returns DataFrame
  - ``highs``   : DataFrame of daily highs  (keyword arg)
  - ``lows``    : DataFrame of daily lows   (keyword arg)
  - ``opens``   : DataFrame of daily opens  (keyword arg)
  - ``volume``  : DataFrame of daily volume (keyword arg, optional)

If ``highs`` / ``lows`` / ``opens`` are not provided, the strategy falls back
to synthetic OHLC derived from close prices and returns (less accurate but
still functional for testing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class VolatilityBreakout(Strategy):
    """
    Intraday range breakout strategy scaled by ATR.

    Parameters
    ----------
    atr_period : int
        ATR lookback period (default 14).
    breakout_multiplier : float
        K in range > K * ATR(N) for signal trigger (default 1.5).
    vol_lookback : int
        Window for realized-vol scaling of position size (default 21).
    vol_floor : float
        Minimum volatility for position sizing (default 0.005).
    target_gross : float
        Target gross exposure (default 1.0).
    holding_days : int
        Number of days to hold a breakout position (default 3).
    volume_confirm : bool
        If True, require volume > volume_multiplier * 20-day avg volume
        for a signal to fire (default True).
    volume_multiplier : float
        Multiplier for volume confirmation (default 1.5).
    volume_avg_period : int
        Lookback for average volume computation (default 20).
    max_weight_per_asset : float
        Maximum absolute weight per asset to prevent concentration
        (default 0.05, i.e. 5%).
    trailing_stop_atr_mult : float
        If return from entry drops below -trailing_stop_atr_mult * ATR,
        the signal is zeroed out (default 1.5).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Donchian (1960s), Bollinger (1980s) — range breakout systems"
        ),
        "historical_evidence": (
            "Popular among CTAs; evidence of profitability in commodities and FX, "
            "weaker in equities"
        ),
        "implementation_risks": (
            "False breakouts, whipsaw in range-bound markets, volume confirmation "
            "reduces but doesn't eliminate false signals"
        ),
        "realistic_expectations": (
            "Research-supported in trend-prone asset classes; equity application "
            "is less well-evidenced"
        ),
    }

    def __init__(
        self,
        atr_period: int = 14,
        breakout_multiplier: float = 1.5,
        vol_lookback: int = 21,
        vol_floor: float = 0.005,
        target_gross: float = 1.0,
        holding_days: int = 3,
        volume_confirm: bool = True,
        volume_multiplier: float = 1.5,
        volume_avg_period: int = 20,
        max_weight_per_asset: float = 0.05,
        trailing_stop_atr_mult: float = 1.5,
    ) -> None:
        params = dict(
            atr_period=atr_period,
            breakout_multiplier=breakout_multiplier,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            target_gross=target_gross,
            holding_days=holding_days,
            volume_confirm=volume_confirm,
            volume_multiplier=volume_multiplier,
            volume_avg_period=volume_avg_period,
            max_weight_per_asset=max_weight_per_asset,
            trailing_stop_atr_mult=trailing_stop_atr_mult,
        )
        super().__init__(name="VolatilityBreakout", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(
        highs: pd.DataFrame,
        lows: pd.DataFrame,
        closes: pd.DataFrame,
        period: int,
    ) -> pd.DataFrame:
        """
        Average True Range:
          TR = max(H-L, |H - prev_C|, |L - prev_C|)
          ATR = RollingMean(TR, period)
        """
        prev_close = closes.shift(1)
        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ],
            axis=0,
        ).groupby(level=0).max()

        # Use Wilder's EWM smoothing (span = period)
        atr = tr.ewm(span=period, min_periods=period).mean()
        return atr

    @staticmethod
    def _synthetic_ohlc(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthetic fallback OHLC from close prices and returns.
        High = close * (1 + 0.5 * |return|)
        Low  = close * (1 - 0.5 * |return|)
        Open = prev_close
        """
        abs_ret = returns.abs()
        highs = prices * (1 + 0.5 * abs_ret)
        lows = prices * (1 - 0.5 * abs_ret)
        opens = prices.shift(1)
        return highs, lows, opens

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
        Generate breakout signals.

        Keyword Args
        ------------
        highs  : pd.DataFrame, optional
        lows   : pd.DataFrame, optional
        opens  : pd.DataFrame, optional
        volume : pd.DataFrame, optional
            Daily volume per asset.  Used for volume confirmation when
            ``volume_confirm`` is True.

        Returns
        -------
        pd.DataFrame
            Signals in {-1, 0, +1}, same shape as *prices*.
        """
        atr_period: int = self.params["atr_period"]
        k: float = self.params["breakout_multiplier"]
        hold_days: int = self.params["holding_days"]

        highs = kwargs.get("highs", None)
        lows = kwargs.get("lows", None)
        opens = kwargs.get("opens", None)
        volume = kwargs.get("volume", None)

        if highs is None or lows is None or opens is None:
            highs, lows, opens = self._synthetic_ohlc(prices, returns)

        # Align all OHLC to the same index/columns as prices
        highs = highs.reindex_like(prices)
        lows = lows.reindex_like(prices)
        opens = opens.reindex_like(prices)

        # ATR
        atr = self._compute_atr(highs, lows, prices, atr_period)

        # Daily range
        daily_range = highs - lows

        # Breakout condition: range > K * ATR
        breakout_flag = daily_range > k * atr

        # --- Volume confirmation ---
        if self.params["volume_confirm"] and volume is not None:
            volume = volume.reindex_like(prices)
            avg_period: int = self.params["volume_avg_period"]
            vol_mult: float = self.params["volume_multiplier"]
            avg_volume = volume.rolling(avg_period, min_periods=avg_period // 2).mean()
            volume_ok = volume > vol_mult * avg_volume
            breakout_flag = breakout_flag & volume_ok

        # Direction: close > open -> bullish breakout (+1), else bearish (-1)
        direction = np.sign(prices - opens).replace(0, np.nan).ffill().fillna(1)

        # Raw signal: direction when breakout, else 0
        raw_signal = direction.where(breakout_flag, other=0.0)

        # --- Trailing stop using ATR ---
        trailing_mult: float = self.params["trailing_stop_atr_mult"]
        if hold_days > 1:
            # Forward-pass simulation to hold positions and apply trailing stop
            n_dates, n_assets = raw_signal.shape
            sig_arr = np.zeros((n_dates, n_assets), dtype=float)
            held_dir = np.zeros(n_assets, dtype=float)
            days_left = np.zeros(n_assets, dtype=int)
            entry_price = np.full(n_assets, np.nan, dtype=float)

            raw_vals = raw_signal.values
            price_vals = prices.values
            atr_vals = atr.reindex_like(prices).values

            for t in range(n_dates):
                for i in range(n_assets):
                    # Check for new breakout signal
                    if raw_vals[t, i] != 0:
                        held_dir[i] = raw_vals[t, i]
                        days_left[i] = hold_days
                        entry_price[i] = price_vals[t, i]

                    if held_dir[i] != 0 and days_left[i] > 0:
                        # Trailing stop check
                        cur_price = price_vals[t, i]
                        cur_atr = atr_vals[t, i] if not np.isnan(atr_vals[t, i]) else 0.0
                        if not np.isnan(entry_price[i]) and cur_atr > 0:
                            if held_dir[i] > 0:
                                pnl = cur_price - entry_price[i]
                            else:
                                pnl = entry_price[i] - cur_price
                            if pnl < -trailing_mult * cur_atr:
                                held_dir[i] = 0.0
                                days_left[i] = 0
                                entry_price[i] = np.nan
                                sig_arr[t, i] = 0.0
                                continue

                        sig_arr[t, i] = held_dir[i]
                        days_left[i] -= 1
                        if days_left[i] <= 0:
                            held_dir[i] = 0.0
                            entry_price[i] = np.nan
                    else:
                        sig_arr[t, i] = 0.0

            signal = pd.DataFrame(sig_arr, index=raw_signal.index, columns=raw_signal.columns)
        else:
            signal = raw_signal

        return signal.fillna(0.0)

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Size positions inversely proportional to realized volatility.

        Parameters
        ----------
        signals : pd.DataFrame
        returns : pd.DataFrame, optional
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        vol_lookback: int = self.params["vol_lookback"]
        vol_floor: float = self.params["vol_floor"]
        max_wt: float = self.params["max_weight_per_asset"]

        if returns is not None:
            realized_vol = (
                returns.rolling(vol_lookback, min_periods=max(1, vol_lookback // 2))
                .std()
                .mul(np.sqrt(252))
                .clip(lower=vol_floor)
            )
            raw_weights = signals / realized_vol
        else:
            raw_weights = signals.copy()

        raw_weights = raw_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        gross = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        # --- Position sizing cap: clip individual asset weights ---
        weights = weights.clip(lower=-max_wt, upper=max_wt)

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
