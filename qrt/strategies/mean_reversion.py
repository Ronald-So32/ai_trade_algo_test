"""
Mean Reversion Strategy
========================
Z-score of price relative to rolling mean.
Signal = -z_score (mean-revert).
Enter when |z| > threshold; exit at 0; hold for at most max_holding days.

Includes trend filter, stop-loss, and drawdown circuit breaker to reduce
MaxDD and improve risk-adjusted returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class MeanReversion(Strategy):
    """
    Rolling z-score mean-reversion strategy.

    Parameters
    ----------
    lookback : int
        Rolling window for mean and std (default 120).
    entry_threshold : float
        |z-score| level required to open a position (default 1.5).
    exit_threshold : float
        |z-score| level at which the position is closed (default 0.5).
    max_holding : int
        Maximum number of days to hold any position (default 5).
    target_gross : float
        Target gross portfolio exposure (default 1.0).
    vol_scale : bool
        If True, further scale weights by 1 / realized_vol (default True).
    vol_lookback : int
        Window for realized-vol scaling (default 21).
    vol_floor : float
        Minimum realized vol to avoid division blow-up (default 0.005).
    trend_sma_period : int
        SMA period for the trend filter (default 200).
    trend_slope_limit : float
        Maximum absolute daily slope of SMA to allow mean-reversion
        signals.  When abs(slope) >= this value the asset is considered
        trending and signals are suppressed (default 0.001).
    stop_loss_pct : float
        Per-position stop-loss threshold expressed as a fraction
        (default 0.03, i.e. 3%).
    dd_circuit_breaker : float
        If cumulative strategy return drops below this level, go flat
        (default -0.15, i.e. -15%).
    dd_cooldown_days : int
        Number of days to stay flat after the circuit breaker fires
        (default 42).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Lo & MacKinlay (1990), Poterba & Summers (1988) — short-term price reversals"
        ),
        "historical_evidence": (
            "Strong at intraday to weekly frequencies; weaker at monthly+"
        ),
        "implementation_risks": (
            "Trend-following regimes, momentum crashes, mean can shift permanently"
        ),
        "realistic_expectations": (
            "Research-supported return premium; may underperform in trending markets; "
            "long drawdowns possible"
        ),
    }

    def __init__(
        self,
        lookback: int = 120,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        max_holding: int = 5,
        target_gross: float = 1.0,
        vol_scale: bool = True,
        vol_lookback: int = 21,
        vol_floor: float = 0.005,
        trend_sma_period: int = 200,
        trend_slope_limit: float = 0.001,
        stop_loss_pct: float = 0.03,
        dd_circuit_breaker: float = -0.15,
        dd_cooldown_days: int = 42,
    ) -> None:
        params = dict(
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            max_holding=max_holding,
            target_gross=target_gross,
            vol_scale=vol_scale,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            trend_sma_period=trend_sma_period,
            trend_slope_limit=trend_slope_limit,
            stop_loss_pct=stop_loss_pct,
            dd_circuit_breaker=dd_circuit_breaker,
            dd_cooldown_days=dd_cooldown_days,
        )
        super().__init__(name="MeanReversion", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Rolling z-score of log prices."""
        log_prices = np.log(prices.clip(lower=1e-8))
        mu = log_prices.rolling(self.params["lookback"], min_periods=self.params["lookback"] // 2).mean()
        sigma = log_prices.rolling(self.params["lookback"], min_periods=self.params["lookback"] // 2).std()
        zscore = (log_prices - mu) / sigma.clip(lower=1e-8)
        return zscore

    def _ou_adaptive_thresholds(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Use Ornstein-Uhlenbeck calibration per Bertram (2010) / Leung & Li
        (2015) to compute per-asset, time-varying optimal entry thresholds.

        Assets with stronger mean-reversion (higher kappa, lower half-life)
        get tighter thresholds, while weak mean-reverters get wider ones.

        Returns a DataFrame of OU-based entry threshold multipliers per
        asset (values typically 1.0 to 3.0 in z-score space).
        """
        try:
            from qrt.models.ornstein_uhlenbeck import OUCalibrator
        except ImportError:
            return pd.DataFrame(
                self.params["entry_threshold"],
                index=prices.index, columns=prices.columns,
            )

        lookback = self.params["lookback"]
        ou = OUCalibrator(min_kappa=0.005, max_half_life=lookback)
        log_prices = np.log(prices.clip(lower=1e-8))

        # Compute OU-optimal thresholds per asset (recalibrate periodically)
        thresholds = pd.DataFrame(
            self.params["entry_threshold"],
            index=prices.index, columns=prices.columns,
        )

        recal_freq = max(21, lookback // 4)

        for col in prices.columns:
            series = log_prices[col].dropna()
            for t in range(lookback, len(series), recal_freq):
                window = series.iloc[max(0, t - lookback):t]
                params = ou.calibrate(window)
                if params is not None and params.half_life < lookback:
                    # OU-optimal: tighter threshold for strong MR
                    opt_thresh = ou.optimal_thresholds(params, cost=0.002)
                    eq_std = params.sigma / np.sqrt(2 * params.kappa)
                    # Convert to z-score units
                    z_entry = abs(opt_thresh.entry_short - params.mu) / max(eq_std, 1e-8)
                    z_entry = np.clip(z_entry, 1.0, 4.0)
                    # Apply forward for recal_freq days
                    end_t = min(t + recal_freq, len(series))
                    dates_range = series.index[t:end_t]
                    thresholds.loc[dates_range, col] = z_entry

        return thresholds

    def _compute_trend_mask(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Return a boolean mask (True = OK to trade) based on the slope of
        the 200-day SMA.  When the trend is strong the mask is False.
        """
        sma_period: int = self.params["trend_sma_period"]
        slope_limit: float = self.params["trend_slope_limit"]

        sma = prices.rolling(sma_period, min_periods=sma_period // 2).mean()
        # Daily slope approximated as first difference of the SMA
        sma_slope = sma.diff() / sma.shift(1).clip(lower=1e-8)
        # Allow trading only when the trend is near-flat
        flat_trend = sma_slope.abs() < slope_limit
        return flat_trend

    def _apply_holding_rules(
        self,
        raw_signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enforce entry/exit threshold, max holding period, and stop-loss
        via a forward-pass simulation.

        raw_signals contains the raw -z_score values (continuous).
        We apply entry/exit rules to produce discrete {-1, 0, +1} signals.
        """
        entry: float = self.params["entry_threshold"]
        exit_thr: float = self.params["exit_threshold"]
        max_hold: int = self.params["max_holding"]
        stop_loss: float = self.params["stop_loss_pct"]

        n_dates, n_assets = raw_signals.shape
        signals_arr = np.zeros((n_dates, n_assets), dtype=float)
        position = np.zeros(n_assets, dtype=float)   # current held signal
        hold_counter = np.zeros(n_assets, dtype=int)  # days held
        entry_price = np.full(n_assets, np.nan, dtype=float)  # price at entry

        # raw_signals already = -z_score; we use z_score directly for thresholds
        # We need z_score = -raw_signals to evaluate entry/exit
        z_arr = (-raw_signals).values  # shape (n_dates, n_assets)
        price_arr = prices.values

        for t in range(n_dates):
            z_t = z_arr[t]

            for i in range(n_assets):
                z = z_t[i]
                if np.isnan(z):
                    position[i] = 0.0
                    hold_counter[i] = 0
                    entry_price[i] = np.nan
                    continue

                if position[i] != 0:
                    hold_counter[i] += 1

                    # Stop-loss check
                    cur_price = price_arr[t, i]
                    if not np.isnan(entry_price[i]) and not np.isnan(cur_price):
                        if position[i] > 0:
                            pnl_pct = (cur_price - entry_price[i]) / entry_price[i]
                        else:
                            pnl_pct = (entry_price[i] - cur_price) / entry_price[i]
                        if pnl_pct < -stop_loss:
                            position[i] = 0.0
                            hold_counter[i] = 0
                            entry_price[i] = np.nan
                            signals_arr[t, i] = 0.0
                            continue

                    # Exit conditions
                    if (
                        hold_counter[i] >= max_hold
                        or abs(z) <= exit_thr
                        or (position[i] > 0 and z < -entry)   # signal flipped strongly
                        or (position[i] < 0 and z > entry)
                    ):
                        position[i] = 0.0
                        hold_counter[i] = 0
                        entry_price[i] = np.nan

                # Entry condition (only if flat)
                if position[i] == 0:
                    if z > entry:
                        position[i] = -1.0   # mean-revert: price high -> short
                        hold_counter[i] = 0
                        entry_price[i] = price_arr[t, i]
                    elif z < -entry:
                        position[i] = 1.0    # price low -> long
                        hold_counter[i] = 0
                        entry_price[i] = price_arr[t, i]

                signals_arr[t, i] = position[i]

        return pd.DataFrame(signals_arr, index=raw_signals.index, columns=raw_signals.columns)

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
        Compute mean-reversion signals with entry/exit rules.

        Returns
        -------
        pd.DataFrame
            Signals in {-1, 0, +1}, same shape as *prices*.
        """
        zscore = self._compute_zscore(prices)
        raw_signals = -zscore  # mean-reversion: fade deviations
        signals = self._apply_holding_rules(raw_signals, prices)

        # --- Trend filter: suppress signals when a strong trend exists ---
        trend_ok = self._compute_trend_mask(prices)
        signals = signals.where(trend_ok, other=0.0)

        # --- Drawdown circuit breaker ---
        dd_threshold: float = self.params["dd_circuit_breaker"]
        cooldown: int = self.params["dd_cooldown_days"]

        # Compute per-bar strategy return (equal-weighted proxy from signals
        # and returns).  This is an approximation used only for the breaker.
        strat_ret = (signals.shift(1) * returns).sum(axis=1)
        cum_ret = strat_ret.cumsum()

        # Track running high-water mark and drawdown
        hwm = cum_ret.cummax()
        drawdown = cum_ret - hwm

        # Identify bars where the drawdown exceeds the threshold
        breaker_on = drawdown < dd_threshold
        # Forward-fill the breaker for cooldown_days
        if breaker_on.any():
            breaker_mask = breaker_on.copy()
            idx = breaker_mask.index
            # Expand each True to cover the next `cooldown` days
            true_locs = np.where(breaker_mask.values)[0]
            mask_arr = breaker_mask.values.copy()
            for loc in true_locs:
                end = min(loc + cooldown, len(mask_arr))
                mask_arr[loc:end] = True
            breaker_mask = pd.Series(mask_arr, index=idx)
            signals[breaker_mask.values] = 0.0

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Optionally volatility-scale and gross-normalize.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals``.
        returns : pd.DataFrame, optional
            Daily returns for vol scaling.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]

        if self.params["vol_scale"] and returns is not None:
            vol_lookback: int = self.params["vol_lookback"]
            vol_floor: float = self.params["vol_floor"]
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
