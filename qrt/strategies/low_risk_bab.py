"""
Low-Risk / Betting-Against-Beta Strategy
==========================================
Two modes of operation:

1. **BAB (Betting Against Beta)** -- rank assets by rolling beta to the market,
   go long the low-beta quintile and short the high-beta quintile, then lever
   each leg so the portfolio is beta-neutral (Frazzini & Pedersen, 2014).

2. **Low-Volatility** -- rank assets by realised volatility, go long the lowest
   quintile and short the highest quintile with equal weights within each leg.

Both modes rebalance on a monthly cadence (every *rebalance_freq* trading days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class LowRiskBAB(Strategy):
    """
    Low-Risk / Betting-Against-Beta strategy.

    Parameters
    ----------
    vol_window : int
        Rolling window (days) for realised volatility estimation (default 63).
    beta_window : int
        Rolling window (days) for beta estimation (default 252).
    long_pct : float
        Fraction of the universe forming the long (low-risk) leg (default 0.20).
    short_pct : float
        Fraction of the universe forming the short (high-risk) leg (default 0.20).
    target_gross : float
        Target gross exposure after normalisation (default 1.0).
    rebalance_freq : int
        Rebalance every *rebalance_freq* trading days (default 21).
    mode : str
        ``"bab"`` for beta-neutral BAB spread, ``"low_vol"`` for simple
        low-volatility ranking (default ``"bab"``).
    beta_neutral : bool
        Whether to lever the long/short legs to achieve beta neutrality in
        BAB mode (default True).  Ignored when *mode* is ``"low_vol"``.
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Frazzini & Pedersen (2014) — \"Betting Against Beta\""
        ),
        "historical_evidence": (
            "Low-beta stocks historically outperform on risk-adjusted basis; "
            "strong global evidence"
        ),
        "implementation_risks": (
            "Can reverse sharply during momentum rallies, interest rate sensitivity, "
            "leverage constraints"
        ),
        "realistic_expectations": (
            "Research-supported premium driven by leverage constraints; "
            "can have extended periods of underperformance"
        ),
    }

    def __init__(
        self,
        vol_window: int = 63,
        beta_window: int = 252,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
        target_gross: float = 1.0,
        rebalance_freq: int = 21,
        mode: str = "bab",
        beta_neutral: bool = True,
    ) -> None:
        if mode not in ("bab", "low_vol"):
            raise ValueError(f"mode must be 'bab' or 'low_vol', got '{mode}'")
        params = dict(
            vol_window=vol_window,
            beta_window=beta_window,
            long_pct=long_pct,
            short_pct=short_pct,
            target_gross=target_gross,
            rebalance_freq=rebalance_freq,
            mode=mode,
            beta_neutral=beta_neutral,
        )
        super().__init__(name="LowRiskBAB", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _market_return(
        returns: pd.DataFrame,
        market_caps: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Compute the market (benchmark) return series.

        If *market_caps* is provided, returns are cap-weighted; otherwise
        equal-weighted across all available assets each day.
        """
        if market_caps is not None:
            # Align market caps to returns, forward-fill gaps
            caps = market_caps.reindex_like(returns).ffill()
            cap_weights = caps.div(caps.sum(axis=1), axis=0)
            mkt = (returns * cap_weights).sum(axis=1)
        else:
            mkt = returns.mean(axis=1)
        return mkt

    @staticmethod
    def _rolling_beta(
        asset_returns: pd.DataFrame,
        market_returns: pd.Series,
        window: int,
    ) -> pd.DataFrame:
        """Compute rolling OLS beta of each asset to the market.

        beta_i = cov(r_i, r_m) / var(r_m)  over *window* days.
        """
        min_periods = max(1, window // 2)
        mkt = market_returns.reindex(asset_returns.index)

        # Rolling variance of market
        mkt_var = mkt.rolling(window, min_periods=min_periods).var()

        betas = pd.DataFrame(np.nan, index=asset_returns.index, columns=asset_returns.columns)
        for col in asset_returns.columns:
            cov = (
                asset_returns[col]
                .rolling(window, min_periods=min_periods)
                .cov(mkt)
            )
            betas[col] = cov / mkt_var.replace(0, np.nan)

        return betas

    @staticmethod
    def _realised_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Annualised realised volatility (rolling std * sqrt(252))."""
        min_periods = max(1, window // 2)
        return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)

    @staticmethod
    def _downside_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Annualised downside volatility (std of negative returns only)."""
        min_periods = max(1, window // 4)
        neg_returns = returns.clip(upper=0.0)
        return neg_returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)

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
        Generate long/short signals based on risk ranking.

        For each rebalance date the method:

        1. Computes each asset's rolling beta to the market and realised vol.
        2. Ranks assets by beta (BAB mode) or realised vol (low-vol mode).
        3. Assigns +1 to the low-risk quintile (long) and -1 to the high-risk
           quintile (short).

        Keyword Args
        ------------
        market_caps : pd.DataFrame, optional
            Market-capitalisation DataFrame (dates x assets) used to compute a
            cap-weighted market return.  If omitted, an equal-weight average is
            used as the market proxy.

        Returns
        -------
        pd.DataFrame
            Signal DataFrame, same shape as *prices*.  Values are +1 (long),
            -1 (short), or 0 (flat).
        """
        vol_window: int = self.params["vol_window"]
        beta_window: int = self.params["beta_window"]
        long_pct: float = self.params["long_pct"]
        short_pct: float = self.params["short_pct"]
        rebalance_freq: int = self.params["rebalance_freq"]
        mode: str = self.params["mode"]

        market_caps: pd.DataFrame | None = kwargs.get("market_caps", None)

        # Pre-compute risk measures
        mkt_ret = self._market_return(returns, market_caps)
        betas = self._rolling_beta(returns, mkt_ret, beta_window)
        vols = self._realised_vol(returns, vol_window)
        _downside_vols = self._downside_vol(returns, vol_window)  # stored for diagnostics

        # Choose ranking metric
        rank_metric = betas if mode == "bab" else vols

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Minimum history required before generating signals
        min_history = max(vol_window, beta_window)

        # Determine rebalance dates
        valid_dates = prices.index[min_history:]
        rebalance_indices = list(range(0, len(valid_dates), rebalance_freq))
        rebalance_dates = valid_dates[rebalance_indices]

        current_signal = pd.Series(0.0, index=prices.columns)

        for i, date in enumerate(prices.index):
            if date in rebalance_dates:
                row = rank_metric.loc[date].dropna()
                if len(row) < max(3, int(1 / min(long_pct, short_pct))):
                    # Not enough assets to form quintiles; keep previous signal
                    signals.loc[date] = current_signal
                    continue

                n_long = max(1, int(np.floor(len(row) * long_pct)))
                n_short = max(1, int(np.floor(len(row) * short_pct)))

                ranked = row.rank(ascending=True)

                # Low-risk leg: lowest beta/vol -> long
                long_mask = ranked <= n_long
                # High-risk leg: highest beta/vol -> short
                short_mask = ranked >= (ranked.max() - n_short + 1)

                sig = pd.Series(0.0, index=prices.columns)
                sig[long_mask[long_mask].index] = 1.0
                sig[short_mask[short_mask].index] = -1.0
                current_signal = sig

            signals.loc[date] = current_signal

        # Zero out rows before sufficient history
        signals.iloc[:min_history] = 0.0

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert raw long/short signals into portfolio weights.

        Within each leg, assets are equal-weighted.  In BAB mode with
        *beta_neutral=True*, the long leg is scaled by ``1 / avg_beta_long``
        and the short leg by ``1 / avg_beta_short`` so both legs contribute
        unit beta, achieving a beta-neutral portfolio.

        The final weights are normalised so gross exposure equals
        *target_gross*.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of ``generate_signals``.
        returns : pd.DataFrame, optional
            Daily returns; required for BAB beta-neutral scaling.

        Keyword Args
        ------------
        market_caps : pd.DataFrame, optional
            Passed through to market-return computation for beta estimation.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]
        mode: str = self.params["mode"]
        beta_neutral: bool = self.params["beta_neutral"]
        beta_window: int = self.params["beta_window"]

        market_caps: pd.DataFrame | None = kwargs.get("market_caps", None)

        # Pre-compute betas if needed for beta-neutral scaling
        betas: pd.DataFrame | None = None
        if mode == "bab" and beta_neutral and returns is not None:
            mkt_ret = self._market_return(returns, market_caps)
            betas = self._rolling_beta(returns, mkt_ret, beta_window)

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            row = signals.loc[date]
            long_assets = row[row > 0].index
            short_assets = row[row < 0].index

            if len(long_assets) == 0 and len(short_assets) == 0:
                continue

            # Equal weight within each leg
            w = pd.Series(0.0, index=signals.columns)
            if len(long_assets) > 0:
                w[long_assets] = 1.0 / len(long_assets)
            if len(short_assets) > 0:
                w[short_assets] = -1.0 / len(short_assets)

            # Beta-neutral scaling in BAB mode
            if mode == "bab" and beta_neutral and betas is not None:
                beta_row = betas.loc[date]

                if len(long_assets) > 0:
                    avg_beta_long = beta_row[long_assets].mean()
                    if np.isfinite(avg_beta_long) and avg_beta_long > 1e-6:
                        w[long_assets] *= 1.0 / avg_beta_long

                if len(short_assets) > 0:
                    avg_beta_short = beta_row[short_assets].mean()
                    if np.isfinite(avg_beta_short) and avg_beta_short > 1e-6:
                        w[short_assets] *= 1.0 / avg_beta_short

            # Normalise to target gross exposure
            gross = w.abs().sum()
            if gross > 1e-10:
                w = w * (target_gross / gross)

            weights.loc[date] = w

        return weights.fillna(0.0)

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
