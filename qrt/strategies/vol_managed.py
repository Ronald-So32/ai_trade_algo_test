"""
Volatility-Managed Portfolio Overlay
======================================
Scale an existing portfolio's exposure by the ratio of target volatility to
realised portfolio volatility.  Based on Moreira & Muir (2017).

  managed_weight_t = base_weight_t × min(target_vol / realised_vol_{t-1}, max_leverage)

The overlay can wrap any strategy that produces a weights DataFrame or it can
directly accept weights from an external source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class VolatilityManagedOverlay(Strategy):
    """
    Volatility-targeting overlay applied to a base portfolio.

    Parameters
    ----------
    target_vol : float
        Target annualised portfolio volatility (default 0.10 → 10 %).
    vol_lookback : int
        Window (in days) for estimating realised portfolio volatility
        (default 21 trading days ≈ 1 month).
    max_leverage : float
        Hard cap on gross leverage (default 2.0).
    vol_floor : float
        Minimum realised vol to prevent extreme leverage (default 0.005).
    ewm_span : int | None
        If set, use an EWM-smoothed vol estimator with this span instead of
        a simple rolling std.  Useful to de-lag vol estimates (default None).
    annualisation_factor : float
        Multiplier to annualise daily vol  (default 252).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Moreira & Muir (2017) — \"Volatility-Managed Portfolios\""
        ),
        "historical_evidence": (
            "Improves Sharpe on many factors; debated whether improvements "
            "survive transaction costs"
        ),
        "implementation_risks": (
            "Whipsaw in volatile regimes, over-leverage in calm periods, "
            "cost of frequent rebalancing"
        ),
        "realistic_expectations": (
            "Research suggests improved risk-adjusted returns; real-world benefit "
            "depends on cost management"
        ),
    }

    def __init__(
        self,
        target_vol: float = 0.10,
        vol_lookback: int = 21,
        max_leverage: float = 2.0,
        vol_floor: float = 0.005,
        ewm_span: int | None = None,
        annualisation_factor: float = 252.0,
    ) -> None:
        params = dict(
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
            vol_floor=vol_floor,
            ewm_span=ewm_span,
            annualisation_factor=annualisation_factor,
        )
        super().__init__(name="VolatilityManagedOverlay", params=params)

    # ------------------------------------------------------------------
    # Realised-vol estimation
    # ------------------------------------------------------------------

    def _estimate_portfolio_vol(
        self,
        portfolio_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute a rolling estimate of annualised portfolio volatility.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily P&L series of the base portfolio.

        Returns
        -------
        pd.Series
            Annualised vol estimate, same index as *portfolio_returns*.
        """
        ann = self.params["annualisation_factor"]
        span = self.params["ewm_span"]
        lb = self.params["vol_lookback"]
        vf = self.params["vol_floor"]

        if span is not None:
            # EWM variance estimator
            ewm_var = portfolio_returns.ewm(span=span, min_periods=max(1, span // 4)).var()
            vol = (ewm_var * ann).pow(0.5).clip(lower=vf)
        else:
            vol = (
                portfolio_returns.rolling(lb, min_periods=max(1, lb // 2))
                .std()
                .mul(np.sqrt(ann))
                .clip(lower=vf)
            )

        return vol

    # ------------------------------------------------------------------
    # Scaling logic
    # ------------------------------------------------------------------

    def _compute_scale(self, portfolio_returns: pd.Series) -> pd.Series:
        """
        Compute the scaling factor for each day based on *previous* day's vol.

        scale_t = clip(target_vol / vol_{t-1}, 0, max_leverage)
        """
        target_vol: float = self.params["target_vol"]
        max_lev: float = self.params["max_leverage"]

        vol = self._estimate_portfolio_vol(portfolio_returns)
        # Use lagged vol so we don't use today's information
        lagged_vol = vol.shift(1).bfill().clip(lower=self.params["vol_floor"])
        scale = (target_vol / lagged_vol).clip(lower=0.0, upper=max_lev)
        return scale

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
        The overlay itself does not generate directional signals independently.
        If ``base_weights`` is supplied (via kwargs), it is used to compute the
        base portfolio return series for vol estimation; otherwise the equally-
        weighted portfolio is used.

        Returns a DataFrame of scale factors (one column per asset, but all
        columns share the same scalar scale on a given day), representing the
        vol-managed version of a flat +1 signal.

        Keyword Args
        ------------
        base_weights : pd.DataFrame, optional
            Pre-computed base portfolio weights to overlay.

        Returns
        -------
        pd.DataFrame
            Scale factor multiplied by 1.0, shape same as *prices*.
            Useful as a pass-through for compute_weights to multiply against.
        """
        base_weights: pd.DataFrame | None = kwargs.get("base_weights", None)

        if base_weights is not None:
            port_ret = (base_weights.shift(1) * returns).sum(axis=1)
        else:
            # Equal-weight base portfolio
            port_ret = returns.mean(axis=1)

        scale = self._compute_scale(port_ret)

        # Broadcast scale to all assets
        signals = pd.DataFrame(
            np.outer(scale.values, np.ones(prices.shape[1])),
            index=prices.index,
            columns=prices.columns,
        )
        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        base_weights: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply volatility scaling to base portfolio weights.

        Parameters
        ----------
        signals : pd.DataFrame
            Scale-factor DataFrame from ``generate_signals``.
        base_weights : pd.DataFrame, optional
            Weights of the underlying strategy.  If None, a uniform long-only
            portfolio (equal weight) is assumed.

        Returns
        -------
        pd.DataFrame
            Vol-managed portfolio weights, same shape as *signals*.
        """
        if base_weights is None:
            base_weights = kwargs.get("base_weights", None)

        if base_weights is None:
            n_assets = signals.shape[1]
            base_weights = pd.DataFrame(
                np.full_like(signals.values, 1.0 / n_assets),
                index=signals.index,
                columns=signals.columns,
            )

        # Scale factor is constant across columns (same value each row)
        scale_factor = signals.iloc[:, 0]  # one column suffices

        managed_weights = base_weights.mul(scale_factor, axis=0)
        return managed_weights.fillna(0.0)

    # ------------------------------------------------------------------
    # Overlay helper
    # ------------------------------------------------------------------

    def overlay(
        self,
        base_weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convenience method: directly overlay volatility management on a
        pre-existing base weight DataFrame.

        Parameters
        ----------
        base_weights : pd.DataFrame
            Portfolio weights from any underlying strategy.
        returns : pd.DataFrame
            Daily asset returns.

        Returns
        -------
        pd.DataFrame
            Vol-managed weights, same shape as *base_weights*.
        """
        port_ret = (base_weights.shift(1) * returns).sum(axis=1)
        scale = self._compute_scale(port_ret)
        managed = base_weights.mul(scale, axis=0)
        return managed.fillna(0.0)

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        base_weights: pd.DataFrame | None = None,
        **kwargs,
    ) -> dict:
        """
        End-to-end: (optionally wrap a base strategy) → vol-managed weights
        → backtest summary.

        Parameters
        ----------
        prices : pd.DataFrame
        returns : pd.DataFrame
        base_weights : pd.DataFrame, optional
            If provided, this strategy's weights are vol-managed.  Otherwise
            an equal-weight portfolio is assumed.
        """
        signals = self.generate_signals(
            prices, returns, base_weights=base_weights, **kwargs
        )
        weights = self.compute_weights(
            signals, base_weights=base_weights, **kwargs
        )
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
