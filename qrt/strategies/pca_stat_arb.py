"""
PCA Statistical Arbitrage Strategy
=====================================
1. Apply PCA to the returns cross-section to extract the top K eigenportfolios
   (systematic factors).
2. Compute idiosyncratic residuals: actual returns minus factor model predicted
   returns.
3. Construct a rolling z-score of the cumulative residual (price-equivalent).
4. Mean-revert on residuals: go long assets with large negative z-score, short
   assets with large positive z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .base import Strategy


class PCAStatArb(Strategy):
    """
    PCA-based statistical arbitrage (eigenportfolio mean reversion).

    Parameters
    ----------
    n_components : int
        Number of PCA factors to extract (default 5).
    lookback : int
        Rolling window (days) for PCA estimation and residual z-score (default 252).
    entry_z : float
        |z-score| threshold for entering a position (default 1.5).
    exit_z : float
        |z-score| threshold for exiting a position (default 0.5).
    max_holding : int
        Maximum days to hold a position (default 20).
    target_gross : float
        Target gross exposure (default 1.0).
    refit_freq : int
        How often (in days) to refit the PCA model (default 63).
    zscore_window : int
        Window for computing z-score of residuals (default 60).
    min_variance_explained : float
        Minimum fraction of variance the K components must explain; if not met,
        more components are added automatically (default 0.5).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Avellaneda & Lee (2010) — \"Statistical Arbitrage in the US Equities Market\""
        ),
        "historical_evidence": (
            "Profitable in academic settings; real-world alpha has declined with competition"
        ),
        "implementation_risks": (
            "Model misspecification, changing factor structure, slow mean reversion"
        ),
        "realistic_expectations": (
            "Research-supported premium from idiosyncratic mean reversion; "
            "capacity limited, alpha decayed post-publication"
        ),
    }

    def __init__(
        self,
        n_components: int = 5,
        lookback: int = 252,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        max_holding: int = 20,
        target_gross: float = 1.0,
        refit_freq: int = 63,
        zscore_window: int = 60,
        min_variance_explained: float = 0.5,
    ) -> None:
        params = dict(
            n_components=n_components,
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding=max_holding,
            target_gross=target_gross,
            refit_freq=refit_freq,
            zscore_window=zscore_window,
            min_variance_explained=min_variance_explained,
        )
        super().__init__(name="PCAStatArb", params=params)

    # ------------------------------------------------------------------
    # PCA fitting
    # ------------------------------------------------------------------

    def _fit_pca(
        self,
        returns_window: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA on the returns window.

        Returns
        -------
        (components, explained_variance_ratio)
        components shape: (n_components, n_assets)
        """
        n_components: int = self.params["n_components"]
        min_var: float = self.params["min_variance_explained"]

        # Drop columns with all-NaN; fill remaining NaN with 0
        clean = returns_window.dropna(axis=1, how="all").fillna(0.0)
        n_assets = clean.shape[1]
        k = min(n_components, n_assets - 1, clean.shape[0] - 1)
        if k < 1:
            return np.zeros((1, n_assets)), np.array([0.0])

        pca = PCA(n_components=k)
        pca.fit(clean.values)  # fit on (n_dates, n_assets)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        # Automatically expand k if variance threshold not reached
        if cum_var[-1] < min_var and k < n_assets - 1:
            k_new = min(
                np.searchsorted(cum_var, min_var) + 1,
                n_assets - 1,
                clean.shape[0] - 1,
            )
            if k_new > k:
                pca = PCA(n_components=int(k_new))
                pca.fit(clean.values)

        # Return loadings aligned back to original columns (zeros for dropped cols)
        components_clean = pca.components_  # shape (k, n_clean_assets)
        full_components = np.zeros((components_clean.shape[0], returns_window.shape[1]))
        col_idx = [returns_window.columns.get_loc(c) for c in clean.columns]
        full_components[:, col_idx] = components_clean

        return full_components, pca.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residuals(
        returns: pd.DataFrame,
        components: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute idiosyncratic residuals: e = r - loadings @ (loadings.T @ r.T).T

        Parameters
        ----------
        returns : pd.DataFrame, shape (n_dates, n_assets)
        components : np.ndarray, shape (k, n_assets)

        Returns
        -------
        pd.DataFrame of residuals, same shape as returns.
        """
        R = returns.fillna(0.0).values  # (n_dates, n_assets)
        L = components  # (k, n_assets)

        # Project onto factor space and subtract
        factor_returns = R @ L.T  # (n_dates, k)
        systematic = factor_returns @ L  # (n_dates, n_assets)
        residuals = R - systematic

        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

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
        Generate mean-reversion signals on PCA residuals.

        Returns
        -------
        pd.DataFrame
            Signals in {-1, 0, +1}, same shape as *prices*.
        """
        lookback: int = self.params["lookback"]
        entry_z: float = self.params["entry_z"]
        exit_z: float = self.params["exit_z"]
        max_hold: int = self.params["max_holding"]
        refit_freq: int = self.params["refit_freq"]
        z_win: int = self.params["zscore_window"]

        dates = prices.index
        n_dates = len(dates)
        signals_out = pd.DataFrame(0.0, index=dates, columns=prices.columns)

        if n_dates <= lookback:
            return signals_out

        # State
        position = pd.Series(0.0, index=prices.columns)
        hold_counter = pd.Series(0, index=prices.columns)

        # Cumulative idiosyncratic residual (think of it as a synthetic "price")
        cum_resid = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        current_components: np.ndarray | None = None
        last_refit: int = -refit_freq

        for t in range(lookback, n_dates):
            # ---- Refit PCA if scheduled ----
            if t - last_refit >= refit_freq:
                ret_window = returns.iloc[t - lookback: t]
                current_components, _ = self._fit_pca(ret_window)
                last_refit = t

            if current_components is None:
                continue

            # ---- Compute residuals for the most recent return ----
            single_ret = returns.iloc[[t]].fillna(0.0)
            single_resid = self._compute_residuals(single_ret, current_components)
            cum_resid.iloc[t] = cum_resid.iloc[t - 1] + single_resid.iloc[0]

            # ---- Z-score of cumulative residual ----
            w_start = max(0, t - z_win)
            window_cum = cum_resid.iloc[w_start: t + 1]
            mu = window_cum.mean()
            sigma = window_cum.std().clip(lower=1e-10)
            z = (cum_resid.iloc[t] - mu) / sigma

            # ---- Update positions ----
            for col in prices.columns:
                z_i = float(z[col])
                if np.isnan(z_i):
                    continue

                pos = float(position[col])

                if pos != 0:
                    hold_counter[col] += 1
                    # Exit conditions
                    if (
                        int(hold_counter[col]) >= max_hold
                        or abs(z_i) <= exit_z
                    ):
                        position[col] = 0.0
                        hold_counter[col] = 0
                        pos = 0.0

                if pos == 0:
                    if z_i > entry_z:
                        position[col] = -1.0  # cumulative residual high → mean revert down
                        hold_counter[col] = 0
                    elif z_i < -entry_z:
                        position[col] = 1.0   # cumulative residual low → mean revert up
                        hold_counter[col] = 0

                signals_out.iloc[t, signals_out.columns.get_loc(col)] = float(position[col])

        return signals_out

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Gross-normalise to target_gross exposure.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross: float = self.params["target_gross"]

        if returns is not None:
            lookback: int = self.params["lookback"]
            realized_vol = (
                returns.rolling(lookback // 4, min_periods=5)
                .std()
                .mul(np.sqrt(252))
                .clip(lower=1e-6)
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
        """End-to-end: signals → weights → backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, returns=returns, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
