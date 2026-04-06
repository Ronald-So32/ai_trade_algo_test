"""
Risk Decomposition
==================
Component and marginal risk contribution charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _component_risk_contributions(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Return the vector of component risk contributions.

    CRC_i = w_i * (Sigma * w)_i / sqrt(w^T Sigma w)
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    portfolio_vol = np.sqrt(w @ sigma @ w)
    if portfolio_vol < 1e-12:
        return np.zeros_like(w)
    marginal = sigma @ w
    return w * marginal / portfolio_vol


def _generate_palette(n: int) -> list[str]:
    """Return n visually distinct hex colours."""
    cmap = plt.get_cmap("tab20" if n <= 20 else "turbo")
    return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


class RiskDecomposition:
    """
    Risk decomposition charts: component risk, marginal risk, and
    stacked risk contribution breakdowns.
    """

    # ------------------------------------------------------------------
    # Risk Contribution by Strategy (stacked bar, time-series)
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contribution_by_strategy(
        strategy_weights: pd.DataFrame,
        strategy_returns: pd.DataFrame,
        covariance: Optional[pd.DataFrame] = None,
        window: int = 63,
        title: str = "Risk Contribution by Strategy",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Stacked-bar chart of rolling risk contribution by strategy.

        Parameters
        ----------
        strategy_weights:
            DataFrame (dates x strategies) of portfolio weights.
        strategy_returns:
            DataFrame (dates x strategies) of strategy returns.
        covariance:
            Optional fixed (n_strategies x n_strategies) covariance matrix as
            a DataFrame.  When *None* the rolling sample covariance of
            *strategy_returns* is used.
        window:
            Rolling window for covariance estimation when *covariance* is None.
        """
        weights_df = strategy_weights.dropna(how="all")
        returns_df = strategy_returns.dropna(how="all")
        strategies = list(weights_df.columns)
        n = len(strategies)
        palette = _generate_palette(n)

        records: list[dict] = []
        dates = weights_df.index

        for date in dates:
            w = weights_df.loc[date].values
            if covariance is not None:
                cov = covariance.loc[strategies, strategies].values
            else:
                # Use trailing window returns to estimate cov
                hist = returns_df.loc[:date].tail(window)
                if len(hist) < 5:
                    continue
                cov = hist[strategies].cov().values
            crc = _component_risk_contributions(w, cov)
            total = crc.sum()
            pct = (crc / total * 100) if total > 1e-12 else crc * 0
            row = dict(zip(strategies, pct))
            row["date"] = date
            records.append(row)

        if not records:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(title, fontweight="bold")
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            _save(fig, save_path)
            return fig

        df = pd.DataFrame(records).set_index("date")

        fig, ax = plt.subplots(figsize=(14, 6))
        bottom = np.zeros(len(df))
        for i, col in enumerate(strategies):
            vals = df[col].fillna(0).values
            ax.bar(
                df.index,
                vals,
                bottom=bottom,
                color=palette[i],
                label=col,
                width=max(1, (df.index[-1] - df.index[0]).days / len(df) * 0.9),
                align="center",
            )
            bottom += vals

        ax.axhline(100, color="#374151", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Risk Contribution (%)")
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            frameon=True,
        )
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Risk Contribution by Asset (static snapshot)
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contribution_by_asset(
        weights: Union[pd.Series, np.ndarray],
        asset_returns: pd.DataFrame,
        covariance: Optional[pd.DataFrame] = None,
        top_n: int = 20,
        title: str = "Risk Contribution by Asset",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of component risk contribution per asset.

        Parameters
        ----------
        weights:
            Asset weights as a Series (index = asset names) or ndarray.
        asset_returns:
            DataFrame of asset returns used to estimate the covariance
            when *covariance* is None.
        covariance:
            Optional pre-computed covariance matrix DataFrame.
        top_n:
            Show only the top-N contributors.
        """
        if isinstance(weights, pd.Series):
            labels = list(weights.index)
            w = weights.values.astype(float)
        else:
            w = np.asarray(weights, dtype=float)
            labels = list(asset_returns.columns[: len(w)])

        if covariance is not None:
            cov = covariance.loc[labels, labels].values
        else:
            cov = asset_returns[labels].cov().values

        crc = _component_risk_contributions(w, cov)
        total = crc.sum()
        pct = crc / total * 100 if total > 1e-12 else crc

        series = pd.Series(pct, index=labels).sort_values(ascending=False)
        series = series.head(top_n)

        colors = [
            "#2563EB" if v >= 0 else "#DC2626"
            for v in series.values
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, len(series) * 0.38 + 1.5)))
        ax.barh(series.index[::-1], series.values[::-1], color=colors[::-1], height=0.65)
        ax.axvline(0, color="#374151", linewidth=0.8)

        for i, (label, val) in enumerate(zip(series.index[::-1], series.values[::-1])):
            ax.text(
                val + 0.2,
                i,
                f"{val:.1f}%",
                va="center",
                fontsize=8.5,
                color="#1F2937",
            )

        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Risk Contribution (%)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Marginal Risk Contribution
    # ------------------------------------------------------------------

    @staticmethod
    def marginal_risk_contribution(
        weights: Union[pd.Series, np.ndarray],
        covariance: Union[pd.DataFrame, np.ndarray],
    ) -> pd.Series:
        """
        Compute the marginal risk contribution (dVol/dw_i).

        Parameters
        ----------
        weights:
            Portfolio weights.
        covariance:
            Covariance matrix.

        Returns
        -------
        pd.Series of marginal risk contributions (same index as *weights*).
        """
        if isinstance(weights, pd.Series):
            labels = weights.index
            w = weights.values.astype(float)
        else:
            w = np.asarray(weights, dtype=float)
            labels = pd.RangeIndex(len(w))

        if isinstance(covariance, pd.DataFrame):
            sigma = covariance.values.astype(float)
        else:
            sigma = np.asarray(covariance, dtype=float)

        portfolio_vol = np.sqrt(w @ sigma @ w)
        if portfolio_vol < 1e-12:
            mrc = np.zeros_like(w)
        else:
            mrc = (sigma @ w) / portfolio_vol

        return pd.Series(mrc, index=labels, name="marginal_risk_contribution")
