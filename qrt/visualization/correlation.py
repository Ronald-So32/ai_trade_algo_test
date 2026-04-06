"""
Correlation Visualizer
======================
Heatmaps and rolling-correlation charts for strategies and assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

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
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_DIVERG_CMAP = sns.diverging_palette(220, 10, as_cmap=True)


def _save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _mask_upper(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[np.triu_indices(n, k=1)] = True
    return mask


class CorrelationVisualizer:
    """
    Correlation heatmaps and rolling pair-wise correlation charts.
    """

    # ------------------------------------------------------------------
    # Strategy Correlation Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_correlation_heatmap(
        strategy_returns_dict: Dict[str, pd.Series],
        title: str = "Strategy Correlation",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Lower-triangular Seaborn correlation heatmap for a set of strategies.

        Parameters
        ----------
        strategy_returns_dict:
            Mapping of strategy name -> daily returns Series.
        title:
            Figure title.
        save_path:
            Optional file path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = pd.DataFrame(strategy_returns_dict).dropna()
        corr = df.corr()
        n = len(corr)

        fig_size = max(6, n * 0.75 + 1)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        mask = _mask_upper(n)
        sns.heatmap(
            corr,
            ax=ax,
            mask=mask,
            cmap=_DIVERG_CMAP,
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="#E5E7EB",
            square=True,
            cbar_kws={"shrink": 0.7, "label": "Pearson Correlation"},
        )

        ax.set_title(title, fontweight="bold", pad=14)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Asset Correlation Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def asset_correlation_heatmap(
        returns: pd.DataFrame,
        top_n: int = 30,
        title: str = "Asset Correlation",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Clustered (hierarchical) Seaborn heatmap for assets.

        Parameters
        ----------
        returns:
            DataFrame of asset returns (columns = asset names).
        top_n:
            Maximum number of assets to display, selected by highest variance.
        title:
            Figure title.
        save_path:
            Optional file path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = returns.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # Select top_n by variance
        if df.shape[1] > top_n:
            top_cols = df.var().nlargest(top_n).index
            df = df[top_cols]

        corr = df.corr()
        n = len(corr)

        # Hierarchical clustering to reorder
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            dist = np.clip(1 - corr.values, 0, 2)
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist)
            Z = linkage(condensed, method="ward")
            order = leaves_list(Z)
            corr = corr.iloc[order, :].iloc[:, order]
        except ImportError:
            pass  # scipy not available – skip reordering

        fig_size = max(8, n * 0.38 + 2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

        annot = n <= 25
        sns.heatmap(
            corr,
            ax=ax,
            cmap=_DIVERG_CMAP,
            center=0,
            vmin=-1,
            vmax=1,
            annot=annot,
            fmt=".1f" if annot else "",
            linewidths=0.3 if n > 15 else 0.5,
            linecolor="#E5E7EB",
            square=True,
            cbar_kws={"shrink": 0.6, "label": "Pearson Correlation"},
        )

        ax.set_title(title, fontweight="bold", pad=14)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Rolling Correlation
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_correlation(
        returns: pd.DataFrame,
        pairs: Sequence[Tuple[str, str]],
        window: int = 63,
        title: str = "Rolling Correlation",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Line chart of rolling pairwise correlation for specified pairs.

        Parameters
        ----------
        returns:
            DataFrame of returns (columns = asset / strategy names).
        pairs:
            List of (col_a, col_b) tuples to plot.
        window:
            Rolling window in periods.
        title:
            Figure title.
        save_path:
            Optional file path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = returns.dropna(how="all")
        n_pairs = len(pairs)
        palette = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(13, 5))

        for i, (a, b) in enumerate(pairs):
            if a not in df.columns or b not in df.columns:
                continue
            roll_corr = df[a].rolling(window).corr(df[b])
            label = f"{a} / {b}"
            color = matplotlib.colors.to_hex(palette(i % 10))
            ax.plot(roll_corr.index, roll_corr.values, linewidth=1.4, label=label, color=color)

        ax.axhline(0, color="#374151", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="#9CA3AF", linewidth=0.5, linestyle=":")
        ax.axhline(-1, color="#9CA3AF", linewidth=0.5, linestyle=":")

        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{title}  (window = {window} periods)", fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            frameon=True,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig
