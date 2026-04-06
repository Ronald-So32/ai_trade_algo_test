"""
Regime Visualizer
=================
Overlay regime states on price charts, visualise transition matrices,
and display regime probability distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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

# Default regime colour palette (up to 8 regimes)
_REGIME_COLORS = [
    "#DBEAFE",  # regime 0 – light blue
    "#FEF9C3",  # regime 1 – light yellow
    "#DCFCE7",  # regime 2 – light green
    "#FCE7F3",  # regime 3 – light pink
    "#FEE2E2",  # regime 4 – light red
    "#EDE9FE",  # regime 5 – light purple
    "#FFEDD5",  # regime 6 – light orange
    "#F3F4F6",  # regime 7 – light grey
]

_REGIME_LINE_COLORS = [
    "#1D4ED8",
    "#CA8A04",
    "#15803D",
    "#BE185D",
    "#B91C1C",
    "#7C3AED",
    "#C2410C",
    "#374151",
]


def _save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _normalise_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    return s


class RegimeVisualizer:
    """
    Visualise market regimes:

    - Coloured background overlays on price charts.
    - Regime transition-probability heatmaps.
    - Stacked area / bar charts of smoothed regime probabilities.
    """

    # ------------------------------------------------------------------
    # Regime Overlay
    # ------------------------------------------------------------------

    @staticmethod
    def regime_overlay(
        prices: pd.Series,
        regime_labels: pd.Series,
        regime_names: Optional[Dict[int, str]] = None,
        title: str = "Regime Overlay",
        save_path: Optional[Union[str, Path]] = None,
        secondary_series: Optional[pd.Series] = None,
        secondary_label: str = "Secondary",
    ) -> plt.Figure:
        """
        Price chart with a coloured background shaded by regime.

        Parameters
        ----------
        prices:
            Price (or index) time series.
        regime_labels:
            Integer regime labels aligned on the same index as *prices*.
        regime_names:
            Optional mapping of integer label -> display name, e.g.
            ``{0: "Bull", 1: "Bear", 2: "Sideways"}``.
        title:
            Figure title.
        save_path:
            Optional file path.
        secondary_series:
            Optional second series plotted on a twin y-axis.
        secondary_label:
            Label for the secondary series.

        Returns
        -------
        matplotlib.figure.Figure
        """
        prices = _normalise_index(prices.dropna())
        regime_labels = _normalise_index(regime_labels).reindex(prices.index, method="ffill").fillna(0).astype(int)

        unique_regimes = sorted(regime_labels.unique())
        if regime_names is None:
            regime_names = {r: f"Regime {r}" for r in unique_regimes}

        n_subplots = 2 if secondary_series is not None else 1
        fig, axes = plt.subplots(
            n_subplots, 1,
            figsize=(14, 5 * n_subplots),
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        if n_subplots == 1:
            axes = [axes]

        ax = axes[0]

        # Shade background by regime
        regime_arr = regime_labels.values
        dates = prices.index
        i = 0
        while i < len(regime_arr):
            current = regime_arr[i]
            j = i + 1
            while j < len(regime_arr) and regime_arr[j] == current:
                j += 1
            color_idx = current % len(_REGIME_COLORS)
            ax.axvspan(
                mdates.date2num(dates[i]),
                mdates.date2num(dates[min(j, len(dates) - 1)]),
                facecolor=_REGIME_COLORS[color_idx],
                alpha=0.55,
                zorder=0,
            )
            i = j

        ax.plot(dates, prices.values, color="#1E3A5F", linewidth=1.5, zorder=3, label="Price")
        ax.set_ylabel("Price")
        ax.set_title(title, fontweight="bold", pad=12)

        # Build legend patches
        patches = [
            mpatches.Patch(
                facecolor=_REGIME_COLORS[r % len(_REGIME_COLORS)],
                edgecolor=_REGIME_LINE_COLORS[r % len(_REGIME_LINE_COLORS)],
                label=regime_names.get(r, f"Regime {r}"),
                alpha=0.7,
            )
            for r in unique_regimes
        ]
        ax.legend(
            handles=patches,
            loc="upper left",
            framealpha=0.85,
            title="Regimes",
            title_fontsize=9,
        )

        if secondary_series is not None and n_subplots > 1:
            sec = _normalise_index(secondary_series.dropna())
            ax2 = axes[1]
            # Shade same regimes on secondary panel
            i = 0
            while i < len(regime_arr):
                current = regime_arr[i]
                j = i + 1
                while j < len(regime_arr) and regime_arr[j] == current:
                    j += 1
                color_idx = current % len(_REGIME_COLORS)
                ax2.axvspan(
                    mdates.date2num(dates[i]),
                    mdates.date2num(dates[min(j, len(dates) - 1)]),
                    facecolor=_REGIME_COLORS[color_idx],
                    alpha=0.45,
                    zorder=0,
                )
                i = j
            ax2.plot(sec.index, sec.values, color="#374151", linewidth=1.2, label=secondary_label)
            ax2.set_ylabel(secondary_label)
            ax2.legend(loc="upper left")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Transition Matrix Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def transition_matrix_heatmap(
        transition_matrix: Union[pd.DataFrame, np.ndarray],
        regime_names: Optional[List[str]] = None,
        title: str = "Regime Transition Matrix",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Seaborn heatmap of the regime transition probability matrix.

        Parameters
        ----------
        transition_matrix:
            Square (n_regimes x n_regimes) matrix where entry [i, j] is
            the probability of transitioning from regime i to regime j.
        regime_names:
            Optional list of labels for each regime.
        title:
            Figure title.
        save_path:
            Optional file path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(transition_matrix, pd.DataFrame):
            tm = transition_matrix.copy()
        else:
            tm = pd.DataFrame(transition_matrix)

        n = len(tm)
        if regime_names is not None:
            tm.index = regime_names
            tm.columns = regime_names
        else:
            tm.index = [f"Regime {i}" for i in range(n)]
            tm.columns = [f"Regime {i}" for i in range(n)]

        tm.index.name = "From"
        tm.columns.name = "To"

        fig_size = max(5, n * 1.1 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

        cmap = sns.light_palette("#2563EB", as_cmap=True)
        sns.heatmap(
            tm,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="#E5E7EB",
            square=True,
            cbar_kws={"shrink": 0.7, "label": "Transition Probability"},
        )

        ax.set_title(title, fontweight="bold", pad=14)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Regime Distribution
    # ------------------------------------------------------------------

    @staticmethod
    def regime_distribution(
        regime_probabilities: pd.DataFrame,
        regime_names: Optional[Dict[int, str]] = None,
        title: str = "Regime Probability Distribution",
        save_path: Optional[Union[str, Path]] = None,
        plot_type: str = "area",
    ) -> plt.Figure:
        """
        Stacked area (or bar) chart of smoothed regime probabilities over time.

        Parameters
        ----------
        regime_probabilities:
            DataFrame (dates x n_regimes) of smoothed posterior probabilities,
            where each row sums to (approximately) 1.
        regime_names:
            Optional mapping of column index -> display name.
        title:
            Figure title.
        save_path:
            Optional file path.
        plot_type:
            ``"area"`` (default) for stacked area chart, ``"bar"`` for
            stacked bar chart.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = regime_probabilities.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        n_regimes = df.shape[1]
        if regime_names is not None:
            df.columns = [regime_names.get(c, f"Regime {c}") for c in df.columns]
        else:
            df.columns = [f"Regime {c}" for c in df.columns]

        palette = [_REGIME_LINE_COLORS[i % len(_REGIME_LINE_COLORS)] for i in range(n_regimes)]
        fill_palette = [_REGIME_COLORS[i % len(_REGIME_COLORS)] for i in range(n_regimes)]

        fig, axes = plt.subplots(
            2, 1,
            figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            sharex=True,
        )
        ax_main, ax_dominant = axes

        if plot_type == "bar":
            bottom = np.zeros(len(df))
            bar_width = max(1, (df.index[-1] - df.index[0]).days / len(df) * 0.85)
            for i, col in enumerate(df.columns):
                ax_main.bar(
                    df.index,
                    df[col].values,
                    bottom=bottom,
                    color=fill_palette[i],
                    edgecolor=palette[i],
                    linewidth=0.3,
                    width=bar_width,
                    label=col,
                )
                bottom += df[col].values
        else:
            # Stacked area
            baseline = np.zeros(len(df))
            for i, col in enumerate(df.columns):
                top = baseline + df[col].values
                ax_main.fill_between(
                    df.index,
                    baseline,
                    top,
                    color=fill_palette[i],
                    alpha=0.85,
                    label=col,
                    step=None,
                )
                ax_main.plot(df.index, top, color=palette[i], linewidth=0.4, alpha=0.5)
                baseline = top

        ax_main.set_ylim(0, 1.02)
        ax_main.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax_main.set_ylabel("Regime Probability")
        ax_main.set_title(title, fontweight="bold", pad=12)
        ax_main.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            frameon=True,
        )

        # Bottom panel: dominant regime
        dominant = df.idxmax(axis=1)
        regime_int = pd.Categorical(dominant, categories=df.columns).codes
        regime_series = pd.Series(regime_int, index=df.index)

        for i, col in enumerate(df.columns):
            mask = regime_series.values == i
            if mask.any():
                ax_dominant.fill_between(
                    df.index,
                    0,
                    1,
                    where=mask,
                    color=fill_palette[i],
                    step="pre",
                    alpha=0.90,
                    label=col,
                )

        ax_dominant.set_ylim(0, 1)
        ax_dominant.set_yticks([0.5])
        ax_dominant.set_yticklabels(["Dominant\nRegime"], fontsize=8)
        ax_dominant.set_xlabel("Date")
        ax_dominant.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig
