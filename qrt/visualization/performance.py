"""
Performance Visualizer
======================
Publication-quality charts for portfolio performance analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_PALETTE = {
    "primary": "#2563EB",
    "benchmark": "#F59E0B",
    "drawdown": "#DC2626",
    "positive": "#16A34A",
    "negative": "#DC2626",
    "neutral": "#6B7280",
}


def _save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _normalise_index(s: pd.Series) -> pd.Series:
    """Return a copy whose index is a DatetimeIndex if it isn't already."""
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    return s


class PerformanceVisualizer:
    """
    Factory class for performance-related figures.

    All public methods return a :class:`matplotlib.figure.Figure` and
    optionally write it to *save_path*.
    """

    # ------------------------------------------------------------------
    # Equity Curve
    # ------------------------------------------------------------------

    @staticmethod
    def equity_curve(
        portfolio_values: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot an indexed equity curve.

        Parameters
        ----------
        portfolio_values:
            Raw portfolio NAV / value series (not normalised).
        benchmark:
            Optional benchmark series aligned on the same index.
        title:
            Figure title.
        save_path:
            If provided the figure is also written to disk.

        Returns
        -------
        matplotlib.figure.Figure
        """
        pv = _normalise_index(portfolio_values.dropna())
        normalised_pv = pv / pv.iloc[0] * 100

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            normalised_pv.index,
            normalised_pv.values,
            color=_PALETTE["primary"],
            linewidth=1.8,
            label="Portfolio",
            zorder=3,
        )

        if benchmark is not None:
            bm = _normalise_index(benchmark.dropna()).reindex(pv.index, method="ffill")
            normalised_bm = bm / bm.iloc[0] * 100
            ax.plot(
                normalised_bm.index,
                normalised_bm.values,
                color=_PALETTE["benchmark"],
                linewidth=1.4,
                linestyle="--",
                label="Benchmark",
                zorder=2,
            )

        ax.fill_between(
            normalised_pv.index,
            normalised_pv.values,
            100,
            where=normalised_pv.values >= 100,
            alpha=0.08,
            color=_PALETTE["positive"],
            label="_nolegend_",
        )
        ax.fill_between(
            normalised_pv.index,
            normalised_pv.values,
            100,
            where=normalised_pv.values < 100,
            alpha=0.08,
            color=_PALETTE["negative"],
            label="_nolegend_",
        )

        ax.axhline(100, color=_PALETTE["neutral"], linewidth=0.8, linestyle=":")

        # Annotations
        total_return = normalised_pv.iloc[-1] - 100
        ax.annotate(
            f"Total Return: {total_return:+.1f}%",
            xy=(normalised_pv.index[-1], normalised_pv.iloc[-1]),
            xytext=(-10, 10),
            textcoords="offset points",
            fontsize=9,
            color=_PALETTE["positive"] if total_return >= 0 else _PALETTE["negative"],
            ha="right",
        )

        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Indexed Value (Base = 100)")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Drawdown Curve
    # ------------------------------------------------------------------

    @staticmethod
    def drawdown_curve(
        returns: pd.Series,
        title: str = "Drawdown",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot the underwater / drawdown curve.

        Parameters
        ----------
        returns:
            Period returns (not cumulative).
        """
        r = _normalise_index(returns.dropna())
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max * 100  # in percent

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color=_PALETTE["drawdown"],
            alpha=0.45,
            label="Drawdown",
        )
        ax.plot(drawdown.index, drawdown.values, color=_PALETTE["drawdown"], linewidth=0.8)

        ax.annotate(
            f"Max DD: {max_dd:.1f}%",
            xy=(max_dd_date, max_dd),
            xytext=(0, -18),
            textcoords="offset points",
            fontsize=9,
            color=_PALETTE["drawdown"],
            arrowprops=dict(arrowstyle="->", color=_PALETTE["drawdown"], lw=0.8),
            ha="center",
        )

        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 63,
        title: str = "Rolling Sharpe",
        save_path: Optional[Union[str, Path]] = None,
        annualisation_factor: float = 252.0,
    ) -> plt.Figure:
        """
        Plot the rolling Sharpe ratio.

        Parameters
        ----------
        returns:
            Daily (or period) returns series.
        window:
            Look-back window in periods (default 63 ~ 3 months).
        annualisation_factor:
            Periods per year used to annualise (default 252).
        """
        r = _normalise_index(returns.dropna())
        rolling_mean = r.rolling(window).mean()
        rolling_std = r.rolling(window).std()
        sharpe = rolling_mean / rolling_std * np.sqrt(annualisation_factor)

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(sharpe.index, sharpe.values, color=_PALETTE["primary"], linewidth=1.4)
        ax.fill_between(
            sharpe.index,
            sharpe.values,
            0,
            where=sharpe.values >= 0,
            alpha=0.15,
            color=_PALETTE["positive"],
        )
        ax.fill_between(
            sharpe.index,
            sharpe.values,
            0,
            where=sharpe.values < 0,
            alpha=0.15,
            color=_PALETTE["negative"],
        )

        ax.axhline(0, color=_PALETTE["neutral"], linewidth=0.8, linestyle="--")
        ax.axhline(1, color=_PALETTE["positive"], linewidth=0.6, linestyle=":", alpha=0.7)
        ax.axhline(-1, color=_PALETTE["negative"], linewidth=0.6, linestyle=":", alpha=0.7)

        ax.set_title(f"{title}  (window = {window} periods)", fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualised Sharpe Ratio")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Rolling Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 63,
        title: str = "Rolling Volatility",
        save_path: Optional[Union[str, Path]] = None,
        annualisation_factor: float = 252.0,
    ) -> plt.Figure:
        """
        Plot annualised rolling volatility.

        Parameters
        ----------
        returns:
            Daily (or period) returns series.
        window:
            Look-back window in periods.
        annualisation_factor:
            Periods per year for annualisation.
        """
        r = _normalise_index(returns.dropna())
        rolling_vol = r.rolling(window).std() * np.sqrt(annualisation_factor) * 100

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(rolling_vol.index, rolling_vol.values, color=_PALETTE["primary"], linewidth=1.4)
        ax.fill_between(
            rolling_vol.index,
            rolling_vol.values,
            0,
            alpha=0.12,
            color=_PALETTE["primary"],
        )

        mean_vol = rolling_vol.mean()
        ax.axhline(
            mean_vol,
            color=_PALETTE["neutral"],
            linewidth=1.0,
            linestyle="--",
            label=f"Mean {mean_vol:.1f}%",
        )

        ax.set_title(f"{title}  (window = {window} periods)", fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualised Volatility (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        _save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Monthly Returns Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def monthly_heatmap(
        returns: pd.Series,
        title: str = "Monthly Returns",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Seaborn heatmap of calendar monthly returns.

        Parameters
        ----------
        returns:
            Daily (or sub-daily) returns series with a DatetimeIndex.
        """
        r = _normalise_index(returns.dropna())
        monthly = (1 + r).resample("ME").prod() - 1

        df = monthly.to_frame(name="return")
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot(index="year", columns="month", values="return") * 100
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot.index.name = "Year"
        pivot.columns.name = "Month"

        # Annual return column
        annual = pivot.sum(axis=1)  # approximate; sum of monthly % is close enough

        n_years = len(pivot)
        fig_height = max(4, n_years * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(15, fig_height))

        cmap = sns.diverging_palette(10, 145, as_cmap=True)
        abs_max = np.nanmax(np.abs(pivot.values))

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            annot=True,
            fmt=".1f",
            linewidths=0.4,
            linecolor="#E5E7EB",
            cbar_kws={"label": "Monthly Return (%)", "shrink": 0.6},
        )

        # Overlay annual returns on right side
        for i, (yr, val) in enumerate(annual.items()):
            color = _PALETTE["positive"] if val >= 0 else _PALETTE["negative"]
            ax.text(
                len(pivot.columns) + 0.55,
                i + 0.5,
                f"{val:+.1f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

        ax.text(
            len(pivot.columns) + 0.55,
            -0.6,
            "Annual",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=_PALETTE["neutral"],
        )

        ax.set_title(title, fontweight="bold", pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        _save(fig, save_path)
        return fig
