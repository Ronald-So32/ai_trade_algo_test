"""
Risk Geometry
=============
Interactive Plotly figures for PCA risk space, efficient frontier,
risk contribution surface, and correlation network (matplotlib/networkx).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


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
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for this function. Install with: pip install plotly"
        )


def _require_networkx() -> None:
    if not _NX_AVAILABLE:
        raise ImportError(
            "networkx is required for this function. Install with: pip install networkx"
        )


def _save_fig(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _save_plotly(fig: "go.Figure", save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".html":
            fig.write_html(str(path))
        elif suffix in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}:
            fig.write_image(str(path))
        else:
            fig.write_html(str(path))


class RiskGeometry:
    """
    Geometric / factor-space views of risk using PCA, the efficient frontier,
    risk-contribution surfaces, and correlation networks.
    """

    # ------------------------------------------------------------------
    # PCA Risk Space – 3-D Scatter
    # ------------------------------------------------------------------

    @staticmethod
    def pca_risk_space_3d(
        returns: pd.DataFrame,
        title: str = "PCA Risk Space (3D)",
        n_components: int = 3,
        save_path: Optional[Union[str, Path]] = None,
    ) -> "go.Figure":
        """
        Project assets into the first three PCA principal components and
        display a 3-D scatter plot coloured by PC-1 loading magnitude.

        Parameters
        ----------
        returns:
            DataFrame of asset returns (columns = asset names).
        title:
            Figure title.
        n_components:
            Must be 3 for the 3D scatter.
        save_path:
            Optional path for HTML or image output.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()
        df = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
        df = df.fillna(df.median())

        cov = df.cov().values
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        loadings = eigenvectors[:, :3]  # shape: (n_assets, 3)
        explained = eigenvalues[:3] / eigenvalues.sum() * 100

        assets = list(df.columns)
        color_vals = np.abs(loadings[:, 0])  # magnitude of PC1 loading

        fig = go.Figure(
            data=go.Scatter3d(
                x=loadings[:, 0],
                y=loadings[:, 1],
                z=loadings[:, 2],
                mode="markers+text",
                text=assets,
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=7,
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="|PC1 Loading|"),
                    opacity=0.85,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial"), x=0.05),
            scene=dict(
                xaxis_title=f"PC1 ({explained[0]:.1f}%)",
                yaxis_title=f"PC2 ({explained[1]:.1f}%)",
                zaxis_title=f"PC3 ({explained[2]:.1f}%)",
                xaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
                yaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
                zaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
            ),
            width=900,
            height=700,
            paper_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=11),
        )

        _save_plotly(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Risk Contribution Surface
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contribution_surface(
        weights_range: np.ndarray,
        covariance: Union[pd.DataFrame, np.ndarray],
        asset_names: Optional[list[str]] = None,
        title: str = "Risk Contribution Surface",
        save_path: Optional[Union[str, Path]] = None,
    ) -> "go.Figure":
        """
        Surface plot of portfolio volatility over a 2D grid of two asset
        weight parameters (remaining weight distributed equally among others).

        Parameters
        ----------
        weights_range:
            1D array of weight values (0–1) swept for the first two assets.
        covariance:
            Square covariance matrix.
        asset_names:
            Optional list of asset labels.
        title:
            Figure title.
        save_path:
            Optional path for HTML or image output.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        if isinstance(covariance, pd.DataFrame):
            sigma = covariance.values.astype(float)
            if asset_names is None:
                asset_names = list(covariance.columns)
        else:
            sigma = np.asarray(covariance, dtype=float)

        n_assets = sigma.shape[0]
        if asset_names is None:
            asset_names = [f"Asset {i+1}" for i in range(n_assets)]

        w_grid = np.asarray(weights_range, dtype=float)
        w1_vals = w_grid
        w2_vals = w_grid
        Z = np.zeros((len(w1_vals), len(w2_vals)))

        for i, w1 in enumerate(w1_vals):
            for j, w2 in enumerate(w2_vals):
                remaining = max(0.0, 1.0 - w1 - w2)
                if n_assets > 2:
                    w_rest = remaining / (n_assets - 2)
                    w = np.array([w1, w2] + [w_rest] * (n_assets - 2))
                else:
                    w = np.array([w1, w2])
                w = np.clip(w, 0, None)
                if w.sum() > 1e-8:
                    w = w / w.sum()
                Z[i, j] = np.sqrt(w @ sigma @ w) * np.sqrt(252) * 100  # annualised %

        fig = go.Figure(
            data=go.Surface(
                x=w2_vals,
                y=w1_vals,
                z=Z,
                colorscale="RdYlGn_r",
                colorbar=dict(title="Ann. Vol (%)"),
                opacity=0.92,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                ),
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial"), x=0.05),
            scene=dict(
                xaxis_title=f"{asset_names[1]} Weight",
                yaxis_title=f"{asset_names[0]} Weight",
                zaxis_title="Ann. Volatility (%)",
                xaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
                yaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
                zaxis=dict(backgroundcolor="#F9FAFB", gridcolor="#E5E7EB"),
            ),
            width=900,
            height=700,
            paper_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=11),
        )

        _save_plotly(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Efficient Frontier
    # ------------------------------------------------------------------

    @staticmethod
    def efficient_frontier(
        returns: pd.DataFrame,
        n_portfolios: int = 1000,
        risk_free_rate: float = 0.0,
        title: str = "Efficient Frontier",
        save_path: Optional[Union[str, Path]] = None,
        annualisation_factor: float = 252.0,
    ) -> "go.Figure":
        """
        Monte-Carlo simulated efficient frontier coloured by Sharpe ratio.

        Parameters
        ----------
        returns:
            DataFrame of asset returns (columns = asset names).
        n_portfolios:
            Number of random portfolios to simulate.
        risk_free_rate:
            Annual risk-free rate for Sharpe computation.
        title:
            Figure title.
        save_path:
            Optional path for HTML or image output.
        annualisation_factor:
            Periods per year.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()
        rng = np.random.default_rng(42)
        df = returns.dropna(axis=1, how="all").dropna(axis=0, how="all").fillna(0)
        n_assets = df.shape[1]
        mu = df.mean().values * annualisation_factor
        sigma = df.cov().values * annualisation_factor
        labels = list(df.columns)

        vols, rets, sharpes, all_weights = [], [], [], []

        for _ in range(n_portfolios):
            raw = rng.exponential(1.0, n_assets)
            w = raw / raw.sum()
            p_ret = w @ mu
            p_vol = np.sqrt(w @ sigma @ w)
            p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 1e-12 else 0.0
            vols.append(p_vol * 100)
            rets.append(p_ret * 100)
            sharpes.append(p_sharpe)
            all_weights.append(w)

        # Min-volatility portfolio
        min_vol_idx = int(np.argmin(vols))
        # Max-Sharpe portfolio
        max_sharpe_idx = int(np.argmax(sharpes))

        hover_texts = []
        for i, w in enumerate(all_weights):
            top3 = sorted(zip(labels, w), key=lambda x: x[1], reverse=True)[:3]
            txt = "<br>".join([f"{lbl}: {wt:.1%}" for lbl, wt in top3])
            hover_texts.append(f"Ret: {rets[i]:.1f}%  Vol: {vols[i]:.1f}%  SR: {sharpes[i]:.2f}<br>Top weights:<br>{txt}")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=vols,
                y=rets,
                mode="markers",
                marker=dict(
                    size=5,
                    color=sharpes,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio", x=1.02),
                    opacity=0.7,
                    line=dict(width=0),
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                name="Simulated Portfolios",
            )
        )

        # Highlight special portfolios
        for idx, label, color, symbol in [
            (min_vol_idx, "Min Volatility", "#2563EB", "star"),
            (max_sharpe_idx, "Max Sharpe", "#16A34A", "diamond"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[vols[idx]],
                    y=[rets[idx]],
                    mode="markers+text",
                    marker=dict(size=14, color=color, symbol=symbol, line=dict(width=1.5, color="white")),
                    text=[label],
                    textposition="top right",
                    textfont=dict(size=11, color=color),
                    name=label,
                    hovertemplate=f"{label}<br>Vol: {vols[idx]:.1f}%  Ret: {rets[idx]:.1f}%  SR: {sharpes[idx]:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial"), x=0.05),
            xaxis=dict(
                title="Annualised Volatility (%)",
                gridcolor="#E5E7EB",
                zeroline=False,
            ),
            yaxis=dict(
                title="Annualised Return (%)",
                gridcolor="#E5E7EB",
                zeroline=False,
            ),
            legend=dict(
                orientation="v",
                x=1.12,
                y=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#D1D5DB",
                borderwidth=1,
            ),
            width=900,
            height=600,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F9FAFB",
            font=dict(family="Arial", size=11),
        )

        _save_plotly(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Correlation Network
    # ------------------------------------------------------------------

    @staticmethod
    def correlation_network(
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5,
        title: str = "Correlation Network",
        save_path: Optional[Union[str, Path]] = None,
        node_size_factor: float = 800.0,
        figsize: tuple[int, int] = (14, 12),
    ) -> plt.Figure:
        """
        NetworkX / matplotlib graph where nodes are assets and edges
        represent absolute correlation above *threshold*.

        Parameters
        ----------
        correlation_matrix:
            Square symmetric correlation matrix (assets x assets).
        threshold:
            Minimum |correlation| for an edge to be drawn.
        title:
            Figure title.
        save_path:
            Optional path to save the figure.
        node_size_factor:
            Scaling factor for node sizes (proportional to mean |correlation|).
        figsize:
            Figure dimensions in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _require_networkx()

        corr = correlation_matrix.copy()
        assets = list(corr.columns)
        n = len(assets)

        G = nx.Graph()
        G.add_nodes_from(assets)

        for i in range(n):
            for j in range(i + 1, n):
                c = corr.iloc[i, j]
                if abs(c) >= threshold:
                    G.add_edge(assets[i], assets[j], weight=abs(c), correlation=c)

        # Layout
        if len(G.edges) > 0:
            pos = nx.spring_layout(G, weight="weight", k=2.5 / np.sqrt(n + 1), seed=42, iterations=80)
        else:
            pos = nx.circular_layout(G)

        # Node sizes proportional to average absolute correlation
        node_sizes = []
        for node in G.nodes:
            nbrs = list(G.neighbors(node))
            if nbrs:
                avg_corr = np.mean([abs(G[node][nb]["correlation"]) for nb in nbrs])
            else:
                avg_corr = 0.2
            node_sizes.append(node_size_factor * avg_corr)

        # Edge colours: positive = green spectrum, negative = red spectrum
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            c = data["correlation"]
            if c >= 0:
                r, g, b = 0.09, 0.39 + 0.5 * c, 0.16
            else:
                r, g, b = 0.55 + 0.45 * abs(c), 0.13, 0.13
            edge_colors.append((r, g, b, 0.7))
            edge_widths.append(1.0 + 3.0 * abs(c))

        # Detect communities via greedy modularity
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx
            cmap = plt.get_cmap("tab20")
            node_colors = [
                matplotlib.colors.to_hex(cmap(community_map.get(node, 0) % 20))
                for node in G.nodes
            ]
        except Exception:
            node_colors = ["#2563EB"] * len(G.nodes)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F9FAFB")

        if len(G.edges) > 0:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.65,
            )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.90,
            linewidths=0.8,
            edgecolors="#374151",
        )

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=7.5,
            font_color="#111827",
            font_weight="bold",
        )

        # Edge weight legend
        for corr_val, label in [(0.9, "|r| = 0.9"), (0.7, "|r| = 0.7"), (threshold, f"|r| = {threshold:.1f}")]:
            ax.plot([], [], linewidth=1.0 + 3.0 * corr_val, color="#6B7280", label=label)

        ax.legend(
            title="Edge Weight",
            loc="lower right",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.85,
        )

        # Stats annotation
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        ax.text(
            0.01, 0.99,
            f"Nodes: {n_nodes}  |  Edges: {n_edges}  |  Threshold: |r| ≥ {threshold}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="left",
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.8),
        )

        ax.set_title(title, fontweight="bold", pad=14)
        ax.axis("off")
        fig.tight_layout()

        _save_fig(fig, save_path)
        return fig
