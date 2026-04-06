"""
DashboardGenerator: Produces interactive HTML dashboards using Jinja2 templates
and embedded Plotly charts for quantitative research workflows.

Generates individual dashboards AND a unified combined dashboard with tabbed
navigation accessible via header buttons.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from jinja2 import Environment, BaseLoader

# ---------------------------------------------------------------------------
# HTML base template (individual dashboards)
# ---------------------------------------------------------------------------

_BASE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: right; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        h1 { color: #2c3e50; } h2 { color: #34495e; }
    </style>
</head>
<body>
<div class="container">
    <h1>{{ title }}</h1>
    {{ content }}
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Combined dashboard template with tabbed navigation
# ---------------------------------------------------------------------------

_COMBINED_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Quantitative Research Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; }

        /* ── Header / Nav ── */
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 0;
            position: sticky; top: 0; z-index: 1000;
            box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        }
        .header-inner {
            max-width: 1500px; margin: 0 auto;
            display: flex; align-items: center;
            padding: 0 24px;
        }
        .header-title {
            color: #e2e8f0; font-size: 18px; font-weight: 700;
            padding: 16px 0; margin-right: 32px; white-space: nowrap;
            letter-spacing: 0.5px;
        }
        .header-title span { color: #64b5f6; }
        .nav-tabs {
            display: flex; gap: 4px; overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        .nav-btn {
            background: transparent; border: none; color: #94a3b8;
            padding: 16px 20px; font-size: 14px; font-weight: 500;
            cursor: pointer; white-space: nowrap; border-bottom: 3px solid transparent;
            transition: all 0.2s ease;
        }
        .nav-btn:hover { color: #e2e8f0; background: rgba(255,255,255,0.05); }
        .nav-btn.active {
            color: #64b5f6; border-bottom-color: #64b5f6;
            background: rgba(100,181,246,0.08);
        }

        /* ── Content ── */
        .container { max-width: 1500px; margin: 0 auto; padding: 20px 24px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card {
            background: white; border-radius: 10px; padding: 24px; margin: 16px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .metric {
            text-align: center; padding: 16px; background: #f8fafc;
            border-radius: 8px; border: 1px solid #e2e8f0;
        }
        .metric-value { font-size: 22px; font-weight: 700; color: #1e293b; }
        .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; }
        th { background: #f8fafc; font-weight: 600; color: #475569; }
        td:first-child, th:first-child { text-align: left; }
        h2 { color: #1e293b; font-size: 18px; margin-bottom: 12px; }
        .section-title { color: #0f3460; font-size: 24px; font-weight: 700; margin: 8px 0 20px; }
    </style>
</head>
<body>
<div class="header">
    <div class="header-inner">
        <div class="header-title"><span>QRT</span> Research Platform</div>
        <div class="nav-tabs">
            {% for tab in tabs %}
            <button class="nav-btn{% if loop.first %} active{% endif %}"
                    onclick="switchTab('{{ tab.id }}', this)">{{ tab.label }}</button>
            {% endfor %}
        </div>
    </div>
</div>
<div class="container">
    {% for tab in tabs %}
    <div id="{{ tab.id }}" class="tab-content{% if loop.first %} active{% endif %}">
        <div class="section-title">{{ tab.title }}</div>
        {{ tab.content }}
    </div>
    {% endfor %}
</div>
<script>
function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
    // Trigger Plotly resize for any charts in the newly visible tab
    setTimeout(() => {
        var plots = document.getElementById(tabId).querySelectorAll('.js-plotly-plot');
        plots.forEach(p => Plotly.Plots.resize(p));
    }, 50);
}
</script>
</body>
</html>"""

_jinja_env = Environment(loader=BaseLoader())


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _render_html(title: str, content: str) -> str:
    """Render the base Jinja2 template with the given title and content."""
    tmpl = _jinja_env.from_string(_BASE_TEMPLATE)
    return tmpl.render(title=title, content=content)


def _render_combined(tabs: list[dict]) -> str:
    """Render the combined tabbed dashboard."""
    tmpl = _jinja_env.from_string(_COMBINED_TEMPLATE)
    return tmpl.render(tabs=tabs)


def _fig_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an embeddable HTML snippet."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write(path: str, html: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _date_strings(index) -> list[str]:
    """Convert any index to clean date strings for Plotly x-axis."""
    idx = pd.Index(index)
    try:
        dt = pd.to_datetime(idx)
        return [d.strftime("%Y-%m-%d") for d in dt]
    except Exception:
        return [str(x) for x in idx]


def _metric_card(metrics: dict[str, Any]) -> str:
    items = "".join(
        f'<div class="metric">'
        f'<div class="metric-value">{v}</div>'
        f'<div class="metric-label">{k}</div>'
        f"</div>"
        for k, v in metrics.items()
    )
    return f'<div class="card"><div class="metric-grid">{items}</div></div>'


def _df_to_html_table(df: pd.DataFrame, index: bool = True) -> str:
    table = df.to_html(
        index=index, border=0, classes="",
        float_format=lambda x: f"{x:.4f}",
    )
    return f'<div class="card">{table}</div>'


def _card(html_content: str, heading: str = "") -> str:
    heading_html = f"<h2>{heading}</h2>" if heading else ""
    return f'<div class="card">{heading_html}{html_content}</div>'


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def _annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    total = (1 + returns).prod()
    n = len(returns)
    if n == 0:
        return float("nan")
    return float(total ** (periods_per_year / n) - 1)


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std()
    if downside == 0:
        return float("nan")
    return float(returns.mean() / downside * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    dd = _compute_drawdown(equity)
    return float(dd.min())


def _calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    cagr = _annualised_return(returns, periods_per_year)
    mdd = abs(_max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return float(cagr / mdd)


def _rolling_sharpe(
    returns: pd.Series, window: int = 63, periods_per_year: int = 252,
) -> pd.Series:
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(periods_per_year)).rename("Rolling Sharpe")


def _monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period("M")
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack(level=1)
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    return pivot


def _cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / n_years) - 1)


def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gain/loss above threshold."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess <= 0].sum())
    if losses == 0:
        return float("nan")
    return float(gains / losses)


def _tail_ratio(returns: pd.Series) -> float:
    """95th percentile / abs(5th percentile) — measures tail asymmetry."""
    p95 = np.percentile(returns.dropna(), 95)
    p5 = abs(np.percentile(returns.dropna(), 5))
    if p5 == 0:
        return float("nan")
    return float(p95 / p5)


def _profit_factor(returns: pd.Series) -> float:
    """Sum of positive returns / abs(sum of negative returns)."""
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    if neg == 0:
        return float("nan")
    return float(pos / neg)


# ---------------------------------------------------------------------------
# DashboardGenerator
# ---------------------------------------------------------------------------

class DashboardGenerator:
    """
    Generates interactive HTML research dashboards with embedded Plotly charts.

    All ``generate_*`` methods return the section HTML content string AND
    optionally write to a standalone file. The ``generate_combined_dashboard``
    method assembles all sections into a single tabbed HTML page.
    """

    # ------------------------------------------------------------------
    # 1. Performance Dashboard
    # ------------------------------------------------------------------

    def generate_performance_dashboard(
        self,
        backtest_result: dict[str, Any],
        save_path: str = "reports/performance_dashboard.html",
    ) -> str:
        returns: pd.Series = pd.Series(backtest_result.get("returns", [])).dropna()
        if returns.empty:
            raise ValueError("backtest_result must contain a non-empty 'returns' series.")

        equity: pd.Series = backtest_result.get("equity", (1 + returns).cumprod())
        equity = pd.Series(equity)
        turnover = backtest_result.get("turnover", None)

        dates = _date_strings(equity.index)

        cagr = _annualised_return(returns)
        sharpe = _sharpe(returns)
        sortino = _sortino(returns)
        mdd = _max_drawdown(returns)
        calmar = _calmar(returns)
        avg_turnover: float
        if turnover is None:
            avg_turnover = float("nan")
        elif hasattr(turnover, "__len__"):
            avg_turnover = float(pd.Series(turnover).mean())
        else:
            avg_turnover = float(turnover)

        # Additional robust metrics
        cagr_val = _cagr(returns)
        omega = _omega_ratio(returns)
        tail = _tail_ratio(returns)
        profit_f = _profit_factor(returns)
        win_rate = float((returns > 0).mean())
        ann_vol = float(returns.std() * np.sqrt(252))

        key_metrics = {
            "CAGR": f"{cagr_val:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Max Drawdown": f"{mdd:.2%}",
            "Calmar": f"{calmar:.2f}",
            "Omega": f"{omega:.2f}" if not np.isnan(omega) else "N/A",
            "Tail Ratio": f"{tail:.2f}" if not np.isnan(tail) else "N/A",
            "Profit Factor": f"{profit_f:.2f}" if not np.isnan(profit_f) else "N/A",
            "Win Rate": f"{win_rate:.1%}",
            "Ann. Vol": f"{ann_vol:.2%}",
            "Avg Turnover": f"{avg_turnover:.2%}" if not np.isnan(avg_turnover) else "N/A",
        }

        # Equity curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=dates, y=equity.values.tolist(),
            mode="lines", name="Equity", line=dict(color="#2980b9", width=2),
        ))
        fig_equity.update_layout(
            title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value",
            template="plotly_white", height=350,
        )

        # Drawdown
        drawdown = _compute_drawdown(equity)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=drawdown.values.tolist(),
            mode="lines", fill="tozeroy", name="Drawdown",
            line=dict(color="#e74c3c", width=1.5),
        ))
        fig_dd.update_layout(
            title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown",
            yaxis_tickformat=".1%", template="plotly_white", height=280,
        )

        # Monthly returns heatmap
        try:
            pivot = _monthly_returns_table(returns)
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot.values.tolist(), x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                colorscale="RdYlGn", zmid=0,
                text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
                texttemplate="%{text}", showscale=True, colorbar=dict(tickformat=".0%"),
            ))
            fig_heatmap.update_layout(title="Monthly Returns Heatmap", template="plotly_white", height=300)
            heatmap_html = _fig_html(fig_heatmap)
        except Exception:
            heatmap_html = "<p>Monthly returns heatmap not available.</p>"

        # Rolling Sharpe
        roll_sharpe = _rolling_sharpe(returns)
        rs_dates = _date_strings(roll_sharpe.index)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=rs_dates, y=roll_sharpe.values.tolist(),
            mode="lines", name="Rolling Sharpe (63d)", line=dict(color="#8e44ad", width=1.5),
        ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_rs.update_layout(
            title="Rolling Sharpe Ratio (63-day)", xaxis_title="Date", yaxis_title="Sharpe",
            template="plotly_white", height=280,
        )

        content = (
            _metric_card(key_metrics)
            + _card(_fig_html(fig_equity), "Equity Curve")
            + _card(_fig_html(fig_dd), "Drawdown")
            + _card(heatmap_html, "Monthly Returns Heatmap")
            + _card(_fig_html(fig_rs), "Rolling Sharpe Ratio")
        )
        html = _render_html("Performance Dashboard", content)
        _write(save_path, html)
        return content  # Return section content for combined dashboard

    # ------------------------------------------------------------------
    # 2. Strategy Diagnostics
    # ------------------------------------------------------------------

    def generate_strategy_diagnostics(
        self,
        strategy_results: dict[str, Any],
        save_path: str = "reports/strategy_diagnostics.html",
    ) -> str:
        if not strategy_results:
            raise ValueError("strategy_results must not be empty.")

        # Per-strategy equity curves
        fig_eq = go.Figure()
        returns_dict: dict[str, pd.Series] = {}
        for name, res in strategy_results.items():
            # Support both {name: series} and {name: {"returns": series}}
            if isinstance(res, dict):
                r = pd.Series(res.get("returns", [])).dropna()
            else:
                r = pd.Series(res).dropna()
            returns_dict[name] = r
            if r.empty:
                continue
            equity = (1 + r).cumprod()
            dates = _date_strings(equity.index)
            fig_eq.add_trace(go.Scatter(
                x=dates, y=equity.values.tolist(), mode="lines", name=name,
            ))
        fig_eq.update_layout(
            title="Per-Strategy Equity Curves", xaxis_title="Date",
            yaxis_title="Cumulative Return", template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )

        # Metrics table
        metrics_rows: list[dict] = []
        for name, r in returns_dict.items():
            if r.empty:
                continue
            metrics_rows.append({
                "Strategy": name,
                "CAGR": _cagr(r),
                "Sharpe": _sharpe(r),
                "Sortino": _sortino(r),
                "Max DD": _max_drawdown(r),
                "Calmar": _calmar(r),
                "Omega": _omega_ratio(r),
                "Win Rate": float((r > 0).mean()),
                "Profit Factor": _profit_factor(r),
            })
        metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy") if metrics_rows else pd.DataFrame()

        # Correlation heatmap
        returns_frame = pd.DataFrame({name: r for name, r in returns_dict.items() if not r.empty})
        if returns_frame.shape[1] > 1:
            corr = returns_frame.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values.tolist(), x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
            ))
            fig_corr.update_layout(title="Strategy Correlation Heatmap", template="plotly_white", height=400)
            corr_html = _fig_html(fig_corr)
        else:
            corr_html = "<p>Correlation heatmap requires at least two strategies.</p>"

        # Signal distributions
        sig_figs_html = ""
        for name, res in strategy_results.items():
            if not isinstance(res, dict):
                continue
            signals = res.get("signals")
            if signals is None:
                continue
            # Flatten DataFrame signals to a single distribution
            if isinstance(signals, pd.DataFrame):
                s = signals.values.flatten()
                s = pd.Series(s).dropna()
            else:
                s = pd.Series(signals).dropna()
            if s.empty or len(s) < 10:
                continue
            # Sample if too large
            if len(s) > 50000:
                s = s.sample(50000, random_state=42)
            fig_hist = px.histogram(
                s, nbins=50, title=f"Signal Distribution — {name}",
                labels={"value": "Signal", "count": "Frequency"},
                template="plotly_white",
            )
            fig_hist.update_layout(height=280, showlegend=False)
            sig_figs_html += _fig_html(fig_hist)

        if not sig_figs_html:
            sig_figs_html = "<p>No signal data provided for histogram plots.</p>"

        content = (
            _card(_fig_html(fig_eq), "Per-Strategy Equity Curves")
            + (_df_to_html_table(metrics_df) if not metrics_df.empty else "")
            + _card(corr_html, "Strategy Correlation")
            + _card(sig_figs_html, "Signal Distributions")
        )
        html = _render_html("Strategy Diagnostics", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 3. Risk Geometry Dashboard
    # ------------------------------------------------------------------

    def generate_risk_geometry_dashboard(
        self,
        returns: pd.DataFrame | pd.Series,
        weights: np.ndarray | list[float],
        covariance: np.ndarray | pd.DataFrame,
        save_path: str = "reports/risk_geometry.html",
    ) -> str:
        returns_df = pd.DataFrame(returns) if not isinstance(returns, pd.DataFrame) else returns.copy()
        weights_arr = np.asarray(weights, dtype=float)
        cov_arr = np.asarray(covariance, dtype=float)
        asset_names = list(returns_df.columns)
        n_assets = len(asset_names)

        # PCA risk-space 3D
        try:
            from sklearn.decomposition import PCA
            clean = returns_df.dropna()
            n_components = min(3, clean.shape[1])
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(clean)
            explained = pca.explained_variance_ratio_

            if n_components == 3:
                fig_pca = go.Figure(data=go.Scatter3d(
                    x=components[:, 0].tolist(), y=components[:, 1].tolist(),
                    z=components[:, 2].tolist(), mode="markers",
                    marker=dict(size=3, color=components[:, 0].tolist(), colorscale="Viridis", opacity=0.7),
                    text=_date_strings(clean.index),
                ))
                fig_pca.update_layout(
                    title=f"PCA Risk Space 3D (explained: {explained[0]:.1%}, {explained[1]:.1%}, {explained[2]:.1%})",
                    scene=dict(
                        xaxis_title=f"PC1 ({explained[0]:.1%})",
                        yaxis_title=f"PC2 ({explained[1]:.1%})",
                        zaxis_title=f"PC3 ({explained[2]:.1%})",
                    ),
                    template="plotly_white", height=500,
                )
            else:
                fig_pca = go.Figure(data=go.Scatter(
                    x=components[:, 0].tolist(),
                    y=components[:, 1].tolist() if n_components > 1 else [0] * len(components),
                    mode="markers", marker=dict(size=4, color="#2980b9", opacity=0.6),
                    text=_date_strings(clean.index),
                ))
                fig_pca.update_layout(
                    title="PCA Risk Space (2D)",
                    xaxis_title=f"PC1 ({explained[0]:.1%})",
                    yaxis_title=f"PC2 ({explained[1]:.1%})" if n_components > 1 else "—",
                    template="plotly_white", height=400,
                )
            pca_html = _fig_html(fig_pca)
        except ImportError:
            pca_html = "<p>scikit-learn not installed — PCA plot unavailable.</p>"
        except Exception as exc:
            pca_html = f"<p>PCA plot error: {exc}</p>"

        # Risk contribution pie chart
        marginal_risk = cov_arr @ weights_arr
        risk_contributions = weights_arr * marginal_risk
        total_portfolio_var = weights_arr @ cov_arr @ weights_arr
        rc_pct = risk_contributions / total_portfolio_var if total_portfolio_var > 0 else risk_contributions

        fig_rc = go.Figure(data=go.Pie(
            labels=asset_names, values=rc_pct.tolist(), hole=0.35, textinfo="label+percent",
        ))
        fig_rc.update_layout(title="Risk Contribution by Asset", template="plotly_white", height=400)

        # Efficient frontier (Monte Carlo)
        try:
            port_vols: list[float] = []
            port_rets: list[float] = []
            rng = np.random.default_rng(42)
            for _ in range(500):
                w = rng.dirichlet(np.ones(n_assets))
                pv = float(np.sqrt(w @ cov_arr @ w * 252))
                pr = float(w @ returns_df.mean() * 252)
                port_vols.append(pv)
                port_rets.append(pr)

            current_vol = float(np.sqrt(weights_arr @ cov_arr @ weights_arr * 252))
            current_ret = float(weights_arr @ returns_df.mean() * 252)

            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=port_vols, y=port_rets, mode="markers",
                marker=dict(
                    size=4,
                    color=[pr / pv if pv > 0 else 0 for pr, pv in zip(port_rets, port_vols)],
                    colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe"), opacity=0.6,
                ),
                name="Random Portfolios",
            ))
            fig_ef.add_trace(go.Scatter(
                x=[current_vol], y=[current_ret], mode="markers",
                marker=dict(size=12, color="red", symbol="star"), name="Current Portfolio",
            ))
            fig_ef.update_layout(
                title="Efficient Frontier (Monte Carlo)",
                xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
                xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                template="plotly_white", height=450,
            )
            ef_html = _fig_html(fig_ef)
        except Exception as exc:
            ef_html = f"<p>Efficient frontier error: {exc}</p>"

        # Correlation network
        try:
            corr_mat = returns_df.corr().values
            threshold = 0.3
            node_x, node_y = [], []
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            angles = np.linspace(0, 2 * np.pi, n_assets, endpoint=False)
            for a in angles:
                node_x.append(float(np.cos(a)))
                node_y.append(float(np.sin(a)))
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    if abs(corr_mat[i, j]) >= threshold:
                        edge_x += [node_x[i], node_x[j], None]
                        edge_y += [node_y[i], node_y[j], None]

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="#bdc3c7"), hoverinfo="none", name="Correlation Edge",
            ))
            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=14, color="#3498db", line=dict(width=1, color="white")),
                text=asset_names, textposition="top center", name="Asset",
            ))
            fig_net.update_layout(
                title=f"Correlation Network (|rho| >= {threshold})",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white", height=450,
            )
            net_html = _fig_html(fig_net)
        except Exception as exc:
            net_html = f"<p>Correlation network error: {exc}</p>"

        content = (
            _card(pca_html, "PCA Risk Space")
            + _card(_fig_html(fig_rc), "Risk Contribution")
            + _card(ef_html, "Efficient Frontier")
            + _card(net_html, "Correlation Network")
        )
        html = _render_html("Risk Geometry Dashboard", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 4. Regime Analysis
    # ------------------------------------------------------------------

    def generate_regime_analysis(
        self,
        regime_labels: pd.Series,
        returns: pd.Series,
        save_path: str = "reports/regime_analysis.html",
    ) -> str:
        regime_labels = pd.Series(regime_labels).dropna()
        returns = pd.Series(returns).dropna()
        unique_regimes = sorted(regime_labels.unique())
        palette = px.colors.qualitative.Plotly

        # Regime timeline
        equity = (1 + returns).cumprod()
        dates = _date_strings(equity.index)
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=dates, y=equity.values.tolist(), mode="lines",
            name="Equity", line=dict(color="#2c3e50", width=1.5),
        ))

        common_idx = regime_labels.index.intersection(equity.index)
        if len(common_idx) > 0:
            rl = regime_labels.reindex(common_idx)
            eq = equity.reindex(common_idx)

            # Build regime color bands (no per-segment annotations to avoid clutter)
            regime_color = {r: palette[i % len(palette)] for i, r in enumerate(unique_regimes)}
            prev_regime = rl.iloc[0]
            start_date = str(pd.Timestamp(common_idx[0]).strftime("%Y-%m-%d"))
            for i in range(1, len(rl)):
                curr_regime = rl.iloc[i]
                if curr_regime != prev_regime or i == len(rl) - 1:
                    end_date = str(pd.Timestamp(common_idx[i]).strftime("%Y-%m-%d"))
                    fig_timeline.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor=regime_color[prev_regime], opacity=0.15, line_width=0,
                    )
                    prev_regime = curr_regime
                    start_date = end_date

            # Add invisible traces as legend entries for each regime color
            for regime_name, color in regime_color.items():
                fig_timeline.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=12, color=color, opacity=0.4, symbol="square"),
                    name=str(regime_name), showlegend=True,
                ))

        fig_timeline.update_layout(
            title="Equity Curve with Regime Timeline",
            xaxis_title="Date", yaxis_title="Cumulative Return",
            template="plotly_white", height=400,
        )

        # Performance by regime
        aligned = pd.concat(
            [returns.rename("returns"), regime_labels.rename("regime")], axis=1
        ).dropna()

        regime_stats: list[dict] = []
        for regime in unique_regimes:
            r = aligned.loc[aligned["regime"] == regime, "returns"]
            if r.empty:
                continue
            regime_stats.append({
                "Regime": str(regime),
                "Count": len(r),
                "Ann. Return": _annualised_return(r),
                "Sharpe": _sharpe(r),
                "Volatility": float(r.std() * np.sqrt(252)),
                "Max DD": _max_drawdown(r),
                "Win Rate": float((r > 0).mean()),
            })
        regime_df = pd.DataFrame(regime_stats).set_index("Regime") if regime_stats else pd.DataFrame()

        # Transition matrix
        try:
            transitions = pd.crosstab(
                regime_labels.iloc[:-1].values,
                regime_labels.iloc[1:].values,
                normalize="index",
            )
            transitions.index.name = "From \\ To"
            fig_trans = go.Figure(data=go.Heatmap(
                z=transitions.values.tolist(),
                x=[str(c) for c in transitions.columns],
                y=[str(r) for r in transitions.index],
                colorscale="Blues", zmin=0, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in transitions.values],
                texttemplate="%{text}",
            ))
            fig_trans.update_layout(
                title="Regime Transition Matrix",
                xaxis_title="Next Regime", yaxis_title="Current Regime",
                template="plotly_white", height=350,
            )
            trans_html = _fig_html(fig_trans)
        except Exception as exc:
            trans_html = f"<p>Transition matrix error: {exc}</p>"

        content = (
            _card(_fig_html(fig_timeline), "Regime Timeline")
            + (_df_to_html_table(regime_df) if not regime_df.empty else "")
            + _card(trans_html, "Regime Transition Matrix")
        )
        html = _render_html("Regime Analysis Dashboard", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 5. Cost Analysis
    # ------------------------------------------------------------------

    def generate_cost_analysis(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        cost_breakdown: dict[str, float],
        save_path: str = "reports/cost_analysis.html",
    ) -> str:
        gross_returns = pd.Series(gross_returns).dropna()
        net_returns = pd.Series(net_returns).dropna()

        gross_equity = (1 + gross_returns).cumprod()
        net_equity = (1 + net_returns).cumprod()
        dates_gross = _date_strings(gross_equity.index)
        dates_net = _date_strings(net_equity.index)

        # Gross vs net equity
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=dates_gross, y=gross_equity.values.tolist(),
            mode="lines", name="Gross", line=dict(color="#27ae60", width=2),
        ))
        fig_eq.add_trace(go.Scatter(
            x=dates_net, y=net_equity.values.tolist(),
            mode="lines", name="Net", line=dict(color="#e74c3c", width=2),
        ))
        fig_eq.update_layout(
            title="Gross vs Net Equity Curves",
            xaxis_title="Date", yaxis_title="Cumulative Return",
            template="plotly_white", height=380,
        )

        # Cost drag
        common = gross_equity.index.intersection(net_equity.index)
        cost_drag = gross_equity.reindex(common) - net_equity.reindex(common)
        drag_dates = _date_strings(cost_drag.index)

        fig_drag = go.Figure()
        fig_drag.add_trace(go.Scatter(
            x=drag_dates, y=cost_drag.values.tolist(),
            mode="lines", fill="tozeroy", name="Cost Drag",
            line=dict(color="#e67e22", width=1.5),
        ))
        fig_drag.update_layout(
            title="Cumulative Cost Drag Over Time",
            xaxis_title="Date", yaxis_title="Equity Difference",
            template="plotly_white", height=300,
        )

        # Cost breakdown pie
        if cost_breakdown:
            labels = list(cost_breakdown.keys())
            values = [float(v) for v in cost_breakdown.values()]
            fig_pie = go.Figure(data=go.Pie(
                labels=labels, values=values, hole=0.35, textinfo="label+percent",
            ))
            fig_pie.update_layout(title="Cost Breakdown", template="plotly_white", height=380)
            pie_html = _fig_html(fig_pie)
        else:
            pie_html = "<p>No cost breakdown data provided.</p>"

        # Daily cost
        daily_cost = gross_returns.reindex(net_returns.index) - net_returns
        daily_cost = daily_cost.dropna()
        cost_dates = _date_strings(daily_cost.index)

        fig_cost_ts = go.Figure()
        fig_cost_ts.add_trace(go.Bar(
            x=cost_dates, y=daily_cost.values.tolist(),
            name="Daily Cost", marker_color="#c0392b", opacity=0.7,
        ))
        rolling_avg = daily_cost.rolling(21).mean()
        fig_cost_ts.add_trace(go.Scatter(
            x=cost_dates, y=rolling_avg.values.tolist(),
            mode="lines", name="21d Rolling Avg", line=dict(color="#2c3e50", width=2),
        ))
        fig_cost_ts.update_layout(
            title="Daily Transaction Cost (Gross - Net)",
            xaxis_title="Date", yaxis_title="Cost", yaxis_tickformat=".3%",
            template="plotly_white", height=300, barmode="overlay",
        )

        # Summary metrics
        total_cost = float(cost_drag.iloc[-1]) if len(cost_drag) else float("nan")
        ann_gross = _annualised_return(gross_returns)
        ann_net = _annualised_return(net_returns)
        cost_drag_bps = (ann_gross - ann_net) * 10_000

        summary = {
            "Gross CAGR": f"{ann_gross:.2%}",
            "Net CAGR": f"{ann_net:.2%}",
            "CAGR Drag (bps)": f"{cost_drag_bps:.1f}",
            "Total Cumulative Drag": f"{total_cost:.4f}" if not np.isnan(total_cost) else "N/A",
            "Avg Daily Cost": f"{daily_cost.mean():.4%}" if not daily_cost.empty else "N/A",
        }

        content = (
            _metric_card(summary)
            + _card(_fig_html(fig_eq), "Gross vs Net Equity Curves")
            + _card(_fig_html(fig_drag), "Cumulative Cost Drag")
            + _card(pie_html, "Cost Breakdown")
            + _card(_fig_html(fig_cost_ts), "Turnover / Daily Cost Analysis")
        )
        html = _render_html("Cost Analysis Dashboard", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 6. Adaptive Allocation Dashboard
    # ------------------------------------------------------------------

    def generate_adaptive_dashboard(
        self,
        weights_df: pd.DataFrame,
        diagnostics_df: pd.DataFrame,
        static_returns: pd.Series,
        dynamic_returns: pd.Series,
        tail_scaling: pd.Series | None = None,
        save_path: str = "reports/adaptive_allocation.html",
    ) -> str:
        """
        Generate dashboard showing adaptive allocation mechanics.

        Shows:
        - Dynamic vs static equity curves
        - Strategy weight evolution over time (stacked area)
        - Rolling Sharpe per strategy
        - Tail risk scaling factors
        - Comparison metrics table
        """
        dates = _date_strings(weights_df.index)

        # ── Equity comparison: Static vs Dynamic ──
        static_eq = (1 + static_returns).cumprod()
        dynamic_eq = (1 + dynamic_returns).cumprod()
        static_dates = _date_strings(static_eq.index)
        dynamic_dates = _date_strings(dynamic_eq.index)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=static_dates, y=static_eq.values.tolist(),
            mode="lines", name="Static Risk Parity",
            line=dict(color="#95a5a6", width=2, dash="dash"),
        ))
        fig_eq.add_trace(go.Scatter(
            x=dynamic_dates, y=dynamic_eq.values.tolist(),
            mode="lines", name="Dynamic Adaptive",
            line=dict(color="#2980b9", width=2),
        ))
        fig_eq.update_layout(
            title="Static vs Dynamic Allocation — Equity Curves",
            xaxis_title="Date", yaxis_title="Cumulative Return",
            template="plotly_white", height=400,
        )

        # ── Strategy weight stacked area ──
        fig_weights = go.Figure()
        for col in weights_df.columns:
            fig_weights.add_trace(go.Scatter(
                x=dates, y=weights_df[col].values.tolist(),
                mode="lines", name=col, stackgroup="weights",
            ))
        fig_weights.update_layout(
            title="Dynamic Strategy Allocation Over Time",
            xaxis_title="Date", yaxis_title="Weight",
            yaxis_tickformat=".0%", template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

        # ── Rolling Sharpe per strategy ──
        sharpe_cols = [c for c in diagnostics_df.columns if c.endswith("_sharpe")]
        fig_sharpe = go.Figure()
        for col in sharpe_cols:
            name = col.replace("_sharpe", "")
            fig_sharpe.add_trace(go.Scatter(
                x=dates, y=diagnostics_df[col].values.tolist(),
                mode="lines", name=name, opacity=0.8,
            ))
        fig_sharpe.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_sharpe.update_layout(
            title="Rolling 126-Day Sharpe Ratio by Strategy",
            xaxis_title="Date", yaxis_title="Sharpe Ratio",
            template="plotly_white", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

        # ── Tail risk scaling ──
        tail_html = ""
        if tail_scaling is not None:
            tail_dates = _date_strings(tail_scaling.index)
            fig_tail = go.Figure()
            fig_tail.add_trace(go.Scatter(
                x=tail_dates, y=tail_scaling.values.tolist(),
                mode="lines", name="Tail Risk Scaling",
                line=dict(color="#e74c3c", width=1.5),
                fill="tozeroy",
            ))
            fig_tail.add_hline(y=1.0, line_dash="dash", line_color="grey")
            fig_tail.update_layout(
                title="Tail Risk Exposure Scaling (1.0 = full, lower = reduced)",
                xaxis_title="Date", yaxis_title="Scaling Factor",
                template="plotly_white", height=280,
            )
            tail_html = _card(_fig_html(fig_tail), "Tail Risk Management")

        # ── Comparison metrics ──
        metrics_data = []
        for label, rets in [("Static Risk Parity", static_returns), ("Dynamic Adaptive", dynamic_returns)]:
            metrics_data.append({
                "Portfolio": label,
                "CAGR": _cagr(rets),
                "Sharpe": _sharpe(rets),
                "Sortino": _sortino(rets),
                "Max DD": _max_drawdown(rets),
                "Calmar": _calmar(rets),
                "Omega": _omega_ratio(rets),
                "Tail Ratio": _tail_ratio(rets),
                "Win Rate": float((rets > 0).mean()),
                "Ann. Vol": float(rets.std() * np.sqrt(252)),
            })
        metrics_df = pd.DataFrame(metrics_data).set_index("Portfolio")

        content = (
            _df_to_html_table(metrics_df)
            + _card(_fig_html(fig_eq), "Equity Curve Comparison")
            + _card(_fig_html(fig_weights), "Dynamic Strategy Weights")
            + _card(_fig_html(fig_sharpe), "Rolling Strategy Sharpe Ratios")
            + tail_html
        )
        html = _render_html("Adaptive Allocation Dashboard", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 7. Monte Carlo Risk Dashboard
    # ------------------------------------------------------------------

    def generate_monte_carlo_dashboard(
        self,
        mc_results: dict,
        save_path: str = "reports/monte_carlo.html",
    ) -> str:
        """Generate Monte Carlo risk analysis dashboard with equity fan, drawdown
        distribution, Sharpe distribution, and leverage stress results."""

        content_parts = []

        # --- Bootstrap equity fan chart ---
        bs = mc_results.get("bootstrap", {})
        bs_paths = bs.get("simulated_paths")
        bs_metrics = bs.get("risk_metrics")

        if bs_paths is not None and not bs_paths.empty:
            n_display = min(200, bs_paths.shape[1])
            sample_cols = bs_paths.columns[:n_display]

            traces = []
            for col in sample_cols:
                traces.append(f"{{x: {list(range(len(bs_paths)))}, y: {bs_paths[col].tolist()}, "
                              f"type: 'scatter', mode: 'lines', opacity: 0.05, "
                              f"line: {{color: '#3498db', width: 0.5}}, showlegend: false}}")

            # Median path
            median_path = bs_paths.median(axis=1)
            traces.append(f"{{x: {list(range(len(median_path)))}, y: {median_path.tolist()}, "
                          f"type: 'scatter', mode: 'lines', name: 'Median', "
                          f"line: {{color: '#e74c3c', width: 3}}}}")

            # P5 and P95
            p5 = bs_paths.quantile(0.05, axis=1)
            p95 = bs_paths.quantile(0.95, axis=1)
            traces.append(f"{{x: {list(range(len(p5)))}, y: {p5.tolist()}, "
                          f"type: 'scatter', mode: 'lines', name: 'P5', "
                          f"line: {{color: '#f39c12', width: 2, dash: 'dash'}}}}")
            traces.append(f"{{x: {list(range(len(p95)))}, y: {p95.tolist()}, "
                          f"type: 'scatter', mode: 'lines', name: 'P95', "
                          f"line: {{color: '#27ae60', width: 2, dash: 'dash'}}}}")

            content_parts.append(f"""
            <div class="chart-container">
            <h3>Block Bootstrap Equity Paths ({bs_paths.shape[1]} simulations)</h3>
            <div id="mc-equity-fan"></div>
            <script>
            Plotly.newPlot('mc-equity-fan', [{','.join(traces)}],
                {{title: 'Simulated Wealth Paths', xaxis: {{title: 'Trading Days'}},
                  yaxis: {{title: 'Wealth (starting at 1.0)'}},
                  showlegend: true, height: 500}});
            </script>
            </div>
            """)

        # --- Drawdown distribution ---
        if bs_metrics is not None and "max_drawdown" in bs_metrics.columns:
            dd_vals = (bs_metrics["max_drawdown"] * 100).tolist()
            content_parts.append(f"""
            <div class="chart-container">
            <h3>Maximum Drawdown Distribution (Bootstrap)</h3>
            <div id="mc-dd-hist"></div>
            <script>
            Plotly.newPlot('mc-dd-hist',
                [{{x: {dd_vals}, type: 'histogram', nbinsx: 50,
                   marker: {{color: '#e74c3c', opacity: 0.7}},
                   name: 'Max Drawdown'}}],
                {{title: 'Distribution of Maximum Drawdowns',
                  xaxis: {{title: 'Max Drawdown (%)'}},
                  yaxis: {{title: 'Frequency'}}, height: 400}});
            </script>
            </div>
            """)

        # --- Sharpe distribution ---
        if bs_metrics is not None and "sharpe" in bs_metrics.columns:
            sharpe_vals = bs_metrics["sharpe"].tolist()
            content_parts.append(f"""
            <div class="chart-container">
            <h3>Sharpe Ratio Distribution (Bootstrap)</h3>
            <div id="mc-sharpe-hist"></div>
            <script>
            Plotly.newPlot('mc-sharpe-hist',
                [{{x: {sharpe_vals}, type: 'histogram', nbinsx: 50,
                   marker: {{color: '#3498db', opacity: 0.7}},
                   name: 'Sharpe'}}],
                {{title: 'Distribution of Sharpe Ratios',
                  xaxis: {{title: 'Sharpe Ratio'}},
                  yaxis: {{title: 'Frequency'}}, height: 400}});
            </script>
            </div>
            """)

        # --- Terminal wealth distribution ---
        if bs_metrics is not None and "terminal_wealth" in bs_metrics.columns:
            tw_vals = bs_metrics["terminal_wealth"].tolist()
            content_parts.append(f"""
            <div class="chart-container">
            <h3>Terminal Wealth Distribution</h3>
            <div id="mc-tw-hist"></div>
            <script>
            Plotly.newPlot('mc-tw-hist',
                [{{x: {tw_vals}, type: 'histogram', nbinsx: 50,
                   marker: {{color: '#27ae60', opacity: 0.7}},
                   name: 'Terminal Wealth'}}],
                {{title: 'Distribution of Terminal Wealth',
                  xaxis: {{title: 'Terminal Wealth (starting at 1.0)'}},
                  yaxis: {{title: 'Frequency'}}, height: 400}});
            </script>
            </div>
            """)

        # --- Summary metrics table ---
        summary = mc_results.get("summary", {})
        bs_summary = summary.get("bootstrap", {})
        perm_summary = summary.get("permutation", {})
        lev_summary = summary.get("leverage_optimal", {})

        content_parts.append(f"""
        <div class="chart-container">
        <h3>Monte Carlo Summary</h3>
        <table class="metrics-table">
        <tr><th>Metric</th><th>Bootstrap</th><th>Permutation</th></tr>
        <tr><td>Median Sharpe</td><td>{bs_summary.get('median_sharpe', 0):.3f}</td><td>{perm_summary.get('median_sharpe', 0):.3f}</td></tr>
        <tr><td>Median Max DD</td><td>{bs_summary.get('median_max_drawdown', 0):.2%}</td><td>{perm_summary.get('median_max_drawdown', 0):.2%}</td></tr>
        <tr><td>Median CAGR</td><td>{bs_summary.get('median_cagr', 0):.2%}</td><td>{perm_summary.get('median_cagr', 0):.2%}</td></tr>
        <tr><td>Prob. of Ruin</td><td>{bs_summary.get('probability_of_ruin', 0):.2%}</td><td>{perm_summary.get('probability_of_ruin', 0):.2%}</td></tr>
        <tr><td>P5 Terminal Wealth</td><td>{bs_summary.get('p5_terminal_wealth', 0):.2f}</td><td>—</td></tr>
        <tr><td>P50 Terminal Wealth</td><td>{bs_summary.get('p50_terminal_wealth', 0):.2f}</td><td>—</td></tr>
        <tr><td>P95 Terminal Wealth</td><td>{bs_summary.get('p95_terminal_wealth', 0):.2f}</td><td>—</td></tr>
        <tr><td>Optimal Leverage</td><td colspan="2">{lev_summary.get('optimal_leverage', 1.0):.1f}x — {lev_summary.get('reason', '')}</td></tr>
        </table>
        </div>
        """)

        # --- Leverage stress comparison ---
        lev_stress = mc_results.get("leverage_stress", {})
        lev_df = lev_stress.get("summary")
        if lev_df is not None and not lev_df.empty:
            lev_levels = lev_df["leverage"].tolist()
            sharpe_by_lev = lev_df["median_sharpe"].tolist()
            dd_by_lev = (lev_df["median_max_drawdown"] * 100).tolist()
            ruin_by_lev = (lev_df["probability_of_ruin"] * 100).tolist()

            content_parts.append(f"""
            <div class="chart-container">
            <h3>Leverage Stress Test</h3>
            <div id="mc-lev-stress"></div>
            <script>
            Plotly.newPlot('mc-lev-stress', [
                {{x: {lev_levels}, y: {sharpe_by_lev}, type: 'bar', name: 'Median Sharpe',
                  marker: {{color: '#3498db'}}, yaxis: 'y'}},
                {{x: {lev_levels}, y: {dd_by_lev}, type: 'scatter', mode: 'lines+markers',
                  name: 'Median MaxDD (%)', line: {{color: '#e74c3c', width: 3}}, yaxis: 'y2'}},
                {{x: {lev_levels}, y: {ruin_by_lev}, type: 'scatter', mode: 'lines+markers',
                  name: 'Ruin Prob (%)', line: {{color: '#f39c12', width: 3, dash: 'dot'}}, yaxis: 'y2'}}
            ], {{title: 'Leverage vs Risk Metrics',
                 xaxis: {{title: 'Leverage'}},
                 yaxis: {{title: 'Sharpe', side: 'left'}},
                 yaxis2: {{title: '% (DD / Ruin)', side: 'right', overlaying: 'y'}},
                 barmode: 'group', height: 450}});
            </script>
            </div>
            """)

        content = "\n".join(content_parts)
        html = _render_html("Monte Carlo Risk Analysis", content)
        _write(save_path, html)
        return content

    # ------------------------------------------------------------------
    # 8. Combined Dashboard (all sections in one tabbed page)
    # ------------------------------------------------------------------

    def generate_combined_dashboard(
        self,
        sections: dict[str, str],
        save_path: str = "reports/dashboard.html",
    ) -> str:
        """
        Assemble multiple dashboard sections into a single tabbed HTML page.

        Parameters
        ----------
        sections:
            Mapping of section_id -> HTML content (returned by generate_* methods).
            Expected keys: "performance", "strategy", "regime", "cost"
        save_path:
            Output file path.
        """
        tab_config = {
            "performance": {"label": "Performance", "title": "Portfolio Performance"},
            "strategy": {"label": "Strategies", "title": "Strategy Diagnostics"},
            "adaptive": {"label": "Adaptive", "title": "Adaptive Allocation"},
            "monte_carlo": {"label": "Monte Carlo", "title": "Monte Carlo Risk Analysis"},
            "regime": {"label": "Regimes", "title": "Regime Analysis"},
            "cost": {"label": "Costs", "title": "Cost Analysis"},
        }

        tabs = []
        for section_id, content in sections.items():
            cfg = tab_config.get(section_id, {"label": section_id.title(), "title": section_id.title()})
            tabs.append({
                "id": section_id,
                "label": cfg["label"],
                "title": cfg["title"],
                "content": content,
            })

        html = _render_combined(tabs)
        _write(save_path, html)
        return html
