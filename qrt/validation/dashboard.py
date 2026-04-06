"""
Validation Dashboard Generator
================================
Produces an HTML dashboard with panels required by the validation spec:
  - Research grounding panel
  - Drawdown analysis panel
  - Regime performance panel
  - Benchmark comparison panel
  - Validation audit report panel
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Environment, BaseLoader

from .benchmark import compute_metrics, drawdown_analysis, BenchmarkComparison

logger = logging.getLogger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f0f2f5; }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 20px 40px; color: #e2e8f0;
        }
        .header h1 { margin: 0; font-size: 24px; }
        .header .status { font-size: 18px; margin-top: 8px; }
        .status-validated { color: #48bb78; font-weight: bold; }
        .status-not-validated { color: #fc8181; font-weight: bold; }
        .container { max-width: 1500px; margin: 20px auto; padding: 0 20px; }
        .tabs { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab-btn {
            padding: 10px 20px; border: none; background: #e2e8f0; cursor: pointer;
            border-radius: 6px 6px 0 0; font-size: 14px; font-weight: 600;
        }
        .tab-btn.active { background: white; color: #2c3e50; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .metric { text-align: center; padding: 12px; background: #f8f9fa; border-radius: 6px; }
        .metric-value { font-size: 22px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid #eee; font-size: 13px; }
        th { background: #f8f9fa; font-weight: 600; text-align: left; }
        td:first-child, th:first-child { text-align: left; }
        h2 { color: #2c3e50; margin: 20px 0 10px; }
        h3 { color: #34495e; margin: 15px 0 8px; }
        .finding { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 13px; }
        .finding-pass { background: #f0fff4; border-left: 4px solid #48bb78; }
        .finding-fail { background: #fff5f5; border-left: 4px solid #fc8181; }
        .finding-warning { background: #fffbeb; border-left: 4px solid #ecc94b; }
        .grounding-card { background: #f7fafc; border-radius: 8px; padding: 16px; margin: 10px 0;
                          border-left: 4px solid #4299e1; }
        .grounding-card h4 { margin: 0 0 8px; color: #2b6cb0; }
        .grounding-card p { margin: 4px 0; font-size: 13px; line-height: 1.5; }
        .grounding-label { font-weight: 600; color: #4a5568; }
    </style>
</head>
<body>
<div class="header">
    <h1>{{ title }}</h1>
    <div class="status">
        Status: <span class="{{ status_class }}">{{ status }}</span>
        &nbsp;|&nbsp; Data Source: {{ data_source }}
        &nbsp;|&nbsp; Passed: {{ pass_count }} | Failed: {{ fail_count }} | Warnings: {{ warn_count }}
    </div>
</div>
<div class="container">
    <div class="tabs">
        {% for tab in tabs %}
        <button class="tab-btn {% if loop.first %}active{% endif %}"
                onclick="showTab('{{ tab.id }}')">{{ tab.label }}</button>
        {% endfor %}
    </div>

    {% for tab in tabs %}
    <div id="{{ tab.id }}" class="tab-content {% if loop.first %}active{% endif %}">
        {{ tab.content }}
    </div>
    {% endfor %}
</div>
<script>
function showTab(id) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
    // Resize plotly charts
    setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
}
</script>
</body>
</html>"""


class ValidationDashboardGenerator:
    """Generate the validation dashboard with all required panels."""

    def __init__(self) -> None:
        self._env = Environment(loader=BaseLoader())

    def generate(
        self,
        audit_report,
        strategy_results: dict,
        benchmark_comparison: pd.DataFrame | None = None,
        regime_performance: pd.DataFrame | None = None,
        composite_results: pd.DataFrame | None = None,
        portfolio_returns: pd.Series | None = None,
        trade_stats: dict | None = None,
        save_path: str = "reports/validation_dashboard.html",
    ) -> str:
        """Generate the full validation dashboard."""
        tabs = []

        # Tab 1: Audit Report
        tabs.append({
            "id": "audit",
            "label": "Validation Audit",
            "content": self._render_audit_panel(audit_report),
        })

        # Tab 2: Research Grounding
        tabs.append({
            "id": "research",
            "label": "Research Grounding",
            "content": self._render_research_panel(strategy_results),
        })

        # Tab 3: Benchmark Comparison
        if benchmark_comparison is not None:
            tabs.append({
                "id": "benchmark",
                "label": "Benchmark Comparison",
                "content": self._render_benchmark_panel(
                    benchmark_comparison, strategy_results
                ),
            })

        # Tab 4: Trade Statistics
        tabs.append({
            "id": "trades",
            "label": "Trade Statistics",
            "content": self._render_trade_stats_panel(
                strategy_results, portfolio_returns, trade_stats
            ),
        })

        # Tab 5: Drawdown Analysis
        tabs.append({
            "id": "drawdown",
            "label": "Drawdown Analysis",
            "content": self._render_drawdown_panel(strategy_results, portfolio_returns),
        })

        # Tab 6: Regime Performance
        if regime_performance is not None and not regime_performance.empty:
            tabs.append({
                "id": "regime",
                "label": "Regime Performance",
                "content": self._render_regime_panel(regime_performance),
            })

        # Tab 7: Composite Testing
        if composite_results is not None and not composite_results.empty:
            tabs.append({
                "id": "composite",
                "label": "Factor Composites",
                "content": self._render_composite_panel(composite_results),
            })

        status = audit_report.determine_status()
        template = self._env.from_string(_TEMPLATE)
        html = template.render(
            title="Quantitative Platform Validation Report",
            status=status,
            status_class="status-validated" if status == "VALIDATED" else "status-not-validated",
            data_source=audit_report.data_source,
            pass_count=audit_report.pass_count,
            fail_count=audit_report.fail_count,
            warn_count=audit_report.warning_count,
            tabs=tabs,
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(html)
        logger.info(f"Validation dashboard saved to {save_path}")
        return html

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_audit_panel(self, report) -> str:
        """Render the validation audit findings."""
        sections = {}
        for f in report.findings:
            sections.setdefault(f.category, []).append(f)

        html_parts = ['<div class="card"><h2>Backtest Integrity Audit</h2>']

        for cat, findings in sorted(sections.items()):
            html_parts.append(f'<h3>{cat.replace("_", " ").title()}</h3>')
            for f in findings:
                css = f"finding-{f.status.lower()}"
                icon = {"PASS": "&#10003;", "FAIL": "&#10007;", "WARNING": "&#9888;"}[f.status]
                loc = f' <small>({f.file_path}:{f.line_number})</small>' if f.file_path else ""
                html_parts.append(
                    f'<div class="finding {css}">'
                    f'{icon} <strong>{f.check_name}</strong>: {f.detail}{loc}'
                    f'</div>'
                )

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_research_panel(self, strategy_results: dict) -> str:
        """Render research grounding for each strategy."""
        html_parts = ['<div class="card"><h2>Strategy Research Grounding</h2>']
        html_parts.append(
            '<p><em>All strategies represent research-supported return premia. '
            'No strategy is described as guaranteed or proven. '
            'Past performance does not predict future results.</em></p>'
        )

        for name, res in sorted(strategy_results.items()):
            strategy = res.get("strategy")
            if strategy is None:
                continue

            grounding = getattr(strategy, "RESEARCH_GROUNDING", {})
            if not grounding:
                continue

            html_parts.append(f'<div class="grounding-card">')
            html_parts.append(f'<h4>{name.replace("_", " ").title()}</h4>')

            labels = {
                "academic_basis": "Academic Basis",
                "historical_evidence": "Historical Evidence",
                "implementation_risks": "Implementation Risks",
                "realistic_expectations": "Realistic Expectations",
            }
            for key, label in labels.items():
                val = grounding.get(key, "Not specified")
                html_parts.append(
                    f'<p><span class="grounding-label">{label}:</span> {val}</p>'
                )

            html_parts.append('</div>')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_trade_stats_panel(
        self,
        strategy_results: dict,
        portfolio_returns: pd.Series | None,
        trade_stats: dict | None,
    ) -> str:
        """Render trade statistics panel with win rate, trade frequency, profit factor."""
        html_parts = ['<div class="card"><h2>Trade Statistics</h2>']
        html_parts.append(
            '<p><em>Stop-loss engine: vol-scaled stops (k=2&sigma;) with '
            'regime-aware crisis tightening. Per Kaminski &amp; Lo (2014), '
            'Arratia &amp; Dorador (2019).</em></p>'
        )

        # Build trade stats from strategy_results
        rows = []
        for name, res in sorted(strategy_results.items()):
            ts = res.get("trade_stats", {})
            if trade_stats and name in trade_stats:
                ts = trade_stats[name]
            if not ts:
                continue
            rows.append({
                "strategy": name,
                "win_rate": ts.get("win_rate", ts.get("trade_win_rate", 0)),
                "total_trades": ts.get("total_trades", ts.get("total_trades_approx", 0)),
                "trades_per_year": ts.get("trades_per_year", 0),
                "profit_factor": ts.get("profit_factor", 0),
                "avg_holding": ts.get("avg_holding_days", 0),
                "avg_return": ts.get("avg_return", ts.get("avg_trade_return", 0)),
                "avg_winner": ts.get("avg_winner", 0),
                "avg_loser": ts.get("avg_loser", 0),
                "stop_exits": ts.get("stop_loss_pct", 0),
            })

        if rows:
            # Summary metrics
            avg_wr = np.mean([r["win_rate"] for r in rows])
            total_trades = sum(r["total_trades"] for r in rows)
            html_parts.append('<div class="metric-grid">')
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{avg_wr:.1%}</div>'
                f'<div class="metric-label">Avg Win Rate</div></div>'
            )
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{total_trades:,}</div>'
                f'<div class="metric-label">Total Trades</div></div>'
            )
            if portfolio_returns is not None:
                port_wr = float((portfolio_returns > 0).mean())
                monthly = portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
                mwr = float((monthly > 0).mean()) if len(monthly) > 0 else 0
                html_parts.append(
                    f'<div class="metric"><div class="metric-value">{port_wr:.1%}</div>'
                    f'<div class="metric-label">Portfolio Daily Win Rate</div></div>'
                )
                html_parts.append(
                    f'<div class="metric"><div class="metric-value">{mwr:.1%}</div>'
                    f'<div class="metric-label">Portfolio Monthly Win Rate</div></div>'
                )
            html_parts.append('</div>')

            # Table
            html_parts.append('<h3>Per-Strategy Trade Metrics</h3>')
            html_parts.append('<table>')
            html_parts.append(
                '<tr><th>Strategy</th><th>Win Rate</th><th>Trades</th>'
                '<th>Trades/Yr</th><th>Profit Factor</th>'
                '<th>Avg Hold (d)</th><th>Avg Return</th>'
                '<th>Avg Winner</th><th>Avg Loser</th><th>Stop Exits</th></tr>'
            )
            for r in rows:
                html_parts.append(
                    f'<tr><td>{r["strategy"]}</td>'
                    f'<td>{r["win_rate"]:.1%}</td>'
                    f'<td>{r["total_trades"]}</td>'
                    f'<td>{r["trades_per_year"]:.0f}</td>'
                    f'<td>{r["profit_factor"]:.2f}</td>'
                    f'<td>{r["avg_holding"]:.1f}</td>'
                    f'<td>{r["avg_return"]:.3%}</td>'
                    f'<td>{r["avg_winner"]:.3%}</td>'
                    f'<td>{r["avg_loser"]:.3%}</td>'
                    f'<td>{r["stop_exits"]:.1%}</td></tr>'
                )
            html_parts.append('</table>')
        else:
            html_parts.append('<p>No trade statistics available.</p>')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_benchmark_panel(
        self,
        comparison_df: pd.DataFrame,
        strategy_results: dict,
    ) -> str:
        """Render benchmark comparison table and charts."""
        html_parts = ['<div class="card"><h2>Benchmark Comparison</h2>']

        # Performance table — include win rate
        display_cols = [
            "annualized_return", "volatility", "sharpe", "sortino",
            "calmar", "max_drawdown", "turnover",
        ]
        available_cols = [c for c in display_cols if c in comparison_df.columns]

        html_parts.append('<table>')
        html_parts.append('<tr><th>Strategy</th>')
        for col in available_cols:
            html_parts.append(f'<th>{col.replace("_", " ").title()}</th>')
        html_parts.append('<th>Win Rate</th>')
        html_parts.append('</tr>')

        for idx, row in comparison_df.iterrows():
            html_parts.append(f'<tr><td>{idx}</td>')
            for col in available_cols:
                val = row.get(col, 0)
                if "return" in col or "drawdown" in col or "volatility" in col or "drag" in col:
                    html_parts.append(f'<td>{val:.2%}</td>')
                elif "turnover" in col:
                    html_parts.append(f'<td>{val:.4f}</td>')
                else:
                    html_parts.append(f'<td>{val:.3f}</td>')
            # Win rate from strategy results
            wr = 0.0
            if idx in strategy_results:
                ts = strategy_results[idx].get("trade_stats", {})
                wr = ts.get("win_rate", ts.get("trade_win_rate", 0))
            html_parts.append(f'<td>{wr:.1%}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')

        # Equity curves chart
        fig = go.Figure()
        for name, res in sorted(strategy_results.items()):
            ret = res["returns"]
            equity = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity.values,
                mode="lines", name=name, line=dict(width=1.5),
            ))
        fig.update_layout(
            title="Cumulative Returns by Strategy",
            yaxis_title="Growth of $1",
            height=450, template="plotly_white",
            legend=dict(font=dict(size=10)),
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_drawdown_panel(
        self,
        strategy_results: dict,
        portfolio_returns: pd.Series | None,
    ) -> str:
        """Render drawdown analysis charts."""
        html_parts = ['<div class="card"><h2>Drawdown Analysis</h2>']

        # Drawdown curves
        fig = make_subplots(rows=1, cols=1)
        dd_summaries = []

        for name, res in sorted(strategy_results.items()):
            ret = res["returns"]
            analysis = drawdown_analysis(ret)
            dd_series = analysis["drawdown_series"]
            fig.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values,
                mode="lines", name=name, line=dict(width=1),
            ))
            dd_summaries.append({
                "strategy": name,
                "max_drawdown": analysis["max_drawdown"],
                "avg_drawdown": analysis["avg_drawdown"],
                "time_in_dd_pct": analysis["time_in_drawdown_pct"],
                "worst_episodes": len(analysis["worst_episodes"]),
            })

        if portfolio_returns is not None:
            port_analysis = drawdown_analysis(portfolio_returns)
            fig.add_trace(go.Scatter(
                x=port_analysis["drawdown_series"].index,
                y=port_analysis["drawdown_series"].values,
                mode="lines", name="Portfolio", line=dict(width=2.5, color="black"),
            ))

        fig.update_layout(
            title="Rolling Drawdown",
            yaxis_title="Drawdown", yaxis_tickformat=".0%",
            height=400, template="plotly_white",
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # Drawdown summary table
        html_parts.append('<h3>Drawdown Summary</h3><table>')
        html_parts.append(
            '<tr><th>Strategy</th><th>Max DD</th><th>Avg DD</th>'
            '<th>Time in DD</th></tr>'
        )
        for s in dd_summaries:
            html_parts.append(
                f'<tr><td>{s["strategy"]}</td>'
                f'<td>{s["max_drawdown"]:.2%}</td>'
                f'<td>{s["avg_drawdown"]:.2%}</td>'
                f'<td>{s["time_in_dd_pct"]:.1%}</td></tr>'
            )
        html_parts.append('</table>')

        # Worst episodes table
        if portfolio_returns is not None:
            analysis = drawdown_analysis(portfolio_returns)
            if analysis["worst_episodes"]:
                html_parts.append('<h3>Worst Portfolio Drawdown Episodes</h3><table>')
                html_parts.append(
                    '<tr><th>Start</th><th>Trough</th><th>End</th>'
                    '<th>Depth</th><th>Duration (days)</th><th>Recovery (days)</th></tr>'
                )
                for ep in analysis["worst_episodes"][:5]:
                    html_parts.append(
                        f'<tr><td>{ep["start"].date()}</td>'
                        f'<td>{ep["trough"].date()}</td>'
                        f'<td>{ep["end"].date()}</td>'
                        f'<td>{ep["depth"]:.2%}</td>'
                        f'<td>{ep["duration_days"]}</td>'
                        f'<td>{ep["recovery_days"]}</td></tr>'
                    )
                html_parts.append('</table>')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_regime_panel(self, regime_df: pd.DataFrame) -> str:
        """Render regime-conditional performance."""
        html_parts = ['<div class="card"><h2>Regime-Conditional Performance</h2>']

        display_cols = ["annualized_return", "volatility", "sharpe", "max_drawdown", "n_days"]
        available = [c for c in display_cols if c in regime_df.columns]

        html_parts.append('<table>')
        html_parts.append('<tr><th>Strategy</th><th>Regime</th>')
        for col in available:
            html_parts.append(f'<th>{col.replace("_", " ").title()}</th>')
        html_parts.append('</tr>')

        for (strat, regime), row in regime_df.iterrows():
            html_parts.append(f'<tr><td>{strat}</td><td>{regime}</td>')
            for col in available:
                val = row.get(col, 0)
                if col == "n_days":
                    html_parts.append(f'<td>{int(val)}</td>')
                elif "return" in col or "drawdown" in col or "volatility" in col:
                    html_parts.append(f'<td>{val:.2%}</td>')
                else:
                    html_parts.append(f'<td>{val:.3f}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_composite_panel(self, composite_df: pd.DataFrame) -> str:
        """Render factor composite testing results."""
        html_parts = ['<div class="card"><h2>Factor Composite Walk-Forward Testing</h2>']
        html_parts.append(
            '<p><em>Combinations ranked by out-of-sample Sharpe ratio. '
            'Walk-forward testing uses rolling train/test splits to prevent overfitting.</em></p>'
        )

        display_cols = [
            "combination", "n_strategies",
            "is_sharpe", "is_annualized_return", "is_max_drawdown",
            "oos_sharpe", "oos_annualized_return", "oos_max_drawdown",
        ]
        available = [c for c in display_cols if c in composite_df.columns]

        html_parts.append('<table>')
        html_parts.append('<tr>')
        col_labels = {
            "combination": "Combination",
            "n_strategies": "# Strats",
            "is_sharpe": "IS Sharpe",
            "is_annualized_return": "IS Return",
            "is_max_drawdown": "IS MaxDD",
            "oos_sharpe": "OOS Sharpe",
            "oos_annualized_return": "OOS Return",
            "oos_max_drawdown": "OOS MaxDD",
        }
        for col in available:
            html_parts.append(f'<th>{col_labels.get(col, col)}</th>')
        html_parts.append('</tr>')

        for _, row in composite_df.head(20).iterrows():
            html_parts.append('<tr>')
            for col in available:
                val = row.get(col, "")
                if isinstance(val, float):
                    if "return" in col or "drawdown" in col:
                        html_parts.append(f'<td>{val:.2%}</td>')
                    else:
                        html_parts.append(f'<td>{val:.3f}</td>')
                elif isinstance(val, int):
                    html_parts.append(f'<td>{val}</td>')
                else:
                    html_parts.append(f'<td>{val}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')

        html_parts.append('</div>')
        return "\n".join(html_parts)
