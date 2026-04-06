"""
Live Trading Dashboard Generator.

Reads from:
  - Alpaca API (current positions, account, order history)
  - reports/trade_history.jsonl (local trade log)

Generates a single self-contained HTML file with:
  - Account overview (equity, P&L, buying power)
  - Current positions table with unrealized P&L
  - Equity curve (from trade history)
  - Trade log (all rebalances with order details)
  - Strategy allocation breakdown
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def generate_live_dashboard(
    account: dict,
    positions: dict[str, dict],
    trade_history: list[dict],
    save_path: str,
):
    """Generate self-contained HTML dashboard for live trading."""

    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    equity = account["equity"]
    cash = account["cash"]
    buying_power = account["buying_power"]

    # Position stats
    n_positions = len(positions)
    total_unrealized = sum(p["unrealized_pl"] for p in positions.values())
    total_market_val = sum(p["market_value"] for p in positions.values())
    n_long = sum(1 for p in positions.values() if float(p["qty"]) > 0)
    n_short = sum(1 for p in positions.values() if float(p["qty"]) < 0)

    # Equity history from trade log
    equity_data = []
    for t in trade_history:
        equity_data.append({
            "date": t.get("date", t["timestamp"][:10]),
            "equity": t.get("equity", 0),
            "n_positions": t.get("n_positions", 0),
            "gross": t.get("gross_exposure", 0),
        })

    # Build positions table rows
    pos_rows = ""
    for sym, p in sorted(positions.items(), key=lambda x: -abs(x[1]["market_value"])):
        pl_color = "#48bb78" if p["unrealized_pl"] >= 0 else "#fc8181"
        direction = "LONG" if float(p["qty"]) > 0 else "SHORT"
        pos_rows += f"""<tr>
            <td style="text-align:left;font-weight:600">{sym}</td>
            <td>{direction}</td>
            <td>{abs(float(p['qty'])):.2f}</td>
            <td>${abs(p['market_value']):,.2f}</td>
            <td>${p['avg_entry']:.2f}</td>
            <td style="color:{pl_color};font-weight:600">${p['unrealized_pl']:+,.2f}</td>
            <td style="color:{pl_color}">{p['unrealized_plpc']:+.2%}</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="7" style="text-align:center;color:#999">No open positions</td></tr>'

    # Trade history rows (most recent first)
    trade_rows = ""
    for t in reversed(trade_history[-50:]):  # Last 50 trades
        ts = t.get("date", t["timestamp"][:10])
        n_ord = t.get("n_orders", 0)
        n_ok = t.get("n_success", 0)
        n_fail = t.get("n_fail", 0)
        eq = t.get("equity", 0)
        gross = t.get("gross_exposure", 0)
        net = t.get("net_exposure", 0)
        status_color = "#48bb78" if n_fail == 0 else "#ecc94b"
        trade_rows += f"""<tr>
            <td style="text-align:left">{ts}</td>
            <td>${eq:,.0f}</td>
            <td>{t.get('n_positions', 0)}</td>
            <td>{n_ord}</td>
            <td style="color:{status_color}">{n_ok}/{n_ord}</td>
            <td>{gross:.1%}</td>
            <td>{net:+.1%}</td>
        </tr>"""

    if not trade_rows:
        trade_rows = '<tr><td colspan="7" style="text-align:center;color:#999">No trades yet</td></tr>'

    # Equity chart data
    eq_dates = json.dumps([e["date"] for e in equity_data])
    eq_values = json.dumps([e["equity"] for e in equity_data])

    # P&L color
    pl_color = "#48bb78" if total_unrealized >= 0 else "#fc8181"
    starting_equity = equity_data[0]["equity"] if equity_data else equity
    total_return = (equity / starting_equity - 1) if starting_equity > 0 else 0
    ret_color = "#48bb78" if total_return >= 0 else "#fc8181"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>QRT Live Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1923; color: #e2e8f0; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 16px 30px; display: flex; justify-content: space-between; align-items: center;
        }}
        .header h1 {{ font-size: 20px; }}
        .header .ts {{ font-size: 12px; color: #718096; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; padding: 15px 30px; }}
        .metric {{ background: #1a2332; border-radius: 8px; padding: 14px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 10px; color: #718096; text-transform: uppercase; margin-top: 4px; }}
        .section {{ padding: 0 30px 15px; }}
        .section h2 {{ font-size: 15px; color: #a0aec0; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .card {{ background: #1a2332; border-radius: 8px; padding: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ font-size: 11px; color: #718096; text-transform: uppercase; padding: 6px 10px; text-align: right; border-bottom: 1px solid #2d3748; }}
        th:first-child {{ text-align: left; }}
        td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #1e2a3a; font-size: 13px; }}
        td:first-child {{ text-align: left; }}
        .green {{ color: #48bb78; }} .red {{ color: #fc8181; }}
        .tabs {{ display: flex; gap: 2px; padding: 0 30px; }}
        .tab-btn {{ padding: 8px 16px; border: none; background: #1a2332; color: #718096; cursor: pointer;
                   border-radius: 6px 6px 0 0; font-size: 12px; font-weight: 600; }}
        .tab-btn.active {{ background: #243447; color: #e2e8f0; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        #equity-chart {{ height: 250px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>QRT Live Trading</h1>
    <div class="ts">Updated: {now} | Paper Trading</div>
</div>

<div class="grid">
    <div class="metric">
        <div class="metric-value">${equity:,.0f}</div>
        <div class="metric-label">Equity</div>
    </div>
    <div class="metric">
        <div class="metric-value" style="color:{ret_color}">{total_return:+.2%}</div>
        <div class="metric-label">Total Return</div>
    </div>
    <div class="metric">
        <div class="metric-value" style="color:{pl_color}">${total_unrealized:+,.0f}</div>
        <div class="metric-label">Unrealized P&L</div>
    </div>
    <div class="metric">
        <div class="metric-value">${cash:,.0f}</div>
        <div class="metric-label">Cash</div>
    </div>
    <div class="metric">
        <div class="metric-value">{n_positions}</div>
        <div class="metric-label">Positions ({n_long}L / {n_short}S)</div>
    </div>
    <div class="metric">
        <div class="metric-value">${buying_power:,.0f}</div>
        <div class="metric-label">Buying Power</div>
    </div>
    <div class="metric">
        <div class="metric-value">{len(trade_history)}</div>
        <div class="metric-label">Total Rebalances</div>
    </div>
</div>

<div class="section">
    <div class="card"><div id="equity-chart"></div></div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="showTab('positions')">Positions</button>
    <button class="tab-btn" onclick="showTab('trades')">Trade History</button>
</div>

<div class="section">
    <div id="positions" class="tab-content active">
        <div class="card">
            <table>
                <thead><tr>
                    <th>Symbol</th><th>Side</th><th>Qty</th><th>Value</th>
                    <th>Entry</th><th>P&L</th><th>P&L%</th>
                </tr></thead>
                <tbody>{pos_rows}</tbody>
            </table>
        </div>
    </div>
    <div id="trades" class="tab-content">
        <div class="card">
            <table>
                <thead><tr>
                    <th>Date</th><th>Equity</th><th>Positions</th><th>Orders</th>
                    <th>Filled</th><th>Gross</th><th>Net</th>
                </tr></thead>
                <tbody>{trade_rows}</tbody>
            </table>
        </div>
    </div>
</div>

<script>
function showTab(id) {{
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
}}

var dates = {eq_dates};
var values = {eq_values};
if (dates.length > 0) {{
    Plotly.newPlot('equity-chart', [{{
        x: dates, y: values, type: 'scatter', mode: 'lines+markers',
        fill: 'tozeroy', fillcolor: 'rgba(72,187,120,0.1)',
        line: {{ color: '#48bb78', width: 2 }},
        marker: {{ size: 4 }}
    }}], {{
        margin: {{ t: 30, r: 20, b: 30, l: 60 }},
        paper_bgcolor: '#1a2332', plot_bgcolor: '#1a2332',
        xaxis: {{ color: '#718096', gridcolor: '#2d3748' }},
        yaxis: {{ color: '#718096', gridcolor: '#2d3748', tickprefix: '$' }},
        title: {{ text: 'Equity Curve', font: {{ color: '#a0aec0', size: 14 }} }}
    }}, {{ responsive: true }});
}} else {{
    document.getElementById('equity-chart').innerHTML = '<p style="text-align:center;color:#718096;padding:40px">No trade history yet. Run auto-trader to start tracking.</p>';
}}
</script>

</body>
</html>"""

    Path(save_path).parent.mkdir(exist_ok=True)
    Path(save_path).write_text(html)
    logger.info(f"Live dashboard saved: {save_path}")
    return save_path
