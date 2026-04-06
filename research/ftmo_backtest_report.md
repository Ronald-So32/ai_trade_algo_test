# FTMO Challenge Backtest Report — TSMOM-Only

Generated: 2026-04-04
Data Source: yfinance daily proxy (53 instruments, 1063 trading days)
Strategy: TSMOM-Only (100% weight)
Profit Target: 10% ($10,000) — FTMO 1-Step
Instruments: 53
OOS Period: 2023-02-22 to 2026-04-03 (811 trading days)

---

## 1. TSMOM Strategy Performance (Out-of-Sample)

### At optimal vol target: 20%

| Metric | Value |
|--------|-------|
| CAGR | 0.1837 |
| Cumulative P&L ($) | $61,815 |
| Hit Rate | 53.6% |
| Payoff Ratio | 1.0123 |
| Volatility (ann.) | 0.2156 |
| Downside Deviation | 0.1673 |
| Max Drawdown | 0.2594 |
| Time Under Water (days) | 235 |
| Sharpe Ratio | 0.6591 |
| Sharpe SE (Lo 2002) | 0.6150 |
| Sharpe 95% CI | [-0.546, 1.864] |
| Sortino Ratio | 0.8492 |
| Calmar Ratio | 0.7083 |
| Profit Factor | 1.1743 |
| Skewness | -0.6806 |
| Excess Kurtosis | 7.5098 |
| Trading Days | 811 |
| Years | 3.2 |

## 2. FTMO Pass Rate by Vol Target

| Vol Target | Sharpe | CAGR | Max DD | Pass Rate | Avg Days to Pass | Fail (MDL/ML/Timeout) |
|------------|--------|------|--------|-----------|------------------|-----------------------|
| 20% ** | 0.659 | 18.4% | 25.9% | 39.5% | 60 | 10/3/10 |
| 25% | 0.685 | 21.2% | 30.7% | 34.2% | 43 | 19/2/4 |
| 30% | 0.692 | 23.1% | 34.5% | 31.6% | 40 | 23/2/1 |
| 35% | 0.689 | 24.3% | 39.1% | 34.2% | 37 | 22/2/1 |

## 3. FTMO Challenge Simulation (Optimal Config)

- **Profit Target**: $5,000 (5%)
- **Total Challenges**: 38
- **Pass Rate**: 39.5%
- **Avg Days to Pass**: 59.8
- **Avg Days to Fail**: 84.7
- **Avg Final P&L**: $3,013
- **Avg Max DD**: $10,164
- **Avg Best Day Ratio**: 17.4%
- **Failures: Daily Loss**: 10
- **Failures: Total Loss**: 3
- **Failures: Timeout**: 10
- **Failures: Best Day**: 0

## 4. Grossman-Zhou Adaptive Sizing

Scales positions based on proximity to FTMO loss limits:
- Daily loss limit: $3,000 (-3%)
- Total loss limit: $10,000 (-10%)
- Position scale = min(daily_headroom, total_headroom)

## 5. Stress Test Results

| Scenario | Sharpe | CAGR | Max DD | FTMO Pass Rate | Avg Days |
|----------|--------|------|--------|----------------|----------|
| baseline | 0.676 | 18.8% | 25.5% | 42.1% | 57 |
| 2x_spread | 0.553 | 15.7% | 25.9% | 39.5% | 53 |
| 2x_slippage | 0.594 | 16.7% | 25.8% | 39.5% | 53 |
| 2x_spread_slippage | 0.470 | 13.7% | 26.4% | 36.8% | 61 |
| weekend_gaps | 0.695 | 22.0% | 29.8% | 34.2% | 29 |
| 2x_swap | 0.622 | 17.4% | 25.7% | 42.1% | 59 |
| all_stress | 0.483 | 15.4% | 32.7% | 36.8% | 35 |

**Stress scenarios**: 2x spread (6 bps), 2x slippage (4 bps), weekend gap shocks (±2% on Mondays), 2x swap costs (100 bps/yr), all combined.

## 6. Next Steps

1. **Re-run on MT5 data** — pass `--mt5-cache path/to/data.parquet` when available
2. **Paper trade 2-4 weeks** on FTMO demo account before live challenge
3. **Monitor stress degradation** — if live costs approach 2x scenario, reduce vol target

## 7. Caveats

- **DATA**: Uses yfinance spot data, NOT MT5 CFD prices (spread/swap/gap differences)
- No intraday MDL tracking (daily bars only — intraday breaches would be worse)
- Best day rule checked at challenge level, not intraday
- Sharpe ratio CI may cross zero — limited OOS history
- TSMOM-only: no diversification benefit from other strategies
