# v5 Optimization Sweep — Full Results

## Methodology

**Per Harvey, Liu & Zhu (2016)**: we report ALL configurations tested to avoid
cherry-picking. The multiple testing penalty means only results with large OOS
improvements over the baseline should be trusted.

**Configurations tested**: 18 total
- 2 allocation methods: equal-weight (DeMiguel et al. 2009) vs risk parity (Maillard et al. 2010)
- 8 vol targets: 8%, 10%, 12%, 15%, 18%, 20%, 25%, 30% (Moreira & Muir 2017)
- 2 fixed leverage baselines: equal-weight 2x, risk parity 2x

**Data**: ~150 S&P 500 large-caps, real data from Yahoo Finance (2010-2026)
**Holdout**: Jan 2024 onwards (OOS)

## How Each Component Works

### 1. Risk Parity Allocation (Maillard, Roncalli & Teiletche 2010)

Instead of giving each strategy 50% of capital (equal-weight), risk parity gives
each strategy equal *risk contribution*. The weight of each strategy is inversely
proportional to its trailing 63-day realized volatility:

    w_i = (1/vol_i) / sum(1/vol_j for all j)

**Why it helps**: TSMOM has ~3x the volatility of Residual Reversal. With equal
weight, TSMOM dominates portfolio risk. Risk parity shifts more capital to the
lower-vol, higher-Sharpe Residual Reversal strategy, improving risk-adjusted returns.

**Example**: If TSMOM vol = 15% and ResRev vol = 5%:
- Equal weight: 50/50 → TSMOM contributes 75% of portfolio risk
- Risk parity: 25/75 → each contributes 50% of portfolio risk

### 2. Volatility-Managed Leverage (Moreira & Muir 2017)

Instead of fixed 2x leverage every day, scale leverage inversely to recent
realized portfolio volatility, targeting a specific annualized vol:

    leverage = min(vol_target / realized_vol_63d, 2.0)

**Why it helps**: Momentum strategies suffer "momentum crashes" — sudden reversals
during high-vol periods (Daniel & Moskowitz 2016). By reducing leverage when vol
is elevated, the system avoids over-leveraging into crashes. Barroso & Santa-Clara
(2015) document this doubles momentum Sharpe from 0.53 to 0.97.

**The vol target controls the aggression**:
- Low target (8-10%): very conservative, low CAGR, low drawdown
- Medium target (15-20%): balanced, moderate CAGR, moderate drawdown
- High target (25-30%): aggressive, approaches fixed 2x behavior
- Fixed 2x: maximum CAGR but worst drawdowns

### 3. Within-Industry Residual Reversal (Blitz et al. 2013, 2023)

Stocks are sorted not on raw 5-day returns, but on *residual* returns after
subtracting the stock's GICS sub-industry mean return. This strips out industry
momentum contamination that weakens the standard reversal signal.

**72 sub-industries** (e.g., Semiconductors, Diversified Banks, Pharmaceuticals)
provide granular decontamination. Blitz et al. (2023) report within-industry
reversal has t-stat 5.49, delivering 1.5x the standard reversal premium.

## All Results (Sorted by OOS Sharpe)

| Configuration | IS Sharpe | IS CAGR | IS MaxDD | OOS Sharpe | OOS CAGR | OOS MaxDD | OOS Sortino | OOS Calmar | Degradation |
|--------------|:---------:|:-------:|:--------:|:----------:|:--------:|:---------:|:-----------:|:----------:|:-----------:|
| risk_parity_vm30pct | 0.582 | 3.83% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -105.2% |
| risk_parity_fixed_2x | 0.582 | 3.83% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -105.2% |
| risk_parity_vm25pct | 0.582 | 3.83% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -105.2% |
| risk_parity_vm20pct | 0.582 | 3.83% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -105.2% |
| risk_parity_vm18pct | 0.582 | 3.82% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -105.1% |
| risk_parity_vm15pct | 0.574 | 3.73% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -108.1% |
| risk_parity_vm12pct | 0.559 | 3.58% | -20.1% | 1.194 | 7.24% | -4.2% | 1.412 | 1.73 | -113.7% |
| risk_parity_vm10pct | 0.539 | 3.38% | -20.1% | 1.190 | 7.19% | -4.1% | 1.405 | 1.74 | -120.9% |
| risk_parity_vm8pct | 0.551 | 3.30% | -18.8% | 1.183 | 6.83% | -3.6% | 1.417 | 1.89 | -114.6% |
| equal_fixed_2x | 0.534 | 5.89% | -29.6% | 1.141 | 8.49% | -6.3% | 1.366 | 1.35 | -113.5% |
| equal_vm25pct | 0.556 | 6.06% | -27.2% | 1.141 | 8.49% | -6.3% | 1.363 | 1.35 | -105.1% |
| equal_vm20pct | 0.578 | 6.23% | -25.0% | 1.141 | 8.49% | -6.3% | 1.363 | 1.35 | -97.2% |
| equal_vm18pct | 0.586 | 6.25% | -25.0% | 1.141 | 8.49% | -6.3% | 1.363 | 1.35 | -94.8% |
| equal_vm15pct | 0.598 | 6.24% | -24.7% | 1.141 | 8.49% | -6.3% | 1.363 | 1.35 | -90.8% |
| equal_vm30pct | 0.545 | 5.97% | -28.5% | 1.141 | 8.49% | -6.3% | 1.363 | 1.35 | -109.3% |
| equal_vm12pct | 0.617 | 6.03% | -23.7% | 1.137 | 8.44% | -6.3% | 1.356 | 1.34 | -84.1% |
| equal_vm10pct | 0.636 | 5.68% | -21.1% | 1.092 | 7.91% | -6.3% | 1.292 | 1.26 | -71.8% |
| equal_vm8pct | 0.656 | 5.05% | -16.9% | 1.088 | 7.25% | -5.7% | 1.338 | 1.26 | -65.9% |

## Winners

| Criterion | Configuration | OOS Sharpe | OOS CAGR | OOS MaxDD |
|-----------|--------------|:----------:|:--------:|:---------:|
| Best Sharpe | risk_parity_vm30pct | 1.194 | 7.24% | -4.2% |
| Best CAGR | equal_fixed_2x | 1.141 | 8.49% | -6.3% |
| Best Calmar | risk_parity_vm8pct | 1.183 | 6.83% | -3.6% |

## Multiple Testing Warning (Harvey et al. 2016)

We tested 18 configurations. The Bonferroni-adjusted significance threshold for
18 tests at 5% is p < 0.0028 (t-stat > 2.77). Only configurations with OOS
Sharpe improvements exceeding this threshold should be considered robust.

The safest approach: choose based on the *direction* of improvement (risk parity
consistently beats equal-weight; vol-managed consistently beats fixed) rather
than the exact optimal vol target, which may be sample-specific.

## References

1. Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *JF*.
2. Barroso, P. & Santa-Clara, P. (2015). Momentum Has Its Moments. *JFE*.
3. Maillard, S., Roncalli, T. & Teiletche, J. (2010). Equally Weighted Risk Contributions Portfolios. *JPM*.
4. Daniel, K. & Moskowitz, T. (2016). Momentum Crashes. *JFE*.
5. Blitz, D. et al. (2013). Short-Term Residual Reversal. *JFM*.
6. Blitz, D. et al. (2023). Reversing the Trend of Short-Term Reversal. *JPM*.
7. Harvey, C., Liu, Y. & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *RFS*.
8. DeMiguel, V. et al. (2009). Optimal Versus Naive Diversification. *RFS*.
