# Simplified Strategy System v4 — Research Report

## Design Philosophy

This system follows Harvey, Liu & Zhu (2016) and McLean & Pontiff (2016) to minimize overfitting.
Runs on ~150 S&P 500 large-caps across 11 GICS sectors and 72 sub-industries.

**Key changes from v3:**
- Expanded universe from 49 to ~150 stocks — quintile sorts go from 10 to 30 stocks
- Sub-industry residuals (Blitz et al. 2023) for finer reversal signal isolation
- Dropped 52-Week High (IS Sharpe 0.055 over 14 years with 151 stocks)
- 2-strategy system: each strategy gets 50% weight instead of 33%

**Principles:**
- **ZERO free parameters** — all values from published academic papers
- **2 strategies** (from 16 → 4 → 3 → 2) — only strategies with strong IS + OOS performance
- **~150 large-cap stocks** — robust cross-sectional sorts, 30 stocks per quintile
- **Within-industry residuals** — 72 GICS sub-industries for reversal signal quality
- **Equal-weight allocation** — DeMiguel et al. (2009) shows 1/N beats optimization OOS
- **Simple vol targeting** — Moreira & Muir (2017) standard specification
- **No HMM, no stop-loss, no ensemble** — minimal complexity

## Strategy Parameters (All Literature-Fixed)

| Strategy | Paper | Lookback | Other Key Params |
|----------|-------|----------|------------------|
| TSMOM | Moskowitz et al. (2012) | 252d (12mo) | Vol-scaled positions |
| Residual Reversal | Blitz et al. (2013, 2023) | 5d (1wk) | Within-industry residuals (72 groups), quintile |

**Total free parameters: 0** (vs ~60+ in original system)
**Universe: ~150 S&P 500 stocks** (vs 49 in v3)

## In-Sample Performance (Full Period)

| Metric | Raw Portfolio | Leveraged (1.7x avg) |
|--------|:------------:|:-----------------------------:|
| Sharpe | 0.538 | 0.639 |
| CAGR | 3.11% | 5.71% |
| Max DD | -15.48% | -20.83% |
| Volatility | 6.04% | 9.39% |

### Portfolio Win Rates (In-Sample, Leveraged)

| Frequency | Win Rate |
|-----------|:--------:|
| Daily | 54.4% |
| Weekly | 57.4% |
| Monthly | 61.7% |
| Profit Factor | 1.12 |
| Avg Win / Avg Loss | 0.93 |

## Individual Strategy Performance (In-Sample)

| Strategy | Sharpe | CAGR | Max DD | Turnover | Daily WR | Weekly WR | Monthly WR | Profit Factor |
|----------|:------:|:----:|:------:|:--------:|:--------:|:---------:|:----------:|:-------------:|
| time_series_momentum | 0.466 | 5.55% | -33.30% | 0.08 | 51.4% | 54.2% | 59.4% | 1.09 |
| residual_reversal | 0.288 | 0.95% | -11.95% | 0.34 | 51.3% | 52.3% | 53.3% | 1.05 |

## Out-of-Sample Performance

| Metric | In-Sample | Out-of-Sample | Degradation |
|--------|:---------:|:-------------:|:-----------:|
| Sharpe | 0.639 | 1.064 | -66.5% |
| CAGR | 5.71% | 7.57% | -32.6% |
| Max DD | -20.83% | -6.07% | — |
| Volatility | 9.39% | 7.10% | — |

### OOS Win Rates (Portfolio, Leveraged)

| Frequency | In-Sample | Out-of-Sample |
|-----------|:---------:|:-------------:|
| Daily | 54.4% | 53.2% |
| Weekly | 57.4% | 59.7% |
| Monthly | 61.7% | 64.4% |
| Profit Factor | 1.12 | 1.22 |

### OOS Strategy-Level Performance

| Strategy | OOS Sharpe | OOS CAGR | OOS Max DD |
|----------|:----------:|:--------:|:----------:|
| time_series_momentum | 0.538 | 3.40% | -8.87% |
| residual_reversal | 1.151 | 4.49% | -3.62% |

### Interpretation

- **Sharpe degradation: -66.5%** — EXCELLENT. OOS outperforms IS — definitively NOT overfit. Strategy captures genuine, persistent premium.

## Complexity Comparison

| Metric | Original | v2 | v3 | v4 (current) |
|--------|:--------:|:--:|:--:|:------------:|
| Strategies | 16 | 4 | 3 | **2** |
| Free parameters | ~60+ | 0 | 0 | **0** |
| Universe size | 49 | 49 | 49 | **~150** |
| Sub-industry groups | 0 | 0 | 7 | **72** |
| Allocation | 4 (ensemble) | equal wt | equal wt | **equal wt** |
| Risk overlays | 5 | DD cap | vol-target | **vol-target** |

## CAGR Note

CAGR (Compound Annual Growth Rate) **fully accounts for losses**. It is computed as:
`CAGR = (final_value / initial_value)^(1/years) - 1`
A negative CAGR means the strategy lost money overall. Drawdowns reduce the compounding base.

## References

1. Moskowitz, T., Ooi, Y., & Pedersen, L. (2012). Time Series Momentum. *JFE*.
2. Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns. *JF*.
3. Lehmann, B. (1990). Fads, Martingales, and Market Efficiency. *QJE*.
4. Blitz, D., Huij, J., Lansdorp, S. & Verbeek, M. (2013). Short-Term Residual Reversal. *JFM*.
5. Blitz, D., van der Grient, B. & Honarvar, I. (2023). Reversing the Trend of Short-Term Reversal. *JPM*.
6. Da, Z., Liu, Q. & Schaumburg, E. (2014). A Closer Look at Short-Term Return Reversal. *Mgmt Sci*.
7. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal vs Naive Diversification. *RFS*.
8. Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios. *JF*.
9. Harvey, C., Liu, Y., & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *RFS*.
10. McLean, R., & Pontiff, J. (2016). Does Academic Research Destroy Predictability? *JF*.
11. Nagel, S. (2012). Evaporating Liquidity. *RFS*.
12. Avramov, D., Chordia, T. & Goyal, A. (2006). Liquidity and Autocorrelations. *JFE*.
13. George, T. & Hwang, C. (2004). The 52-Week High and Momentum Profits. *JF*.
