# v4 (Static) vs v5 (Dynamic) — Comparison Report

## Design Differences

| Aspect | v4 (Static) | v5 (Dynamic) |
|--------|-------------|--------------|
| Strategy allocation | Equal-weight 50/50 | Risk parity (inverse-vol) |
| Leverage | Fixed 2.0x | Vol-managed (target 15%, cap 2x) |
| Residual reversal | Equal-weight within quintile | Liquidity-weighted (Nagel 2012) |
| Free parameters | 0 | 0 (all from literature) |

## Academic Basis for v5

1. **Risk parity** — Maillard, Roncalli & Teiletche (2010): equalizing risk
   contributions outperforms equal-weight when strategies have different volatilities.
2. **Vol-managed leverage** — Moreira & Muir (2017): scaling exposure inversely to
   recent vol improves Sharpe by ~50% for momentum. Barroso & Santa-Clara (2015):
   vol-scaling momentum avoids crashes, improves Sharpe from 0.53 to 0.97.
3. **Liquidity-weighted reversal** — Nagel (2012): reversal premium is liquidity
   provision compensation, strongest for less-liquid stocks. Amihud (2002):
   illiquidity is a priced factor.

## In-Sample Performance

| Metric | v4 (Static) | v5 (Dynamic) | Improvement |
|--------|:-----------:|:------------:|:-----------:|
| Sharpe | 0.538 | 0.588 | +9.4% |
| CAGR | 5.93% | 3.83% | -35.4% |
| Max DD | -29.43% | -19.39% | -34.1% |
| Volatility | 12.09% | 6.79% | — |
| Sortino | 0.627 | 0.743 | — |
| Calmar | 0.20 | 0.20 | — |
| Daily WR | 54.4% | 52.1% | — |
| Monthly WR | 62.2% | 59.3% | — |

## Out-of-Sample Performance

| Metric | v4 (Static) | v5 (Dynamic) | Improvement |
|--------|:-----------:|:------------:|:-----------:|
| Sharpe | 1.110 | 1.117 | +0.6% |
| CAGR | 8.08% | 6.62% | -18.1% |
| Max DD | -6.07% | -3.85% | — |
| Volatility | 7.24% | 5.89% | — |
| Sortino | 1.321 | 1.319 | — |
| Calmar | 1.33 | 1.72 | — |
| Daily WR | 53.2% | 47.0% | — |
| Monthly WR | 64.5% | 64.5% | — |

## Sharpe Degradation (IS → OOS)

| Version | IS Sharpe | OOS Sharpe | Degradation |
|---------|:---------:|:----------:|:-----------:|
| v4 | 0.538 | 1.110 | -106.4% |
| v5 | 0.588 | 1.117 | -89.9% |

v5 shows **higher degradation** — the dynamic elements may be fitting to in-sample patterns.

## References

1. Maillard, S., Roncalli, T. & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contributions Portfolios. *JPM*.
2. Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *JF*.
3. Barroso, P. & Santa-Clara, P. (2015). Momentum Has Its Moments. *JFE*.
4. Nagel, S. (2012). Evaporating Liquidity. *RFS*.
5. Amihud, Y. (2002). Illiquidity and Stock Returns. *JFM*.
6. Avramov, D., Chordia, T. & Goyal, A. (2006). Liquidity and Autocorrelations in Individual Stock Returns. *JFE*.
7. DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *RFS*.
