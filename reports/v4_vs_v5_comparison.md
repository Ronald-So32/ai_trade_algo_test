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
| Sharpe | 0.534 | 0.574 | +7.3% |
| CAGR | 5.89% | 3.73% | -36.6% |
| Max DD | -29.57% | -20.14% | -31.9% |
| Volatility | 12.09% | 6.79% | — |
| Sortino | 0.623 | 0.724 | — |
| Calmar | 0.20 | 0.19 | — |
| Daily WR | 54.5% | 52.4% | — |
| Monthly WR | 62.2% | 58.7% | — |

## Out-of-Sample Performance

| Metric | v4 (Static) | v5 (Dynamic) | Improvement |
|--------|:-----------:|:------------:|:-----------:|
| Sharpe | 1.141 | 1.194 | +4.6% |
| CAGR | 8.49% | 7.24% | -14.8% |
| Max DD | -6.30% | -4.18% | — |
| Volatility | 7.38% | 6.01% | — |
| Sortino | 1.366 | 1.412 | — |
| Calmar | 1.35 | 1.73 | — |
| Daily WR | 53.8% | 47.6% | — |
| Monthly WR | 61.1% | 63.5% | — |

## Sharpe Degradation (IS → OOS)

| Version | IS Sharpe | OOS Sharpe | Degradation |
|---------|:---------:|:----------:|:-----------:|
| v4 | 0.534 | 1.141 | -113.5% |
| v5 | 0.574 | 1.194 | -108.1% |

v5 shows **higher degradation** — the dynamic elements may be fitting to in-sample patterns.

## References

1. Maillard, S., Roncalli, T. & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contributions Portfolios. *JPM*.
2. Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *JF*.
3. Barroso, P. & Santa-Clara, P. (2015). Momentum Has Its Moments. *JFE*.
4. Nagel, S. (2012). Evaporating Liquidity. *RFS*.
5. Amihud, Y. (2002). Illiquidity and Stock Returns. *JFM*.
6. Avramov, D., Chordia, T. & Goyal, A. (2006). Liquidity and Autocorrelations in Individual Stock Returns. *JFE*.
7. DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *RFS*.
