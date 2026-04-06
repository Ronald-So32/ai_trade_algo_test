# Backtest Overfitting Diagnostic Report

**Overall Confidence: MODERATE**

## Strategy-Level Statistical Tests

| Strategy                  |  Sharpe |     DSR |     PSR |  MinBTL |  Signif |
|---------------------------|---------|---------|---------|---------|---------|
| time_series_momentum      |   0.450 |   0.000 |   1.000 |     inf |      NO |
| cross_sectional_momentum  |   0.368 |   0.000 |   1.000 |     inf |      NO |
| volatility_breakout       |   0.301 |   0.000 |   1.000 |     inf |      NO |
| vol_managed               |   1.008 |   0.000 |   1.000 |     inf |      NO |
| pead                      |   0.368 |   0.000 |   1.000 |     inf |      NO |

## Multiple Testing Correction (Harvey et al. 2016)

- Number of strategies tested: 5
- Required t-ratio (Bonferroni): 0.64
- Required t-ratio (BH-FDR 5%): 0.01
- Expected max Sharpe of null: 0.297
- Strategies surviving Bonferroni: 1
- Strategies surviving BH-FDR: 1

## In-Sample vs Out-of-Sample Degradation

- **time_series_momentum**: IS Sharpe=0.614, OOS Sharpe=0.288, Degradation=53.1% [OVERFIT WARNING]
- **cross_sectional_momentum**: IS Sharpe=0.594, OOS Sharpe=0.072, Degradation=88.0% [OVERFIT WARNING]
- **volatility_breakout**: IS Sharpe=0.313, OOS Sharpe=0.298, Degradation=4.7%
- **vol_managed**: IS Sharpe=1.020, OOS Sharpe=0.990, Degradation=3.0%
- **pead**: IS Sharpe=0.290, OOS Sharpe=0.465, Degradation=-60.4%

## White's Reality Check (Bootstrap)

- Best strategy: vol_managed
- Bootstrap p-value: 0.0000
- Significant after data snooping correction: YES

## Leverage Risk Haircut

- Applied leverage: 6.5x
- Raw Sharpe: 0.202
- Haircut Sharpe (OOS estimate): 0.000
- Expected OOS MaxDD: -12.06%
- Vol drag at leverage: 0.18%
- Leverage safety score: 4.7/10

## Warnings

- No strategies pass DSR significance test (0/5 significant)
- Leverage safety score only 4.7/10
