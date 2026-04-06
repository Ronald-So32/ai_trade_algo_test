# Leverage Stress Test Report

**Overall Risk Score: 5.0/10** (lower is riskier)

## Base Metrics
- leverage: 6.5399
- cagr: 0.5301
- maxdd: -0.0403
- vol: 0.1129
- sharpe: 3.8268
- base_vol: 0.0653
- vol_drag: 0.0772
- calmar: 13.1539
- n_observations: 4075

## Volatility Shock Scenarios

| Scenario                  |   Vol Mult |       CAGR |      MaxDD |   Sharpe |
|---------------------------|------------|------------|------------|----------|
| Normal                    |       1.00 |    53.01% |    -4.03% |    3.827 |
| Mild stress (+50% vol)    |       1.50 |    44.01% |    -4.77% |    2.989 |
| Moderate stress (+100% vol) |       2.00 |    29.23% |    -6.52% |    2.353 |
| Severe stress (+200% vol) |       3.00 |    13.69% |    -9.43% |    1.197 |
| Extreme stress (+300% vol) |       4.00 |     8.04% |   -10.44% |    0.894 |

## Drawdown Stress Scenarios

- **Worst historical DD at 6.5x**: MaxDD=-45.68%, Recovery=N/A days
- **2x worst DD at 6.5x**: MaxDD=-62.51%, Recovery=N/A days
- **10% flash crash at 6.5x**: MaxDD=-65.40%, Recovery=N/A days
- **1%/day × 20 days at 6.5x**: MaxDD=-74.15%, Recovery=N/A (scenario) days

## Estimation Error Analysis

- Sharpe estimation SE: 0.025
- 95% CI for Sharpe: [1.275, 1.374]
- At leverage, error amplified by: 6.5x
- Expected OOS Sharpe range: [1.275, 1.374]

## Recommendations

- NOTE: At 6.5x leverage, vol drag = 7.72% annually. Ensure this is accounted for in return expectations.
- CRITICAL: A 10% flash crash would cause -65.40% drawdown at current leverage. Ensure stop-losses or circuit breakers are in place.
