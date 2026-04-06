# Research Card: Post-Volatility-Spike Mean Reversion

## Hypothesis
After extreme volatility spikes (VIX > 30 equivalent), equity indices produce statistically significant positive returns over the subsequent 20-60 trading days as fear premiums dissipate and vol mean-reverts. This is a Phase 3 strategy deployed AFTER the initial crisis shock.

## Economic Mechanism
- Implied volatility systematically overshoots during crises (fear premium)
- Forced selling creates temporary mispricing below fundamental value
- Short-term reversal after panic days reflects liquidity provider compensation
- VIX has a documented half-life of ~17 trading days (mean-reverts reliably)

## Academic Sources
1. Whaley (2009) — "Understanding the VIX", JPM
2. Moreira & Muir (2017) — "Volatility-Managed Portfolios", JF
3. Bali & Peng (2006) — "Is There a Risk-Return Trade-Off?", JFE
4. Hameed & Mian (2015) — "Industries and Stock Return Reversals", JFQA

## Target Instruments
- US500.cash, US100.cash, GER40.cash (most liquid equity indices)
- US2000.cash (small-cap bounces harder but more volatile)

## Holding Period
- Short-term reversal: 1-5 trading days (after single-day crash > 3%)
- Vol mean-reversion: 10-40 trading days (after sustained VIX > 30)

## Signal
- Sub-strategy A (short-term reversal):
  - Daily decline > 2.5% on major index
  - Wait for first green day (bullish reversal candle)
  - Enter long at next open
  - Hold 3-5 days
- Sub-strategy B (vol mean-reversion):
  - Trailing 20-day realized vol > 2x 60-day realized vol
  - Vol has turned down (3-day downtrend in realized vol)
  - Enter long equity indices
  - Scale up as vol continues to decline
  - Exit when vol returns to long-term average

## Position Sizing
- 0.5-1.0% risk per trade
- Stop: 1.5x below the crisis low (tight for short-term reversal)
- Scale in: 50% initial, add 50% on confirmation

## Known Failure Modes
- **Catching a falling knife**: Buying after initial selloff when the real crash is just beginning (Feb 2020 buy -> Mar 2020 crash)
- **Sustained crisis**: If geopolitical situation worsens, vol stays elevated and equities continue lower
- **V-recovery too fast**: Profits hit Best Day Rule concentration
- **Dead cat bounce**: Initial bounce fails and market resumes decline

## FTMO Feasibility
- **MDL (3%)**: MODERATE TO HIGH RISK — buying dips can produce immediate losses if decline continues
- **Best Day Rule**: RISK — violent bounces (+4-5% in one day) can concentrate profits
- **Weekend gap**: Moderate — less relevant as this strategy is entered after confirmation
- **Timing**: CRITICAL — this strategy is ONLY deployed AFTER peak fear, not at crisis onset

## Falsification Criteria
- Post-spike (VIX > 30) 60-day equity returns are negative in > 40% of instances
- Short-term reversal win rate < 55% after daily declines > 3%
- FTMO simulation shows MDL breaches from buying dips > 15% of attempts
- Strategy generates negative alpha vs simple buy-and-hold during vol normalization periods
