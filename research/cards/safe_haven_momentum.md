# Research Card: Safe-Haven Momentum

## Hypothesis
Gold, JPY, and CHF exhibit persistent positive momentum during geopolitical crises, with gold being the most reliable safe haven. Momentum signals on safe havens during risk-off periods produce Sharpe 0.5-0.8.

## Economic Mechanism
- Flight-to-quality flows create sustained buying pressure in safe havens
- Central bank and sovereign wealth fund rebalancing amplifies moves
- Geopolitical uncertainty has no clear resolution timeline, so safe-haven demand persists
- Gold additionally benefits from inflation expectations during commodity supply shocks

## Academic Sources
1. Baur & Lucey (2010) — "Is Gold a Hedge or a Safe Haven?", Finance Research Letters
2. Baur & McDermott (2010) — "Is Gold a Safe Haven? International Evidence", JBF
3. Ranaldo & Soderlind (2010) — "Safe Haven Currencies", Review of Finance
4. O'Connor, Lucey, Batten & Baur (2015) — "The Financial Economics of Gold", IRFA

## Target Instruments
- Primary: XAUUSD
- Secondary: XAUEUR, XAGUSD (higher beta)
- FX: Short USDJPY (conditional on BOJ policy), short USDCHF

## Holding Period
- 5-40 trading days (crisis duration)

## Signal
- 10-day breakout on XAUUSD combined with risk-off confirmation
- Risk-off confirmation: equity index (US500) below 20-day low AND/OR VIX proxy > 25
- Momentum: 5-day and 20-day return both positive for safe haven

## Position Sizing
- 0.5-0.75% risk per trade
- Gold: 2x ATR stop (~$50-80 below entry)
- Max 2 safe-haven positions simultaneously

## Known Failure Modes
- USD liquidity crunch: In March 2020, gold sold off -12% as everything was liquidated for USD cash
- JPY safe-haven status is CONDITIONAL on BOJ policy (failed in 2022)
- Gold can pull back sharply mid-crisis before resuming uptrend
- If crisis is resolved quickly, safe-haven premium evaporates

## FTMO Feasibility
- **MDL (3%)**: Low risk — well-controlled position sizes
- **Best Day Rule**: MODERATE RISK — gold can surge 3-5% in a single crisis day, capturing 30-50% of total challenge profits. Mitigate by scaling out on extreme days.
- **Weekend gap**: Low for gold (trades near 24/5). Moderate for JPY.
- **Spread/swap**: Gold swaps are meaningful for holds > 1 week. Factor into sizing.

## Falsification Criteria
- Gold fails to rally in 2+ consecutive geopolitical crises
- Safe-haven momentum Sharpe < 0.2 in OOS crisis periods
- FTMO simulation shows Best Day Rule violations > 30% of passing attempts
