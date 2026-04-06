# Research Card: Crisis Volatility Breakout

## Hypothesis
During high-volatility geopolitical crisis regimes, Donchian channel breakout systems produce positive convex returns across commodities, forex, and indices, with Sharpe ratios 50-100% higher than calm-period performance.

## Economic Mechanism
- Information diffusion lag across global markets during crises
- Forced selling/deleveraging creates persistent directional moves
- Liquidity withdrawal causes price overshoot beyond fundamentals
- Hedging demand creates sustained order flow in one direction

## Academic Sources
1. Hurst, Ooi & Pedersen (2017) — "A Century of Evidence on Trend-Following Investing", AQR
2. Brock, Lakonishok & LeBaron (1992) — "Simple Technical Trading Rules", JF
3. Kavajecz & Odders-White (2004) — "Technical Analysis and Liquidity Provision", RFS
4. Fung & Hsieh (2001) — "The Risk in Hedge Fund Strategies", RFS

## Target Instruments
- XAUUSD, USOIL.cash, UKOIL.cash, NATGAS.cash (primary)
- US500.cash, GER40.cash, EURUSD, USDJPY (secondary)
- WHEAT.c, COFFEE.c (supply-shock sensitive)

## Holding Period
- 5-30 trading days (trend capture)

## Signal
- 20-day Donchian channel breakout (fast) or 60-day (slow)
- Confirmation: ATR expansion > 1.5x 60-day average
- Vol-regime filter: only trade when trailing 20-day realized vol > 1.2x 60-day vol

## Position Sizing
- 0.5% risk per trade via ATR-based stop (2x ATR)
- Max 6 concurrent positions = 3.0% max risk

## Known Failure Modes
- Whipsaw in range-bound markets (low hit rate ~30-35%)
- V-shaped reversals (COVID March 2020 pattern)
- Spread widening during extreme vol eats into breakout profits

## FTMO Feasibility
- **MDL (3%)**: Safe — max 6 positions x 0.5% = 3.0% worst case, but all hitting stops simultaneously is rare
- **Best Day Rule**: Excellent — profits accumulate over multi-day trends
- **Weekend gap**: Moderate risk — reduce positions by 50% on Friday close
- **Spread/swap**: Commodity spreads widen during crises; budget 2-3x normal spreads

## Falsification Criteria
- OOS Sharpe < 0.3 across 3+ crisis periods
- Hit rate < 25% with payoff ratio < 2.0x
- FTMO challenge simulation pass rate < 30%
