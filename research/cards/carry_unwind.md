# Research Card: Carry Trade Unwind

## Hypothesis
During geopolitical crises, FX carry trades unwind predictably as risk-off triggers deleveraging. Shorting high-yield currencies against funding currencies (JPY, CHF) produces positive returns with Sharpe 0.3-0.5 during confirmed risk-off periods.

## Economic Mechanism
- Carry traders are structurally long high-yield (AUD, NZD, ZAR, MXN) and short funding (JPY, CHF)
- Crisis triggers margin calls and deleveraging
- Unwind is self-reinforcing: losses cause more selling, causing more losses
- Completion in 1-4 weeks followed by slow recovery

## Academic Sources
1. Brunnermeier, Nagel & Pedersen (2008) — "Carry Trades and Currency Crashes", NBER Macro Annual
2. Lustig, Roussanov & Verdelhan (2011) — "Common Risk Factors in Currency Markets", RFS
3. Menkhoff, Sarno, Schmeling & Schrimpf (2012) — "Carry Trades and Global FX Volatility", JF
4. Jurek (2014) — "Crash-Neutral Currency Carry Trades", JFE

## Target Instruments
- Short AUDJPY, short NZDJPY, short GBPJPY
- Short USDZAR, short USDMXN (if risk-off is severe)
- Long USDCHF or long CHFJPY as alternative expression

## Holding Period
- 5-20 trading days (carry crash duration)

## Signal
- Entry trigger (all must be true):
  - Equity index (US500) down > 2% from 5-day high
  - JPY strengthening (USDJPY below 5-day low)
  - Trailing 5-day realized vol > 1.5x 20-day average
- Exit: Vol subsides below 20-day average OR carry pair recovers above pre-crisis level

## Position Sizing
- 0.5-1.0% risk per pair
- Max 3 pairs simultaneously = 1.5-3.0% total risk
- Stop above pre-crisis high (typically 1.5-2.0x ATR)

## Known Failure Modes
- False alarm: geopolitical tension that does NOT escalate
- Central bank intervention to support high-yield currencies
- Low carry environment (near-zero rate differentials) means less carry to unwind
- 2022 showed JPY weakening during crisis due to BOJ policy divergence

## FTMO Feasibility
- **MDL (3%)**: Safe if stops honored. Carry unwinds move in your favor.
- **Best Day Rule**: Good — carry unwinds typically spread over 1-3 weeks, not concentrated
- **Weekend gap**: MODERATE RISK — geopolitical developments over weekends cause FX gaps. Reduce positions on Friday.
- **Spread/swap**: Exotic pairs (ZAR, MXN) have wide spreads. Prefer AUD/JPY and NZD/JPY for tighter spreads.

## Falsification Criteria
- Carry unwinds fail to materialize in 2+ consecutive risk-off events
- JPY strengthening signal accuracy < 50% during confirmed equity drawdowns > 5%
- Strategy produces negative returns net of spreads/swaps
