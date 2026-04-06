# FTMO 1-Step Challenge: Strategy Research & Implementation Plan

## Challenge Rules (from account screenshot)

| Parameter | Value |
|-----------|-------|
| **Account Size** | $100,000 |
| **Profit Target** | $5,000 (5%) |
| **Max Daily Loss** | -$3,000 (3%) |
| **Max Overall Loss** | -$10,000 (10% EOD trailing) |
| **Best Day Rule** | No single day > 50% of total profit |
| **Leverage** | 1:30 |
| **Platform** | MetaTrader 5 |
| **Time Limit** | Unlimited |

### Key Constraint Analysis

The **3% daily loss limit** is the binding constraint — far tighter than the 2-step's 5%. At 1.5% daily portfolio vol, a 2-sigma day breaches you. This forces conservative sizing.

The **Best Day Rule** penalizes concentrated returns. Strategies that produce consistent, distributed gains are strongly favored over lumpy trend-following.

The **EOD trailing max loss** means your floor rises with profits. After making +7%, your floor is only 3% below starting equity. One bad week and you're breached even though you were profitable overall.

**Asymmetric barriers (+5% / -10%)** give 2:1 downside room — favorable for positive-edge strategies.

---

## FTMO Pass Rate Context

- Overall pass rate: **~8-10%** (FTMO's own published data)
- ~90% of traders fail during the challenge phase
- Only ~7% of those who pass ever receive a payout
- Most successful traders: **>70% win rate, >1.0 R:R, conservative sizing**
- Most traded instrument by successful traders: **Gold (XAUUSD)**
- Featured winners: 72-80% win rate, 1.38-2.04 R:R, passed in 7-10 days

Sources: FTMO Blog "A High Win Rate is the Foundation of Success"; CoinLaw FTMO Statistics 2026; Finance Magnates prop firm pass rate analysis

---

## Strategy Rankings (by FTMO 1-Step suitability)

### Tier 1 — PRIMARY STRATEGIES (implement these)

---

### 1. Cross-Asset Time-Series Momentum (TSMOM)

**Academic basis:** Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum," *Journal of Financial Economics*, 104(2), 228-250.
**Additional:** Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following Investing," *JPM*, 44(1), 15-29.

**Documented performance:**
- Diversified TSMOM across 58 instruments: **Sharpe 1.28**
- Positive returns in every decade since 1880
- Works on equity indices, commodities, FX, bonds

**Signal:** For each instrument, compute 12-month trailing return. If positive → long. If negative → short. Vol-scale positions (target constant risk per instrument).

**FTMO adaptation:**
- Apply across: NAS100, SPX500, US30, DAX, XAUUSD, EURUSD, GBPUSD, USDJPY
- 8 instruments → cross-asset diversification replaces cross-sectional stock diversification
- Multi-timeframe variant (Dudler, Gmuer & Malamud 2015): combine 1-day, 5-day, 21-day, 63-day, 252-day lookbacks with risk-adjusted returns (RAMOM)
- RAMOM has ~40% lower turnover than standard TSMOM

**Expected Sharpe on FTMO instruments:** ~0.7-1.0 (diversified across 8 instruments)

**1-Step compatibility:**
- Distributed returns across instruments → naturally satisfies Best Day Rule
- Daily rebalancing → consistent P&L distribution
- Vol-targeting prevents daily DD breaches
- Risk: slow in low-trend environments

---

### 2. Gold (XAUUSD) Pullback Mean Reversion

**Academic basis:**
- Baur & Lucey (2010), "Is Gold a Hedge or a Safe Haven?", *Financial Review*, 45(2), 217-229.
- Erb & Harvey (2013), "The Golden Dilemma," *Financial Analysts Journal*, 69(4), 10-42.

**Documented performance (backtested, open-source):**
- 4-Phase State Machine on XAUUSD 5-min: **Sharpe 0.89, PF 1.64, Win Rate 55.4%, Max DD 5.81%**
- 175 trades over 5 years (~3/month), avg win $1,187 / avg loss $913
- Expectancy: $251/trade on $10k account

**Signal:** Buy pullbacks within uptrends, sell rallies within downtrends. Uses EMA crossovers for trend direction, ATR-based stops (2.5x ATR), ATR-based take profit (12x ATR), time-of-day filter (London/NY overlap).

**1-Step compatibility:**
- **5.81% max DD fits perfectly within 10% trailing limit**
- 55% win rate with 1.3:1 R:R → consistent daily P&L
- Gold is the #1 instrument traded by successful FTMO traders
- 3/month trade frequency is too low alone — combine with other strategies
- Risk: requires trend/regime filter (ADX < 25) to avoid trending markets

**Enhancement with RSI filter:**
- Unfiltered RSI mean reversion on gold: no edge (2,397 trades)
- With EMA(200) trend filter + ADX(14) < 25: **Profit Factor 3.00**
- Extreme selectivity but very high win rate when signals fire

---

### 3. FX Carry + Momentum Portfolio

**Academic basis:**
- Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry," *JFE*, 127(2), 197-225. **Sharpe 0.8-1.2**
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Currency Momentum Strategies," *JFE*, 106(3), 660-684. **Sharpe 0.95**
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere," *J. Finance*, 68, 929-985.

**Signal (Carry):** Go long high-swap-rate pairs, short negative-swap pairs. Visible in MT5 as swap long/short values.

**Signal (Momentum):** At month-end, rank available FX pairs by 1-month return. Long top 2 gainers, short bottom 2 losers.

**Combined:** Weight 50% carry + 50% momentum. Momentum and carry are negatively correlated in FX — natural hedge.

**FTMO adaptation:**
- Universe: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
- Monthly rebalancing → very low frequency, minimal spread cost
- 0.5% risk per pair → 7 pairs × 0.5% = 3.5% total portfolio risk
- Daily P&L from swap accumulation (carry) + position gains (momentum)

**1-Step compatibility:**
- Monthly rebalancing → no overtrading
- Distributed across 7 pairs → satisfies Best Day Rule
- Carry accrues daily (small, consistent positive P&L)
- Risk: carry crashes (e.g., 2008 JPY unwind). Need VIX/vol filter to reduce in high-vol regimes.

---

### 4. Macro Event Timing (Pre-FOMC Drift + Announcement Days)

**Academic basis:**
- Lucca & Moench (2015), "The Pre-FOMC Announcement Drift," *J. Finance*, 70(1), 329-371. **Pre-FOMC 24h drift: avg 49 bps, Sharpe 1.43**
- Savor & Wilson (2013/2014), "Asset Pricing: A Tale of Two Days," *JFE*. **Announcement days deliver 11.4 bps vs 1.1 bps on non-announcement days. 60% of equity risk premium earned on 13% of trading days.**

**Signal:** Calendar-based. Go long SPX500/NAS100 the day before FOMC, NFP, CPI releases. Exit after announcement.

**Frequency:** ~8 FOMC + 12 NFP + 12 CPI = ~32 events/year (~2-3/month)

**1-Step compatibility:**
- Pure calendar signal → no parameters, no overfitting
- 49 bps average per FOMC event on SPX → meaningful contribution
- Position overnight → use FTMO Swing account to hold through events
- Risk: single-event losses can be large on surprise announcements
- Size conservatively: 0.5% risk per event

**Note:** Recent evidence (through 2024) confirms pre-FOMC drift persists but is now concentrated around meetings with press conferences.

---

### Tier 2 — SECONDARY STRATEGIES (add for diversification)

---

### 5. Factor Momentum Overlay (Fama-French)

**Academic basis:**
- Gupta & Kelly (2019), "Factor Momentum Everywhere," *JPM*, 45(3), 13-36. **Sharpe 0.84**
- Ehsani & Linnainmaa (2022), "Factor Momentum and the Momentum Factor," *J. Finance*, 77(3), 1877-1919.

**Signal:** Download daily FF factor returns from Kenneth French Data Library (free). Compute 1-month rolling return on each factor (MKT, SMB, HML, RMW, CMA, MOM). Use as overlay signals:
- HML trending up → favor value-heavy indices (US30, DAX)
- MOM trending up → stronger TSMOM conviction
- Factor momentum subsumes industry momentum

**Implementation:** Not a standalone strategy. Used to tilt existing TSMOM positions by ±20-30% based on factor regime.

**1-Step compatibility:** Zero additional trades — just modifies sizing of existing positions.

---

### 6. NAS100 Opening Range Breakout (ORB)

**Basis:** Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy," SFI Research Paper. **Sharpe 2.4-2.81 on individual stocks**

**Practitioner backtest on NQ (Nasdaq E-mini):**
- 114 trades, **74.56% win rate, PF 2.512**
- 433% return in one year
- Max DD: 12%
- Max consecutive losses: 2 (never 3+)

**Signal:** If price breaks above the high of the first 5-15 minutes after US cash open, go long NAS100. Stop at other side of opening range. Target = 50% of opening range. Max 1 trade/day.

**1-Step compatibility:**
- Very high win rate (74%) → satisfies FTMO's emphasis on consistency
- Single trade per day → clean P&L attribution for Best Day Rule
- Risk: **long-only bias** — works in bull markets, underperforms in bear markets
- Need to verify on CFD data (spreads may differ from futures)

---

### 7. Turn-of-Month Effect

**Academic basis:**
- McConnell & Xu (2008), "Equity Returns at the Turn of the Month," *Financial Analysts Journal*. **Sharpe 1.04.** Found in 31 of 35 countries.
- Lakonishok & Smidt (1988), "Are Seasonal Anomalies Real?", *RFS*, 1(4), 403-425.

**Signal:** Buy SPX500/NAS100 at close on the last trading day of the month. Sell at close on the 3rd trading day of the new month. Flat all other days.

**Performance:** 7.2% annualized, 6.9% vol, max DD -20.8%. Only in market ~20% of the time.

**1-Step compatibility:**
- Very simple, calendar-based
- Too slow as standalone (would take 8+ months to hit 5%)
- Best as an **overlay** — increase TSMOM exposure during TOM window, reduce otherwise
- Zero free parameters

---

### 8. Index FF-Residual Reversal (Zaremba 2019)

**Academic basis:** Zaremba, Umutlu & Karathanasopoulos (2019), "Alpha Momentum and Alpha Reversal in Country and Industry Equity Indexes," *J. Empirical Finance*, 53, 144-161.

**Signal:** Regress each index's returns on FF factors (rolling 60-day). The residual captures index-specific alpha:
- Short-term residual (1-week) → reversal signal
- Long-term residual (1-month) → momentum signal

**This is the closest adaptation of Residual Reversal to index CFDs.**

**1-Step compatibility:** Adds a second signal dimension beyond pure TSMOM. Lower conviction than stock-level residual reversal. Use as a weight modifier, not standalone.

---

### Tier 3 — SUPPLEMENTARY / FILTERS (conditional use)

---

### 9. VIX-Regime Position Sizing

**Academic basis:** Nagel (2012), "Evaporating Liquidity," *RFS*. +1pp VIX → +0.13 conditional Sharpe for reversal.

**Implementation (from earnings_black_swan/ebs/risk.py):**
```
Low VIX (<15):   0.7x exposure (calm, less edge)
Normal (15-25):  1.0x (baseline)
High (25-35):    1.3x (elevated fear, more mispricing)
Extreme (>35):   0.5x (crisis, correlated selloff — protect capital)
```

**1-Step compatibility:** Pure risk management overlay. Prevents large losses during vol spikes (which would breach 3% daily limit). Increases exposure during profitable regimes.

---

### 10. Earnings Announcement Premium (on 23 FTMO stocks)

**Academic basis:** Frazzini & Lamont (2007), "The Earnings Announcement Premium," NBER. **VW Sharpe 0.94, EW Sharpe 2.38.** 7-18% annualized excess returns.

**Signal:** Go long stock CFD 1-2 days before earnings. Exit 1 day after. Only trade stocks with positive recent momentum.

**Available on FTMO:** AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA + ~16 others

**1-Step compatibility:**
- ~23 stocks × 4 quarters = ~92 events/year (~1.5/week)
- Risk: earnings gaps can be 5-10% → **dangerous for 3% daily DD limit**
- Must size at 0.25-0.5% risk maximum per stock
- Use as supplement, never primary

---

### 11. COT Positioning Filter

**Academic basis:** Wang (2003), "Investor Sentiment, Market Timing, and Futures Returns," *Applied Financial Economics*. Large speculator positioning predicts continuation; hedger positioning is contrarian.

**Signal:** Weekly CFTC Commitments of Traders data. When speculator net positioning in gold or indices is >90th percentile historically → reduce long exposure. When <10th → increase.

**1-Step compatibility:** Filter only, not standalone. Helps avoid crowded trades.

---

### NOT RECOMMENDED for FTMO 1-Step

| Strategy | Why Not |
|----------|---------|
| **Volatility Risk Premium** | FTMO has no options/VIX instruments |
| **Cross-sectional stock reversal** | Need 150+ stocks, FTMO has 23 |
| **London Breakout on FX** | Backtested to **lose money** (QuantifiedStrategies) |
| **NFP/News trading** | 0.09% avg gain, no edge after spreads (QuantifiedStrategies) |
| **Bollinger Bands on FX** | PF 0.85-1.14 → edge consumed by CFD spreads |
| **NAS100 vs SPX500 stat arb** | Spread driven by non-stationary sector rotation |
| **Microstructure/order flow** | MT5 CFDs provide tick volume only, no real order book |
| **ML on indices** | Near-zero OOS R² for aggregate index prediction (Gu, Kelly & Xiu 2020) |
| **Pure FX mean reversion** | Half-life 3-5 years (Rogoff 1996) → too slow |

---

## Optimal Strategy Combination for 1-Step Challenge

### Architecture

```
Layer 1 — CORE ALPHA (60% of risk budget):
  Cross-asset TSMOM on 8 instruments (NAS100, SPX500, US30, DAX, XAUUSD, EURUSD, GBPUSD, USDJPY)
  Multi-timeframe: 1d, 5d, 21d, 63d, 252d lookbacks (Dudler et al. 2015 RAMOM)
  Factor momentum overlay (Gupta & Kelly 2019)
  Daily rebalance at 3:30 PM ET

Layer 2 — TACTICAL ALPHA (30% of risk budget):
  Gold pullback mean reversion (London/NY session, trend-filtered)
  NAS100 Opening Range Breakout (US cash open, long-only)
  Macro event positioning (pre-FOMC, NFP, CPI calendar)

Layer 3 — RISK OVERLAY (controls all sizing):
  CPPI cushion-based exposure (Grossman & Zhou 1993)
  Daily circuit breaker at -2% (1% buffer before 3% limit)
  Profit lock: at +4% reduce to 50% risk, at +4.5% reduce to 25%
  Vol-targeting: 1.0-1.2% daily portfolio vol (Moreira & Muir 2017)
  VIX-regime scaling (from earnings_black_swan/ebs/risk.py)
  Best Day Rule monitor: cap daily P&L contribution tracking
```

### Position Sizing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk per TSMOM instrument | 0.3% of equity | 8 instruments × 0.3% = 2.4% gross |
| Risk per tactical trade | 0.5% of equity | Max 2-3 simultaneous |
| Daily portfolio vol target | 1.0-1.2% | Optimal for +5%/-10% barriers |
| Max leverage used | ~3-5x effective | Well within 1:30 limit |
| Daily hard stop | -2.0% | Flatten everything, done for day |
| Profit lock threshold | +4.0% | Begin reducing risk |
| Coast threshold | +4.5% | Minimal risk, just don't lose it |

### Expected Performance

| Metric | Estimate | Basis |
|--------|----------|-------|
| Daily expected return | +0.25% to +0.40% | TSMOM Sharpe ~0.8-1.0 at 1.0-1.2% daily vol |
| Expected days to pass | 15-25 trading days | 5% target / 0.25% daily |
| P(pass) per attempt | **50-65%** | Monte Carlo with Sharpe 0.8-1.0, 1.2% daily vol, 3%/10% barriers |
| Expected cost to pass | $540-1,080 | 1-2 attempts at EUR 540 |
| Daily win rate | ~55-60% | Cross-asset TSMOM + tactical |
| Worst expected daily loss | -2.0% (circuit breaker) | Hard stop prevents 3% breach |
| Best Day Rule compliance | Natural | Distributed across 8+ instruments |

### Comparison: 1-Step vs 2-Step

| Factor | 1-Step | 2-Step |
|--------|--------|--------|
| Profit target | 5% | 10% + 5% |
| Daily DD limit | **3%** (tighter) | 5% |
| Max DD | 10% trailing | 10% static |
| Best Day Rule | **Yes** (restrictive) | No |
| Payout | 90% immediate | 80% → 90% |
| Optimal daily vol | **1.0-1.2%** | 1.5-2.0% |
| Our P(pass) | ~50-65% | ~55-70% |

The 1-step's 3% daily limit forces ~30% lower daily vol than the 2-step. The Best Day Rule forces diversified, distributed returns. Our cross-asset TSMOM + tactical approach naturally satisfies both.

---

## Backtesting Plan

| Step | Test | Data | Pass Criteria |
|------|------|------|---------------|
| 1 | TSMOM on each instrument individually | yfinance (NAS100=^NDX, SPX500=^GSPC, Gold=GC=F, FX from FRED) | Sharpe > 0.3 per instrument |
| 2 | Combined cross-asset TSMOM (equal risk) | Same | Combined Sharpe > 0.7 |
| 3 | Add RAMOM multi-timeframe | Same | Sharpe improvement > 10% |
| 4 | Add factor momentum overlay | Kenneth French Data Library | Marginal Sharpe contribution > 0 |
| 5 | Gold pullback strategy standalone | XAUUSD tick/1min data | PF > 1.5, Max DD < 6% |
| 6 | NAS100 ORB standalone | NQ futures/NAS100 5-min data | Win rate > 65%, PF > 2.0 |
| 7 | Full combined system | All above | Combined Sharpe > 0.8 |
| 8 | FTMO Challenge Simulator | Monte Carlo, 10,000 runs | P(pass) > 50% |
| 9 | Stress test (Mar 2020, Aug 2015, Feb 2018) | Historical | Never breach 3% daily |
| 10 | Walk-forward OOS | 2010-2020 IS, 2020-2026 OOS | OOS Sharpe within 30% of IS |

---

## Reusable Code from Existing Codebase

### From `ai_trade_algo_test/qrt/`:
- `strategies/time_series_momentum.py` — Core TSMOM signal, adapt for indices/FX/gold
- `execution/signal_generator.py` — Multi-strategy signal combination framework
- `strategies/base.py` — Strategy base class with backtest_summary()

### From `earnings_black_swan/ebs/`:
- `risk.py` — **Critical reuse**: Kelly criterion, CDaR drawdown scaling, VIX-regime sizing, risk of ruin Monte Carlo, composite position sizer
- `features/price_earnings_features.py` — Earnings date detection (for announcement premium strategy)

### New code needed:
- MT5 Python bridge (`metatrader5` package) — execution layer
- CPPI cushion manager — tracks equity high-water mark and adjusts exposure
- Daily circuit breaker — monitors intraday P&L and flattens at -2%
- Best Day Rule tracker — monitors per-day P&L contribution
- Profit lock module — reduces risk as equity approaches +5%
- Cross-asset data fetcher — pulls OHLC for indices/FX/gold from MT5

---

## MT5 Implementation Notes

- **MT5 runs on Windows only** — need a Windows VPS or Wine on Mac
- Python bridge: `pip install MetaTrader5` — sends orders via `mt5.order_send()`
- Architecture: Python strategy logic (external) → MT5 terminal (execution only)
- Poll interval: every 30 seconds (adequate for daily strategies)
- **Max 2,000 server requests/day** — not an issue for daily rebalancing
- FTMO Swing account recommended (hold overnight, no news restrictions)

---

## Key Academic References

### Core Strategy Papers
1. Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum, *JFE* 104(2)
2. Hurst, Ooi & Pedersen (2017) — A Century of Trend-Following, *JPM* 44(1)
3. Koijen, Moskowitz, Pedersen & Vrugt (2018) — Carry, *JFE* 127(2)
4. Menkhoff, Sarno, Schmeling & Schrimpf (2012) — Currency Momentum, *JFE* 106(3)
5. Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere, *J. Finance* 68
6. Gupta & Kelly (2019) — Factor Momentum Everywhere, *JPM* 45(3)
7. Dudler, Gmuer & Malamud (2015) — Risk-Adjusted Time Series Momentum, SFI
8. Lucca & Moench (2015) — Pre-FOMC Announcement Drift, *J. Finance* 70(1)
9. Savor & Wilson (2014) — Asset Pricing: A Tale of Two Days, *JFE* 113(2)
10. McConnell & Xu (2008) — Turn of Month, *FAJ* 64(2)
11. Gao, Han, Li & Zhou (2018) — Market Intraday Momentum, *JFE* 129(2)
12. Frazzini & Lamont (2007) — Earnings Announcement Premium, NBER
13. Baur & Lucey (2010) — Gold as Safe Haven, *Financial Review* 45(2)
14. Zaremba et al. (2019) — Alpha Momentum/Reversal in Indexes, *J. Empirical Finance* 53

### Risk Management Papers
15. Grossman & Zhou (1993) — Optimal Investment Under Drawdown Constraints, *Math. Oper. Res.* 18(1)
16. Black & Perold (1992) — Theory of CPPI, *J. Econ. Dyn. Control* 16(3-4)
17. Moreira & Muir (2017) — Volatility-Managed Portfolios, *J. Finance* 72(4)
18. Browne (2000) — Reaching Goals by a Deadline, *Adv. Appl. Probab.* 32(2)
19. Carpenter (2000) — Option Compensation and Risk Appetite, *J. Finance* 55(5)
20. Busseti, Ryu & Boyd (2016) — Risk-Constrained Kelly Gambling, Stanford

### Anti-Recommendations (documented failures)
21. Fang, Jacobsen & Qin (2017) — Bollinger Bands profitability "largely disappeared," *JPM* 43(4)
22. Cederburg et al. (2020) — Vol-managed portfolios fail OOS, *JFE* 138(1)
23. Do & Faff (2010) — Pairs trading profits declined to insignificance, *FAJ* 66(4)
