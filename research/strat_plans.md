# FTMO 1-Step Challenge: Strategy & Implementation Plan

Created: 2026-04-04
Status: PLANNING (no code changes yet)
Challenge: $100K account, 10% profit target ($10,000), unlimited time, 3% MDL, 10% ML

---

## Executive Summary

The current TSMOM-only system at 25% vol target produces ~20% CAGR — reaching $10,000 in ~6 months average. To hit 2 months, we need ~50% annualized return without proportionally increasing drawdown.

**Core insight:** The April 2026 market (Iran war, Hormuz closure) creates the BEST environment for trend-following in years — gold +70%, oil +50% YTD. But equities are whipsaw traps. The system must tilt toward assets with clean trends and away from noisy ones.

**Approach:** Enhanced TSMOM with crisis alpha overlays, expanded instrument universe, and regime-aware position sizing.

---

## Strategy Architecture

### Layer 1: Enhanced TSMOM (Core — 70% of risk budget)

**What changes from current system:**

| Parameter | Current | Proposed | Rationale | Protocol Compliance |
|-----------|---------|----------|-----------|-------------------|
| Profit target | $5,000 (wrong) | **$10,000** | Correct FTMO 1-Step rules | N/A — bug fix |
| Challenge window | 60 days | **180 days** | FTMO has unlimited time | N/A — rule correction |
| Instrument universe | 53 (yfinance) | **100-120** | Full FTMO list where data available | More diversification = higher vol target safely |
| Vol target | 25% | **Test 30%, 35%, 40%** | Higher return needed for 10% target | Walk-forward required per Section 2 |
| Lookbacks | (63, 126, 252) | **Test (21, 63, 126)** | Faster signal for faster profit | Academic: Moskowitz 1-12mo all profitable |
| Blend weights | (0.4, 0.4, 0.2) | **Test (0.5, 0.3, 0.2)** | Emphasize shorter-term | Walk-forward required |
| Trend strength cap | 0.20 | **Test 0.25, 0.30** | Give strong trends full weight | Walk-forward required |
| Min signal threshold | 0.20 | **Test 0.10, 0.15** | Trade more instruments | Walk-forward required |
| VoV reduction | 0.50 binary | **Test 0.30 smooth** | Less aggressive haircut | Moreira & Muir (2017) supports smoother |
| Gap risk reduction | 0.30 uniform | **0.15 large-cap, 0.40 crypto** | Asset-class specific | Data-driven from MT5 |
| Max instrument weight | 0.10 | **Test 0.15** | Allow concentration in gold/oil | Walk-forward required |

**Economic rationale (Section 5.3):** Behavioral underreaction to trends (Moskowitz et al. 2012)

### Layer 2: Pyramided Breakout (Synthetic Gamma — 15% of risk budget)

**New addition to create convex payoff profile.**

**Mechanism:**
- When TSMOM signal is strong AND price breaks above N-day high → add to position
- Start at 0.5% equity risk, add 0.5% per breakout level, max 4 pyramid units = 2% total
- Creates lookback-straddle-like payoff (Fung & Hsieh 2001)
- Trail stop-loss on each pyramid unit independently

**Economic rationale:** Behavioral underreaction to trends + volatility-risk compensation (trend continuation after breakout is documented across all asset classes)

**Academic sources:**
1. Fung & Hsieh (2001) — trend-following replicates lookback straddle
2. Brock, Lakonishok & LeBaron (1992) — breakout rules produce significant returns

**FTMO feasibility:**
- Best Day Rule: pyramiding spreads profit across multiple days (entry at different levels) — helps compliance
- MDL: 4 units x 0.5% = 2% max risk per instrument, within 3% daily limit
- Convex payoff: large winners on strong trends, small losses on failed breakouts

**Implementation: Modify position sizing in TSMOM — add scale-in logic, not a separate strategy.**

### Layer 3: Alternative Data Overlays (15% impact on sizing)

These modify position sizing, not signal direction. They act as filters/adjustments.

#### 3a. COT Positioning Overlay

**Mechanism:**
- Weekly: download CFTC COT data
- Compute percentile rank of net speculative positioning (1-3 year lookback)
- When speculative longs >90th percentile AND TSMOM long → reduce size by 30% (crowded)
- When speculative shorts >90th percentile AND TSMOM long → increase size by 20% (contrarian)
- Symmetric for shorts

**Economic rationale:** Inventory/liquidity effects (De Roon et al. 2000)
**Academic sources:** De Roon, Nijman & Veld (2000); Bessembinder (1992)
**Data:** Free from CFTC, weekly. Python: `cot_reports`
**FTMO impact:** Reduces risk of momentum crash from crowded positioning

#### 3b. GPR Regime Filter

**Mechanism:**
- Daily: check Caldara-Iacoviello GPR index
- When GPR z-score > 2.0: shift allocation toward gold/commodities, away from equities
- When GPR z-score > 3.0: reduce overall gross exposure by 20%

**Economic rationale:** Event-information processing delay + risk compensation
**Academic source:** Caldara & Iacoviello (2022), AER
**Data:** Free CSV from matteoiacoviello.com/gpr.htm
**FTMO impact:** Defensive — reduces drawdowns during geopolitical spikes

#### 3c. Central Bank Calendar Filter

**Mechanism:**
- Flatten or reduce USD-pair FX positions in the 4 hours around FOMC/ECB announcements
- Reduce equity index shorts before FOMC (pre-FOMC drift tends to be positive)

**Economic rationale:** Event-information processing delay
**Academic source:** Lucca & Moench (2015)
**Data:** Public calendars
**FTMO impact:** Avoids binary-outcome days that could cause MDL breaches

---

## Implementation Plan

### Phase 0: Fix Foundation (Day 1)
**No strategy changes — just correct bugs and infrastructure**

- [ ] Revert profit target from $5,000 to $10,000
- [ ] Change challenge simulation window from 60 to 180 days
- [ ] Re-run baseline backtest with correct parameters
- [ ] Record baseline metrics as control group

### Phase 1: Expand Universe (Days 2-3)
**More instruments = more diversification = can run higher vol safely**

- [ ] Map all 166 FTMO instruments to yfinance proxies where available
- [ ] Target: 100+ instruments (currently 53)
- [ ] Add all forex crosses (21 more pairs)
- [ ] Add all indices (8 more)
- [ ] Add metals (XPTUSD, XPDUSD, XCUUSD, gold-cross pairs)
- [ ] Add commodities (NATGAS, COCOA, COFFEE, CORN, etc.)
- [ ] Add crypto (BTC, ETH, SOL, XRP, etc. — at least 10)
- [ ] Re-run backtest with expanded universe, same parameters
- [ ] Record metrics — expect Sharpe improvement from diversification

### Phase 2: Vol Target & Lookback Optimization (Days 4-7)
**Walk-forward validated parameter sweep**

Testing matrix (per RESEARCH_PROTOCOL.md Section 2):
- Vol targets: {25%, 30%, 35%, 40%}
- Lookbacks: {(21,63,126), (63,126,252), (42,126,252)}
- Blend weights: {(0.5,0.3,0.2), (0.4,0.4,0.2), (0.33,0.33,0.34)}

Total variants: 4 x 3 x 3 = 36

**Walk-forward protocol:**
- Training: 1 year (252 days)
- OOS test: 3 months (63 days)
- Expanding window
- Report Deflated Sharpe if >10 variants (we'll have 36 → mandatory)

Metrics per variant:
- FTMO pass rate (with 10% target, 180-day window)
- Average days to pass
- MDL breach count
- Sharpe, CAGR, Max DD
- Best Day Ratio

### Phase 3: Signal Tuning (Days 8-10)
**On the best vol/lookback from Phase 2:**

- [ ] Test min_signal: {0.10, 0.15, 0.20}
- [ ] Test trend_strength_cap: {0.20, 0.25, 0.30}
- [ ] Test VoV reduction: {0.30 smooth, 0.40 smooth, 0.50 binary (current)}
- [ ] Test gap_risk: {0.15 equity / 0.40 crypto, 0.30 uniform (current)}
- [ ] Test max_instrument_weight: {0.10, 0.15}
- [ ] Record all variants for Deflated Sharpe

### Phase 4: Pyramided Breakout Addition (Days 11-13)
**Add scale-in logic to TSMOM**

- [ ] Implement pyramid entry: 0.5% base, add on N-day breakout, max 4 units
- [ ] Test N = {10, 20, 40} day breakout levels
- [ ] Test with and without pyramiding on best Phase 3 config
- [ ] Walk-forward validate
- [ ] FTMO sim with Best Day Rule check (pyramiding should help)

### Phase 5: Alternative Data Overlays (Days 14-18)
**Add COT, GPR, and calendar filters**

- [ ] Download and parse CFTC COT data for available instruments
- [ ] Map COT contracts to FTMO CFD symbols
- [ ] Implement COT positioning percentile overlay
- [ ] Download GPR daily index
- [ ] Implement GPR regime filter
- [ ] Build FOMC/ECB calendar event list
- [ ] Implement calendar position reduction
- [ ] Backtest each overlay independently (A/B test vs no overlay)
- [ ] Backtest all overlays combined
- [ ] Walk-forward validate combined system

### Phase 6: Stress Testing (Days 19-20)
**Per RESEARCH_PROTOCOL.md Section 7**

All stress tests on the final combined system:
- [ ] 2x spreads
- [ ] 2x slippage
- [ ] Weekend gap shocks (±3% Monday open)
- [ ] Volatility regime shift (vol doubles for 1 month)
- [ ] Correlation spike (all assets correlate to 0.8 for 1 week)
- [ ] Higher swap costs (2x)
- [ ] Partial fills (50% fill rate)

**Minimum requirement:** Must remain net profitable under scenarios 1-3.

### Phase 7: Final Report & Go/No-Go (Day 21)

- [ ] Generate comprehensive backtest report
- [ ] Compare all variants with Deflated Sharpe Ratio
- [ ] Compute Probability of Backtest Overfitting
- [ ] IS vs OOS degradation analysis
- [ ] FTMO challenge pass rate at 10% target
- [ ] Estimated days to pass
- [ ] Go/No-Go decision based on:
  - Pass rate > 40% on OOS
  - Avg days to pass < 90
  - MDL breach rate < 15%
  - Stress test profitable on scenarios 1-3
  - Sharpe CI lower bound > -0.5

---

## Risk Budget Allocation

| Component | Risk Budget | Max Daily Loss Contribution |
|-----------|------------|---------------------------|
| TSMOM core positions | 70% | 2.1% |
| Pyramid additions | 15% | 0.45% |
| Buffer for slippage/gaps | 15% | 0.45% |
| **Total** | **100%** | **3.0% (FTMO limit)** |

**Internal limit:** Hard stop at 2.0% daily loss (vs FTMO 3.0%) to maintain safety buffer.

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 0: Fix foundation | 1 | Correct baseline backtest |
| 1: Expand universe | 2-3 | 100+ instruments, diversification backtest |
| 2: Vol/lookback optimization | 4-7 | Best vol target and lookback combo |
| 3: Signal tuning | 8-10 | Optimized signal parameters |
| 4: Pyramided breakout | 11-13 | Convex payoff addition |
| 5: Alt data overlays | 14-18 | COT + GPR + calendar filters |
| 6: Stress testing | 19-20 | Full stress test suite |
| 7: Final report | 21 | Go/No-Go decision |

**Total: ~3 weeks of development before deploying to VPS for paper trading.**

---

## REVISED PLAN (2026-04-04): Restricted Universe + Vol-Targeting

### Key Insight from Backtest Results

Expanding from 53 to 105 instruments HURT performance (Sharpe went negative). The problem was not TSMOM itself but applying it to whipsawing equities and crypto with long lookbacks. Academic evidence (Gorton & Rouwenhorst 2006, Hamilton 2003, Erb & Harvey 2006) confirms commodity TSMOM has the highest Sharpe during supply shocks.

### Revised Architecture

**Universe: 20-25 instruments (RESTRICTED, not expanded)**
- Commodities (60% risk): gold, silver, platinum, copper, oil x2, natgas, soft commodities
- FX (25% risk): 9 major/commodity-currency pairs
- Crypto fast-TSMOM (15% risk): BTC + ETH primary, alts in BTC-bull only

**Three key changes from original plan:**
1. **DROP equities and equity indices** — whipsaw in crisis, negative TSMOM contribution
2. **ADD Moreira-Muir vol-targeting** — scale positions by inverse realized vol, target 12-15% portfolio vol
3. **SEPARATE crypto lookback** — 14-day for crypto, 63/126/252d for everything else

### Implementation (same-day)
1. Rewrite backtest engine with restricted universe + dual lookback + vol-targeting
2. Run backtest with FTMO sim (10% target, 180-day window)
3. Run stress tests on best config
4. Generate report, update memory

---

## Expected Outcome (Revised)

| Metric | Before (105 instruments) | Target (restricted) | Confidence |
|--------|-------------------------|-------------------|------------|
| FTMO pass rate (10% target) | 25-31% | **45-60%** | Medium-High |
| Avg days to pass | 33-43 | **21-40** | Medium |
| Sharpe ratio | -0.1 to 0.1 | **0.6-1.0** | Medium-High |
| MDL breach rate | 66-69% | **20-30%** | High |
| CAGR | -3% to +1% | **15-30%** | Medium |
3. Pyramiding captures the large trends (gold, oil) with convex payoff
4. COT/GPR overlays prevent us from being on the wrong side of crowded/reversing trades

**Risk:** Ceasefire/war resolution would reverse oil and partially reverse gold. The system needs to detect and adapt — VoV reduction and GPR filter help but can't prevent all losses from a regime change.

---

## Compliance with RESEARCH_PROTOCOL.md

| Requirement | How We Meet It |
|-------------|---------------|
| Economic rationale (Section 5.3) | Behavioral underreaction (TSMOM), inventory effects (COT), event processing delay (GPR/calendar) |
| At least 2 academic sources per strategy | All layers have 2+ sources documented |
| Walk-forward validation (Section 2) | 1-year train / 3-month OOS, expanding window |
| Deflated Sharpe (Section 4.3) | Computed for all 36+ variants tested |
| FTMO sim overlay (Section 1.4) | 10% target, 180-day window, MDL/ML/Best Day tracking |
| Stress tests (Section 7) | All 7 scenarios run on final system |
| Point-in-time discipline (Section 3) | All features use trailing data only |
| Cost model (Section 1.2) | Spread + slippage + swap per asset class |
| Research cards (Section 5.1) | Created for each new component |
