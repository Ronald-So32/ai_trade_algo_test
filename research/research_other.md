# FTMO Challenge Research Compendium
## Strategies for High-Volatility, Geopolitical Crisis Environments

Generated: 2026-04-04
Context: Iran-Hormuz war (Feb 28, 2026+), VIX 27, Gold $4,676, Oil $111, BTC $67k

---

## Part 1: Crisis Alpha Strategies

### 1.1 Crisis TSMOM / Trend-Following (A+ evidence)

**Academic Sources:**
- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE
- Fung & Hsieh (2001), "The Risk in Hedge Fund Strategies: Theory and Evidence from Trend Followers"
- Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following Investing"
- Daniel & Moskowitz (2016), "Momentum Crashes", JFE

**How it works:** Long assets with positive trailing returns, short assets with negative. Multi-scale blending (63/126/252 day lookbacks). Inverse-vol position sizing.

**Crisis performance:**
- SG Trend Index: +28% in 2022 (Russia-Ukraine), +18% in 2008 (GFC)
- Fung & Hsieh proved trend-following = lookback straddle — naturally convex payoff
- Daniel & Moskowitz WARNING: equity momentum CRASHES during crisis recoveries. Commodity momentum does NOT exhibit this crash risk.

**Expected Sharpe:** 0.4-0.8 (higher during sustained crises)
**FTMO fit:** Excellent — convex payoff avoids Best Day concentration, positive skewness protects against total loss breach

**Key finding for current market:** Overweight commodity/FX momentum, AVOID equity momentum during crises. Gold TSMOM is the highest-conviction trade.

### 1.2 Commodity Supply Shock Strategy

**Academic Sources:**
- Hamilton (2003), "What is an Oil Shock?", JEconometrics
- Kilian (2009), "Not All Oil Price Shocks Are Alike", AER
- Ready (2018), "Oil Prices and the Stock Market", RFS

**How it works:** Long oil and energy-related commodities during supply disruptions (Hormuz, embargoes). Short energy-importing equity indices (DAX, Nikkei). Historical supply disruptions produced 80-300% oil moves.

**Risk:** Binary reversal on ceasefire/resolution. 2019 Aramco drone attack reversed in 2 weeks.

**Current application:** USOIL.cash, UKOIL.cash long; NATGAS.cash long; GER40.cash short (Europe most exposed to energy shock)

**Expected Sharpe:** 0.5-1.0 during active supply disruption, negative if resolution occurs
**FTMO fit:** High return potential but requires tight stops for reversal risk. Cap daily oil exposure to avoid MDL breach on ceasefire headline.

### 1.3 Safe-Haven Momentum (Gold Focus)

**Academic Sources:**
- Baur & Lucey (2010), "Is Gold a Hedge or a Safe Haven?", JBF
- Baur & McDermott (2010), "Is Gold a Safe Haven? International Evidence"
- Bouri et al. (2020), "Geopolitical Risks and Gold", IRFA

**How it works:** Gold is the most reliable safe haven across all 6 major crises since 1980. During geopolitical crises, gold rallies from both safe-haven demand AND inflation/supply-shock expectations.

**Current state:** Gold $4,676, +70% in 18 months. Multi-year structural bull driven by:
- Central bank buying (record levels)
- Iran war safe-haven premium
- Fed easing expectations
- De-dollarization flows
- JP Morgan target: $5,000; Goldman target: $6,000

**JPY note:** JPY safe-haven status has WEAKENED — failed in 2022 due to BOJ policy divergence. Do NOT rely on JPY as safe haven in this cycle.

**Expected Sharpe:** 0.8-1.2 for gold momentum during crises
**FTMO fit:** Excellent — sustained trend with low reversal risk even on ceasefire (structural demand continues)

### 1.4 Carry Trade Unwind

**Academic Sources:**
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes", NBER Macro Annual
- Lustig, Roussanov & Verdelhan (2011), "Common Risk Factors in Currency Markets", RFS

**How it works:** During risk-off events, high-yielding currencies (AUD, NZD, emerging markets) crash against funding currencies (JPY, CHF, USD). The carry trade unwinds as leveraged positions are liquidated.

**Tradeable on FTMO:** Short AUDJPY, NZDJPY during VIX spikes. Predictable triggers: VIX spike + JPY strengthening + equity drawdown.

**Expected Sharpe:** 0.3-0.6 (episodic, not continuous)
**FTMO fit:** Good — produces quick profits on risk-off days, but signal is intermittent

### 1.5 Post-Shock Mean Reversion (Equity Bounce)

**Academic Sources:**
- Bloom (2009), "The Impact of Uncertainty Shocks", Econometrica
- Baker, Bloom & Davis (2016), "Measuring Economic Policy Uncertainty", QJE

**How it works:** After peak fear (VIX > 40), equities typically recover. Historical: buying S&P when VIX > 40 produced +9.8% avg over 60 days with 78% win rate.

**Risk:** Requires TIMING — buying too early means catching a falling knife. VIX is currently 27, not 40. Deploy ONLY after confirmed peak fear, not at crisis onset.

**Expected Sharpe:** 0.5-0.8 post-peak (timing-dependent)
**FTMO fit:** Strong if timed correctly — large move helps reach 10% target. But timing failure = MDL breach.

---

## Part 2: Qualitative / Alternative Data

### 2.1 COT (Commitment of Traders) Data — RECOMMENDED

**Academic Sources:**
- De Roon, Nijman & Veld (2000), "Hedging Pressure Effects in Futures Markets"
- Bessembinder (1992), "Systematic Risk, Hedging Pressure, and Risk Premiums in Futures Markets"
- Moskowitz, Ooi & Pedersen (2012) — notes positioning interaction with momentum

**What it does:** CFTC reports show commercial vs speculative positioning in futures. Extreme speculative positioning (>90th percentile) signals crowding and reversal risk.

**How to use with TSMOM:**
- When speculative positioning is extreme AND TSMOM signal agrees → reduce size (crowded)
- When speculative positioning is extreme opposite AND TSMOM signal is strong → increase size (contrarian confirmation)

**Data:** Free from CFTC, weekly (Tuesday data, Friday release). Python: `cot_reports` package.
**Implementation complexity:** Low-medium
**Evidence strength:** Strong

### 2.2 Geopolitical Risk Index (GPR) — RECOMMENDED

**Academic Sources:**
- Caldara & Iacoviello (2022), "Measuring Geopolitical Risk", AER

**What it does:** Newspaper-based index of geopolitical tension. GPR spikes correlate with gold rallies, oil spikes, equity drops, safe-haven FX appreciation.

**How to use:** Regime filter — when GPR > 2 std devs above mean:
- Reduce equity momentum positions
- Increase gold/commodity momentum positions
- Tighten daily loss stops

**Data:** Free CSV from matteoiacoviello.com/gpr.htm, daily version available.
**Implementation complexity:** Low
**Evidence strength:** Strong (AER publication)

### 2.3 Central Bank Calendar — RECOMMENDED

**Academic Sources:**
- Lucca & Moench (2015), "The Pre-FOMC Announcement Drift"
- Hansen & McMahon (2016), "Shocking Language: Understanding the Macroeconomic Effects of Central Bank Communication"

**What it does:** S&P 500 earned ~80% of excess returns in 24 hours before FOMC announcements (1994-2011). Effect has weakened since publication but still significant.

**How to use:** 
- Reduce short equity exposure before FOMC
- Flatten FX positions around FOMC/ECB (binary outcomes)
- Zero NLP required — just a calendar

**Data:** Public FOMC/ECB/BoJ calendars
**Implementation complexity:** Very low
**Evidence strength:** Strong

### 2.4 News Sentiment (FinBERT) — SECONDARY

**Academic Sources:**
- Tetlock (2007), "Giving Content to Investor Sentiment", JF
- Loughran & McDonald (2011), "When Is a Liability Not a Liability?", JF
- Heston & Sinha (2017), "News vs. Sentiment: Predicting Stock Returns from News Stories"

**What it does:** NLP-scored news sentiment has 1-2 day predictive power for equity returns. Negative words matter more than positive (asymmetric).

**How to use:** Regime filter — when aggregate financial news sentiment is extremely negative AND momentum is weakening, reduce exposure. NOT a standalone signal.

**Data:** Free via GDELT, RSS feeds + FinBERT model (HuggingFace: `ProsusAI/finbert`)
**Implementation complexity:** Medium-high
**Evidence strength:** Moderate (real but small alpha)

### 2.5 Crypto Fear & Greed Index — CRYPTO ONLY

**Data:** Free API from alternative.me, daily
**Evidence:** Weak-moderate. Contrarian value at extremes (sub-20 = buy, above-80 = sell)
**Current reading:** 12 (Extreme Fear) — contrarian buy signal, but trend is clearly bearish
**Implementation complexity:** Very low

### 2.6 Social Media Sentiment — NOT RECOMMENDED

**Evidence:** Bollen et al. (2011) "Twitter mood predicts stock market" has been widely criticized for overfitting. Subsequent replications are mixed. Signal-to-noise ratio too low for a prop challenge.

---

## Part 3: Options/Futures Synthetic Strategies on CFDs

### 3.1 FTMO Instrument Constraints

**FTMO offers EXCLUSIVELY spot CFDs.** No options, no futures, no VIX, no term structure instruments. This eliminates:
- Calendar spreads
- Basis trades
- Contango/backwardation plays
- Direct volatility trading
- Any strategy requiring options greeks

### 3.2 What IS Possible

#### 3.2.1 Pyramided Breakout (Synthetic Gamma) — RECOMMENDED

**Academic Sources:**
- Fung & Hsieh (2001) — trend-following = lookback straddle (synthetic long gamma)
- Brock, Lakonishok & LeBaron (1992), "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns", JF

**How it works:** Add to winning positions on trend continuation (pyramid). Start with 0.5% risk, add 0.5% at each breakout level, max 4 units = 2% total. Creates convex payoff mimicking long gamma.

**Expected Sharpe:** 0.4-0.7
**FTMO fit:** Excellent — convex payoff helps reach 10% target on strong trends. Must limit to 4 pyramid units at 0.5% risk each to stay within 3% daily limit.
**Implementation:** Modify position sizing in TSMOM to scale into winners rather than rebalancing to fixed weight.

#### 3.2.2 Gold/Silver Ratio Trade — VIABLE

**Academic Sources:**
- Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", RFS
- Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis"

**How it works:** Gold/silver ratio (XAUUSD/XAGUSD) is historically mean-reverting with ~60-day half-life, range 60-100. When ratio > 85, long silver short gold. When < 65, long gold short silver.

**Expected Sharpe:** 0.2-0.5 after CFD spread + swap costs
**FTMO fit:** Low correlation with TSMOM — diversification benefit. But spread costs on silver are high.

#### 3.2.3 WTI/Brent Spread — VIABLE

**How it works:** USOIL.cash vs UKOIL.cash spread is mean-reverting with ~40-day half-life. Trade mean-reversion on the spread.

**Expected Sharpe:** 0.2-0.4 after costs
**FTMO fit:** Diversifying. Small allocation only.

#### 3.2.4 Index Dispersion (Simplified) — MARGINAL

**How it works:** Trade individual stock CFDs vs equity index CFDs. When implied correlation is high, go long individual stocks + short index (bet on correlation decreasing). Without VIX/options data, proxy correlation regime using realized correlation.

**Expected Sharpe:** 0.1-0.3 (hard to implement well without options)
**FTMO fit:** Too many positions, too much margin. Not recommended.

### 3.3 What Does NOT Work on CFDs

| Strategy | Why Not |
|----------|---------|
| Covered calls | No options — take-profit orders don't generate premium |
| Calendar spreads | No expiry dates on CFDs |
| Contango/backwardation | CFDs have no roll yield — Gorton & Rouwenhorst (2006) showed ~60% of commodity futures returns come from roll yield |
| Equity pairs (Gatev method) | CFD spread + swap costs likely eliminate the alpha |
| VIX trading | No VIX CFD on FTMO |
| Any Greeks-based strategy | No options |

---

## Part 4: Current Market Environment (April 2026)

### 4.1 Macro Backdrop

- **Iran war** ongoing since Feb 28, 2026 ("Operation Epic Fury")
- **Strait of Hormuz closed** — ~20% of global oil supply disrupted
- IEA: "largest supply disruption in the history of the global oil market"
- Ceasefire speculation oscillating — binary risk for oil/gold

### 4.2 Asset Class Trends (TSMOM signal quality)

| Asset | Price | 3-6mo Trend | TSMOM Signal Quality |
|-------|-------|-------------|---------------------|
| **XAUUSD** | $4,676 | Strong bull (+70% in 18mo) | **Excellent** — structural + crisis |
| **USOIL.cash** | $111 | Strong bull (+50% YTD) | **Strong but binary reversal risk** |
| **NATGAS.cash** | Elevated | Bull on supply disruption | **Good** |
| **COCOA.c, COFFEE.c** | Elevated | Supply disruption trends | **Moderate** |
| **BTCUSD** | $67k | Bearish (Fear index: 12) | **Moderate short** |
| **ETHUSD** | $2,058 | Bearish | **Moderate short** |
| **GER40.cash** | ~22,945 | Bearish (energy shock) | **Moderate but violent rallies** |
| **US500.cash** | ~5,600 | Choppy/range-bound | **Poor — whipsaw** |
| **US100.cash** | Similar | Choppy | **Poor** |
| **EURUSD** | ~1.15 | Weakening (energy shock) | **Moderate** |
| **USDJPY** | ~152 | Range-bound | **Poor** |
| **DXY.cash** | ~100.5 | Short-term bull (safe haven) | **Moderate** |

### 4.3 Volatility Regime

- **VIX: 27** (elevated, upper quartile)
- Range over past month: 20-35
- Headline-driven spikes — worst for pure TSMOM on equities
- Commodity and gold vol is elevated but DIRECTIONAL (good for TSMOM)

### 4.4 Implications for Strategy

1. **Gold is the #1 TSMOM trade** — structural bull + crisis premium + central bank buying
2. **Oil is high-conviction but dangerous** — ceasefire = instant 20-30% reversal
3. **Commodities broadly trending** — supply disruption affects food, energy, metals
4. **Equities are TRAPS for momentum** — headline whipsaw, no clean trend
5. **Crypto offers clean short momentum** — bearish with extreme fear
6. **Forex has moderate opportunities** — EUR weakness (energy), USD strength (safe haven)

---

## Part 5: Available FTMO Instruments (166 total)

### By Asset Class

| Class | Count | Leverage | Current TSMOM Quality |
|-------|-------|----------|----------------------|
| Forex Majors | 7 | 1:30 | Moderate |
| Forex Crosses | 21 | 1:30 | Mixed |
| Forex Exotics | 15 | 1:30 | Low (high spread) |
| Indices | 14 | 1:20 | Poor (whipsaw) |
| Metals | 9 | 1:10 | **Excellent** |
| Commodities | 11 | 1:10 | **Good-Excellent** |
| US Equities | 45 | 1:5 | Poor (whipsaw) |
| EU Equities | 13 | 1:5 | Poor-Moderate |
| Crypto | 31 | 1:2 | **Moderate (short)** |

### Key Symbols for Current Regime

**High conviction (strong sustained trends):**
XAUUSD, XAGUSD, XPTUSD, USOIL.cash, UKOIL.cash, NATGAS.cash

**Moderate conviction (directional with noise):**
EURUSD (short), GER40.cash (short), BTCUSD (short), ETHUSD (short), COCOA.c, COFFEE.c

**Avoid (whipsaw/no trend):**
US500.cash, US100.cash, US30.cash, most individual equities

---

## References (43 sources)

1. Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE
2. Fung & Hsieh (2001), "The Risk in Hedge Fund Strategies", FAJ
3. Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following"
4. Daniel & Moskowitz (2016), "Momentum Crashes", JFE
5. Hamilton (2003), "What is an Oil Shock?", JEconometrics
6. Kilian (2009), "Not All Oil Price Shocks Are Alike", AER
7. Ready (2018), "Oil Prices and the Stock Market", RFS
8. Baur & Lucey (2010), "Is Gold a Hedge or a Safe Haven?", JBF
9. Baur & McDermott (2010), "Is Gold a Safe Haven? International Evidence"
10. Bouri et al. (2020), "Geopolitical Risks and Gold", IRFA
11. Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes"
12. Lustig, Roussanov & Verdelhan (2011), "Common Risk Factors in Currency Markets", RFS
13. Bloom (2009), "The Impact of Uncertainty Shocks", Econometrica
14. Baker, Bloom & Davis (2016), "Measuring Economic Policy Uncertainty", QJE
15. Caldara & Iacoviello (2022), "Measuring Geopolitical Risk", AER
16. Tetlock (2007), "Giving Content to Investor Sentiment", JF
17. Loughran & McDonald (2011), "When Is a Liability Not a Liability?", JF
18. Heston & Sinha (2017), "News vs. Sentiment"
19. Lucca & Moench (2015), "The Pre-FOMC Announcement Drift"
20. Hansen & McMahon (2016), "Shocking Language"
21. De Roon, Nijman & Veld (2000), "Hedging Pressure Effects in Futures Markets"
22. Bessembinder (1992), "Systematic Risk, Hedging Pressure, and Risk Premiums"
23. Moreira & Muir (2017), "Volatility-Managed Portfolios", JF
24. Grossman & Zhou (1993), "Optimal Investment Strategies for Controlling Drawdowns"
25. Browne (1999), "Beating a Moving Target"
26. Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading", RFS
27. Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis"
28. Gorton & Rouwenhorst (2006), "Facts and Fantasies about Commodity Futures", FAJ
29. Brock, Lakonishok & LeBaron (1992), "Simple Technical Trading Rules", JF
30. Ante (2023), Bitcoin Fear & Greed Index research
31. Bollen, Mao & Zeng (2011), "Twitter mood predicts the stock market" (CRITICIZED)
32. Chen et al. (2014), Seeking Alpha article predictiveness
33. Sanders, Irwin & Merrin (2010), COT positioning and returns
34. Schmeling & Wagner (2019), "Central bank tone predicts bond returns"
35. Lo (2002), "The Statistics of Sharpe Ratios", FAJ
36. Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
37. Kraaijeveld & De Smedt (2020), Twitter sentiment and crypto
38. Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere", JF
39. Baltas & Kosowski (2020), "Demystifying Time-Series Momentum Strategies"
40. Baz et al. (2015), "Dissecting Investment Strategies in the Cross Section and Time Series"
41. Koijen et al. (2018), "Carry", JFE
42. Ilmanen (2011), "Expected Returns", Wiley
43. Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers", JF

---

## Part 6: Strategy Alternative Assessment (April 2026 Update)

### Critical Finding: TSMOM Is Right, Universe Is Wrong

The research conclusively shows TSMOM is the correct core strategy for geopolitical crisis environments (SG Trend Index: +28% in 2022, +18% in 2008). The problem is applying it to 105 instruments including whipsawing equities and crypto.

### Strategies Evaluated and Verdicts

| Strategy | Verdict | Reason |
|----------|---------|--------|
| **Commodity TSMOM** | **ADOPT — core** | Sharpe 0.8-1.2 during supply shocks. Strongest evidence. Hamilton (2003), Gorton & Rouwenhorst (2006) |
| **Vol-targeting overlay** | **MUST ADD** | Moreira & Muir (2017), Barroso & Santa-Clara (2015). Directly fixes MDL breaches (#1 failure). Expected +0.2-0.4 Sharpe improvement |
| **FX TSMOM** | **KEEP** | Menkhoff et al. (2012). Clean trends in safe-haven/commodity currencies |
| **Fast crypto TSMOM** | **ADD — 15% risk** | Liu et al. (2022). 14-day lookback with BTC regime filter. Separate from main system |
| Carry trade | **SKIP** | Crashes during crises (Brunnermeier et al. 2009). Negative skew = MDL killer |
| FX value (PPP) | **SKIP** | Too slow (3-5 year half-life). Taylor & Taylor (2004) |
| Risk parity | **ADOPT — as sizing** | Already doing inverse-vol. Formalize to equal risk contribution |
| Quality equity | **SKIP** | Unnecessary complexity. Drop equities entirely |
| Equity indices | **DROP** | Whipsaw in current regime. Daniel & Moskowitz (2016) momentum crashes |

### Recommended Architecture

**Universe: 20-25 instruments (down from 105)**
- Commodities (60% risk): XAUUSD, XAGUSD, XPTUSD, XCUUSD, USOIL, UKOIL, NATGAS, COCOA, COFFEE, CORN, SOYBEAN, WHEAT, COTTON, SUGAR
- FX (25% risk): EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, USDNOK, USDMXN
- Crypto fast-TSMOM (15% risk): BTC, ETH + alts in BTC-bull only

**Signal: TSMOM with adaptive lookback**
- Commodities/FX: 42/126/252d blend (standard)
- Crypto: 7/14/21d blend (fast)
- Weight shorter lookbacks when realized vol > historical median

**Sizing: Vol-targeted risk parity (Moreira-Muir)**
- Target portfolio vol: 12-15% annualized
- Scale = target_vol / realized_vol(21d)
- VIX > 30: reduce target to 10%. VIX > 40: reduce to 8%
- Equal risk contribution across active positions

**Risk: MDL-aware**
- Internal daily stop: -2.0% (buffer below FTMO's -3%)
- At -1.5%: cut all positions 50%

### Key Academic Sources (added)
44. Moreira & Muir (2017), "Volatility-Managed Portfolios", JF
45. Barroso & Santa-Clara (2015), "Momentum Has Its Moments", JFE
46. Bhardwaj, Gorton & Rouwenhorst (2015), "Commodity Momentum"
47. Erb & Harvey (2006), "Strategic and Tactical Value of Commodity Futures"
48. Liu, Tsyvinski & Wu (2022), "Common Risk Factors in Cryptocurrency", RFS
49. Corbet et al. (2018), "Exploring Dynamic Relationships between Cryptocurrencies"
50. Asness, Frazzini & Pedersen (2019), "Quality Minus Junk", RFS

---

## Part 7: Pairs Trading / Statistical Arbitrage Assessment

### Key Finding: Conditional Add at 10-15% Risk Budget

Pairs trading is negatively correlated with TSMOM (rho -0.2 to -0.4, strengthening to -0.5 to -0.7 during momentum crashes). This is the primary value — reducing MDL breach probability.

### Mathematical Framework

**Cointegration method (Engle-Granger 1987):**
1. OLS: Y_t = α + β × X_t + ε_t
2. ADF test on ε_t (p < 0.05 required)
3. Z-score: z_t = (ε_t - mean(ε)) / std(ε)
4. Entry: |z| > 2.0 | Exit: |z| < 0.5 | Stop: |z| > 4.0
5. Recalibrate β monthly via rolling 252-day window

**Ornstein-Uhlenbeck for half-life:**
- dS = θ(μ - S)dt + σdW
- Half-life = ln(2)/θ
- Estimate via AR(1): S_t = c + φ × S_{t-1} + ε → θ = -ln(φ)

### Top 6 Pairs for FTMO (ranked by evidence + current opportunity)

| Rank | Pair | Type | Half-Life | Est. Sharpe | Current Signal |
|------|------|------|-----------|-------------|---------------|
| 1 | CVX/XOM | Energy | 12-25d | 0.9-1.2 | Good — oil co-movement |
| 2 | AUDUSD/NZDUSD | FX | 15-40d | 0.7-1.0 | Always on |
| 3 | LMT/RTX | Defense | 15-30d | 0.7-0.9 | Excellent — war dislocation |
| 4 | JPM/BAC | Banks | 10-20d | 0.7-1.0 | Good — VIX stress |
| 5 | XAUUSD/XAGUSD | Commodity | 30-60d | 0.5-0.8 | Excellent if ratio >95 |
| 6 | NVDA/AMD | Semis | 8-18d | 0.5-0.9 | Moderate — check stability |

### CFD Cost Analysis
- Round-trip: ~40 bps spread + 18-36 bps swaps (15-day hold) = 60-75 bps total
- Gross returns: 80-150 bps per trade → net 20-75 bps
- Thin but positive for top pairs. Bottom pairs may be negative after costs.

### FTMO Compatibility
- Many small wins → excellent for Best Day Rule
- Correlation rho -0.2 to -0.4 with TSMOM → reduces MDL breaches
- Risk: M&A/earnings can blow up a pair >3% in one day → avoid pairs near events
- Max 3% notional per leg, 8 simultaneous pairs, 30% gross exposure cap

### Expected Impact on Pass Rate
- Conservative: +2-3% (42% → 44-45%)
- Optimistic: +5-8% (42% → 47-50%)
- Depends heavily on whether top pairs maintain cointegration on MT5 data

### Academic Sources
51. Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading", RFS
52. Do & Faff (2010), "Does Simple Pairs Trading Still Work?"
53. Krauss (2017), "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook"
54. Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis"
55. Engle & Granger (1987), "Co-integration and Error Correction"
56. Huck & Afawubo (2015), "Pairs Trading and Selection Methods"

---

## Part 8: Earnings-Based Equity Strategies Assessment

### Verdict: SKIP ALL

Earnings strategies are fundamentally wrong for this challenge. Every variant was evaluated:

**PEAD (Post-Earnings Announcement Drift):**
- Ball & Brown (1968), Bernard & Thomas (1989) established the anomaly
- **Martineau (2022) "Rest in Peace PEAD"** showed it's dead for large-cap US stocks post-2005
- Our 32 stocks are the most-followed mega-caps on Earth — zero alpha remains
- Verdict: SKIP

**Pre-Earnings Volatility:**
- Patell & Wolfson (1981), Dubinsky & Johannes (2006)
- Requires OPTIONS to capture variance risk premium — impossible on CFDs
- Directional proxy has Sharpe ~0.0
- Verdict: SKIP (incompatible with CFDs)

**Earnings Momentum (SUE):**
- Foster, Olsen & Shevlin (1984) — functionally identical to PEAD
- Same decay problems apply
- Verdict: SKIP (repackaged PEAD)

**Earnings Calendar Premium:**
- Savor & Wilson (2016): real risk premium, stocks earn ~7 bps/day around earnings
- But at safe position sizes (2%): only ~$640/year on $100K
- Economically immaterial for reaching 10% target
- Verdict: SKIP (immaterial)

**EBS (existing code in src/strategies/ebs.py):**
- Best of the lot — targets >8% drops with ML model, asymmetric payoff
- But only ~14 trades/year, expected P&L near zero on these mega-caps
- Conditional: maybe at 0-2% risk if validated on MT5 data
- Verdict: SKIP for now, revisit after MT5 validation

### Why Earnings Strategies Don't Fit This Challenge
1. PEAD is arbitraged away for our universe (Martineau 2022)
2. Gap risk directly threatens 3% MDL (±10-20% earnings gaps on single stocks)
3. Only 128 events/year clustered in 12 weeks — dormant 40 weeks
4. At safe sizing: generates ~$640/year, need $10,000
5. Contradicts our own finding that equities are whipsaw traps
6. Every addition has hurt so far — complexity itself is a drag

### Academic Sources
57. Ball & Brown (1968), "An Empirical Evaluation of Accounting Income Numbers", JAR
58. Bernard & Thomas (1989), "Post-Earnings-Announcement Drift", JAR
59. Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift", Critical Finance Review
60. Savor & Wilson (2016), "Earnings Announcements and Systematic Risk", JF
61. Foster, Olsen & Shevlin (1984), "Earnings Releases, Anomalies, and Security Returns", TAR
62. McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?", JF
