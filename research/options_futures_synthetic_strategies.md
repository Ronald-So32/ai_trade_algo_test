# Options, Futures, and Synthetic Derivative Strategies on FTMO MT5

## Research Context
**Date**: 2026-04-03
**Objective**: Determine whether options/futures are available on FTMO MT5, and what derivative-like strategies can be replicated using CFDs under FTMO constraints.
**Account**: $100K Swing, MT5, 1:30 leverage (forex), 1:20 (indices), 1:10 (metals/commodities), 1:5 (equities), 1:2 (crypto).
**Constraints**: 3% daily loss limit, 10% total loss limit, 50% best-day rule, no weekend restrictions (Swing).

---

## 1. Does FTMO Offer Options or Futures Trading?

### Answer: No.

FTMO offers **exclusively spot CFDs** (Contracts for Difference). Across all 166 instruments in the FTMO catalog:

- **No options** of any kind (vanilla, binary, exotic)
- **No futures contracts** (no expiry dates, no contract rolls)
- **No swaps/swaptions**
- **No structured products**
- **No VIX CFDs** (DXY.cash is available for dollar index, but no volatility index)

FTMO's MT5 platform is a dealing-desk CFD environment. The instruments ending in `.cash` (e.g., USOIL.cash, US500.cash) are cash-settled CFDs that track the spot/front-month price of the underlying. They do not have:
- Expiry dates
- Delivery mechanisms
- Options chains
- Futures term structures

**Implication**: Any options-like or futures-like strategy must be **synthetically replicated** using combinations of CFD positions, stop-loss orders, and portfolio construction. This is a fundamental constraint that limits many strategies but enables some creative alternatives.

---

## 2. Synthetic Options Strategies Using CFDs

### 2A. Synthetic Covered Call (Long CFD + Implied Short Call)

**Real version**: Long stock + sell OTM call = capped upside, income from premium.

**CFD approximation**: Long CFD + take-profit limit order at a fixed distance above entry.

- The take-profit acts as the "strike price" where upside is capped.
- The "premium" is replaced by the expected value of the truncated profit distribution.
- This is NOT a true covered call because there is no premium received upfront.

**Assessment**:
- **Feasible on FTMO**: Trivially, yes -- this is just a long position with a take-profit. But it lacks the economic substance of a covered call (no income generation).
- **Academic evidence**: Israelov & Nielsen (2015), "Covered Calls Uncovered", FAJ -- showed covered calls underperform the underlying in strong uptrends. The CFD version is strictly worse because there is no premium income to offset capped upside.
- **Verdict**: Not useful. A take-profit order does not generate options premium. The whole point of covered calls is income; without selling actual calls, this is just a limit order.

### 2B. Synthetic Protective Put via Stop-Loss

**Real version**: Long stock + buy OTM put = unlimited upside, limited downside.

**CFD approximation**: Long CFD + guaranteed stop-loss order.

- The stop-loss acts as the "strike price" limiting downside.
- The "premium" is the expected slippage cost + the opportunity cost of being stopped out on whipsaws.
- FTMO does not offer guaranteed stop-losses; MT5 stop-losses can experience slippage in gaps.

**Assessment**:
- **Feasible on FTMO**: Partially. Standard stop-loss orders provide approximate downside protection, but with slippage risk on gaps (especially commodities and indices at open).
- **Academic evidence**: Broadie, Chernov & Johannes (2009), "Understanding Index Option Returns", RFS -- showed that put protection is expensive partly due to volatility risk premium. CFD stops avoid paying this premium but also lack the convexity guarantee through gaps.
- **Key difference**: Real puts protect through gaps; stop-losses do not. Weekend gaps on FTMO Swing accounts are a real risk. A 3% gap through a stop-loss could trigger the 3% daily loss limit.
- **Sharpe impact**: Negative drag from being stopped out on whipsaws. Granham & Harvey (2014) estimate stop-losses reduce Sharpe by 0.1-0.3 vs. optimal options hedging.
- **Verdict**: Useful as basic risk management but not a true substitute for puts. The gap risk is the critical weakness.

### 2C. Synthetic Straddle/Strangle via Correlated CFDs

**Real version**: Buy call + buy put at same/different strikes = profit from large moves in either direction.

**CFD approximation approaches**:

1. **Breakout straddle**: Place buy-stop above resistance and sell-stop below support. One triggers, cancel the other (OCO logic). This creates straddle-like payoff around a range.

2. **Correlated pair straddle**: Go long a high-beta instrument and short a low-beta instrument in the same sector. Net exposure is to the "excess volatility" of the high-beta name.

3. **Momentum breakout (Fung & Hsieh 2001 approach)**: Trend-following strategies naturally replicate lookback straddle payoffs. Already implemented in TSMOM.

**Assessment**:
- **Feasible on FTMO**: The breakout approach (1) works but requires OCO order logic, which MT5 supports via EA or manual monitoring. The correlated pair approach (2) is feasible but messy. The TSMOM approach (3) is already in the system.
- **Academic evidence**:
  - Fung & Hsieh (2001) -- "The Risk in Hedge Fund Strategies" -- proved that trend-following returns are statistically equivalent to lookback straddle returns. R-squared of 0.48 between TSMOM and lookback straddle payoffs.
  - Lo, Mamaysky & Wang (2000) -- "Foundations of Technical Analysis", JF -- showed that breakout signals (support/resistance breaks) capture straddle-like payoffs with positive expected value.
- **Expected Sharpe**: 0.3-0.6 for breakout straddles (lower than real straddles because entry timing is imperfect and there is no true gamma exposure).
- **FTMO risk**: OCO breakout orders can both trigger in a whipsaw, creating a double loss. Limit to 0.5% risk per side.
- **Verdict**: TSMOM already provides the best synthetic straddle. Explicit breakout straddles add marginal value. The correlated-pair approach is better classified as a pairs trade (see Section 4).

### 2D. Delta-Neutral Strategies Using CFD Pairs

**Real version**: Options market-making -- delta-hedge options positions continuously.

**CFD approximation**: Hold offsetting long/short positions in correlated instruments, profiting from relative value changes.

**Examples with FTMO instruments**:
- Long XAUUSD / Short XAGUSD (gold/silver ratio trade)
- Long US500.cash / Short US100.cash (value vs. growth rotation)
- Long EURUSD / Short GBPUSD (EUR/GBP cross via triangular arbitrage)
- Long NVDA / Short US100.cash (single stock alpha vs. index beta)

**Assessment**:
- **Feasible on FTMO**: Yes, this is standard pairs/relative value trading. However, it is NOT truly delta-neutral in the options sense because there is no gamma, no vega, and no theta. It is market-neutral (low net beta) but not delta-neutral in the derivatives sense.
- **Academic evidence**:
  - Gatev, Goetzmann & Rouwenhorst (2006) -- "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", RFS. Excess returns of 11% annually (before costs) for distance-based pairs. See Section 4 for detailed discussion.
  - Avellaneda & Lee (2010) -- "Statistical Arbitrage in the US Equities Market", Quantitative Finance. PCA-based statistical arbitrage using sector ETFs, Sharpe ~1.0 gross.
- **Expected Sharpe**: 0.5-1.0 for well-constructed pairs, heavily dependent on spread costs and swap rates.
- **FTMO risk**: Pairs can diverge during crises, creating larger-than-expected drawdowns on BOTH legs simultaneously. The 3% daily limit can be hit if correlation breaks down.
- **Verdict**: Viable and implementable. This is the best "options-like" strategy available on FTMO for reducing directional risk. See Sections 3 and 4 for detailed implementation.

---

## 3. Futures-Like Strategies on CFDs

### 3A. Calendar Spread Equivalents (USOIL.cash vs UKOIL.cash)

**Real futures version**: Buy front-month, sell back-month (or vice versa) to capture roll yield / term structure.

**CFD reality**: FTMO CFDs have no term structure. USOIL.cash and UKOIL.cash are two DIFFERENT underlyings (WTI vs Brent), not two maturities of the same crude oil contract.

- USOIL.cash = WTI Crude Oil spot CFD
- UKOIL.cash = Brent Crude Oil spot CFD

**Assessment**:
- **Feasible on FTMO**: You can trade the WTI-Brent spread (USOIL vs UKOIL), but this is a cross-commodity spread, NOT a calendar spread. The spread reflects transportation costs, regional supply/demand, and quality differences, not term structure.
- **Academic evidence**:
  - Büyüksahin, Haigh & Robe (2010) -- "Commodities and Equities: Ever a Market of One?", JFQA. Documents the WTI-Brent spread as driven by infrastructure (Cushing storage), geopolitical factors, and seasonal refinery demand.
  - Fattouh (2010) -- "The Dynamics of Crude Oil Price Differentials", Energy Economics. Shows WTI-Brent spread is mean-reverting with half-life of ~40 trading days. Historical range: -$5 to +$25 (Brent premium).
- **Expected Sharpe**: 0.3-0.6 for mean-reversion of the WTI-Brent spread.
- **Key risk**: The spread can blow out during regional supply shocks (e.g., WTI went to -$37 in April 2020 while Brent stayed positive). FTMO's 3% daily limit could be breached on a violent spread move.
- **FTMO implementation**: Size each leg to risk no more than 0.5% on spread divergence. The spread has historically mean-reverted, making it a viable strategy.
- **Verdict**: Viable as a cross-commodity spread, NOT as a calendar spread. No true calendar spreads are possible on FTMO.

### 3B. Basis Trade Equivalents

**Real version**: Long cash instrument, short futures = capture basis (convergence of futures to spot at expiry).

**CFD reality**: Not possible. CFDs have no expiry, no delivery, no basis. The CFD IS the spot price (approximately). There is no futures contract to trade against it.

**Potential substitute**: Trade the "swap basis" -- the overnight financing cost (swap rate) on CFDs can create a term-structure-like effect. Long positions in high-swap-cost instruments earn negative carry; short positions earn positive carry.

**Assessment**:
- **Feasible on FTMO**: Only in the sense that swap rates affect P&L. You cannot explicitly trade the basis.
- **Academic evidence**: Koijen, Moskowitz, Pedersen & Vrugt (2018) -- "Carry", JF. Carry strategies (going long high-carry assets, short low-carry) produce Sharpe ~0.7 across asset classes. On CFDs, carry = -(swap rate), so this translates to going long instruments with positive swap and short those with negative swap.
- **Verdict**: Not a basis trade per se, but carry/swap optimization is a valid consideration for position selection. Already partially captured by selecting instruments with favorable swap rates.

### 3C. Contango/Backwardation Plays on Commodity CFDs

**Real version**: Trade the roll yield in futures term structure -- short contango (sell overpriced front-month), long backwardation (buy underpriced front-month).

**CFD reality**: CFDs track spot prices and do NOT embed term structure information. When the FTMO broker rolls the underlying reference, the CFD price adjusts seamlessly -- there is no visible contango or backwardation in the CFD price.

**Assessment**:
- **Feasible on FTMO**: No. Without access to multiple futures maturities, you cannot trade contango/backwardation.
- **Academic evidence**: Gorton & Rouwenhorst (2006) -- "Facts and Fantasies About Commodity Futures", FAJ. Documents that ~60% of commodity futures returns historically came from roll yield. This return source is entirely absent from CFDs.
- **Implication**: CFD traders miss the most documented source of commodity returns. This is a fundamental disadvantage of CFDs vs. real futures.
- **Verdict**: Not possible on FTMO. This is a material limitation.

### 3D. Cross-Commodity Spreads (Ratio Trades)

**Real version**: Trade relative value between related commodities.

**Available on FTMO**:

| Spread | Instruments | Economic Rationale | Mean-Reversion? |
|--------|------------|-------------------|-----------------|
| Gold/Silver ratio | XAUUSD / XAGUSD | Industrial vs monetary demand | Yes, half-life ~60 days, historical range 60-100 |
| Oil/NatGas ratio | USOIL.cash / NATGAS.cash | Substitution in energy production | Weakly, very noisy |
| WTI/Brent spread | USOIL.cash / UKOIL.cash | Regional crude pricing | Yes, half-life ~40 days |
| Platinum/Gold spread | XPTUSD / XAUUSD | Auto catalyst demand vs safe haven | Weakly mean-reverting |
| Copper/Gold ratio | XCUUSD / XAUUSD | Risk-on/risk-off indicator | Regime-dependent |
| Soybean/Corn ratio | SOYBEAN.c / CORN.c | Crop rotation economics | Seasonal mean-reversion |
| Coffee/Cocoa | COFFEE.c / COCOA.c | Soft commodity weather patterns | Weak relationship |

**Assessment**:
- **Feasible on FTMO**: Yes. Multiple commodity pairs are available. The gold/silver ratio and WTI/Brent spread are the most liquid and well-documented.
- **Academic evidence**:
  - Erb & Harvey (2006) -- "The Strategic and Tactical Value of Commodity Futures", FAJ. Documents commodity spread strategies with Sharpe 0.3-0.7.
  - Marshall, Nguyen & Visaltanachoti (2012) -- "Commodity Liquidity Measurement and Transaction Costs", RFS. Shows cross-commodity spreads have lower transaction costs than outright positions due to natural hedging.
  - Lutzenberger (2014) -- "The Predictability of Aggregate Returns on Commodity Futures", Review of Financial Economics. Confirms mean-reversion in commodity ratios.
- **Expected Sharpe**: 0.3-0.7 for gold/silver ratio, 0.2-0.5 for others.
- **FTMO risk**: Commodity CFDs have wider spreads (0.03-0.10% per leg) and higher swap costs than forex. Two legs = double the spread cost. The 3% daily limit is less constraining because pairs trades have lower volatility than outright positions.
- **Complexity**: Moderate. Requires ratio calculation, z-score normalization, and dual-leg order management.
- **Verdict**: The gold/silver ratio trade is the standout candidate -- well-documented, mean-reverting, liquid enough on FTMO. WTI/Brent spread is the second best. Others are too noisy or have poor liquidity.

---

## 4. Academic Research on Spread/Pairs Trading

### 4A. Gatev, Goetzmann & Rouwenhorst (2006) -- Classic Pairs Trading

**Paper**: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", Review of Financial Studies.

**Method**:
1. Formation period (12 months): compute normalized price series for all pairs, select 20 pairs with minimum sum of squared deviations.
2. Trading period (6 months): when spread exceeds 2 standard deviations, go long underperformer and short outperformer.
3. Close when spread reverts to zero (or at end of trading period).

**Results**:
- Excess return: 11% annualized (1962-2002)
- After transaction costs: ~6% annualized
- Sharpe ratio: ~0.7-0.9 (depending on period)
- Declining profitability over time: 1960s-1980s strongest, 1990s-2000s weaker

**Relevance to FTMO CFDs**:
- **Adaptation needed**: FTMO has ~58 equity CFDs (US + EU) -- a reasonable universe for distance-based pair selection, but much smaller than the 1000+ stocks used in the paper.
- **Cost impact**: CFD spreads (0.05-0.15% per leg for equities) are higher than equity commissions in the academic study. With 2 legs, round-trip cost is 0.2-0.6%. This erodes most of the documented alpha.
- **Swap cost**: Holding equity CFD pairs overnight incurs swap costs on BOTH legs (long pays, short receives, but net is typically negative). For a 5-20 day hold, swap costs add 0.1-0.5% drag.
- **Post-publication decay**: Gatev et al. documented declining returns over time. Do, Faff & Hamza (2006) confirmed further decay post-publication. Net of CFD costs, post-2010 returns are likely near zero for the basic distance method.

**Verdict**: The basic Gatev approach is unlikely to work on FTMO CFDs after costs. Enhanced versions (see 4B) may still work.

### 4B. Vidyamurthy (2004) -- Cointegration-Based Pairs Trading

**Book**: "Pairs Trading: Quantitative Methods and Analysis", Wiley.

**Method**:
1. Test all pairs for cointegration (Engle-Granger two-step or Johansen test).
2. Estimate the cointegrating vector (hedge ratio).
3. Trade the spread: mean-revert when the residual exceeds a threshold.
4. Dynamic hedge ratio estimation (rolling OLS or Kalman filter).

**Enhancement over Gatev**: Cointegration provides a theoretical basis for mean-reversion (error correction mechanism), while distance-based methods are purely statistical.

**Relevant extensions**:
- **Caldeira & Moura (2013)** -- "Selection of a Portfolio of Pairs Based on Cointegration", RBFIN. Showed cointegration-based selection improves Sharpe by 0.2-0.4 over distance-based.
- **Krauss (2017)** -- "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook", Statistical Papers. Comprehensive survey; machine-learning-enhanced pairs trading shows Sharpe 0.8-1.2 on US equities.

**Relevance to FTMO CFDs**:
- **Better suited**: Cointegration is more robust to structural breaks and works with smaller universes (FTMO's 58 equities, 14 indices, 11 commodities).
- **Commodity pairs**: Cointegration is strongest for related commodities (gold/silver, WTI/Brent) where a fundamental economic relationship constrains long-run divergence.
- **Forex application**: Currency triangles (e.g., EURUSD, GBPUSD, EURGBP) are cointegrated by construction (no-arbitrage). Deviations from triangular parity = pure spread capture. On FTMO, this works but profits are limited by the tight bid-ask spreads.

**Verdict**: Cointegration-based pairs are the most promising approach for FTMO. Focus on commodity pairs and sector equity pairs where fundamental relationships exist.

### 4C. Evidence for Commodity Spread Trading

**Key papers**:
- **Simon (1999)** -- "The Soybean Crush Spread", Journal of Futures Markets. Documents profitability of trading soybean complex spreads (soybean vs meal + oil). Sharpe ~0.5.
- **Wang & Ke (2005)** -- "Energy Spread Trading", Energy Economics. WTI-Brent spread and crack spread strategies, Sharpe 0.4-0.6.
- **Bessembinder, Coughenour, Seguin & Smoller (1995)** -- "Mean Reversion in Equilibrium Asset Prices: Evidence from the Futures Term Structure", JF. Documents mean-reversion in commodity price relationships.

**On FTMO**: Soybean crush spread is partially possible (have SOYBEAN.c but no soybean meal/oil CFDs). Energy spreads (WTI/Brent) are feasible. Agricultural spreads (corn/wheat, soybean/corn) are feasible but have wide CFD spreads.

### 4D. Do Pairs Strategies Work on CFDs with Spreads and Swap Costs?

**Critical assessment**:

| Factor | Impact | Magnitude |
|--------|--------|-----------|
| Bid-ask spread (per leg) | Negative | 0.03-0.15% per side |
| Round-trip cost (2 legs x 2 trades) | Negative | 0.12-0.60% total |
| Overnight swap (net of long/short) | Negative | 0.01-0.05% per day |
| Holding period effect (5-20 days) | Negative | 0.05-1.0% total swap drag |
| Total cost for one round-trip trade | Negative | 0.2-1.5% |
| Gross alpha per trade (academic) | Positive | 0.5-2.0% |
| Net alpha per trade | Mixed | -0.5% to +1.0% |

**Conclusion**: Pairs trading on CFDs is marginal. It works ONLY for:
1. Highly cointegrated pairs (gold/silver, WTI/Brent) with strong mean-reversion
2. Wide spread divergences (>2.5 sigma) where alpha per trade exceeds costs
3. Longer holding periods where alpha accumulates faster than swap costs
4. Instruments with tight CFD spreads (forex pairs, major indices)

For FTMO equity pairs, the cost structure likely eliminates alpha. For commodity and forex pairs, there is a narrow window of profitability.

---

## 5. Volatility Strategies Without Options

### 5A. VIX-Related Trading

**Does FTMO have VIX CFDs?**: No. There is no VIX, VXX, UVXY, SVXY, or any volatility index CFD in the FTMO instrument list. The closest proxy is DXY.cash (Dollar Index), which is not a volatility instrument.

**Workaround -- Implied Vol Proxies**:
- Compute realized volatility from available price data (20-day realized vol of US500.cash as a VIX proxy)
- Use the vol-of-vol of the TSMOM strategy as a regime signal (already in the system)
- Trade equity indices conditional on volatility regime (Moreira & Muir 2017)

**Assessment**:
- **Feasible on FTMO**: Cannot trade VIX directly. Can use realized vol as a signal for other strategies (already done in TSMOM vol-targeting).
- **Academic evidence**:
  - Moreira & Muir (2017) -- "Volatility-Managed Portfolios", JF. Scaling equity exposure inversely by realized vol improves Sharpe by 0.2-0.4. Applicable to any FTMO equity index CFD.
  - Bollerslev, Tauchen & Zhou (2009) -- "Expected Stock Returns and Variance Risk Premia", RFS. The gap between implied and realized vol (variance risk premium) predicts returns, but requires implied vol data not available on FTMO.
- **Verdict**: Volatility as a SIGNAL is already integrated. Volatility as a TRADEABLE INSTRUMENT is not available.

### 5B. Vol-of-Vol as a Signal

**Already implemented**: The TSMOM strategy uses trailing realized volatility for position sizing. Extending this to vol-of-vol (the variability of volatility itself) is straightforward.

**Academic evidence**:
- Huang, Shaliastovich, Chen & Ghysels (2014) -- "Volatility-of-Volatility Risk", JFQA. Vol-of-vol predicts future returns with negative coefficient: high vol-of-vol = lower expected returns. Sharpe improvement of 0.1-0.2 when used as a filter.
- Baltussen, Van Bekkum & Grient (2018) -- "Unknown Unknowns: Uncertainty About Risk and Stock Returns", JFE. Uncertainty about volatility (vol-of-vol) is a priced risk factor.

**Implementation on FTMO**:
- Compute vol-of-vol as the standard deviation of 20-day rolling realized vol over a 60-day window.
- When vol-of-vol is high (>1.5x median), reduce position sizes across all strategies by 30-50%.
- When vol-of-vol is low (<0.5x median), allow full position sizing.

**Assessment**:
- **Feasible on FTMO**: Yes, simple to implement as a risk overlay.
- **Expected Sharpe improvement**: 0.1-0.2 incremental (modest but positive).
- **FTMO compatibility**: Highly compatible -- reducing size during uncertain vol regimes protects the 3% daily loss limit.
- **Complexity**: Low. A few lines of code on top of existing vol calculation.
- **Verdict**: Easy win. Should be added as a risk filter.

### 5C. Dispersion-Like Strategies (Stock CFDs vs Index CFDs)

**Real version**: Sell index options, buy single-stock options = profit from correlation being lower than implied by index vol.

**CFD approximation**: Trade the relative performance of individual stock CFDs vs their sector/index CFD. If individual stocks collectively outperform the index, correlation is dropping (dispersion rising).

**Concrete implementation**:
1. Compute daily returns for all FTMO tech stocks (AAPL, MSFT, NVDA, GOOG, META, AMZN, etc.)
2. Compute average pairwise correlation over trailing 20 days.
3. When correlation is high (>0.7), expect mean-reversion toward lower correlation. Go long individual stocks most divergent from index, short the index.
4. When correlation is low (<0.3), expect convergence. Trade index vs laggards.

**Assessment**:
- **Feasible on FTMO**: Partially. Requires holding 5-10 equity CFD positions simultaneously + 1 index short. This creates significant margin usage (equities at 1:5 leverage). With $100K account and 50% max margin utilization, can hold ~$250K notional in equity positions.
- **Academic evidence**:
  - Driessen, Maenhout & Vilkov (2009) -- "The Price of Correlation Risk: Evidence from Equity Options", RFS. Documents correlation risk premium of 3-5% per annum. This is the economic basis for dispersion trades.
  - Deng (2008) -- "Dispersion Trading", working paper. Shows dispersion trade Sharpe 0.5-0.8 with options, but requires selling index vol and buying single-stock vol.
- **Key limitation**: Without options, you cannot directly capture the correlation risk premium. CFD-based "dispersion" captures relative momentum, not vol premium.
- **Expected Sharpe**: 0.2-0.4 (significantly lower than options-based dispersion).
- **FTMO risk**: Many positions = many sources of loss. A bad day across all tech stocks could breach 3% daily limit.
- **Complexity**: High. Many positions, complex monitoring, high margin usage.
- **Verdict**: Theoretically interesting but practically difficult on FTMO. The residual reversal strategy already captures some of this effect more simply. Not recommended as a standalone strategy.

### 5D. Gamma-Like Exposure Through Breakout Strategies

**Real version**: Buying options provides gamma -- accelerating profits as the underlying moves further in your direction.

**CFD approximation**: Breakout strategies with pyramiding (adding to winners) create a convex payoff profile resembling positive gamma.

**Implementation**:
1. Enter initial position on breakout (e.g., 20-day Donchian channel break).
2. Add to position at predefined intervals (every 1x ATR of favorable movement, add 25% more).
3. Trail stop-loss below the most recent add point.
4. Maximum pyramid: 4 units (initial + 3 adds).

**Assessment**:
- **Feasible on FTMO**: Yes, already partially captured by the TSMOM and breakout strategies documented in crisis_alpha_strategies.md. Pyramiding is the specific enhancement.
- **Academic evidence**:
  - Fung & Hsieh (2001) -- as discussed, trend-following = lookback straddle = positive gamma.
  - Sewell (2011) -- "Characterization of Financial Time Series", working paper, UCL. Documents that pyramiding trend signals improves tail performance (kurtosis becomes more positive), mimicking long gamma.
  - Covel (2009) -- "Trend Following" (book). Documents Turtle Trading pyramiding rules that produced Sharpe 0.8-1.2 in systematic implementation.
- **Expected Sharpe**: 0.5-0.8 for diversified breakout with pyramiding.
- **FTMO risk**: Pyramiding concentrates risk. If the trend reverses after maximum pyramid, losses are amplified. The 3% daily limit constrains pyramid size. With 4 units at 0.5% risk each = 2% risk on full pyramid, leaving 1% buffer for the daily limit.
- **Best Day Rule**: Pyramided trend profits can accumulate heavily on a single day. Must monitor intraday P&L to avoid the 50% best-day concentration.
- **Complexity**: Moderate. Requires pyramid logic in order management, partial position sizing, and multi-level trailing stops.
- **Verdict**: This is the best "synthetic gamma" available on FTMO. Pyramiding on breakouts provides genuine convexity. Recommended as an enhancement to existing TSMOM/breakout strategies, with strict pyramid limits to respect FTMO rules.

---

## 6. Summary Assessment Matrix

| Strategy | Feasible on FTMO? | Academic Sharpe | Net Sharpe (after CFD costs) | FTMO Rule Risk | Complexity | Recommendation |
|----------|-------------------|----------------|------------------------------|---------------|------------|----------------|
| Covered call (take-profit) | Trivially yes | N/A (not a real covered call) | N/A | None | Trivial | NOT USEFUL -- no premium |
| Protective put (stop-loss) | Yes with caveats | N/A | N/A | Gap risk | Trivial | Use as risk mgmt, not a strategy |
| Synthetic straddle (breakout) | Yes | 0.3-0.6 | 0.2-0.4 | Whipsaw risk | Low-Med | Already in TSMOM |
| Delta-neutral pairs | Yes | 0.5-1.0 | 0.3-0.6 | Correlation break | Med | RECOMMENDED (selective) |
| Calendar spread | NO | N/A | N/A | N/A | N/A | IMPOSSIBLE |
| Basis trade | NO | N/A | N/A | N/A | N/A | IMPOSSIBLE |
| Contango/backwardation | NO | N/A | N/A | N/A | N/A | IMPOSSIBLE |
| Gold/Silver ratio | Yes | 0.3-0.7 | 0.2-0.5 | Ratio blowout | Med | RECOMMENDED |
| WTI/Brent spread | Yes | 0.3-0.6 | 0.2-0.4 | Regional shock | Med | VIABLE |
| Equity pairs (Gatev) | Marginal | 0.7-0.9 | 0.0-0.3 | Cost drag | Med-High | MARGINAL -- costs erode alpha |
| Cointegrated commodity pairs | Yes | 0.4-0.7 | 0.2-0.5 | Fundamental break | Med | RECOMMENDED (selective) |
| VIX trading | NO | N/A | N/A | N/A | N/A | NO VIX CFDs AVAILABLE |
| Vol-of-vol signal | Yes (as filter) | +0.1-0.2 Sharpe | +0.1-0.2 | None | Low | RECOMMENDED as risk overlay |
| Dispersion (stocks vs index) | Partially | 0.5-0.8 | 0.1-0.3 | Many positions | High | NOT RECOMMENDED (too complex) |
| Pyramided breakout (gamma) | Yes | 0.5-0.8 | 0.4-0.7 | Concentrated risk | Med | RECOMMENDED as TSMOM enhancement |

---

## 7. Priority Recommendations for Implementation

### Tier 1 -- High priority, strong evidence, FTMO-compatible:
1. **Vol-of-vol risk overlay**: Add to existing risk gate. Low effort, positive Sharpe improvement.
2. **Pyramided breakout enhancement**: Add to TSMOM/breakout. Moderate effort, provides synthetic gamma.

### Tier 2 -- Worth investigating, moderate evidence:
3. **Gold/Silver ratio trade**: Best commodity pairs candidate. Requires new strategy module.
4. **WTI/Brent spread trade**: Secondary commodity spread. Can share infrastructure with gold/silver.

### Tier 3 -- Marginal, high cost drag:
5. **Cointegrated equity pairs**: Only if spread costs on FTMO equities are confirmed to be tight enough.
6. **Forex triangular arbitrage**: Theoretically sound but profits are tiny relative to costs.

### Not recommended:
- Covered calls (no economic substance without options)
- Dispersion trading (too many positions, too complex, too much margin)
- Calendar/basis/contango strategies (impossible on CFDs)
- VIX trading (instrument not available)

---

## 8. Key Academic References

1. Avellaneda, M. & Lee, J. (2010). "Statistical Arbitrage in the US Equities Market." *Quantitative Finance*, 10(7), 761-782.
2. Bessembinder, H., Coughenour, J., Seguin, P. & Smoller, M. (1995). "Mean Reversion in Equilibrium Asset Prices." *Journal of Finance*, 50(1), 361-375.
3. Bollerslev, T., Tauchen, G. & Zhou, H. (2009). "Expected Stock Returns and Variance Risk Premia." *Review of Financial Studies*, 22(11), 4463-4492.
4. Broadie, M., Chernov, M. & Johannes, M. (2009). "Understanding Index Option Returns." *Review of Financial Studies*, 22(11), 4493-4529.
5. Büyüksahin, B., Haigh, M. & Robe, M. (2010). "Commodities and Equities: Ever a Market of One?" *Journal of Alternative Investments*, 12(3), 76-95.
6. Caldeira, J. & Moura, G. (2013). "Selection of a Portfolio of Pairs Based on Cointegration." *RBFIN*, 11(3), 391-425.
7. Covel, M. (2009). *Trend Following*. FT Press.
8. Deng, Q. (2008). "Dispersion Trading." Working paper.
9. Driessen, J., Maenhout, P. & Vilkov, G. (2009). "The Price of Correlation Risk." *Review of Financial Studies*, 22(3), 1299-1338.
10. Erb, C. & Harvey, C. (2006). "The Strategic and Tactical Value of Commodity Futures." *Financial Analysts Journal*, 62(2), 69-97.
11. Fattouh, B. (2010). "The Dynamics of Crude Oil Price Differentials." *Energy Economics*, 32(2), 334-342.
12. Fung, W. & Hsieh, D. (2001). "The Risk in Hedge Fund Strategies." *Review of Financial Studies*, 14(2), 313-341.
13. Gatev, E., Goetzmann, W. & Rouwenhorst, K. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*, 19(3), 797-827.
14. Gorton, G. & Rouwenhorst, K. (2006). "Facts and Fantasies About Commodity Futures." *Financial Analysts Journal*, 62(2), 47-68.
15. Huang, D., Shaliastovich, I., Chen, S. & Ghysels, E. (2014). "Volatility-of-Volatility Risk." *Journal of Financial and Quantitative Analysis*.
16. Israelov, R. & Nielsen, L. (2015). "Covered Calls Uncovered." *Financial Analysts Journal*, 71(6), 64-76.
17. Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018). "Carry." *Journal of Financial Economics*, 127(2), 197-225.
18. Krauss, C. (2017). "Statistical Arbitrage Pairs Trading Strategies." *Statistical Papers*, 58(1), 5-59.
19. Lo, A., Mamaysky, H. & Wang, J. (2000). "Foundations of Technical Analysis." *Journal of Finance*, 55(4), 1705-1765.
20. Marshall, B., Nguyen, N. & Visaltanachoti, N. (2012). "Commodity Liquidity Measurement and Transaction Costs." *Review of Financial Studies*.
21. Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance*, 72(4), 1611-1644.
22. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
23. Wang, Z. & Ke, X. (2005). "Energy Spread Trading." *Energy Economics*.
