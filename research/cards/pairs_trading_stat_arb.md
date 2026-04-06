# Research Card: Pairs Trading / Statistical Arbitrage

**Date**: 2026-04-03
**Status**: RESEARCH COMPLETE -- VERDICT: CONDITIONAL ADD (10-15% risk budget)
**Hypothesis**: Adding a cointegration-based pairs trading sleeve to TSMOM can reduce MDL breach rate and improve FTMO pass rate by exploiting mean-reversion alpha that is negatively correlated with momentum drawdowns.
**Economic mechanism**: Cross-sectional relative mispricing; law of one price; temporary divergence of fundamentally linked assets.

---

## 1. Classic Pairs Trading: Academic Evidence

### 1.1 Gatev, Goetzmann & Rouwenhorst (2006)

**Paper**: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", *Review of Financial Studies* 19(3), 797-827.

**Exact results (1962-2002, US equities)**:
- Average excess return: **11% annualized** (top 5 pairs, fully invested)
- Average excess return: **1.31% per 6-month trading period** (conservative, top 20 pairs)
- Sharpe ratio: approximately **0.35-0.55** (gross), depending on pair selection tier
- After estimated transaction costs (1-way trip of ~50 bps): returns reduced by ~50%, Sharpe falls to **~0.2-0.3**
- 89% of formation periods found at least one pair that opened and closed profitably
- Average holding period per trade: **~3.5 months** (pairs converge slowly)
- Win rate: approximately **63%** of pairs that opened also closed profitably

**Method (the "distance method")**:
1. **Formation period (12 months)**: Normalize all stock prices to start at $1. Compute sum of squared deviations (SSD) between all pairs. Rank by minimum SSD. Select top 5/20 pairs.
2. **Trading period (6 months)**: When normalized price spread exceeds 2 historical standard deviations, open long the underperformer / short the outperformer. Close when spread returns to zero. If spread does not converge by end of trading period, close at market.

**Key mathematical formulation**:
```
Formation:
  P_i(t) = price_i(t) / price_i(0)   [normalized to $1]
  SSD(i,j) = sum_{t=1}^{T} (P_i(t) - P_j(t))^2
  Select pairs with minimum SSD

Trading:
  Spread(t) = P_i(t) - P_j(t)
  Open when |Spread(t) - mean(Spread)| > 2 * std(Spread)
  Close when Spread(t) crosses mean(Spread)
```

### 1.2 Do & Faff (2010, 2012) -- Profitability Decline

**Do & Faff (2010)**: "Does Simple Pairs Trading Still Work?", *Financial Analysts Journal* 66(4), 83-95.

- Replicated Gatev et al. method on 1962-2009 US data
- **Pre-2000**: excess returns of ~1.0% per 6-month period (consistent with Gatev)
- **Post-2002**: excess returns declined to **~0.3-0.4%** per 6-month period
- **Post-2009**: marginal or zero after transaction costs
- Attributed decline to: increased hedge fund competition, lower arbitrage limits, electronic trading speed

**Do & Faff (2012)**: "Are Pairs Trading Profits Robust to Trading Costs?", *Journal of Financial Research* 35(2), 261-287.

- At realistic institutional transaction costs (10-30 bps per side), post-2002 pairs trading profits are **statistically indistinguishable from zero**
- Only the top 1-5 pairs maintain marginal profitability
- Conclusion: **the simple distance method is largely arbitraged away**

### 1.3 Krauss (2017) -- Comprehensive Review

**Paper**: "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook", working paper (subsequently widely cited).

Key findings from the review of 100+ papers:
- **Distance method**: declining post-2000, largely dead post-2010
- **Cointegration method**: more robust than distance, Sharpe ~0.3-0.7 depending on market and period
- **Time-series (OU) approach**: best academic results, Sharpe ~0.5-1.0 in-sample, but sensitive to parameter estimation
- **Machine learning approaches**: promising but high overfitting risk
- **Cross-asset pairs (FX, commodities)**: less studied but potentially less crowded than equities
- **Key insight**: "Pairs trading profits are compensation for convergence risk" -- they are NOT risk-free

### 1.4 Post-2020 Realistic Performance

Based on synthesis of Krauss (2017), Liu et al. (2020), and subsequent work:

- **Equity pairs (US large-cap), daily data, after costs**: Sharpe **0.1-0.4** (net)
- **FX pairs (G10)**: Sharpe **0.2-0.5** (lower costs help)
- **Commodity pairs**: Sharpe **0.3-0.6** (structural relationships more stable)
- **With Kalman filter dynamic hedging**: add ~0.1-0.2 Sharpe vs static OLS
- **Multi-pair portfolio (10-20 pairs)**: diversification brings Sharpe to **0.4-0.7** net

**Honest assessment**: A well-implemented cointegration system on a diversified set of pairs can achieve Sharpe 0.3-0.5 net of realistic costs post-2020. This is modest but valuable as a diversifier.

---

## 2. Cointegration-Based Pairs Trading

### 2.1 Theoretical Foundation

**Engle & Granger (1987)**: "Co-Integration and Error Correction: Representation, Estimation, and Testing", *Econometrica* 55(2), 251-276.

Two time series X(t) and Y(t), each I(1) (unit root), are cointegrated if there exists a linear combination:

```
Y(t) - beta * X(t) = epsilon(t)
```

where epsilon(t) is I(0) (stationary). Beta is the cointegrating coefficient (hedge ratio).

**Two-step Engle-Granger procedure**:
1. Estimate beta by OLS: `Y(t) = alpha + beta * X(t) + epsilon(t)`
2. Test residuals epsilon(t) for stationarity using Augmented Dickey-Fuller (ADF) test
3. If ADF rejects unit root (p < 0.05), the pair is cointegrated

**Johansen (1991)** multivariate extension:
- Tests for cointegration rank among N variables simultaneously
- Uses trace test and maximum eigenvalue test
- Can find multiple cointegrating vectors in a system of N > 2 assets
- More powerful than pairwise Engle-Granger when testing 3+ assets

### 2.2 Vidyamurthy (2004)

**Book**: *Pairs Trading: Quantitative Methods and Analysis*, Wiley.

Key practical contributions:
- Formalized the cointegration approach for trading (vs. Gatev's distance approach)
- Introduced the concept of "spread" as the error-correction term
- Recommended z-score-based entry/exit on the residual series
- Noted that hedge ratio should be re-estimated periodically (not static)
- Warned that cointegration can break down (regime changes, M&A, fundamental shifts)

### 2.3 Likely Cointegrated Pairs in the FTMO 45-Stock Universe (US + EU)

**High probability of cointegration** (structural/economic linkage):

| Pair | Sector | Rationale | Expected Half-Life |
|------|--------|-----------|-------------------|
| CVX / XOM | Energy | Both integrated oil majors, 90%+ revenue overlap | 5-15 days |
| JPM / BAC | Financials | Both money-center banks, driven by yield curve, credit cycle | 8-20 days |
| GOOG / META | Tech/Advertising | Both ad-revenue dependent, similar macro sensitivity | 10-25 days |
| MSFT / GOOG | Tech | Both cloud + enterprise, but less tight than GOOG/META | 15-30 days |
| BA / RTX | Industrials/Defense | Both aerospace/defense, government contract cycle | 10-25 days |
| LMT / RTX | Defense | Both pure defense contractors, correlated order book | 8-20 days |
| AAPL / MSFT | Tech | Mega-cap tech, but increasingly divergent business models | 15-40 days |
| KO / MCD | Consumer Staples | Both defensive consumer, similar macro beta | 10-25 days |
| BMW / MBG | EU Auto | Both German luxury auto, same regulatory/macro environment | 5-15 days |
| ALVG / DBKGn | EU Financials | Both German financial, ECB rate sensitivity | 8-20 days |
| TTE / XOM | Global Energy | Both supermajor integrated oil | 10-20 days |

**Moderate probability** (same sector but divergent business models):
| Pair | Notes |
|------|-------|
| NVDA / AMD | Both semiconductors but NVDA has unique AI positioning -- may break |
| NFLX / DIS | Both streaming but very different business mixes |
| V / JPM | Payments vs banking -- weak structural link |
| JNJ / PFE | Both pharma but different pipeline risk |

**Low probability / likely NOT cointegrated**:
- TSLA / any auto stock (TSLA trades as a tech/momentum stock)
- GME / any stock (meme stock dynamics destroy cointegration)
- MSTR / any stock (effectively a levered BTC vehicle)
- Cross-sector pairs generally (AAPL / JPM, etc.)

### 2.4 Half-Life of Mean Reversion

The half-life is the expected time for the spread to revert halfway to its mean:

```
Half-life = ln(2) / theta
```

where theta is the mean-reversion speed parameter from the OU process (see Section 3c).

**Typical values from academic literature and practitioner experience**:

| Pair Type | Half-Life Range | Tradeable? |
|-----------|----------------|------------|
| Same-subsector equities (CVX/XOM) | 5-15 trading days | YES - ideal |
| Same-sector equities (JPM/BAC) | 8-25 trading days | YES |
| Cross-sector equities | 30-90+ trading days | Marginal to NO |
| FX carry pairs | 10-30 trading days | YES |
| Commodity substitutes (WTI/Brent) | 2-8 trading days | YES - fast |

**Rule of thumb**: Only trade pairs with half-life between 5 and 30 trading days. Below 5 = transaction costs dominate. Above 30 = capital is tied up too long, cointegration may break before convergence.

### 2.5 Stability of Cointegration Relationships

**Critical finding**: Cointegration relationships are NOT permanent. They break down.

Evidence:
- **Bossaerts (1988)**: Cointegration among stock prices is much less stable than among economic time series
- **Huck & Afawubo (2015)**: ~40-50% of pairs that pass cointegration tests in one 12-month window fail the test in the next 12-month window
- **Clegg & Krauss (2018)**: Rolling cointegration tests show that only ~25-35% of equity pairs maintain cointegration for >2 consecutive years

**Causes of breakdown**:
1. M&A activity (one stock gets acquired -- spread blows out permanently)
2. Fundamental business divergence (e.g., one company pivots to AI, the other doesn't)
3. Regulatory change affecting one but not the other
4. Earnings surprise creating permanent re-rating
5. Index inclusion/exclusion changing ownership base

**Implication for FTMO**: Must re-test cointegration at least monthly. Must have hard stop-losses in case of permanent breakdown. A pair that was cointegrated last quarter may not be cointegrated this quarter.

---

## 3. Mathematical Models for Pairs Trading

### 3a. Distance Method (Gatev et al.)

**Formation Period** (length F, typically 252 trading days):

```
1. For each stock i, normalize: P_i(t) = S_i(t) / S_i(t_0)
2. For each pair (i,j), compute:
     SSD(i,j) = (1/F) * sum_{t=1}^{F} [P_i(t) - P_j(t)]^2
3. Rank all C(N,2) pairs by SSD ascending
4. Select top K pairs (typically K = 5 or 20)
5. Compute formation-period statistics:
     mu_ij = mean(P_i(t) - P_j(t))
     sigma_ij = std(P_i(t) - P_j(t))
```

**Trading Period** (length T, typically 126 trading days):

```
Spread(t) = P_i(t) - P_j(t)

Entry Rules:
  IF Spread(t) > mu_ij + 2 * sigma_ij:
    Short stock i, Long stock j (spread is too wide, expect convergence)
  IF Spread(t) < mu_ij - 2 * sigma_ij:
    Long stock i, Short stock j

Exit Rules:
  Close when Spread(t) crosses mu_ij (mean reversion complete)
  OR close at end of trading period (forced liquidation)

Position Sizing:
  Equal dollar amounts: $1 long / $1 short (dollar-neutral)
```

**Strengths**: Simple, no distributional assumptions, no parameter estimation beyond mean/std.
**Weaknesses**: Ignores dynamic hedge ratios, doesn't account for non-stationarity, largely arbitraged away post-2010.

### 3b. Cointegration Method (Engle-Granger)

**Step 1: Estimate hedge ratio via OLS**

```
Y(t) = alpha + beta * X(t) + epsilon(t)

where:
  Y(t) = log price of stock Y at time t
  X(t) = log price of stock X at time t
  beta  = hedge ratio (units of X to short per unit of Y long)
  alpha = intercept (long-run spread level)
  epsilon(t) = residual (the tradeable spread)

Estimation: OLS on rolling window (typically 60-252 days)
```

**Step 2: Test residuals for stationarity (ADF test)**

```
ADF regression: delta(epsilon_t) = gamma * epsilon_{t-1} + sum_{k=1}^{p} phi_k * delta(epsilon_{t-k}) + u_t

H0: gamma = 0  (unit root, NOT cointegrated)
H1: gamma < 0  (stationary, IS cointegrated)

Reject H0 if ADF test statistic < critical value:
  1% CV: -3.90  (for 2-variable Engle-Granger, not standard ADF tables!)
  5% CV: -3.34
  10% CV: -3.04

IMPORTANT: Use Engle-Granger critical values, NOT standard ADF tables.
The critical values are more negative because the residuals are estimated, not observed.
```

**Step 3: Trade the residual using z-scores**

```
z(t) = (epsilon(t) - mean(epsilon)) / std(epsilon)

Entry Rules:
  Long spread  (long Y, short beta*X) when z(t) < -2.0
  Short spread (short Y, long beta*X)  when z(t) > +2.0

Exit Rules:
  Close long  when z(t) > -0.5  (or crosses 0)
  Close short when z(t) < +0.5  (or crosses 0)

Stop-Loss:
  Close long  if z(t) < -4.0  (spread blowing out, cointegration breaking)
  Close short if z(t) > +4.0

Position Sizing (dollar-neutral):
  Long leg:  $N in stock Y
  Short leg: $N * beta in stock X
  Net dollar exposure: ~zero
  Net beta exposure: ~zero (if same sector)
```

**Typical z-score thresholds from literature**:

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Entry | +/-2.5 | +/-2.0 | +/-1.5 |
| Exit | +/-0.5 | +/-0.0 | +/-0.75 |
| Stop-loss | +/-4.0 | +/-3.5 | +/-3.0 |

**Trade-off**: Wider entry = fewer trades, higher win rate, lower total return. Narrower entry = more trades, lower win rate, more transaction costs.

### 3c. Ornstein-Uhlenbeck (OU) Process

The spread S(t) is modeled as a continuous-time mean-reverting process:

```
dS(t) = theta * (mu - S(t)) * dt + sigma * dW(t)

Parameters:
  theta = mean-reversion speed (higher = faster reversion)
  mu    = long-run equilibrium level of the spread
  sigma = volatility of the spread
  W(t)  = standard Brownian motion
```

**Discrete-time estimation (from daily data)**:

```
The OU process has an exact discrete solution:
  S(t+1) = S(t) * exp(-theta*dt) + mu * (1 - exp(-theta*dt)) + eta(t)

This is equivalent to an AR(1) model:
  S(t+1) = a + b * S(t) + eta(t)

where:
  b = exp(-theta * dt)        =>  theta = -ln(b) / dt
  a = mu * (1 - b)            =>  mu = a / (1 - b)
  Var(eta) = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)

Estimation: OLS regression of S(t+1) on S(t)
  => get a_hat, b_hat, sigma_eta_hat
  => theta_hat = -ln(b_hat) / dt     (dt = 1/252 for daily)
  => mu_hat = a_hat / (1 - b_hat)
  => sigma_hat = sigma_eta_hat * sqrt(2 * theta_hat / (1 - b_hat^2))
```

**Half-life of mean reversion**:

```
Half-life = ln(2) / theta

In terms of the AR(1) coefficient b:
  Half-life = -ln(2) / ln(b)    [in units of dt]

Example: b = 0.95 daily => theta = -ln(0.95)/1 = 0.0513
         Half-life = ln(2)/0.0513 = 13.5 trading days
```

**Optimal entry/exit (under OU dynamics)**:

Bertram (2010), "Analytic Solutions for Optimal Statistical Arbitrage Trading":

```
Optimal entry and exit thresholds for a symmetric OU strategy:
  Entry: |S(t) - mu| > sigma / sqrt(2*theta) * sqrt(2*ln(sigma/(c*sqrt(2*pi*theta))))
  Exit:  S(t) = mu  (at the mean)

where c = round-trip transaction cost per unit of spread

Simplified practical rule:
  Entry when S is approximately 1.5-2.0 sigma_spread from mu
  Exit when S crosses mu
  
The exact optimal thresholds depend on theta, sigma, and transaction costs.
Faster mean-reversion (higher theta) => tighter entry thresholds are optimal.
Higher transaction costs => wider entry thresholds are optimal.
```

**Kalman Filter for Dynamic Hedge Ratio**:

Instead of static OLS beta, estimate beta dynamically:

```
State equation:   beta(t) = beta(t-1) + w(t),    w(t) ~ N(0, Q)
Observation eq:   Y(t) = alpha + beta(t)*X(t) + v(t),  v(t) ~ N(0, R)

Kalman Filter recursion:
  Predict:
    beta_pred(t) = beta(t-1)           [state prediction]
    P_pred(t) = P(t-1) + Q            [covariance prediction]
  
  Update:
    Innovation: e(t) = Y(t) - alpha - beta_pred(t) * X(t)
    Innovation variance: F(t) = X(t)^2 * P_pred(t) + R
    Kalman gain: K(t) = P_pred(t) * X(t) / F(t)
    beta(t) = beta_pred(t) + K(t) * e(t)
    P(t) = (1 - K(t)*X(t)) * P_pred(t)

Tuning:
  Q = state noise variance (controls how fast beta can change)
    - Small Q (~1e-6): beta changes slowly, more like OLS
    - Large Q (~1e-3): beta adapts quickly, more noise
  R = observation noise variance (spread around the hedge ratio)
    - Estimate from OLS residual variance as starting point

The Kalman filter automatically adapts the hedge ratio as the
relationship between the two stocks evolves over time.
```

**Advantages of Kalman filter over rolling OLS**:
- Smoother transitions (no window edge effects)
- Properly handles time-varying relationships
- Provides uncertainty estimates for the hedge ratio
- Widely used in practice (Elliott, van der Hoek & Malcolm, 2005)

### 3d. Copula-Based Approach

**Concept**: Model each stock's marginal distribution separately, then join them via a copula to model the dependence structure.

```
Step 1: Transform each stock's returns to uniform marginals
  U(t) = F_X(X(t))    [CDF transform of stock X returns]
  V(t) = F_Y(Y(t))    [CDF transform of stock Y returns]
  
  Where F_X, F_Y are estimated empirically or parametrically
  (e.g., Student-t marginals)

Step 2: Fit a copula C(u,v) to the pair (U,V)
  Common choices:
  - Gaussian copula:  C(u,v) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)
  - Student-t copula: accounts for tail dependence
  - Clayton copula: asymmetric lower-tail dependence
  - Gumbel copula: asymmetric upper-tail dependence

Step 3: Compute conditional probabilities
  P(V < v | U = u) = partial C / partial u  evaluated at (u,v)
  
  This gives the "expected" quantile of stock Y given where stock X is.

Step 4: Trading rule
  If P(V < v | U = u) < 0.05:
    Stock Y is unusually low relative to X => Long Y, Short X
  If P(V < v | U = u) > 0.95:
    Stock Y is unusually high relative to X => Short Y, Long X
  Exit when conditional probability returns to 0.3-0.7 range
```

**References**: 
- Liew & Wu (2013), "Pairs Trading: A Copula Approach", *Journal of Derivatives and Hedge Funds*
- Xie, Liew & Wu (2016), "Pairs Trading with Copulas"

**Practical assessment for daily CFD data**:
- **Feasibility**: LOW to MODERATE. Copula estimation requires substantial data (500+ observations minimum) and is sensitive to the choice of copula family.
- **Advantage over cointegration**: Can capture nonlinear dependence, works for non-cointegrated pairs.
- **Disadvantage**: More parameters to estimate, higher model risk, harder to interpret economically.
- **Verdict for FTMO**: Likely not worth the complexity. The cointegration/OU approach is simpler, more robust, and better understood. Copulas add implementation risk without clear alpha improvement for daily equity CFD pairs.

---

## 4. Pairs Trading on CFDs Specifically

### 4.1 CFD Spread Costs

Typical FTMO MT5 spreads for equity CFDs:

| Pair Leg | Typical Spread | In bps |
|----------|---------------|--------|
| AAPL | 0.10-0.30 | 5-15 bps |
| MSFT | 0.10-0.30 | 5-15 bps |
| XOM | 0.05-0.15 | 5-10 bps |
| CVX | 0.05-0.15 | 5-10 bps |
| JPM | 0.05-0.15 | 5-10 bps |
| GOOG | 0.30-0.80 | 10-20 bps |
| META | 0.30-0.80 | 10-20 bps |
| NVDA | 0.20-0.50 | 10-20 bps |

**Round-trip cost for a pair trade (both legs, entry + exit)**:
```
Total cost = 2 legs * 2 trips * avg_spread_bps
           = 4 * 10 bps = 40 bps per complete round-trip (typical)
           
For high-spread pairs (GOOG/META): 4 * 15 bps = 60 bps
For low-spread pairs (CVX/XOM):    4 * 7 bps  = 28 bps
```

**Impact on profitability**:
If a typical pair trade earns 80-150 bps gross (academic average), and costs 30-60 bps round-trip, that leaves **20-120 bps net per trade**. This is thin but positive for the best pairs.

### 4.2 Overnight Swap Costs

For a pairs trade (long one equity CFD, short another):

```
Long leg swap:  typically negative (you pay financing, ~5-8% annualized)
Short leg swap: typically positive (you receive rebate, ~1-3% annualized)
Net swap per day: -(5-8%) + (1-3%) = net cost of ~3-6% annualized
                  = ~1.2-2.4 bps per day

For average 15-day holding period: 18-36 bps swap cost
```

**The swaps do NOT cancel out** because:
- Long side: you pay the broker's financing rate (benchmark + markup)
- Short side: you receive the broker's lending rate (benchmark - markup)
- The markup asymmetry creates a net drag of ~3-6% annualized

**For FTMO specifically**: Swap rates are on the symbol_info. The net carry cost is a meaningful drag on pairs with long holding periods. Prefer pairs with half-life < 15 days to minimize swap impact.

### 4.3 Leverage Effects

FTMO equity CFD leverage: 1:5 (20% margin per position).

```
For a dollar-neutral pair ($5,000 long + $5,000 short on $100K account):
  Margin used: $5,000/5 + $5,000/5 = $2,000 (2% of account)
  
For 5 simultaneous pairs at $5,000 per leg:
  Margin used: 5 * $2,000 = $10,000 (10% of account)
  
For 10 simultaneous pairs at $3,000 per leg:
  Margin used: 10 * $1,200 = $12,000 (12% of account)
```

**Leverage is helpful** because:
1. Pairs trading returns are low (0.5-2% per trade on notional) -- leverage amplifies to meaningful account returns
2. Dollar-neutral means net market exposure is ~zero, so leverage doesn't amplify market risk
3. 50% margin utilization limit (per ftmo_rules.yaml) allows running 20-25 pairs at moderate size

**Leverage is dangerous** because:
1. A pair blowout (one leg gaps 10%) is amplified -- but this is bounded by position size, not leverage itself
2. Must size conservatively: each pair should risk no more than 0.3-0.5% of account

---

## 5. Correlation with TSMOM

### 5.1 Academic Evidence on Momentum vs. Mean-Reversion Correlation

**Gatev et al. (2006)**: Noted that pairs trading profits are "highest following market declines and during periods of high dispersion" -- exactly when cross-sectional and time-series momentum tend to underperform.

**Asness, Moskowitz & Pedersen (2013)**: "Value and Momentum Everywhere", *Journal of Finance*:
- Documented that value (a form of mean reversion) and momentum have **negative correlation of approximately -0.4 to -0.6** across asset classes
- Combining value and momentum produces Sharpe improvements of 40-60% vs either alone

**Baltas & Kosowski (2013)**: "Momentum Strategies in Futures Markets and Trend-Following Funds":
- Mean-reversion strategies act as a natural hedge to momentum crash risk
- Momentum crashes (2009 Q1, 2020 March) coincide with mean-reversion outperformance

**Expected correlation between pairs trading and TSMOM**:
```
Empirical estimates from literature:
  rho(pairs, TSMOM) = -0.2 to -0.4 (most estimates)
  
  During normal markets:   rho ~ -0.1 to -0.2 (low negative)
  During momentum crashes: rho ~ -0.5 to -0.7 (strongly negative = diversifying)
  During trending markets: rho ~ 0.0 to +0.1  (uncorrelated -- pairs generates small steady gains)
```

### 5.2 Impact on FTMO Pass Rate

This negative correlation is precisely what we need for FTMO:

```
Scenario Analysis (simplified):

Current TSMOM-only:
  - Pass rate: 42.1%
  - Main failure mode: MDL breach during momentum reversals (whipsaw days)
  - Typical MDL breach: TSMOM loses 3%+ when trend reverses sharply

With 85% TSMOM + 15% pairs:
  - On TSMOM drawdown days, pairs likely gains 0.1-0.3%
  - This reduces the -3.0% day to approximately -2.7% to -2.9%
  - Small reduction, but MDL breaches are clustered near -3.0%, so even
    0.2-0.3% buffer converts some breach days to near-miss days
  
  Estimated impact:
  - MDL breach rate reduction: ~5-10% (relative, not absolute)
  - Pass rate improvement: +2 to +5 percentage points
  - New estimated pass rate: 44-47%
```

**Caveat**: The pairs component also introduces its own risk of blowout (see Section 6). If one pair blows up on the same day TSMOM draws down, the combination is WORSE. This is why position sizing per pair must be conservative (0.3-0.5% risk per pair).

### 5.3 Could 10-20% Weight Reduce MDL Breaches?

Yes, but the effect is moderate, not transformative:

```
Portfolio daily return volatility (simplified):
  sigma_p^2 = w1^2 * sigma_TSMOM^2 + w2^2 * sigma_pairs^2 + 2*w1*w2*rho*sigma_TSMOM*sigma_pairs

  With w1=0.85, w2=0.15, rho=-0.3:
  sigma_TSMOM = 1.0% daily, sigma_pairs = 0.5% daily

  sigma_p^2 = 0.85^2 * 1.0^2 + 0.15^2 * 0.5^2 + 2*0.85*0.15*(-0.3)*1.0*0.5
            = 0.7225 + 0.005625 - 0.03825
            = 0.689875
  sigma_p = 0.830%

  Reduction: 1.0% -> 0.83% daily vol = 17% reduction in portfolio volatility
  
  At the -3.0% MDL boundary (3.0 sigma event for TSMOM-only):
  New distance: 3.0 / 0.83 = 3.61 sigma (harder to breach)
  P(breach) drops from ~0.13% to ~0.015% per day (rough, Gaussian approx)
```

This is a meaningful improvement in tail risk, though the Gaussian approximation understates the true tail probabilities.

---

## 6. FTMO-Specific Considerations

### 6.1 Can a Pair Blow Up > 3% in One Day?

**YES. This is the #1 risk.**

Scenarios where a "dollar-neutral" pair can have large one-day loss:

1. **M&A announcement**: Stock A announces acquisition of Stock B. A drops 5%, B jumps 30%. If you're long A / short B, you lose ~17.5% on the pair notional. At 2% account allocation per pair, that's 0.35% account loss. Manageable -- but at 5% allocation, it's 0.875%.

2. **Earnings surprise (divergent)**: Stock A beats earnings (+8%), Stock B misses (-6%). If you're short A / long B, that's 14% pair loss. At 2% allocation: 0.28%.

3. **FDA/regulatory event**: One pharma stock gets approval, the other gets rejection. Can easily be 15-20% divergence.

4. **Overnight gap**: US equity CFDs don't trade overnight. A pair can gap 3-5% on after-hours news.

**Risk mitigation**:
```
Hard rules for pairs trading within FTMO:
1. Max notional per pair: 3% of account per leg (6% gross, 0% net)
   => Max pair loss in a 15% divergence event: 0.9% of account
2. No pairs with earnings within 3 trading days
3. No pharma pairs during FDA decision windows
4. No pairs where one stock has pending M&A
5. Maximum total pairs exposure: 30% gross (15% long, 15% short)
   => Even if ALL pairs blow up simultaneously by 10%, max loss = 3.0%
6. Pre-market check: if overnight gap on any pair > 3%, close at open
```

### 6.2 Best Day Rule Compatibility

**Pairs trading is EXCELLENT for the Best Day Rule.**

```
Best Day Rule: max_day_profit / sum(positive_day_profits) <= 0.50

Pairs trading return profile:
  - Many small wins (0.1-0.3% per day when a pair converges)
  - Few large losses (pair divergence / cointegration breakdown)
  - Typical win rate: 55-65%
  - Typical payoff ratio: 0.8-1.2 (roughly symmetric)

This means profits are spread across many days, making it nearly
impossible for any single day to exceed 50% of total positive P&L.

Compare to TSMOM:
  - Few large wins (trend days: +1-3%)
  - Many small losses (chop days: -0.2 to -0.5%)
  - This CONCENTRATES profit in a few days -- Best Day Rule risk

The COMBINATION of pairs + TSMOM naturally improves Best Day compliance
because pairs adds many small positive days to the denominator.
```

### 6.3 Margin Considerations

```
Long + Short equity CFD pair at FTMO:
  Margin per leg = notional / leverage = notional / 5
  Total margin for pair = 2 * notional / 5 = 40% of per-leg notional

Example: $5,000 per leg pair
  Margin = 2 * $5,000 / 5 = $2,000

With 10 simultaneous pairs at $5,000/leg:
  Total margin = 10 * $2,000 = $20,000 (20% of $100K account)
  
FTMO margin utilization limit: 50% = $50,000
Remaining for TSMOM: $50,000 - $20,000 = $30,000

This is workable but constraining. With 5 pairs at $3,000/leg:
  Total margin = 5 * $1,200 = $6,000 (6% of account)
  Remaining for TSMOM: $44,000 -- much more comfortable
```

**Recommendation**: Run 5-8 pairs at moderate size rather than 10-20 pairs. This balances diversification against margin consumption.

### 6.4 How Many Pairs Are Tradeable?

From the 45 US+EU equity CFDs: C(45,2) = 990 possible pairs.

**Filtering pipeline**:
```
990 total pairs
  -> ~200 same-sector or economically linked (pre-filter)
  -> ~40-80 pass cointegration test (ADF p < 0.05) in any given month
  -> ~15-30 have half-life between 5-30 days
  -> ~10-15 have acceptable spread costs (cost < 40% of expected return)
  -> ~5-10 pass all filters simultaneously at any given time
```

Additionally, from cross-asset pairs:
```
FX pairs: EURUSD/GBPUSD, AUDUSD/NZDUSD, etc. -- 3-5 tradeable
Commodity pairs: USOIL/UKOIL (very strong), possibly XAUUSD/XAGUSD -- 1-3 tradeable
Index pairs: US500/US100, GER40/EU50 -- 1-3 tradeable
```

**Total tradeable universe at any given time: approximately 8-15 pairs**, of which you would trade 5-8 simultaneously.

---

## 7. Implementation Complexity

### 7.1 Code Estimate

A basic cointegration pairs trading system requires:

```
Component                              Lines of Code (Python)
---------------------------------------------------------------
Cointegration test (Engle-Granger)     40-60
ADF test wrapper                       20-30
Rolling hedge ratio (OLS)              30-40
Kalman filter hedge ratio              60-100
OU parameter estimation                40-60
Z-score computation & signal gen       30-50
Pair selection & filtering             60-100
Position sizing (dollar-neutral)       40-60
Risk management (stop-loss, earnings)  50-80
Integration with existing framework    80-120
Backtest adapter                       60-80
---------------------------------------------------------------
TOTAL                                  510-780 lines

Using statsmodels for ADF/cointegration: saves ~100 lines
Using pykalman or filterpy for Kalman: saves ~50 lines
Realistic estimate with libraries: ~400-600 lines
```

This fits within a new `src/strategies/pairs_trading.py` module.

### 7.2 Rolling Recalibration Frequency

| Parameter | Recalibration Frequency | Rationale |
|-----------|------------------------|-----------|
| Pair selection (cointegration test) | Monthly | Cointegration status changes slowly; weekly is too noisy |
| Hedge ratio (OLS) | Weekly | Balance between responsiveness and stability |
| Hedge ratio (Kalman) | Daily (automatic) | Kalman filter updates continuously; no manual recal needed |
| OU parameters (theta, mu, sigma) | Weekly | Half-life estimation needs fresh data but daily is noisy |
| Z-score mean/std | Rolling 60-day window, updated daily | Must track recent spread dynamics |
| Universe filter (earnings, M&A) | Daily | Must avoid event-risk pairs |

**Recommendation**: Use Kalman filter for hedge ratio (eliminates weekly OLS recalibration), monthly cointegration re-screening, daily z-score updates.

### 7.3 Optimal Number of Simultaneous Pairs

| # Pairs | Pros | Cons |
|---------|------|------|
| 1-2 | Simple | Concentrated risk; one blowup = big loss |
| 3-5 | Moderate diversification | Still somewhat concentrated |
| **5-8** | **Good diversification; manageable margin** | **RECOMMENDED** |
| 10-15 | High diversification | High margin usage; transaction costs; complexity |
| 20+ | Maximum diversification | Unrealistic for 45-stock universe; margin-heavy |

**Academic guidance**: Krauss (2017) notes that diversification benefits plateau at ~10-15 pairs for a typical equity universe. For our constrained 45-stock universe with 5:1 leverage, **5-8 pairs is optimal**.

---

## 8. Honest Verdict: Can Pairs Trading Push 42% Pass Rate Higher?

### 8.1 Expected Contribution

```
Standalone pairs trading performance (estimated, post-cost):
  - Sharpe: 0.3-0.5
  - Annual return: 4-8% (unlevered)
  - Max drawdown: 5-10%
  - FTMO pass rate (standalone): ~15-25% (too slow to reach 10% target alone)

As a 15% complement to TSMOM:
  - Diversification benefit: -0.2 to -0.4 correlation with TSMOM
  - MDL breach reduction: ~5-10% relative improvement
  - Best Day Rule improvement: significant (adds many small positive days)
  - Net pass rate improvement: +2 to +5 percentage points
  - Expected new pass rate: ~44-47%
```

### 8.2 What Must Be True for This to Work

1. We can find 5+ cointegrated pairs at any given time in the FTMO universe
2. Half-lives are 5-25 days (fast enough to generate P&L within challenge window)
3. CFD spreads + swaps consume < 40% of gross pair alpha
4. No catastrophic pair blowup during the challenge (M&A, earnings)
5. Cointegration relationships remain stable for at least 2-3 months

### 8.3 What Could Go Wrong

1. **CFD costs eat all alpha**: At 40 bps round-trip + 20 bps swap, the hurdle is ~60 bps per trade. If average gross return is only 80 bps, net is 20 bps -- barely worth the complexity.
2. **Cointegration instability**: In the current high-vol regime (Iran/Hormuz), correlations and cointegration relationships may be less stable. Sector dispersion is elevated.
3. **Simultaneous pair blowouts**: In a market crash, "dollar-neutral" doesn't mean risk-neutral. All long-value/short-growth pairs can lose simultaneously.
4. **Implementation risk**: Adding 400-600 lines of new code increases system complexity and debugging surface.
5. **Margin competition with TSMOM**: Using 10-20% of margin for pairs reduces TSMOM capacity.

### 8.4 Final Recommendation

**CONDITIONAL ADD at 10-15% of risk budget, with strict guard rails.**

The expected +2 to +5 pp improvement in pass rate (42% -> 44-47%) is modest but real. The improvement comes primarily from:
1. Reduced MDL breach probability (negative correlation with TSMOM)
2. Better Best Day Rule compliance (many small wins)

**However**, this should be LOWER PRIORITY than:
- Optimizing TSMOM parameters (Phase 2-3 in strat_plans.md)
- Adding pyramided breakout (Phase 4)
- Alt data overlays (Phase 5)

These are likely to have larger marginal impact on pass rate than pairs trading.

**Implementation should be gated on**: First confirming that at least 5 pairs in the FTMO universe pass the cointegration test with half-life < 25 days and positive expected return after CFD costs. Run this screen before writing any trading code.

### 8.5 Guard Rails if Implemented

```
1. Max 15% of risk budget (max 0.45% daily loss contribution)
2. Max 8 simultaneous pairs
3. Max 3% notional per leg per pair (0.6% of account per pair)
4. No pairs with earnings within 5 days
5. No pairs with pending M&A
6. Hard stop at z = 4.0 (pair divergence > 4 sigma = assume cointegration broken)
7. Monthly cointegration re-screening (drop pairs that fail, add new ones)
8. Kill switch: if pairs sleeve loses > 1.5% cumulative, halt for 5 days
9. Prefer CVX/XOM, JPM/BAC, USOIL/UKOIL as core stable pairs
10. Use Kalman filter for hedge ratio, not static OLS
```

---

## 9. Academic References (Complete)

1. **Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006)**. "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies* 19(3), 797-827.

2. **Do, B. & Faff, R. (2010)**. "Does Simple Pairs Trading Still Work?" *Financial Analysts Journal* 66(4), 83-95.

3. **Do, B. & Faff, R. (2012)**. "Are Pairs Trading Profits Robust to Trading Costs?" *Journal of Financial Research* 35(2), 261-287.

4. **Krauss, C. (2017)**. "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook." Working paper, Friedrich-Alexander-Universitat Erlangen-Nurnberg.

5. **Engle, R.F. & Granger, C.W.J. (1987)**. "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55(2), 251-276.

6. **Johansen, S. (1991)**. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59(6), 1551-1580.

7. **Vidyamurthy, G. (2004)**. *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.

8. **Bertram, W.K. (2010)**. "Analytic Solutions for Optimal Statistical Arbitrage Trading." *Physica A* 389(11), 2234-2243.

9. **Elliott, R.J., van der Hoek, J. & Malcolm, W.P. (2005)**. "Pairs Trading." *Quantitative Finance* 5(3), 271-276.

10. **Asness, C.S., Moskowitz, T.J. & Pedersen, L.H. (2013)**. "Value and Momentum Everywhere." *Journal of Finance* 68(3), 929-985.

11. **Baltas, A.N. & Kosowski, R. (2013)**. "Momentum Strategies in Futures Markets and Trend-Following Funds." Working paper, SSRN.

12. **Liew, R.Q. & Wu, Y. (2013)**. "Pairs Trading: A Copula Approach." *Journal of Derivatives and Hedge Funds* 19(1), 12-30.

13. **Huck, N. & Afawubo, K. (2015)**. "Pairs Trading and Selection Methods: Is Cointegration Superior?" *Applied Economics* 47(6), 599-613.

14. **Clegg, M. & Krauss, C. (2018)**. "Pairs Trading with Partial Cointegration." *Quantitative Finance* 18(1), 121-138.

15. **Bossaerts, P. (1988)**. "Common Non-Stationary Components of Asset Prices." *Journal of Economic Dynamics and Control* 12(2-3), 347-364.

---

## 10. Falsification Criteria

This strategy hypothesis should be REJECTED if:

1. Fewer than 3 pairs in the FTMO universe pass the cointegration test with half-life < 25 days in backtesting
2. Net-of-cost Sharpe ratio is < 0.15 on 3+ years of OOS data
3. Correlation with TSMOM is > 0 (positive, not diversifying)
4. Any single pair event causes > 1.5% account loss in backtesting
5. Adding pairs at 10-15% weight does NOT reduce MDL breach rate in FTMO simulation
6. Best Day Rule compliance does NOT improve with the pairs sleeve added
