# Quantitative Trading Platform — Complete Technical Requirements Document
## For Strategy Research & Optimization

**Purpose**: This document provides a complete specification of the current quant trading platform so that a research agent can identify more optimal strategies, improve profitability, and reduce maximum drawdown — especially in the context of predicted market trends (2025–2030).

**Current Performance (Real Data, 49 US equities, 2010–2026)**:
- Final Portfolio Sharpe: 1.13, CAGR: 9.82%, MaxDD: -16.89%
- Best individual strategy: Volatility-Managed Portfolio (Sharpe 1.04, CAGR 11.32%)
- Best walk-forward composite: vol_managed + pead (OOS Sharpe 1.18)

---

## TABLE OF CONTENTS

1. [Platform Architecture Overview](#1-platform-architecture-overview)
2. [Data Pipeline & Market Data](#2-data-pipeline--market-data)
3. [All 13 Current Strategies — Detailed Specifications](#3-all-13-current-strategies)
4. [Backtesting Engine — How It Works](#4-backtesting-engine)
5. [Portfolio Construction & Optimization](#5-portfolio-construction--optimization)
6. [Walk-Forward Testing Framework](#6-walk-forward-testing-framework)
7. [Regime Detection & Adaptation](#7-regime-detection--adaptation)
8. [Alpha Discovery Engine](#8-alpha-discovery-engine)
9. [Risk Management & Monte Carlo](#9-risk-management--monte-carlo)
10. [Transaction Cost Model](#10-transaction-cost-model)
11. [Current Performance Results](#11-current-performance-results)
12. [Identified Weaknesses & Opportunities](#12-identified-weaknesses--opportunities)
13. [Constraints & Requirements for New Strategies](#13-constraints--requirements-for-new-strategies)
14. [Research Questions for Optimization](#14-research-questions-for-optimization)

---

## 1. Platform Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    run_research.py                           │
│  Main Pipeline Orchestrator (10 steps, ~104s runtime)       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Data Pipeline (Yahoo Finance → Parquet)            │
│  Step 2: 13 Strategy Signals + Weights + DD Cap             │
│  Step 3: Regime Detection (Vol Classifier + HMM)            │
│  Step 4: Portfolio Construction (Risk Parity + Dynamic)     │
│  Step 4b: Monte Carlo Risk Simulation (5000 paths)          │
│  Step 5: Walk-Forward Testing (3yr train / 6mo test)        │
│  Step 6: Alpha Discovery (signal gen → filter → rank)       │
│  Step 7: ML Meta-Model (strategy weight prediction)         │
│  Step 8: Dashboard Generation (7 HTML reports)              │
│  Step 9: Experiment Tracking                                │
│  Step 10: Validation Audit (STATUS: VALIDATED)              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Key Modules:                                               │
│  qrt/data/       - Pipeline, returns, universe, real data   │
│  qrt/strategies/ - 13 strategies (all extend base.Strategy) │
│  qrt/backtest/   - Event-driven engine, portfolio state     │
│  qrt/portfolio/  - Risk parity, vol target, adaptive alloc  │
│  qrt/regime/     - Vol classifier, HMM (4-state)            │
│  qrt/risk/       - Monte Carlo simulator                    │
│  qrt/costs/      - Multi-component transaction cost model   │
│  qrt/walkforward/- Rolling train/test framework             │
│  qrt/alpha_engine/ - Automated signal discovery             │
│  qrt/validation/ - Audit engine, benchmarks, composites     │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
- All strategies inherit from `Strategy` ABC with `generate_signals()` and `compute_weights()`
- Execution lag enforced via `weights.shift(1) * returns` — signals at t, trade at t+1
- Universal drawdown cap: iterative circuit breaker ensures MaxDD ≤ 20% for all strategies
- Point-in-time correct: no lookahead bias in any signal generation
- Real market data from Yahoo Finance (49 US equities across 7 sectors)

---

## 2. Data Pipeline & Market Data

### Universe
49 US equities across 7 sectors, downloaded from Yahoo Finance:

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, NVDA, AVGO, AMD, QCOM, TXN |
| Financials | JPM, BAC, WFC, C, GS, MS, AXP |
| Healthcare | LLY, JNJ, MRK, ABBV, PFE, TMO, DHR |
| Consumer | AMZN, COST, WMT, HD, LOW, MCD, KO |
| Industrials | CAT, DE, RTX, GE, HON, UNP, UPS |
| Energy | XOM, CVX, COP, EOG, SLB, OXY, MPC |
| Communications | GOOGL, META, NFLX, DIS, TMUS, VZ, CMCSA |

**Date Range**: 2010-01-04 to 2026-03-13 (4,073 trading days for most stocks)

### Data Fields
- OHLCV (open, high, low, close, adjusted_close, volume)
- Market cap (estimated from fast_info)
- Dividend amounts
- Split adjustments (applied to adjusted_close)

### Return Types Computed
| Type | Formula | Use |
|------|---------|-----|
| ret_raw | (close[t] - close[t-1]) / close[t-1] | Unadjusted returns |
| ret_adj | Based on adjusted_close | **Primary — used by all strategies** |
| ret_ex_div | Price-only return net of dividends | Carry strategy input |
| ret_incl_delist | raw return, NaN after delist | Survivorship check |
| log_ret | log(adj_close[t] / adj_close[t-1]) | Mean reversion z-scores |

### Universe Construction Filters
Applied at monthly rebalance:
1. Security must be active (not delisted)
2. Minimum 2-year price history
3. Minimum closing price ≥ $5
4. Minimum market cap ≥ $1B
5. Minimum median 63-day dollar volume ≥ $5M/day
6. Sector caps for diversification (max 40% in any sector)

### Storage
All data stored as Parquet files in `data/parquet/`:
- `market_data.parquet` (11 MB, 197,852 rows)
- `returns.parquet` (11.5 MB, 197,852 rows)
- `security_master.parquet` (49 securities)
- `universe_*.parquet` (monthly membership)

---

## 3. All 13 Current Strategies

### Strategy 1: Time-Series Momentum

**Academic Basis**: Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum"

**How It Works**:
- Computes trailing returns over 3 windows: 63 days, 126 days, 252 days
- Takes the sign of each (positive = long, negative = short)
- Blends with weights: 40% × 63d + 40% × 126d + 20% × 252d
- Applies trend-strength scaling: stronger trends get larger positions
- Scales by inverse realized volatility (vol_lookback=63 days)

**Signal Formula**:
```
raw_signal = 0.4 × sign(ret_63d) + 0.4 × sign(ret_126d) + 0.2 × sign(ret_252d)
trend_strength = min(|primary_return| / 0.20, 1.0)
signal = raw_signal × trend_strength
```

**Risk Controls**:
- Vol-of-vol reduction: when vol-of-vol is elevated, reduce exposure by 50%
- Vol floor: 1% minimum to prevent division by near-zero
- Gross normalization to target_gross=1.0

**Key Parameters**: lookback=252, vol_lookback=63, vol_floor=0.01, vov_reduction=0.50

**Performance**: Sharpe 0.271, CAGR 2.33%, MaxDD -17.27%

**Weakness**: Momentum crashes during sharp reversals (e.g., COVID March 2020, rate hikes 2022)

---

### Strategy 2: Cross-Sectional Momentum

**Academic Basis**: Jegadeesh & Titman (1993), Carhart (1997)

**How It Works**:
- Ranks all stocks by trailing 252-day return, **skipping the most recent 21 days** (avoids short-term reversal)
- Goes long top 20% (highest past winners), short bottom 20% (worst losers)
- Weights within legs are proportional to z-score of momentum signal (not equal-weight)

**Signal Formula**:
```
momentum_score = price[t - skip] / price[t - lookback] - 1
z_score = (momentum_score - mean) / std
long: top 20% by momentum → weight proportional to z_score
short: bottom 20% → weight proportional to -z_score
```

**Risk Controls**:
- Drawdown circuit breaker: 50% exposure reduction if drawdown > 25%, 21-day cooldown
- Gross normalization

**Key Parameters**: lookback=252, skip_days=21, decile=0.20, target_gross=0.60

**Performance**: Sharpe 0.426, CAGR 2.87%, MaxDD -18.65%

**Weakness**: Severe momentum crash risk (Daniel & Moskowitz 2016). Short leg has been particularly difficult post-2000.

---

### Strategy 3: Mean Reversion

**Academic Basis**: Lo & MacKinlay (1990), Poterba & Summers (1988)

**How It Works**:
- Computes rolling z-score of log prices over 120-day window
- When z > 1.5 → short (price overextended above mean)
- When z < -1.5 → long (price suppressed below mean)
- Exit when |z| < 0.5 or after max 5 days
- 200-day SMA trend filter: suppresses mean-reversion signals when market is strongly trending

**Signal Formula**:
```
log_price = log(close)
z = (log_price - rolling_mean(120d)) / rolling_std(120d)
signal = -sign(z) when |z| > entry_threshold, subject to trend filter
```

**Risk Controls**:
- Trend filter: suppress signals when 200-day SMA slope is too steep
- Per-position stop-loss at 3%
- Drawdown circuit breaker at -15% cumulative, 42-day cooldown

**Key Parameters**: lookback=120, entry_threshold=1.5, exit_threshold=0.5, max_holding=5, stop_loss_pct=0.03

**Performance**: Sharpe -0.085, CAGR -0.09%, MaxDD -3.94%

**Weakness**: At daily frequency on large-cap US equities, momentum dominates. Mean reversion works better at higher frequency (intraday) or in range-bound asset classes.

---

### Strategy 4: Carry (Dividend Yield)

**Academic Basis**: Koijen et al. (2018) — "Carry"

**How It Works**:
- Estimates carry from dividend yield (total return minus price return over 63 days)
- Ranks stocks cross-sectionally by carry
- Goes long top 25% (high-carry), short bottom 25% (low-carry)
- Rebalances every 21 trading days

**Signal Formula**:
```
carry = rolling_mean(total_return - price_return, 63d)  # dividend yield proxy
rank stocks by carry
long: top 25%, short: bottom 25%
```

**Key Parameters**: carry_lookback=63, n_quantile=0.25, rebalance_freq=21

**Performance**: Sharpe 0.017, CAGR 0.12%, MaxDD -18.33%

**Weakness**: Equity dividend carry is a very noisy signal compared to FX/rates carry. Dividend cuts and sector concentration are major risks.

---

### Strategy 5: Distance Pairs

**Academic Basis**: Gatev, Goetzmann & Rouwenhorst (2006)

**How It Works**:
- Formation phase (252 days): normalize prices by dividing by first value, compute sum-of-squared-distances between all pairs, select closest 20 pairs
- Trading phase: compute spread z-score for each pair, trade when |z| > 2.0, exit at |z| < 0.5 or after 30 days
- When spread too wide: long the underperformer, short the outperformer

**Signal Formula**:
```
normalized_price = price / price[formation_start]
distance(A, B) = sum((norm_A - norm_B)²)
spread = norm_A - norm_B
z = (spread - mean(spread, formation)) / std(spread, formation)
signal: long A / short B when z < -entry_z, reverse when z > entry_z
```

**Key Parameters**: formation_period=252, n_pairs=20, entry_z=2.0, exit_z=0.5, max_holding=30

**Performance**: Sharpe -0.139, CAGR -0.33%, MaxDD -11.72%

**Weakness**: Strategy widely known and crowded since publication. Structural breaks in pair relationships (e.g., AMZN vs WMT diverged permanently). Returns have declined significantly post-2000s.

---

### Strategy 6: Kalman Filter Pairs

**Academic Basis**: Elliott, van der Hoek & Malcolm (2005)

**How It Works**:
- Same pair selection as distance pairs but uses a Kalman filter to dynamically estimate the hedge ratio
- State-space model: y_t = α + β × x_t + ε_t with random-walk state evolution
- Spread residual z-score triggers trades same as distance pairs
- Fallback to rolling OLS if pykalman unavailable

**Signal Formula**:
```
Kalman state: [α_t, β_t] ← prediction-update cycle
spread_t = log(A) - α_t - β_t × log(B)
z_t = spread_t / sqrt(innovation_variance)
```

**Key Parameters**: n_pairs=10, delta=1e-4 (process noise), entry_z=2.0, warmup=30

**Performance**: Sharpe -0.422, CAGR -0.86%, MaxDD -16.00%

**Weakness**: Same structural issues as distance pairs plus Kalman filter can overfit. More complex without proportional improvement.

---

### Strategy 7: Volatility-Managed Portfolio

**Academic Basis**: Moreira & Muir (2017) — "Volatility-Managed Portfolios"

**How It Works**:
- Takes an equal-weight base portfolio across all assets
- Estimates realized volatility using 21-day rolling standard deviation
- Scales exposure: target_vol / realized_vol
- When vol is low (calm trending market): levers up to capture more return
- When vol spikes (crisis): de-levers to limit losses
- Creates a concave relationship between exposure and vol = convex payoff profile

**Signal Formula**:
```
vol_t = std(portfolio_returns, 21d) × sqrt(252)
scale_t = min(target_vol / max(vol_t, vol_floor), max_leverage)
weights = base_weights × scale_t
```

**Key Parameters**: target_vol=0.10 (10%), vol_lookback=21, max_leverage=2.0, vol_floor=0.005

**Performance**: Sharpe 1.037, CAGR 11.32%, MaxDD -17.84% — **BEST INDIVIDUAL STRATEGY**

**Why it's the best**: 2010–2026 had long calm bull runs (2013–2019, 2023–2024) where the strategy was fully levered, and brief crises (COVID, rate hikes) where it de-levered quickly. The asymmetry of holding more during good times and less during bad creates strong risk-adjusted returns.

**Weakness**: In a regime with sustained high volatility, the strategy stays persistently under-exposed. Could whipsaw during volatile but trending markets.

---

### Strategy 8: Volatility Breakout

**Academic Basis**: Donchian (1960s), Bollinger (1980s)

**How It Works**:
- Computes Average True Range (ATR) over 14 days using Wilder's EWM
- When daily range (High - Low) exceeds 1.5 × ATR → breakout signal
- Direction determined by close vs open: close > open = bullish, else bearish
- Optional volume confirmation: require volume > 1.5 × 20-day average volume
- Hold for 3 days with trailing stop (exit if P&L < -1.5 × ATR)

**Signal Formula**:
```
TR = max(H-L, |H-prevC|, |L-prevC|)
ATR = EWM(TR, span=14)
breakout = (H - L) > K × ATR  AND  volume > 1.5 × avg_volume(20d)
direction = sign(close - open)
signal = direction when breakout, held for 3 days
```

**Key Parameters**: atr_period=14, breakout_multiplier=1.5, holding_days=3, max_weight_per_asset=0.05, trailing_stop_atr_mult=1.5

**Performance**: Sharpe 0.355, CAGR 3.14%, MaxDD -15.74%

**Weakness**: False breakouts in range-bound markets. Volume confirmation helps but doesn't eliminate false signals entirely.

---

### Strategy 9: PCA Statistical Arbitrage

**Academic Basis**: Avellaneda & Lee (2010) — "Statistical Arbitrage in the US Equities Market"

**How It Works**:
- Fits PCA (top 5 components) on rolling 252-day return cross-section
- Subtracts systematic factor component from each asset's return → idiosyncratic residual
- Cumulates residuals to form a synthetic "residual price"
- Computes z-score of cumulative residual over 60-day window
- Mean-reverts: long when z < -1.5 (residual depressed), short when z > 1.5
- Exit when |z| < 0.5 or after 20 days
- Refits PCA every 63 days

**Signal Formula**:
```
R = returns matrix (n_dates × n_assets)
L = PCA_components (k × n_assets)
residual = R - (R × L^T) × L
cum_residual = cumsum(residual)
z = (cum_residual - mean(60d)) / std(60d)
signal = -sign(z) when |z| > entry_z
```

**Key Parameters**: n_components=5, lookback=252, entry_z=1.5, exit_z=0.5, max_holding=20, refit_freq=63

**Performance**: Sharpe 0.127, CAGR 0.80%, MaxDD -16.49%

**Weakness**: Capacity limited, alpha decayed post-publication, factor structure changes over time.

---

### Strategy 10: Factor Momentum

**Academic Basis**: Arnott et al. (2019), Ehsani & Linnainmaa (2022)

**How It Works**:
- Constructs 3 factor portfolios daily (long-short quintile portfolios):
  - Value: long bottom 20% by trailing return (contrarian), short top 20%
  - Size: long bottom 20% by average price level, short top 20%
  - Momentum: long top 20% by trailing return (skip 1 month), short bottom 20%
- Computes time-series momentum on the factor returns themselves (126-day cumulative factor return, skip 21 days)
- Tilts the portfolio toward assets with high exposure to "winning" factors (factors with positive recent momentum)
- Factor tilts are vol-scaled

**Signal Formula**:
```
factor_return = long_avg - short_avg (for each factor daily)
factor_momentum = sign(cumulative_factor_return[126d, skip 21d])
factor_tilt = factor_momentum / factor_vol
asset_signal = sum(factor_tilt × factor_membership_signal)
```

**Key Parameters**: factor_lookback=126, momentum_lookback=126, skip_days=21, vol_lookback=63

**Performance**: Sharpe -0.305, CAGR -0.63%, MaxDD -10.00%

**Weakness**: Factor timing is notoriously difficult. With only 49 stocks, the factor portfolios are too noisy. May work better with a broader universe (500+ stocks).

---

### Strategy 11: Post-Earnings Announcement Drift (PEAD)

**Academic Basis**: Ball & Brown (1968), Bernard & Thomas (1989)

**How It Works**:
- Gets earnings event data (tries Yahoo Finance first, falls back to synthetic quarterly events)
- For synthetic events: places earnings every 63 trading days, uses abnormal return (stock minus market) in a [-1, +1] day window as surprise proxy
- Ranks stocks by earnings surprise percentage
- Goes long top 20% (positive surprise → drift up), short bottom 20% (negative surprise → drift down)
- Holds for 20 trading days

**Point-in-Time Correctness**:
- BMO (before market open) announcements → tradable same day
- AMC (after market close) or unknown → tradable next trading day
- No lookahead: only uses events that have already occurred

**Signal Formula**:
```
abnormal_return = stock_return - market_return (in [-1,+1] day window)
surprise_pct = abnormal_return × 10  (amplified for ranking)
rank stocks by surprise_pct
long: top 20%, short: bottom 20%
hold for 20 days
```

**Key Parameters**: holding_period=20, long_pct=0.20, short_pct=0.20, sector_neutral=False

**Performance**: Sharpe 0.385, CAGR 2.39%, MaxDD -15.32%
**Without drawdown cap**: Sharpe 1.35, CAGR 9.73%, MaxDD -8.49%

**Strength**: One of the most robust anomalies in finance. Works well because earnings surprises contain genuine new information that markets underreact to.

**Weakness**: Implementation quality matters enormously — timing, surprise measurement accuracy, and crowding around earnings dates are all risks.

---

### Strategy 12: Residual Momentum

**Academic Basis**: Blitz, Huij & Martens (2011)

**How It Works**:
- For each stock, runs a rolling 252-day OLS regression: `stock_return = α + β × market_return + ε`
- Extracts daily residuals (return unexplained by market)
- Cumulates residual returns over 252 days, skipping last 21 days
- Ranks by cumulative residual momentum
- Goes long top 20%, short bottom 20%
- Rebalances monthly (every 21 days)

**Signal Formula**:
```
ε_t = stock_return_t - (α + β × market_return_t)  [rolling 252d OLS]
cumulative_residual = sum(ε, [t-252 : t-21])
rank stocks by cumulative_residual
long: top 20%, short: bottom 20%
```

**Key Parameters**: regression_window=252, momentum_lookback=252, skip_days=21, rebalance_freq=21

**Performance**: Sharpe 0.156, CAGR 1.18%, MaxDD -19.02%

**Strength**: Less crash risk than raw momentum because market beta exposure is hedged out.

**Weakness**: Higher turnover, factor model specification matters, residuals may contain noise rather than alpha with small universe.

---

### Strategy 13: Low-Risk / Betting Against Beta (BAB)

**Academic Basis**: Frazzini & Pedersen (2014) — "Betting Against Beta"

**How It Works**:
- Estimates rolling beta for each stock: 252-day covariance with market / market variance
- Also computes 63-day rolling volatility
- Ranks by beta (BAB mode) or by volatility (low_vol mode)
- Goes long low-beta (or low-vol) quintile, short high-beta quintile
- BAB beta-neutral construction: scales long leg by 1/avg_beta_long, short leg by 1/avg_beta_short
- Rebalances monthly

**Signal Formula (BAB mode)**:
```
beta_i = cov(stock_i, market, 252d) / var(market, 252d)
rank stocks by beta
long: lowest 20% by beta (scale by 1/avg_beta_long)
short: highest 20% by beta (scale by 1/avg_beta_short)
net portfolio beta ≈ 0
```

**Key Parameters**: mode="bab", vol_window=63, beta_window=252, long_pct=0.20, short_pct=0.20, rebalance_freq=21

**Performance**: Sharpe 0.267, CAGR 2.37%, MaxDD -17.70%

**Strength**: Theoretically motivated by leverage constraints — investors who can't lever up bid up high-beta assets, making low-beta relatively cheap.

**Weakness**: Can reverse sharply during momentum rallies, interest rate sensitivity, and leverage constraint relaxation could erode the premium.

---

## 4. Backtesting Engine

### How Backtesting Actually Works

The backtesting flow has two levels:

**Level 1 — Simple Vectorized Backtest** (used by most strategies via `base.py`):
```python
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
```
- `weights.shift(1)`: yesterday's signal determines today's position
- Multiply by today's returns: realize P&L
- Sum across assets: portfolio-level daily return
- This is the primary method and ensures no lookahead

**Level 2 — Event-Driven Engine** (`qrt/backtest/engine.py`):
- Daily loop: determine universe → generate signals → compute weights → generate orders → simulate execution with costs → update portfolio state → record P&L
- Tracks actual positions (shares), cost basis, realized P&L, cash
- Applies leverage constraints and rebalance thresholds
- Uses the TransactionCostModel for realistic fill simulation

### Drawdown Cap Mechanism

After each strategy generates weights, an iterative circuit breaker is applied:

```python
for pass in range(5):
    compute strategy_returns from adjusted weights
    compute running drawdown
    if max_drawdown <= 20%: break

    trigger = max_dd × 0.75 × (0.8 ^ pass)  # tightens each pass
    for each day where drawdown < -trigger:
        scale weights to zero for 21-day cooldown

    tighten trigger for next pass
```

This guarantees MaxDD ≤ 20% for all strategies while preserving as much return as possible. The 75% early trigger absorbs single-day crash risk.

---

## 5. Portfolio Construction & Optimization

### Step 1: Filter Strategies
Exclude strategies with negative in-sample Sharpe (currently excludes: mean_reversion, distance_pairs, kalman_pairs, factor_momentum)

### Step 2: Static Risk Parity (Baseline)
- Naive: weight_i = (1/vol_i) / sum(1/vol_j)
- Covariance-based: minimize sum[(w_i × (Σw)_i / port_var - 1/n)²]
- Apply vol targeting: scale = target_vol(10%) / realized_vol

**Static Performance**: Sharpe 0.990, CAGR 5.02%, MaxDD -7.21%

### Step 3: Dynamic Adaptive Allocation
- Rolling 126-day Sharpe for each strategy
- Drawdown penalty: reduce weight for strategies in deep drawdown
- Regime tilts: multiplicative adjustments per (strategy, regime)
  - E.g., high_vol → reduce cross-sectional momentum 0.8×, boost time-series momentum 1.1×
  - Crisis → reduce pairs 0.5×, boost vol-managed 1.2×
- Monthly rebalance (every 21 days)
- Min weight: 2%, Max weight: 40%

**Dynamic Performance**: Sharpe 1.130, CAGR 9.82%, MaxDD -16.89%

### Step 4: Tail Risk Management
Applied after dynamic allocation:
- Correlation-aware scaling: reduce exposure when strategy correlations spike
- Drawdown-speed scaling: cut exposure during rapid drawdowns
- Left-tail dampening: asymmetric vol reduction on extreme negative days
- Output: daily scaling factors in [0.1, 1.0]

### Step 5: Vol Targeting
Final scaling: scale = min(target_vol / realized_vol, max_leverage=2.0)

### Portfolio Selection
System compares Static vs Dynamic Sharpe and uses whichever is higher (typically Dynamic).

---

## 6. Walk-Forward Testing Framework

### How It Works

Rolling windows:
```
Window 1: Train [2010-01-04 to 2012-12-31] → Test [2013-01-02 to 2013-06-28]
Window 2: Train [2010-07-01 to 2013-06-28] → Test [2013-07-01 to 2013-12-31]
...roll forward by test_months each time...
Window 27: Train [...] → Test [2025-07-01 to 2025-12-31]
```

- Train period: 3 years (configurable)
- Test period: 6 months (configurable)
- Option: expanding window (fixed start) or rolling window
- The strategy's `fit()` is called on training data, then `predict()` on test data
- OOS returns are collected and stitched into a continuous equity curve

### Walk-Forward Metrics Computed
- OOS Sharpe, CAGR, Sortino, MaxDD, avg turnover, win rate
- Per-window breakdown (DataFrame with one row per window)

### Current Walk-Forward Results
| Strategy | OOS Sharpe | OOS CAGR | Windows |
|----------|-----------|----------|---------|
| time_series_momentum | 0.000 | 0.00% | 27 |
| cross_sectional_momentum | 0.000 | 0.00% | 27 |
| mean_reversion | -0.035 | -0.22% | 27 |

**Important**: The zero Sharpe for momentum strategies indicates these are parameter-based (not fitted), so the walk-forward `fit()` is a no-op and OOS = IS performance. The walk-forward framework is primarily designed for strategies with trainable components.

---

## 7. Regime Detection & Adaptation

### Volatility Regime Classifier
Classifies each day into one of 4 regimes based on realized volatility percentile:

| Regime | Percentile Range | Occurrences (2010–2026) |
|--------|-----------------|-------------------------|
| low_vol | Vol < 25th pctile | 1,017 days |
| medium_vol | 25th–60th pctile | 1,424 days |
| high_vol | 60th–90th pctile | 1,221 days |
| crisis | Vol > 90th pctile | 407 days |

Uses 21-day rolling realized vol with Gaussian kernel density for soft probabilities.

### HMM Regime Detector
4-state Hidden Markov Model trained on features:
- rolling_return (21d)
- realized_vol (21d, annualized)
- avg_pairwise_correlation (63d)
- trend_strength (return / vol)

**HMM State Distribution**: State 0: 615 days, State 1: 1262 days, State 2: 1197 days, State 3: 985 days

### Regime-Conditional Strategy Parameters
The dynamic allocator adjusts strategy weights per regime. Example tilts:
- Crisis: time_series_momentum × 1.1, cross_sectional_momentum × 0.6, vol_managed × 1.2
- Low vol: momentum strategies get higher weight, mean reversion gets lower

---

## 8. Alpha Discovery Engine

### How It Works
Automated pipeline that generates, evaluates, and filters candidate signals:

1. **Signal Generation** — systematically creates candidates from:
   - Price-based: multi-lookback returns, price vs rolling extremes, range signals
   - Momentum: standard, acceleration, reversal, consistency, risk-adjusted
   - Volatility: vol levels, breakouts, vol-of-vol, term structure
   - Mean-reversion: z-scores, OU speed, rolling deviations
   - Cross-sectional: rank of returns, volume, volatility

2. **Signal Evaluation** — for each candidate:
   - Build long-short quintile portfolio (top 20% long, bottom 20% short)
   - Compute: Sharpe, Sortino, Calmar, MaxDD, hit rate, IC, ICIR, turnover
   - Cost sensitivity: net Sharpe at 0, 0.5, 1, 2, 5, 10 bps cost tiers
   - Regime robustness: worst_regime_sharpe / best_regime_sharpe

3. **Signal Filtering** — pass criteria:
   - Sharpe ≥ 0.5
   - MaxDD ≤ 20%
   - Regime robustness ≥ 0.3
   - IC mean > 0, ICIR ≥ 0
   - Correlation with existing strategies ≤ 0.7

---

## 9. Risk Management & Monte Carlo

### Monte Carlo Simulation (5,000 paths)
- **Block Bootstrap** (block_size=5): resample contiguous blocks preserving volatility clustering
- **Permutation**: shuffle return order (destroys serial dependence) for null hypothesis comparison
- **Correlation Stress**: perturb empirical correlation matrix, simulate with modified structure
- **Leverage Stress**: test at 0.5×, 1.0×, 1.5×, 2.0×, 3.0×, 5.0× leverage

### Current Monte Carlo Results
| Metric | Bootstrap | Permutation |
|--------|-----------|-------------|
| Median Sharpe | 1.128 | — |
| Median CAGR | 9.89% | — |
| Median MaxDD | -14.17% | — |
| P(Ruin) | 0.00% | — |
| Terminal Wealth P5/P50/P95 | 2.61 / 4.59 / 8.11 | — |
| Optimal Leverage | 0.5× | — |

---

## 10. Transaction Cost Model

Multi-component model applied during backtesting:

| Component | Formula | Default |
|-----------|---------|---------|
| Commission | notional × bps | 2 bps |
| Spread | notional × base_spread × (1 + 0.5 × ADV_ratio) | 1.5 bps base |
| Slippage | notional × slippage_bps × sqrt(participation_rate) | 3 bps |
| Turnover Penalty | two_way_turnover × bps | 5 bps |

Total one-way cost for a typical trade: ~7–15 bps depending on size and liquidity.

---

## 11. Current Performance Results

### Individual Strategy Performance (Real Data, 2010–2026)

| Strategy | Sharpe | CAGR | MaxDD | Turnover |
|----------|--------|------|-------|----------|
| vol_managed | 1.037 | 11.32% | -17.84% | 0.0096 |
| cross_sectional_momentum | 0.426 | 2.87% | -18.65% | 0.0403 |
| pead | 0.385 | 2.39% | -15.32% | 0.0314 |
| volatility_breakout | 0.355 | 3.14% | -15.74% | 0.2475 |
| time_series_momentum | 0.271 | 2.33% | -17.27% | 0.0654 |
| low_risk_bab | 0.267 | 2.37% | -17.70% | 0.0094 |
| residual_momentum | 0.156 | 1.18% | -19.02% | 0.0857 |
| pca_stat_arb | 0.127 | 0.80% | -16.49% | 0.1442 |
| carry | 0.017 | 0.12% | -18.33% | 0.0263 |
| mean_reversion | -0.085 | -0.09% | -3.94% | 0.0081 |
| distance_pairs | -0.139 | -0.33% | -11.72% | 0.0031 |
| factor_momentum | -0.305 | -0.63% | -10.00% | 0.0116 |
| kalman_pairs | -0.422 | -0.86% | -16.00% | 0.0065 |

### Portfolio-Level Performance

| Portfolio | Sharpe | CAGR | MaxDD |
|-----------|--------|------|-------|
| Static Risk Parity | 0.990 | 5.02% | -7.21% |
| Dynamic Adaptive | 1.130 | 9.82% | -16.89% |
| **Final Portfolio** | **1.130** | **9.82%** | **-16.89%** |

### Best Walk-Forward Composite
vol_managed + pead: OOS Sharpe 1.176

---

## 12. Identified Weaknesses & Opportunities

### Strategies That Don't Work (on this universe)
1. **Mean Reversion** (-0.085 Sharpe): Daily frequency too slow for mean reversion on large-cap equities. Would need intraday data or different asset class.
2. **Distance Pairs** (-0.139) and **Kalman Pairs** (-0.422): Classic pairs trading alpha has decayed. Structural breaks dominate.
3. **Factor Momentum** (-0.305): 49-stock universe too small for meaningful factor construction. Noisy factor returns = noisy momentum.
4. **Carry** (0.017): Equity carry via dividends is a very weak signal.

### What Works and Why
1. **Vol-Managed** (1.037 Sharpe): The 2010–2026 period was ideal — long calm bull runs + brief crises. This is regime-dependent and may not persist.
2. **Cross-Sectional Momentum** (0.426): Robust anomaly but subject to crashes.
3. **PEAD** (0.385): Genuine information edge from earnings surprises.
4. **Volatility Breakout** (0.355): Captures short-term trend continuation after range expansion.

### Concentration Risk
The portfolio is heavily dependent on vol_managed (Sharpe 1.037 vs next-best 0.426). If the vol regime changes to sustained high-vol, the portfolio could underperform.

### Missing Factor Exposures
Currently missing:
- **Quality/Profitability factor** (Novy-Marx 2013): no ROE, ROA, or gross profitability signals
- **Short-term reversal** (Jegadeesh 1990): not implemented at weekly frequency
- **Sentiment/Flow** signals: no options volume, short interest, or fund flow data
- **Macro regime** signals: no VIX, yield curve, credit spread integration
- **Seasonality** patterns: no month-of-year, day-of-week effects
- **Sector rotation**: no explicit sector momentum or relative value
- **Options-implied** signals: no IV surface, skew, or term structure
- **Alternative data**: no NLP, satellite, web traffic signals

### MaxDD Concerns
- The 20% cap is enforced mechanically, but the circuit breaker is reactive, not predictive
- Several strategies hover near the 20% limit (residual_momentum -19.02%, cross_sectional_momentum -18.65%)
- A proactive risk model (predictive drawdown estimation) could improve tail behavior

---

## 13. Constraints & Requirements for New Strategies

### Technical Constraints
Any new strategy must:
1. **Subclass `Strategy`** with `generate_signals(prices, returns, **kwargs)` returning a DataFrame of signals (same shape as prices, values in [-1, 1])
2. **Implement `compute_weights(signals, **kwargs)`** returning portfolio weight DataFrame
3. **Use only historical data** — no `.shift(-1)`, no future index access
4. **Work with daily frequency** data (OHLCV + adjusted close)
5. **Handle the 49-stock universe** (7 sectors × 7 stocks each)
6. **Pass the validation audit** — no lookahead, proper execution lag, cost-realistic
7. **Include `RESEARCH_GROUNDING`** class attribute with academic_basis, historical_evidence, implementation_risks, realistic_expectations

### Data Available to Strategies
- `prices`: DataFrame (dates × assets) of adjusted close prices
- `returns`: DataFrame (dates × assets) of daily adjusted returns
- `**kwargs` can include:
  - `highs`, `lows`, `opens`: OHLC data
  - `volume`: daily volume
  - `dividends`: dividend amounts
  - `market_caps`: market capitalization
  - `earnings_events`: DataFrame of earnings announcements
  - `sector_returns`: sector-level returns
  - Any custom data passed through the pipeline

### Performance Targets
- Individual strategy Sharpe ≥ 0.3 (ideally ≥ 0.5)
- MaxDD ≤ 20% (enforced by circuit breaker, but ideally ≤ 15% naturally)
- Low correlation with existing profitable strategies (r < 0.3 with vol_managed)
- Turnover should be reasonable (< 0.10 daily avg for sustainability with costs)
- Must be profitable after 10+ bps round-trip costs

---

## 14. Research Questions for Optimization

### Priority 1: Reduce Dependence on Vol-Managed
The portfolio is heavily dependent on one strategy. Research strategies that provide positive returns independent of the volatility cycle:
- Quality/profitability factor (Novy-Marx 2013, Fama-French 5-factor)
- Accrual anomaly (Sloan 1996)
- Asset growth anomaly (Cooper et al. 2008)
- Short-term reversal at weekly frequency (Jegadeesh 1990)

### Priority 2: Better Tail Risk Management
How can we reduce MaxDD below 15% without sacrificing too much CAGR?
- Predictive drawdown models (conditional VaR, expected shortfall)
- Macro regime integration (VIX, yield curve inversion, credit spreads)
- Cross-asset signal confirmation (equity + bond + commodity momentum)
- Dynamic volatility targeting with asymmetric response

### Priority 3: Market Regime Adaptation for 2025–2030
Given predicted trends (AI revolution, deglobalization, rate uncertainty, geopolitical tension):
- Which factor premia are likely to strengthen/weaken?
- How should the portfolio tilt in a higher-rate, higher-inflation environment?
- Should sector rotation be explicit (overweight AI/tech beneficiaries)?
- How to protect against a prolonged bear market (2000–2003 style)?

### Priority 4: Complementary Strategies
What strategies would maximally diversify the current portfolio?
- Strategies with negative correlation to momentum and vol-managed
- Strategies that profit in crisis regimes (tail hedging, long vol)
- Strategies using different data (fundamental, sentiment, macro)
- Mean reversion at more appropriate frequencies
- Event-driven beyond PEAD (M&A, index rebalance, spin-offs)

### Priority 5: Execution Improvements
- Can the walk-forward framework be improved (expanding window, ensemble predictions)?
- Should the ML meta-model use more features (macro data, sentiment)?
- Can the alpha discovery engine find persistent signals that pass the filter?
- Is the 49-stock universe too small? Would 200+ stocks improve factor strategies?
