# Signal Architecture & Strategy Research Requirements Document

## Purpose

This document provides a comprehensive technical description of a quantitative trading research platform's signal architecture. The goal is to enable deep research into new ways to improve signal generation, processing, combination, evaluation, and overall strategy performance. Use this document to identify gaps, suggest novel signal types, propose better combination methods, and recommend improvements to the signal-to-trade pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Signal Generation Layer](#2-signal-generation-layer)
3. [Signal Evaluation Layer](#3-signal-evaluation-layer)
4. [Signal Filtering Layer](#4-signal-filtering-layer)
5. [Strategy Definitions & Signal Consumption](#5-strategy-definitions--signal-consumption)
6. [Signal-to-Weight Conversion Pipeline](#6-signal-to-weight-conversion-pipeline)
7. [Portfolio Allocation & Multi-Strategy Combination](#7-portfolio-allocation--multi-strategy-combination)
8. [Risk Management & Signal Validation](#8-risk-management--signal-validation)
9. [Backtest Engine & Execution Simulation](#9-backtest-engine--execution-simulation)
10. [End-to-End Data Flow Diagram](#10-end-to-end-data-flow-diagram)
11. [Configuration & Parameters](#11-configuration--parameters)
12. [Known Limitations & Research Opportunities](#12-known-limitations--research-opportunities)

---

## 1. System Overview

### Architecture

The system is an **end-to-end quantitative trading research platform** with a signal-centric architecture:

```
RAW MARKET DATA → SIGNAL GENERATION → SIGNAL EVALUATION → SIGNAL FILTERING
       ↓
STRATEGY EXECUTION → PORTFOLIO CONSTRUCTION → BACKTEST ENGINE → RESULTS
```

### Core Components

| Component | Role |
|-----------|------|
| **Signal Generator** | Produces 100+ candidate signals from 5 families |
| **Signal Evaluator** | Scores each signal on 15+ quality metrics |
| **Signal Filter** | Cascading filter to select ~10-30 robust signals |
| **Strategy Library** | 10 concrete trading strategies consuming signals |
| **Portfolio Allocator** | Dynamic strategy weighting with regime adaptation |
| **Backtest Engine** | Event-driven daily simulation with cost modeling |
| **Risk Management** | Multi-layered controls (signal, strategy, portfolio levels) |

### Data Inputs

- **Price data**: Daily OHLCV (Open, High, Low, Close, Volume)
- **Returns**: Daily log or simple returns
- **Universe**: Configurable set of securities (synthetic or real)
- **Dividend yields**: Optional, used by Carry strategy

---

## 2. Signal Generation Layer

### Overview

The `SignalGenerator` class produces **100+ candidate signals** organized into **5 families**. Each signal is a DataFrame indexed by (date x security), with values representing the signal strength.

### Signal Family 1: Price-Based Signals

| Signal Name | Calculation | Parameters |
|-------------|-------------|------------|
| `return_Xd` | Cumulative return over X days | Lookbacks: 5, 10, 21, 63, 126, 252 days |
| `price_vs_max_Xd` | Close / rolling max(close, X days) | Lookbacks: 63, 126, 252 days |
| `price_vs_min_Xd` | Close / rolling min(close, X days) | Lookbacks: 63, 126, 252 days |
| `high_low_range_Xd` | Rolling mean of (high - low) / close | Lookbacks: 21, 63 days |
| `close_in_range_Xd` | (Close - Low) / (High - Low) rolling mean | Lookbacks: 21, 63 days |

**What these capture**: Trend strength, proximity to extremes, intraday volatility patterns, and close position relative to daily range.

### Signal Family 2: Momentum Signals

| Signal Name | Calculation | Parameters |
|-------------|-------------|------------|
| `momentum_Xd` | Cumulative return over X days | Lookbacks: 21, 63, 126, 252 days |
| `momentum_accel_X_Y` | momentum_Xd - momentum_Yd | Pairs: (21,63), (63,126), (126,252) |
| `momentum_reversal_Xd` | -1 × return_Xd (inverted short-term) | Lookbacks: 5, 10, 21 days |
| `momentum_consistency_Xd` | Fraction of positive return days | Lookbacks: 63, 126, 252 days |
| `risk_adj_momentum_Xd` | momentum_Xd / realized_vol_Xd | Lookbacks: 63, 126, 252 days |

**What these capture**: Absolute trend following, trend acceleration/deceleration, short-term reversal (contrarian), win-rate consistency, and volatility-normalized momentum.

### Signal Family 3: Volatility Signals

| Signal Name | Calculation | Parameters |
|-------------|-------------|------------|
| `volatility_Xd` | Annualized rolling std of returns | Lookbacks: 21, 63, 126 days |
| `vol_zscore_X_Y` | (vol_Xd - mean(vol_Yd)) / std(vol_Yd) | Pairs: (21,126), (21,252), (63,252) |
| `vol_breakout_X_Y` | vol_Xd / rolling_max(vol_Yd) | Pairs: (21,126), (21,252) |
| `vol_mean_reversion_X_Y` | vol_Xd / rolling_min(vol_Yd) | Pairs: (21,126), (21,252) |
| `vol_of_vol_Xd` | Rolling std of rolling volatility | Lookbacks: 63, 126 days |
| `vol_term_structure_X_Y` | vol_Xd / vol_Yd (short/long ratio) | Pairs: (21,63), (21,126), (63,126) |

**What these capture**: Volatility level, volatility regime transitions, volatility breakouts/compression, second-order volatility, and term structure inversions.

### Signal Family 4: Mean-Reversion Signals

| Signal Name | Calculation | Parameters |
|-------------|-------------|------------|
| `zscore_Xd` | (return - rolling_mean) / rolling_std | Lookbacks: 21, 63, 126, 252 days |
| `cumret_zscore_Xd` | Z-score of cumulative returns | Lookbacks: 63, 126, 252 days |
| `ou_speed_Xd` | Ornstein-Uhlenbeck mean reversion speed via AR(1) | Lookbacks: 126, 252 days |
| `return_deviation_Xd` | (return - rolling_mean) / rolling_vol | Lookbacks: 63, 126 days |

**What these capture**: Statistical overextension from mean, price-level mean reversion, speed of mean reversion (OU process), and volatility-adjusted deviation.

**OU Speed Calculation Detail**: Regresses return_t on return_{t-1} via OLS. The AR(1) coefficient β estimates mean-reversion speed as -ln(β)/dt. Higher values = faster mean reversion.

### Signal Family 5: Cross-Sectional Signals

| Signal Name | Calculation | Parameters |
|-------------|-------------|------------|
| `xsec_return_rank_Xd` | Percentile rank of return_Xd across universe | Lookbacks: 21, 63, 126, 252 days |
| `xsec_vol_rank_Xd` | Percentile rank of volatility_Xd | Lookbacks: 21, 63 days |
| `xsec_low_vol_rank_Xd` | 1.0 - vol_rank (inverted for low-vol factor) | Lookbacks: 21, 63 days |
| `xsec_vol_change_rank_X_Y` | Rank of (vol_Xd / vol_Yd) | Pairs: (21,63), (21,126) |
| `xsec_vwret_rank_Xd` | Rank of volume-weighted return | Lookbacks: 21, 63 days |
| `xsec_turnover_rank_X_Y` | Rank of (recent_vol / hist_vol) | Pairs: (21,63), (21,126) |

**What these capture**: Relative momentum, relative volatility, low-volatility anomaly, volatility regime change, smart-money flow (volume-weighted), and liquidity regime shifts.

### Post-Generation Signal Processing

After generation, all signals undergo:

1. **Winsorization**: Values clipped to 1st and 99th percentiles (cross-sectional, per date)
2. **Z-Score Normalization**: Cross-sectional standardization to mean=0, std=1 per date
3. **Reindexing**: Aligned to match the returns DataFrame shape (dates × securities)

**Current limitation**: Processing is purely cross-sectional. No time-series normalization, no adaptive windowing, no non-linear transformations.

---

## 3. Signal Evaluation Layer

### Overview

The `SignalEvaluator` evaluates each candidate signal by constructing a **quintile long-short portfolio** and computing 15+ quality metrics.

### Long-Short Portfolio Construction

For each signal on each date:
1. Rank all securities by signal value (cross-sectionally)
2. **Long leg**: Top 20% (quintile 5) — equal-weighted
3. **Short leg**: Bottom 20% (quintile 1) — equal-weighted
4. **Daily P&L**: Long return - Short return (next-day forward return)
5. **Alignment**: Signal on date t predicts return on date t+1

### Core Performance Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Sharpe Ratio** | mean(daily_pnl) / std(daily_pnl) × √252 | Risk-adjusted return |
| **Sharpe Stability** | std(rolling_sharpe_63d) | Consistency of Sharpe over time |
| **Annualized Return** | mean(daily_pnl) × 252 | Absolute return |
| **Annualized Volatility** | std(daily_pnl) × √252 | Return variability |
| **Hit Rate** | fraction of days where pnl > 0 | Win rate |

### Information Coefficient (IC) Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **IC Mean** | mean(daily Spearman corr(signal, forward_return)) | Average predictive power |
| **IC Std** | std(daily Spearman corr) | IC variability |
| **ICIR** | (IC_mean / IC_std) × √252 | Annualized information ratio |

**Note**: IC is computed as Spearman rank correlation between signal values and next-day returns, computed daily then averaged.

### Risk Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Max Drawdown** | max peak-to-trough decline of cumulative P&L | Worst-case loss |
| **Calmar Ratio** | annualized_return / max_drawdown | Return per unit of drawdown |
| **Skewness** | 3rd moment of daily P&L | Tail asymmetry |
| **Kurtosis** | 4th moment of daily P&L | Fat-tail risk |

### Trading/Capacity Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Turnover** | mean daily one-way weight change | Trading cost sensitivity |
| **Cost Sensitivity** | Sharpe at {0, 0.5, 1, 2, 5, 10} bps cost | How costs erode alpha |

### Regime Analysis

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Per-Regime Sharpe** | Sharpe ratio computed within each volatility regime | Regime-specific performance |
| **Regime Robustness** | worst_regime_sharpe / best_regime_sharpe | Cross-regime consistency |

**Current limitation**: Evaluation uses only a single forward return horizon (1-day). No multi-horizon IC analysis. No decay analysis of signal predictive power over time.

---

## 4. Signal Filtering Layer

### Overview

The `SignalFilter` applies **cascading filters** to reduce ~100 candidate signals to ~10-30 high-quality signals. Each filter is applied sequentially; signals must pass ALL criteria.

### Filter Cascade (Applied in Order)

| Step | Filter | Default Threshold | Rationale |
|------|--------|-------------------|-----------|
| 1 | Sharpe Ratio | ≥ 0.5 | Minimum risk-adjusted return |
| 2 | Max Drawdown | ≤ 20% | Limit downside risk |
| 3 | Regime Robustness | ≥ 0.3 | Avoid regime-specific overfit |
| 4 | IC Mean | > 0.0 | Must have positive predictive power |
| 5 | ICIR | ≥ 0.0 | Information ratio floor |
| 6 | Correlation | ≤ 0.7 with existing strategies | Diversification requirement |

### Correlation Filter Detail

- Computes **Spearman rank correlation** between candidate signal's daily P&L and each existing strategy's daily P&L
- If |correlation| > 0.7 with ANY existing strategy, the signal is rejected
- Purpose: Ensure new signals add diversification, not redundancy

### Filter Output

- List of passing signal names
- Per-signal transparency: which filter(s) each rejected signal failed
- Correlation matrix of all signal P&Ls (for diversification analysis)

**Current limitation**: Filters are static thresholds. No adaptive or data-driven threshold selection. No multi-objective optimization (e.g., Pareto frontier of Sharpe vs. drawdown). Correlation filter is pairwise, not portfolio-level (doesn't consider joint diversification benefit).

---

## 5. Strategy Definitions & Signal Consumption

### Base Strategy Interface

All 10 strategies inherit from `BaseStrategy`:

```python
class BaseStrategy:
    def generate_signals(prices, returns, **kwargs) -> pd.DataFrame
        # Returns signals in [-1, 1] range (dates × securities)

    def compute_weights(signals, **kwargs) -> pd.DataFrame
        # Converts signals to portfolio weights (dates × securities)

    def backtest_summary(weights, returns) -> dict
        # Returns: total_return, annual_return, annual_vol, sharpe, max_dd, turnover
```

---

### Strategy 1: Time Series Momentum

**Academic Basis**: Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum"

**Signal Generation**:
1. Compute cumulative returns over 3 lookback windows: 63d, 126d, 252d
2. Blend with weights: (0.4 × 63d, 0.4 × 126d, 0.2 × 252d) — default for medium-vol regime
3. Compute **trend strength**: min(1.0, |trailing_return_126d| / 0.20)
4. Final signal = sign(blended_momentum) × trend_strength
5. Signal range: [-1, +1] continuous

**Weight Computation**:
1. Raw weight = signal / realized_vol(63d) — volatility scaling
2. If vol-of-vol > median: reduce weight proportionally (uncertainty scaling)
3. Gross normalize: sum(|weights|) = target_gross (default 1.0)

**Regime-Conditional Parameters**:

| Regime | Lookback | Blend Weights (63/126/252) | Target Gross |
|--------|----------|---------------------------|--------------|
| Low Vol | 252d | (0.2, 0.4, 0.4) | 1.2 |
| Medium Vol | 126d | (0.4, 0.4, 0.2) | 1.0 |
| High Vol | 63d | (0.5, 0.3, 0.2) | 0.7 |
| Crisis | 63d | (0.6, 0.3, 0.1) | 0.3 |

**Key Properties**: Continuous signal, vol-scaled weights, regime-adaptive parameters, daily rebalance.

---

### Strategy 2: Cross-Sectional Momentum

**Academic Basis**: Jegadeesh & Titman (1993), Carhart (1997)

**Signal Generation**:
1. Compute trailing 252-day return, **skipping the most recent 30 days** (avoids short-term reversal)
2. Rank securities cross-sectionally by this return
3. Top 20% → long signal (+1), Bottom 20% → short signal (-1), Middle 60% → no signal (0)
4. Within each leg: z-score weighting (stronger momentum → larger weight)

**Weight Computation**:
1. Gross exposure = 1.0 (0.5 long, 0.5 short)
2. **Circuit breaker**: If cumulative strategy return drops > 25% from peak, reduce exposure proportionally

**Key Properties**: Discrete signal (long/short/flat), cross-sectional ranking, skip-period for reversal avoidance, circuit breaker for crash protection.

---

### Strategy 3: Mean Reversion

**Academic Basis**: Lo & MacKinlay (1990), Poterba & Summers (1988)

**Signal Generation**:
1. Compute z-score of log prices over 120-day rolling window
2. **Entry signal**: Triggered when |z| > 1.5
3. **Exit signal**: Triggered when |z| < 0.5 OR position held > 5 days
4. **Trend filter**: Suppress signals when 200-day SMA slope > 0.001 (strong uptrend — mean reversion unreliable)
5. Signal direction: Negative of z-score (buy when oversold, sell when overbought)

**Weight Computation**:
1. Vol-scaling: weight = signal / realized_vol(21d)
2. Gross normalization to 1.0
3. **Stop-loss**: Close position if loss exceeds 3% per position
4. **Circuit breaker**: Flatten ALL positions if cumulative strategy return < -15%

**Key Properties**: Threshold-based entry/exit (not continuous), holding period limit, trend filter, per-position stop-loss, strategy-level circuit breaker.

---

### Strategy 4: Carry

**Academic Basis**: Koijen et al. (2018) — "Carry"

**Signal Generation**:
1. Compute carry proxy: dividend yield (if available) or total_return - price_return
2. Smooth carry estimate over 63-day rolling window
3. Rank cross-sectionally
4. Top 25% → long (high carry), Bottom 25% → short (low carry)

**Weight Computation**:
1. Equal-weight within each leg
2. Gross normalization to 1.0
3. **Rebalance frequency**: Every 21 days (carry-forward weights between rebalances)

**Key Properties**: Slow-moving signal, infrequent rebalancing (monthly), wider quantile bands (25% vs 20%), carry-forward between rebalances.

---

### Strategy 5: Volatility Breakout

**Academic Basis**: Donchian (1960s), Bollinger (1980s)

**Signal Generation**:
1. Compute **ATR** (Average True Range) over 14 days
2. **Breakout condition**: Daily range (high - low) > 1.5 × ATR
3. **Direction**: If close > open → long (+1); if close < open → short (-1)
4. **Volume confirmation**: Only trigger if volume > 1.5 × 20-day average volume
5. **Holding**: Maximum 3 days per trade

**Weight Computation**:
1. Vol-scaling: weight = signal / realized_vol
2. Per-asset cap: |weight| ≤ 5%
3. Gross normalization to 1.0
4. **Trailing stop**: Exit if unrealized PnL < -1.5 × current ATR

**Key Properties**: Event-driven (not always in market), requires volume confirmation, short holding period, trailing stop instead of fixed stop.

---

### Strategy 6: Low-Risk / Betting Against Beta (BAB)

**Academic Basis**: Frazzini & Pedersen (2014) — "Betting Against Beta"

**Signal Generation — Two Modes**:

**Mode A (BAB)**:
1. Estimate rolling beta to market (252-day regression window)
2. Rank by beta
3. Bottom 20% (low beta) → long, Top 20% (high beta) → short
4. **Beta-neutral scaling**: Lever each leg by 1 / average_beta_of_leg

**Mode B (Low-Vol)**:
1. Compute rolling realized volatility (63-day window)
2. Bottom 20% (low vol) → long, Top 20% (high vol) → short

**Weight Computation**:
1. Equal-weight within each leg
2. Rebalance every 21 days
3. Optional beta-neutral leverage scaling

**Key Properties**: Exploits low-risk anomaly, optional beta-neutral construction, slow rebalancing.

---

### Strategy 7: Residual Momentum

**Academic Basis**: Blitz, Huij & Martens (2011) — "Residual Momentum"

**Signal Generation**:
1. Run OLS regression: asset_return = α + β_market × market_return + β_sector × sector_return + ε
2. Regression window: 252 days
3. Accumulate residuals (ε) over 252 days, **skipping last 21 days**
4. Rank by cumulative residual return
5. Top 20% → long, Bottom 20% → short

**Weight Computation**:
1. Equal-weight within legs
2. Optional vol-scaling
3. Rebalance every 21 days

**Key Properties**: Removes systematic exposure before ranking momentum, lower crash risk than raw momentum, skip-period to avoid short-term reversal.

---

### Strategy 8: Factor Momentum

**Academic Basis**: Arnott et al. (2019) — "Factor Momentum Everywhere"

**Signal Generation**:
1. **Define 3 factors** as quintile long-short portfolios:
   - Value: Long lowest-return quintile, short highest (contrarian)
   - Size: Long lowest-price quintile, short highest
   - Momentum: Long highest-return quintile, short lowest (126d return)
2. Compute factor returns (daily P&L of each factor portfolio)
3. Apply **time-series momentum to factors**: 126-day return of each factor, skip 21 days
4. Compute **factor tilt**: momentum signal for each factor, vol-scaled
5. Compute **asset signal**: For each asset, signal = Σ(factor_tilt_i × asset_factor_loading_i)

**Weight Computation**:
1. Vol-scaling of asset-level signals
2. Gross normalization to 1.0

**Key Properties**: Meta-strategy (momentum of factors), combines cross-sectional and time-series elements, factor loading determines asset exposure.

---

### Strategy 9: PCA Statistical Arbitrage

**Academic Basis**: Avellaneda & Lee (2010) — "Statistical Arbitrage in the U.S. Equities Market"

**Signal Generation**:
1. Fit PCA on returns (5 components, auto-adjust if <50% variance explained)
2. Compute factor loadings and residuals (idiosyncratic returns)
3. Compute z-score of cumulative residual returns
4. **Entry**: |z| > 1.5
5. **Exit**: |z| < 0.5 OR held > 20 days
6. **Refit PCA**: Every 63 days

**Weight Computation**:
1. Forward-pass simulation tracking per-asset position state
2. Vol-scaling
3. Gross normalization to 1.0

**Key Properties**: Dynamic factor model (PCA refitted periodically), mean reversion on idiosyncratic component only, longer holding period than simple mean reversion.

---

### Strategy 10a: Kalman Pairs Trading

**Academic Basis**: Dynamic hedge ratio estimation via Kalman filter

**Signal Generation**:
1. **Pair selection**: Screen for cointegration (ADF test or Hurst exponent < 0.5)
2. Select top 10 pairs by cointegration strength
3. For each pair (X, Y):
   - Kalman filter state: [α, β] where spread = Y - α - β × X
   - Process noise: 1e-4 (how fast hedge ratio changes)
   - Observation noise: 1.0
4. Compute z-score of spread (normalized by rolling std)
5. **Entry**: |z| > 2.0
6. **Exit**: |z| < 0.5 OR held > 30 days
7. **Rescreen pairs**: Every 252 days

**Weight Computation**:
- Long one side, short the other (hedge ratio from Kalman state)
- Position sizing proportional to signal strength

**Key Properties**: Dynamic hedge ratio (adapts over time), cointegration-based pair selection, long holding period for convergence.

---

### Strategy 10b: Distance Pairs Trading

**Academic Basis**: Gatev, Goetzmann & Rouwenhorst (2006)

**Signal Generation**:
1. **Formation period**: 252 days
2. Normalize prices to start at 1.0 during formation
3. Compute sum of squared distances (SSD) between all pairs
4. Select 20 closest pairs (smallest SSD)
5. Compute spread: normalized_price_A - normalized_price_B
6. Compute z-score of spread
7. **Entry**: |z| > 2.0
8. **Exit**: |z| < 0.5 OR held > 30 days
9. **Rescreen pairs**: Every 252 days

**Key Properties**: Non-parametric pair selection (no cointegration assumption), simpler than Kalman but less adaptive, known declining returns post-2000.

---

## 6. Signal-to-Weight Conversion Pipeline

### Standard Pipeline (Per Strategy)

Every strategy follows this transformation chain:

```
Step 1: RAW SIGNAL
   Source: Strategy-specific (momentum, z-score, rank, breakout)
   Range: Typically [-1, +1] or continuous

Step 2: THRESHOLD APPLICATION (if applicable)
   Mean reversion: Entry |z| > 1.5, exit |z| < 0.5
   Volatility breakout: Range > 1.5 × ATR + volume confirm
   Pairs: Entry |z| > 2.0, exit |z| < 0.5

Step 3: VOLATILITY SCALING (optional, most strategies use this)
   weight_raw = signal / realized_vol(lookback)
   Effect: Larger positions in low-vol assets, smaller in high-vol

Step 4: SPECIAL ADJUSTMENTS
   TSM: Vol-of-vol reduction, trend strength scaling
   XS Momentum: Circuit breaker if drawdown > 25%
   Mean Reversion: Stop-loss at 3%, circuit breaker at -15%
   Vol Breakout: Per-asset cap at 5%, trailing stop at 1.5 ATR
   BAB: Beta-neutral leverage adjustment

Step 5: GROSS NORMALIZATION
   sum(|weights|) = target_gross (typically 1.0)
   Ensures consistent leverage across strategies

Step 6: OUTPUT
   weights: DataFrame (dates × securities)
   Values: Target portfolio weights (positive = long, negative = short)
```

### Signal Types by Strategy

| Strategy | Signal Type | Entry | Exit | Rebalance |
|----------|-------------|-------|------|-----------|
| TS Momentum | Continuous [-1,1] | Always in market | N/A | Daily |
| XS Momentum | Discrete (long/short/flat) | Quintile rank | Rank change | Daily |
| Mean Reversion | Threshold-based | \|z\| > 1.5 | \|z\| < 0.5 or 5d | Daily check |
| Carry | Discrete (long/short/flat) | Quantile rank | Rank change | Every 21d |
| Vol Breakout | Event-driven | Range > 1.5×ATR | 3d or trailing stop | Daily check |
| BAB / Low-Vol | Discrete (long/short/flat) | Quintile rank | Rank change | Every 21d |
| Residual Momentum | Discrete (long/short/flat) | Quintile rank | Rank change | Every 21d |
| Factor Momentum | Continuous | Always in market | N/A | Daily |
| PCA Stat Arb | Threshold-based | \|z\| > 1.5 | \|z\| < 0.5 or 20d | Daily check |
| Kalman Pairs | Threshold-based | \|z\| > 2.0 | \|z\| < 0.5 or 30d | Daily check |
| Distance Pairs | Threshold-based | \|z\| > 2.0 | \|z\| < 0.5 or 30d | Daily check |

---

## 7. Portfolio Allocation & Multi-Strategy Combination

### Dynamic Strategy Allocation

**Module**: `adaptive_allocation.py`

Combines multiple strategies using dynamic weights:

1. **Rolling Sharpe Estimation**: 126-day rolling Sharpe ratio per strategy
2. **Drawdown Penalty**: Reduce allocation to strategies currently in drawdown beyond threshold
3. **Regime Tilts**: Adjust strategy weights based on current volatility regime

**Regime Tilt Matrix**:

| Regime | Momentum | Carry | Mean Reversion | Vol Breakout | BAB |
|--------|----------|-------|----------------|--------------|-----|
| Low Vol | +20% | +30% | 0% | 0% | 0% |
| Medium Vol | 0% | 0% | 0% | 0% | 0% |
| High Vol | 0% | 0% | +20% | +30% | 0% |
| Crisis | +30% (TSM) | -50% | 0% | 0% | 0% |

**Weight Bounds**: 2% minimum, 40% maximum per strategy

**Rebalance**: Every 21 days (monthly)

### Volatility Targeting

**Module**: `vol_targeting.py`

Scales total portfolio exposure to target a specific volatility level:

- **Target volatility**: 10% annualized (configurable)
- **Lookback**: 63 days for realized vol estimation
- **Scaling factor**: target_vol / realized_vol
- **Leverage cap**: 2.0x maximum

### Bayesian Kelly Sizing

**Module**: `bayesian_kelly.py`

Optimal position sizing with Bayesian shrinkage:

1. **Expected returns estimation**:
   - Raw mean returns from historical data
   - **James-Stein shrinkage**: Shrink toward grand mean
   - Shrinkage intensity: λ = min(1, (N-2) × σ²_pool / (T × ‖μ̂ - μ₀‖²))

2. **Kelly criterion**:
   - Diagonal: f_i = μ_bayes_i / σ²_i
   - Full covariance: f = Σ⁻¹ × μ
   - **Fractional Kelly**: w = 0.25 × f (quarter Kelly — default)

3. **Constraints applied sequentially**:
   - Per-asset cap: |w_i| ≤ 5%
   - Gross leverage: Σ|w_i| ≤ 2.0
   - Strategy exposure: gross_long ≤ 30%, gross_short ≤ 30%

---

## 8. Risk Management & Signal Validation

### Layer 1: Signal-Level Validation (Signal Generator)

| Check | Action |
|-------|--------|
| Winsorization | Clip to 1st/99th percentile cross-sectionally |
| Z-score normalization | Standardize to mean=0, std=1 per date |
| Reindexing | Align signal to returns shape |

### Layer 2: Signal Quality Evaluation (Signal Evaluator)

| Check | Action |
|-------|--------|
| Trivial P&L detection | Skip if std(pnl) < 1e-12 |
| Minimum assets per quintile | Require ≥ 5 assets in long/short legs |
| NaN handling | Skip dates with insufficient data |

### Layer 3: Signal Filtering (Signal Filter)

| Check | Threshold | Action |
|-------|-----------|--------|
| Sharpe ratio | ≥ 0.5 | Reject low-alpha signals |
| Max drawdown | ≤ 20% | Reject high-risk signals |
| Regime robustness | ≥ 0.3 | Reject regime-dependent signals |
| IC mean | > 0.0 | Reject non-predictive signals |
| Correlation | ≤ 0.7 | Reject redundant signals |

### Layer 4: Strategy-Level Risk Controls

| Control | Strategies | Parameters |
|---------|-----------|------------|
| Stop-loss | Mean Reversion | 3% per position |
| Trailing stop | Vol Breakout | 1.5 × ATR |
| Holding limit | Mean Rev (5d), Vol Break (3d), PCA (20d), Pairs (30d) | Max days held |
| Circuit breaker | Mean Rev (-15%), XS Mom (-25%) | Flatten positions if breached |
| Trend filter | Mean Reversion | Suppress if 200d SMA slope > 0.001 |
| Position cap | Vol Breakout | 5% per asset max |
| Volume confirmation | Vol Breakout | Require 1.5× avg volume |

### Layer 5: Portfolio-Level Risk Controls

| Control | Description |
|---------|-------------|
| **Drawdown scaler** | Smooth, continuous exposure reduction as drawdown deepens (CDaR-inspired) |
| **Correlation scaling** | Reduce exposure when pairwise strategy correlations spike |
| **Drawdown speed scaling** | Cut exposure when drawdown velocity > -0.5% per 10 days |
| **Asymmetric vol targeting** | More aggressive reduction on large negative days (z-score < -2.0) |
| **Gross leverage cap** | Hard cap at 2.0x |
| **Per-asset cap** | Hard cap at 5% |
| **Strategy exposure limits** | 30% gross long, 30% gross short |

---

## 9. Backtest Engine & Execution Simulation

### Engine Architecture

**Module**: `engine.py`

Event-driven daily simulation:

```
For each trading date t:

  1. SIGNAL GENERATION
     For each active strategy:
       signals = strategy.generate_signals(prices[:t], returns[:t])

  2. SIGNAL NORMALIZATION
     Z-score normalize combined signals
     Ensure sum(|normalized_signal|) = 1.0

  3. POSITION SIZING
     Apply regime adjustments to strategy parameters
     target_weights = strategy.compute_weights(signals)
     Apply leverage constraints via PositionSizer

  4. ORDER GENERATION
     delta = target_weights - current_weights
     Filter: only trade if |delta| > rebalance_threshold (0.1%)
     Filter: only trade if notional > min_trade_notional ($100)

  5. EXECUTION SIMULATION
     For each order:
       fill_price = market_price
       costs = commission_bps + slippage_bps + spread_bps
       net_proceeds = notional × (1 - costs/10000)

  6. POSITION UPDATE
     Update: share counts, cost basis, cash balance
     Record: realized P&L from closed positions

  7. VALUATION
     Mark-to-market all positions
     Record: portfolio value, leverage, exposures, daily return
```

### Cost Model

| Cost Component | Default Value |
|----------------|---------------|
| Commission | 2.0 bps |
| Spread | 5.0 bps |
| Slippage | 3.0 bps |
| **Total round-trip** | **20.0 bps** |

### Backtest Output

- Daily portfolio returns and cumulative P&L
- Metrics: Sharpe, max drawdown, Calmar, win rate, turnover
- Trade ledger: all fills with entry/exit prices and costs
- Attribution: by strategy, by sector, by theme
- Drawdown time series

---

## 10. End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW MARKET DATA                                   │
│  (Daily OHLCV, Returns, Volumes, Dividend Yields)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 SIGNAL GENERATION LAYER                              │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Price    │ │ Momentum │ │Volatility│ │ Mean Rev │ │Cross-Sec │ │
│  │ Signals  │ │ Signals  │ │ Signals  │ │ Signals  │ │ Signals  │ │
│  │ (~15)    │ │ (~15)    │ │ (~20)    │ │ (~10)    │ │ (~15)    │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                                      │
│  Post-processing: Winsorize → Z-score normalize → Reindex           │
│  Output: ~100 candidate signals (DataFrame dict)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 SIGNAL EVALUATION LAYER                              │
│                                                                      │
│  For each signal:                                                    │
│    1. Build quintile long-short portfolio (top/bottom 20%)          │
│    2. Compute daily P&L (1-day forward returns)                     │
│    3. Calculate 15+ metrics:                                        │
│       - Sharpe, IC mean/std/IR, max DD, Calmar                     │
│       - Hit rate, skewness, kurtosis                                │
│       - Turnover, cost sensitivity                                  │
│       - Per-regime Sharpe, regime robustness                        │
│                                                                      │
│  Output: metrics_df (100+ signals × 15+ metrics)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 SIGNAL FILTERING LAYER                               │
│                                                                      │
│  Cascading filters (in order):                                      │
│    1. Sharpe ≥ 0.5                                                  │
│    2. Max Drawdown ≤ 20%                                            │
│    3. Regime Robustness ≥ 0.3                                       │
│    4. IC Mean > 0                                                   │
│    5. ICIR ≥ 0                                                      │
│    6. |Correlation with existing| ≤ 0.7                             │
│                                                                      │
│  Output: ~10-30 filtered signals + correlation matrix               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────────────────┐
│  STRATEGY LIBRARY        │  │  ALPHA RESEARCH OUTPUTS              │
│  (10 concrete strategies)│  │  - Signal rankings                   │
│                          │  │  - Correlation structure              │
│  Each strategy:          │  │  - Regime analysis                   │
│  1. generate_signals()   │  │  - Cost sensitivity                  │
│  2. compute_weights()    │  └──────────────────────────────────────┘
│  3. Apply risk controls  │
└────────────┬─────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PORTFOLIO ALLOCATION LAYER                              │
│                                                                      │
│  1. Regime Classification (HMM / vol-based)                         │
│     → Low Vol | Medium Vol | High Vol | Crisis                      │
│                                                                      │
│  2. Regime-Conditional Parameter Adjustment                         │
│     → Strategy lookbacks, blend weights, target gross               │
│                                                                      │
│  3. Dynamic Strategy Allocation                                     │
│     → Rolling Sharpe (126d) + drawdown penalty + regime tilts       │
│     → Weight bounds: [2%, 40%] per strategy                         │
│                                                                      │
│  4. Volatility Targeting                                            │
│     → Scale to 10% annualized vol target                            │
│     → Leverage cap: 2.0x                                            │
│                                                                      │
│  5. Bayesian Kelly Sizing                                           │
│     → James-Stein shrinkage → fractional Kelly (0.25)              │
│     → Constraints: 5% per-asset, 2.0x leverage, 30% strategy       │
│                                                                      │
│  Output: final_weights (dates × securities)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BACKTEST ENGINE (Event-Driven Daily)                    │
│                                                                      │
│  For each date:                                                      │
│    1. Generate target weights from strategies + allocation          │
│    2. Generate orders (if Δweight > 0.1% threshold)                 │
│    3. Simulate execution (commission + spread + slippage)           │
│    4. Update portfolio state (positions, cash, P&L)                 │
│    5. Record daily returns, exposures, metrics                      │
│                                                                      │
│  Cost model: 2 bps commission + 5 bps spread + 3 bps slippage      │
│                                                                      │
│  Output: BacktestResult                                             │
│    - Daily returns, cumulative P&L, drawdown series                 │
│    - Sharpe, max DD, Calmar, win rate, turnover                    │
│    - Trade ledger, P&L attribution                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. Configuration & Parameters

### Signal/Alpha Configuration

```yaml
alpha:
  min_sharpe: 0.5
  max_drawdown: 0.20
  min_regime_robustness: 0.3
  max_correlation_existing: 0.7
  min_ic_mean: 0.0
  min_icir: 0.0
```

### Backtest Configuration

```yaml
backtest:
  initial_capital: 10_000_000  # $10M
  commission_bps: 2.0
  spread_bps: 5.0
  slippage_bps: 3.0
  max_leverage: 2.0
  target_volatility: 0.10  # 10%
  rebalance_threshold: 0.001  # 0.1%
  min_trade_notional: 100  # $100
```

### Position Sizing Configuration

```yaml
sizing:
  kelly_fraction: 0.25        # Quarter Kelly
  max_asset_exposure: 0.05    # 5% per asset
  max_leverage: 2.0           # 2x gross leverage
  max_strategy_exposure: 0.30 # 30% per strategy leg
```

### Strategy-Specific Parameters

```yaml
time_series_momentum:
  primary_lookback: 126
  multi_scale_lookbacks: [63, 126, 252]
  multi_scale_weights: [0.4, 0.4, 0.2]
  target_gross: 1.0
  trend_strength_divisor: 0.20

cross_sectional_momentum:
  lookback: 252
  skip_period: 30
  quantile: 0.20
  drawdown_circuit_breaker: -0.25

mean_reversion:
  zscore_lookback: 120
  entry_threshold: 1.5
  exit_threshold: 0.5
  max_holding_days: 5
  stop_loss: 0.03
  circuit_breaker: -0.15
  trend_filter_slope: 0.001

carry:
  smoothing_window: 63
  quantile: 0.25
  rebalance_frequency: 21

volatility_breakout:
  atr_period: 14
  breakout_multiplier: 1.5
  volume_confirmation: 1.5
  max_holding_days: 3
  trailing_stop_atr_multiple: 1.5
  max_position: 0.05

low_risk_bab:
  beta_window: 252
  vol_window: 63
  quantile: 0.20
  rebalance_frequency: 21

residual_momentum:
  regression_window: 252
  momentum_window: 252
  skip_period: 21
  quantile: 0.20
  rebalance_frequency: 21

factor_momentum:
  factor_lookback: 126
  momentum_lookback: 126
  skip_period: 21
  target_gross: 1.0

pca_stat_arb:
  n_components: 5
  min_variance_explained: 0.50
  refit_frequency: 63
  entry_threshold: 1.5
  exit_threshold: 0.5
  max_holding_days: 20

kalman_pairs:
  n_pairs: 10
  process_noise: 0.0001
  observation_noise: 1.0
  entry_threshold: 2.0
  exit_threshold: 0.5
  max_holding_days: 30
  rescreen_frequency: 252

distance_pairs:
  n_pairs: 20
  formation_period: 252
  entry_threshold: 2.0
  exit_threshold: 0.5
  max_holding_days: 30
  rescreen_frequency: 252
```

---

## 12. Known Limitations & Research Opportunities

### Signal Generation Gaps

1. **No alternative data signals**: No sentiment, news, earnings, or macro signals
2. **No microstructure signals**: No order flow, bid-ask spread, or tick-level features
3. **No fundamental signals**: No value (P/E, P/B), quality, or profitability factors
4. **No inter-asset signals**: No lead-lag, spillover, or network-based signals
5. **No options-derived signals**: No implied volatility, skew, or put-call ratio
6. **No seasonal/calendar signals**: No day-of-week, month-of-year, or holiday effects
7. **Linear signals only**: No non-linear transformations, interaction terms, or ML-derived features

### Signal Processing Gaps

1. **Static lookback windows**: No adaptive or data-driven window selection
2. **No signal decay modeling**: No analysis of how fast signal predictive power decays
3. **No multi-horizon evaluation**: Only 1-day forward return IC computed
4. **No signal combination optimization**: Signals are used independently by strategies, not combined into composite alpha
5. **No online learning**: Signals don't adapt parameters based on recent performance
6. **No non-linear combination**: No ML models for combining signals (only linear blending in TSM)

### Evaluation Gaps

1. **Single-horizon IC**: Only next-day returns used; no 5d, 21d, or 63d IC analysis
2. **No IC decay curve**: Don't measure how IC changes with forward horizon
3. **No conditional IC**: Don't measure IC conditioned on regime, volatility, or other signals
4. **No out-of-sample evaluation**: All evaluation appears in-sample
5. **No Bayesian evaluation**: No posterior predictive checks or uncertainty in metrics
6. **No turnover-adjusted metrics**: Sharpe is pre-cost; need net-of-cost optimization

### Portfolio Construction Gaps

1. **No covariance-aware allocation**: Strategy weights ignore cross-strategy covariance
2. **No risk parity across strategies**: Equal vol contribution not implemented
3. **No dynamic Kelly updating**: Kelly parameters fixed; not updated adaptively
4. **No tail-risk hedging**: No explicit protection against extreme events beyond circuit breakers
5. **No transaction cost optimization**: Weights don't account for trading costs in optimization

### Risk Management Gaps

1. **Fixed circuit breaker thresholds**: Not adaptive to regime or recent performance
2. **No expected shortfall (CVaR) optimization**: Risk measured by volatility, not tail risk
3. **No correlation regime detection**: No dynamic monitoring of cross-strategy correlations
4. **No crowding detection**: No measurement of strategy crowdedness
5. **No liquidity adjustment**: Position sizes don't account for market liquidity

### Architecture Gaps

1. **Daily frequency only**: No intraday signals or execution
2. **No streaming/real-time**: Batch processing only
3. **No walk-forward optimization**: No expanding/rolling window parameter selection
4. **No ensemble methods**: No model averaging, stacking, or boosting for signal combination
5. **No causal inference**: No methods to distinguish spurious from causal signal relationships

---

## Research Questions for Deep Investigation

### Signal Enhancement
1. How can we add **non-linear signal transformations** (e.g., polynomial features, wavelet decomposition, Fourier analysis) to capture patterns that linear signals miss?
2. What **interaction signals** between existing families (e.g., momentum × volatility, carry × mean-reversion) could improve prediction?
3. How can **adaptive lookback windows** (e.g., based on regime, autocorrelation, or Hurst exponent) improve signal quality?
4. What **alternative data** signals (sentiment, flow, macro) would be most complementary to existing price-based signals?

### Signal Combination
5. Can **ML meta-models** (gradient boosting, neural networks) combine signals more effectively than quintile sorting or linear blending?
6. How should we build **composite alpha** from multiple signals while managing multicollinearity?
7. What is the optimal **signal weighting scheme** — equal weight, IC-weighted, Sharpe-weighted, or learned?
8. How can we use **attention mechanisms** or transformer architectures to dynamically weight signals based on current market conditions?

### Signal Evaluation
9. What is the **IC decay curve** for each signal family — how fast does predictive power fade from 1-day to 63-day horizons?
10. How does **conditional IC** (IC during high vs. low volatility, or up vs. down markets) vary across signals?
11. Can we use **Bayesian methods** to better estimate signal quality with uncertainty bounds?
12. How should we implement **walk-forward evaluation** to get unbiased performance estimates?

### Portfolio Construction
13. How can we move from **strategy-level allocation** to **signal-level allocation** for finer-grained portfolio construction?
14. What **risk parity** or **hierarchical risk parity** methods would improve multi-strategy combination?
15. How can we incorporate **transaction costs directly into the optimization** rather than as a post-hoc adjustment?
16. What **tail risk measures** (CVaR, expected shortfall) should replace or supplement volatility in the allocation framework?

### Risk Management
17. How can **adaptive circuit breakers** (based on regime or realized tail risk) improve on fixed thresholds?
18. What **crowding indicators** could warn when our strategies are too correlated with the broader market?
19. How can **dynamic correlation monitoring** between strategies trigger preemptive de-risking?
20. What **stress testing** methods should be applied to validate signal robustness to extreme scenarios?

---

*Document generated for deep research into signal architecture improvements. All signal families, strategies, processing steps, evaluation metrics, and risk controls are described as implemented in the current codebase.*
