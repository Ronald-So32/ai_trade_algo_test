# Trading Strategy Research Brief

> **Purpose**: Comprehensive documentation of all 10 trading strategies currently deployed in a quantitative equity research platform. Use this document to research complementary strategies, identify gaps in the current strategy mix, and propose higher-returning or more diversifying additions.
>
> **Universe**: 49 US large-cap equities across 7 sectors (Tech, Financials, Healthcare, Consumer, Industrials, Energy, Communications)
> **Data**: Daily OHLCV bars, 2010-01-04 to 2026-03-13 (real Yahoo Finance data)
> **Frequency**: All strategies operate on daily bars
> **Risk Constraint**: Every strategy has an iterative drawdown circuit breaker that caps individual MaxDD at 20%

---

## Current Performance Summary (Real Data, With 20% MaxDD Cap)

| Strategy | Sharpe | Ann. Return | MaxDD | Turnover | Category |
|---|---|---|---|---|---|
| vol_managed | 1.038 | 11.34% | -17.84% | 0.010 | Overlay / Risk Mgmt |
| cross_sectional_momentum | 0.426 | 2.87% | -18.65% | 0.040 | Momentum |
| time_series_momentum | 0.270 | 2.32% | -17.27% | 0.065 | Momentum |
| volatility_breakout | 0.355 | 3.13% | -15.74% | 0.248 | Short-Term / Breakout |
| pca_stat_arb | 0.128 | 0.81% | -16.49% | 0.144 | Statistical Arbitrage |
| carry | 0.017 | 0.12% | -18.33% | 0.026 | Income / Carry |
| distance_pairs | -0.139 | -0.33% | -11.72% | 0.003 | Pairs Trading |
| kalman_pairs | -0.422 | -0.86% | -16.00% | 0.007 | Pairs Trading |
| mean_reversion | -0.085 | -0.09% | -3.94% | 0.008 | Mean Reversion |
| factor_momentum | -0.305 | -0.63% | -10.00% | 0.012 | Factor / Multi-Factor |
| **DYNAMIC PORTFOLIO** | **1.308** | **12.73%** | **-14.29%** | — | Combined |

---

## Portfolio Construction

The strategies are combined using:
1. **Static Risk Parity** (inverse-variance weighting) as baseline
2. **Dynamic Adaptive Allocation** — rolling 126-day Sharpe-weighted with regime tilts, drawdown penalty (halves weight when DD > -10%), and monthly rebalance
3. **Tail Risk Management** — correlation-aware scaling, drawdown-speed scaling, left-tail dampening
4. **Vol Targeting** — scales portfolio to 10% annualized volatility
5. Dynamic is chosen over static when it has higher Sharpe (it usually does: 1.308 vs 0.897)

**Monte Carlo Analysis** (5,000 block-bootstrap simulations):
- Median Sharpe: 1.309 | Median CAGR: 13.04% | Median MaxDD: -14.51%
- Probability of ruin: 0.00% | P5/P50/P95 terminal wealth: 3.86 / 7.25 / 13.66

---

## Strategy 1: Time-Series Momentum

**Category**: Absolute Momentum / Trend Following
**Academic Basis**: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
**Sharpe**: 0.270 | **MaxDD**: -17.27%

### Signal Generation
- Blends momentum at **three time scales**: 63-day, 126-day, and 252-day trailing returns
- Signal = weighted sum of sign(trailing_return) at each scale: `0.4 × sign(63d) + 0.4 × sign(126d) + 0.2 × sign(252d)`
- **Trend-strength scaling**: signal is multiplied by `min(1, |trailing_return| / 0.20)` — weak trends produce smaller positions
- Requires a full 252-day window before producing signals

### Position Sizing
- Positions are **volatility-scaled**: `weight_i = signal_i / realized_vol_i` (63-day rolling std, annualized)
- **Vol-of-vol reduction**: when an asset's rolling vol-of-vol exceeds its median, position is reduced by 50% — this prevents blowup when volatility itself is unstable
- All weights normalized to target gross exposure of 1.0

### Risk Controls
- Vol floor of 1% to prevent leverage blow-up on low-vol assets
- Universal 20% MaxDD iterative circuit breaker

### Parameters
```
lookback=252, vol_lookback=63, target_gross=1.0, vol_floor=0.01
multi_scale_weights=(0.4, 0.4, 0.2), trend_strength_cap=0.20
vov_reduction=0.50, vov_lookback=63
```

### Strengths
- Captures sustained trends across asset-level time series
- Multi-scale blending reduces whipsaws compared to single-lookback momentum
- Vol-of-vol scaling is a defensive innovation beyond standard vol scaling

### Weaknesses
- Underperforms in choppy, range-bound markets (no trend to follow)
- 252-day lookback is slow to react to regime shifts
- Currently Sharpe = 0.27 on real data — mediocre standalone

---

## Strategy 2: Cross-Sectional Momentum

**Category**: Relative Momentum / Rank-Based
**Academic Basis**: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
**Sharpe**: 0.426 | **MaxDD**: -18.65%

### Signal Generation
- For each date, compute **momentum return** = `P(t - skip) / P(t - lookback) - 1` with skip_days=21 to avoid short-term reversal
- Rank all assets by momentum return
- **Top quintile** (20% of assets) → long; **bottom quintile** → short
- Weights within each leg are proportional to **cross-sectional z-scores** of momentum returns (not equal-weighted)

### Position Sizing
- Z-score weighted: assets with more extreme momentum get larger positions within each leg
- Long leg sums to `target_gross / 2 = 0.30` (reduced from 0.50); short leg sums to `-0.30`
- Note: `target_gross` reduced to 0.60 (from default 1.0) specifically for risk control

### Risk Controls
- Built-in drawdown circuit breaker: when strategy DD exceeds -25%, exposure is halved for 21 days
- Plus the universal 20% MaxDD cap applied on top
- 21-day skip period avoids short-term reversal effect

### Parameters
```
lookback=252, skip_days=21, decile=0.20, target_gross=0.60
dd_threshold=0.25, dd_reduction=0.50, dd_cooldown=21
```

### Strengths
- Cross-sectional design is market-neutral (dollar-neutral long-short)
- Z-score weighting concentrates in strongest signals vs equal-weight
- Well-established academic premium

### Weaknesses
- Momentum crashes: can experience severe drawdowns when trends reverse abruptly (e.g., March 2009 momentum crash)
- The "loser" leg can produce large losses during sharp V-shaped recoveries
- Real-data Sharpe of 0.43 is modest — the premium may have decayed in recent years

---

## Strategy 3: Mean Reversion

**Category**: Short-Term Reversal / Statistical
**Academic Basis**: Poterba & Summers (1988), Lo & MacKinlay (1990)
**Sharpe**: -0.085 | **MaxDD**: -3.94%

### Signal Generation
- Computes **z-score of log prices** vs their 120-day rolling mean/std
- Signal = `-z_score` (fade deviations: buy low, sell high)
- **Entry/exit rules**: enter when |z| > 1.5 (2.0 in current config), exit when |z| < 0.5
- **Maximum holding period**: 5 days (forces position closure)
- **Stop-loss**: exits if position PnL drops below -3%

### Filters
- **Trend filter**: suppresses signals when the 200-day SMA slope exceeds ±0.1% per day — won't mean-revert against a strong trend
- **Drawdown circuit breaker**: goes flat if cumulative strategy return drops below -15%, stays flat for 42 days

### Position Sizing
- Volatility-scaled: `weight = signal / realized_vol` (21-day lookback, annualized)
- Normalized to target gross = 1.0

### Parameters
```
lookback=21 (overridden from default 120), entry_threshold=2.0
exit_threshold=0.5, max_holding=5, target_gross=1.0
vol_scale=True, vol_lookback=21, vol_floor=0.005
trend_sma_period=200, trend_slope_limit=0.001
stop_loss_pct=0.03, dd_circuit_breaker=-0.15, dd_cooldown_days=42
```

### Strengths
- Very low MaxDD (-3.94%) — extremely conservative
- Complementary to momentum (negative correlation)
- Trend filter prevents fighting strong directional moves

### Weaknesses
- Negative Sharpe on real data — the strategy isn't profitable with current parameters
- Short holding period (5 days) may be too restrictive for the 21-day lookback
- Aggressive trend filter may suppress too many valid signals
- Mean reversion in individual US equities is a weak effect vs. index/ETF level

---

## Strategy 4: Distance-Based Pairs Trading

**Category**: Statistical Arbitrage / Pairs Trading
**Academic Basis**: Gatev, Goetzmann & Rouwenhorst (2006)
**Sharpe**: -0.139 | **MaxDD**: -11.72%

### Signal Generation
- **Formation period** (252 days): normalizes prices to start at 1.0, then finds the 20 closest pairs by sum-of-squared-distances
- **Trading period**: for each pair (A, B), computes spread = norm_A - norm_B
- Z-scores the spread using formation-period mean/std
- **Enter** when |z| > 2.0: if spread too wide (z > 2), short A / long B; if spread too narrow (z < -2), long A / short B
- **Exit** when |z| < 0.5 or after 30 days
- Pairs are re-screened every 252 days

### Position Sizing
- Signals from all pairs for each asset are summed (clipped to [-1, +1])
- Equal-weight within active positions, scaled to target gross = 1.0

### Parameters
```
formation_period=252, n_pairs=20, entry_z=2.0, exit_z=0.5
max_holding=30, target_gross=1.0, rebalance_freq=252
```

### Strengths
- Market-neutral by construction (every trade is a pair)
- No directional market exposure
- Low turnover

### Weaknesses
- Negative Sharpe on real data — distance-based pair selection doesn't capture cointegration well
- Sum-of-squared-distances selects similar price paths, not mean-reverting pairs
- 252-day formation/rebalance is very slow to adapt
- In a trending market, all stocks may diverge from formation-period relationships

---

## Strategy 5: Kalman Filter Pairs Trading

**Category**: Statistical Arbitrage / Adaptive Pairs
**Academic Basis**: Elliott, van der Hoek & Malcolm (2005), Triantafyllopoulos & Montana (2011)
**Sharpe**: -0.422 | **MaxDD**: -16.00%

### Signal Generation
- **Pair selection**: ranks pairs by residual variance of OLS spread (lower variance = tighter pair), selects top 10
- **Dynamic hedge ratio**: uses a Kalman filter with state vector `[α, β]` modeled as a random walk:
  - `y_t = α_t + β_t × x_t + ε_t` (where y = log(price_A), x = log(price_B))
  - Process noise parameter `delta = 1e-4`, observation noise = 1.0
  - Falls back to rolling 60-day OLS if pykalman is unavailable
- **Z-score**: computed from rolling residual mean/std (last 30 observations)
- Entry at |z| > 2.0, exit at |z| < 0.5 or after 30 days
- 30-day warmup before trading

### Position Sizing
- Same as distance pairs: sum of pair signals per asset, clipped to [-1, +1], normalized to target gross = 1.0

### Parameters
```
n_pairs=10, formation_period=252, entry_z=2.0, exit_z=0.5
max_holding=30, delta=1e-4, obs_cov=1.0
rebalance_freq=252, target_gross=1.0, warmup=30
```

### Strengths
- Kalman filter adapts hedge ratios in real-time (vs static OLS)
- Theoretically better at tracking non-stationary relationships
- Market-neutral

### Weaknesses
- Negative Sharpe — not profitable on this universe
- Kalman filter needs careful tuning of delta/obs_cov (current defaults may not suit equities)
- Pair selection via residual variance is a weak proxy for cointegration
- Small number of pairs (10) + infrequent re-screening (252 days) limits alpha capture

---

## Strategy 6: Volatility Breakout

**Category**: Short-Term Event-Driven / Breakout
**Academic Basis**: Keltner Channels, ATR breakout systems
**Sharpe**: 0.355 | **MaxDD**: -15.74%

### Signal Generation
- Computes **ATR** (Average True Range) using Wilder's EWM smoothing over 14 days
- Signal fires when **daily range (H - L) > 1.5 × ATR(14)**
- Direction: if close > open → bullish (+1), else bearish (-1)
- **Volume confirmation**: requires volume > 1.5× the 20-day average volume
- **Holding period**: 3 days after breakout
- **Trailing stop**: exits if PnL drops below -1.5 × ATR from entry

### Position Sizing
- Inversely proportional to realized volatility (21-day lookback)
- Per-asset weight capped at 5% to prevent concentration
- Normalized to target gross = 1.0

### Parameters
```
atr_period=14, breakout_mult=1.5, vol_lookback=21, vol_floor=0.005
target_gross=1.0, holding_days=3(overridden to 5)
volume_confirm=True, volume_multiplier=1.5, volume_avg_period=20
max_weight_per_asset=0.05, trailing_stop_atr_mult=1.5
```

### Strengths
- Captures short-term directional moves after volatility expansion
- Volume confirmation filters out false breakouts
- Trailing stop limits losses on failed breakouts
- Higher turnover but good risk-adjusted performance (Sharpe 0.36)

### Weaknesses
- High turnover (0.25) increases transaction costs
- Short holding period means many small trades
- Falls back to synthetic OHLC when real high/low/open data isn't provided (loses accuracy)
- Performance depends heavily on ATR period and breakout multiplier tuning

---

## Strategy 7: Carry Strategy

**Category**: Income / Cross-Sectional Value
**Academic Basis**: Koijen, Moskowitz, Pedersen & Vrugt (2018) "Carry"
**Sharpe**: 0.017 | **MaxDD**: -18.33%

### Signal Generation
- **Carry signal**: ranks assets by dividend yield or carry proxy
  - If dividend yields DataFrame provided → uses directly
  - If total returns provided → carry = total_return - price_return (dividend component)
  - Fallback → rolling 63-day average of excess return vs cross-sectional mean
- Ranks all assets: **top 25%** → long (high carry), **bottom 25%** → short (low carry)
- Rebalances every **21 days** (carry-forward between rebalances)

### Position Sizing
- Equal-weight within long and short legs
- Each leg gets `target_gross / 2 = 0.50`

### Parameters
```
carry_lookback=63, n_quantile=0.25, target_gross=1.0
min_assets=4, rebalance_freq=21
```

### Strengths
- Low turnover (monthly rebalance)
- Theoretically captures the carry premium (compensation for risk)
- Market-neutral (long-short)

### Weaknesses
- Near-zero Sharpe on real data — carry is very weak in US equities without dividend yield data
- The fallback carry proxy (rolling excess return) is noisy and may capture momentum instead
- In the current implementation, dividends are sparse in daily data, making the carry signal unreliable
- Carry is much stronger in FX, rates, and commodities than equities

---

## Strategy 8: Factor Momentum

**Category**: Multi-Factor / Cross-Sectional + Time-Series
**Academic Basis**: Arnott, Clements, Kalesnik & Linnainmaa (2023) "Factor Momentum Everywhere", Ehsani & Linnainmaa (2022)
**Sharpe**: -0.305 | **MaxDD**: -10.00%

### Signal Generation
1. **Constructs 3 factor portfolios** each day from the cross-section:
   - **Value**: long bottom quintile by trailing 126-day return (contrarian — "cheap" stocks)
   - **Size**: long bottom quintile by average 63-day price level (small-cap proxy)
   - **Momentum**: long top quintile by trailing 126-day return (skip 21 days)
2. **Computes factor returns** as long-short equal-weighted portfolio returns
3. **Applies time-series momentum** to factor returns: if a factor's cumulative 126-day return (skip 21 days) is positive, tilt toward that factor
4. **Factor vol scaling**: tilts are scaled by `sign(factor_return) / factor_vol` (63-day lookback)
5. **Aggregates**: each asset's signal = sum of (factor_tilt × factor_membership)

### Position Sizing
- Volatility-scaled: `weight = signal / realized_vol` (63-day lookback)
- Normalized to target gross = 1.0

### Parameters
```
factor_lookback=126, momentum_lookback=126, skip_days=21
vol_lookback=63, target_gross=1.0, quantile=0.20
```

### Strengths
- Second-order alpha: timing factors themselves, not just stocks
- Diversifies across value, size, and momentum premia
- Academic evidence for factor momentum is strong

### Weaknesses
- Negative Sharpe on real data — factor definitions may be too simplistic for equities
- Using price level as "size" proxy is crude (should use market cap)
- Value defined as contrarian return is really short-term reversal, not book/price
- Requires many assets for clean factor construction; 49 may be insufficient

---

## Strategy 9: PCA Statistical Arbitrage

**Category**: Statistical Arbitrage / Eigenportfolio Mean Reversion
**Academic Basis**: Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"
**Sharpe**: 0.128 | **MaxDD**: -16.49%

### Signal Generation
1. **PCA decomposition**: fits PCA on 252-day rolling returns window, extracts top 5 components (auto-expands if <50% variance explained)
2. **Idiosyncratic residuals**: `residual = actual_return - systematic_component` where systematic = projection onto PCA factors
3. **Cumulative residual**: tracks running sum of daily residuals per asset (synthetic "price")
4. **Z-score**: rolling 60-day z-score of cumulative residual
5. **Mean-revert on residuals**: long when z < -1.5 (residual too low), short when z > 1.5
6. **Exit**: when |z| < 0.5 or after 20 days
7. PCA is refitted every **63 days**

### Position Sizing
- Volatility-scaled: `weight = signal / realized_vol` (63-day window, annualized)
- Normalized to target gross = 1.0

### Parameters
```
n_components=5, lookback=252, entry_z=1.5(overridden to 2.0), exit_z=0.5
max_holding=20, target_gross=1.0, refit_freq=63
zscore_window=60, min_variance_explained=0.5
```

### Strengths
- Market-neutral: strips out systematic risk, trades only idiosyncratic moves
- PCA is data-driven — no need to pre-specify factors
- Refitting every 63 days adapts to evolving factor structure
- Positive Sharpe (0.13) on real data

### Weaknesses
- Modest alpha — PCA residuals may not be truly stationary
- High turnover (0.14) relative to return
- Forward-pass simulation per-asset is computationally expensive
- Only uses 5 PCA components — may not capture all systematic risk in 49 stocks

---

## Strategy 10: Volatility-Managed Portfolio Overlay

**Category**: Risk Management / Overlay
**Academic Basis**: Moreira & Muir (2017) "Volatility-Managed Portfolios"
**Sharpe**: 1.038 | **MaxDD**: -17.84%

### Signal Generation
- Not a directional strategy — it's an **overlay** that scales any base portfolio's exposure
- Scale factor: `scale_t = min(target_vol / realized_vol_{t-1}, max_leverage)`
- Uses **lagged** realized vol (no look-ahead): either rolling 63-day std or EWM
- Currently wraps an **equal-weight long-only portfolio** of all 49 stocks

### How It Works
1. Estimate portfolio realized vol from prior day's rolling window
2. If vol is low (e.g., 5%) and target is 10% → lever up 2x
3. If vol is high (e.g., 20%) and target is 10% → scale down to 0.5x
4. Hard cap at 2.0x leverage, floor at 0.5% vol

### Why It Produces the Best Returns
- **Buys low vol, sells high vol**: when markets are calm, it levers up; during crises, it scales down — this is empirically the correct timing
- **Variance timing premium**: high-vol periods have worse risk-adjusted returns (documented by Moreira & Muir); avoiding them is profitable
- **Convex payoff**: compounds more in calm uptrends (levered), loses less in crashes (de-levered)
- **No directional prediction needed**: only scales exposure based on vol forecast, which is highly predictable

### Parameters
```
target_vol=0.10, vol_lookback=63, max_leverage=2.0
vol_floor=0.005, ewm_span=None, annualisation_factor=252
```

### Strengths
- Highest Sharpe (1.04) by a wide margin
- Can overlay on ANY base strategy, not just equal-weight
- Simple, robust, hard to overfit (one key parameter: target_vol)
- Backed by strong academic evidence

### Weaknesses
- Currently overlays on naive equal-weight — performance depends on the underlying portfolio
- Max leverage of 2.0 means it benefits from calm bull markets (2013-2019, 2023-2024)
- Could underperform if volatility becomes persistently high
- Transaction costs from leverage changes not fully accounted for

---

## Universal Risk Controls

All strategies share the following risk infrastructure:

### 1. Iterative Drawdown Circuit Breaker (in `Strategy.apply_drawdown_cap()`)
- **Trigger**: 75% of max_dd threshold (i.e., at -15% when cap is -20%)
- **Action**: zeroes out all weights for 21 days (cooldown)
- **Iterative**: runs up to 5 passes, tightening the trigger by 20% each pass, until MaxDD converges under the target
- This prevents single-day crashes from blowing through the cap

### 2. Regime-Adaptive Parameter Overrides
Different strategy parameters are used for each volatility regime:
- **low_vol** (vol < 25th percentile): higher target_gross, tighter entry thresholds
- **medium_vol** (25th-60th percentile): default parameters
- **high_vol** (60th-90th percentile): reduced target_gross, wider thresholds
- **crisis** (vol > 90th percentile): minimal exposure, very wide thresholds

### 3. Dynamic Allocation
- Rolling 126-day Sharpe-weighted strategy allocation (monthly rebalance)
- Drawdown penalty: strategy weight halved when DD > -10%
- Regime tilts: overweight vol_managed and mean_reversion in crisis, overweight momentum in low_vol
- Min/max weight bounds: 2%–40% per strategy

### 4. Tail Risk Management
- **Correlation-aware scaling**: reduces 40% when average strategy correlation > 0.50
- **Drawdown-speed scaling**: halves exposure when DD velocity > -0.5%/day over 10 days
- **Left-tail dampening**: cuts 30% on >2-sigma negative portfolio return days

---

## Strategy Categorization & Correlation Structure

| Category | Strategies | Expected Correlation |
|---|---|---|
| **Momentum** | time_series_momentum, cross_sectional_momentum, factor_momentum | High within category |
| **Mean Reversion** | mean_reversion, distance_pairs, kalman_pairs, pca_stat_arb | Moderate within category; **negatively correlated with momentum** |
| **Breakout / Event** | volatility_breakout | Low correlation with others |
| **Income / Value** | carry | Low correlation with momentum strategies |
| **Risk Management** | vol_managed | Correlated with underlying (equal-weight market) |

---

## Identified Gaps & Opportunities for Complementary Strategies

Based on the current mix, the following strategy types are **missing or underrepresented**:

### 1. Missing Strategy Categories
- **Trend Following on Futures/Macro**: all strategies trade individual equities — no exposure to bonds, commodities, FX, or rates
- **Options-Based Strategies**: no volatility selling (put writing), dispersion trading, or volatility carry
- **Sentiment / Alternative Data**: no NLP signals from news, earnings calls, social media, or analyst revisions
- **High-Frequency / Microstructure**: all strategies are daily — no intraday alpha
- **Event-Driven**: no earnings momentum, M&A arbitrage, or corporate event strategies
- **Quality / Defensive**: no quality factor (ROE, margins, earnings stability), low-vol anomaly, or profitability factor
- **Seasonality**: no monthly/quarterly seasonality effects (e.g., turn-of-month, January effect)

### 2. Weaknesses in Current Mix
- **4 out of 10 strategies have negative Sharpe** — they are drag on the portfolio even with dynamic allocation
- **Pairs trading is broken**: both distance and Kalman pairs are unprofitable — possibly need cointegration-based pair selection (Engle-Granger, Johansen) or shorter formation/trading periods
- **Carry is too weak** in equities without proper dividend yield data
- **Factor momentum uses crude factor definitions** — needs proper market cap for size, book-to-market for value
- **Mean reversion works poorly** on individual stocks at daily frequency — may be better at sector/ETF level or intraday

### 3. Suggestions for Research
- **Machine Learning Strategies**: gradient-boosted return prediction, neural network alpha signals, reinforcement learning for portfolio optimization
- **Cross-Asset Momentum**: extend momentum to trade equity sectors alongside treasury bonds and commodities (all-weather portfolio)
- **Short-Term Reversal (1-5 day)**: the Jegadeesh (1990) short-term reversal premium, properly implemented at the portfolio level
- **Earnings Momentum (PEAD)**: post-earnings announcement drift is one of the strongest and most persistent anomalies
- **Low-Volatility Anomaly**: long low-vol stocks / short high-vol stocks — defensive, high Sharpe, low correlation with momentum
- **Dispersion Trading**: sell index vol, buy single-stock vol when implied correlation is high
- **Pair Selection via Cointegration**: replace distance/Kalman pair selection with proper Engle-Granger or VECM-based testing
- **Sector Rotation**: momentum/mean-reversion at the GICS sector level rather than individual stocks
- **Risk Premia Harvesting**: systematic strategies targeting specific risk premia (variance risk premium, term premium, credit spread)

---

## Technical Implementation Details

### Data Pipeline
- Real data fetched from Yahoo Finance via `yfinance` library
- 49 tickers across 7 GICS sectors (Technology, Financials, Healthcare, Consumer Discretionary, Industrials, Energy, Communications)
- Daily OHLCV bars from 2010-01-04 to present
- Adjusted close prices used for returns calculation
- Dividends captured for carry strategy

### Framework Architecture
```
qrt/
├── strategies/          # All 10 strategy implementations
│   ├── base.py          # Abstract base class with drawdown cap
│   ├── time_series_momentum.py
│   ├── cross_sectional_momentum.py
│   ├── mean_reversion.py
│   ├── distance_pairs.py
│   ├── kalman_pairs.py
│   ├── volatility_breakout.py
│   ├── carry.py
│   ├── factor_momentum.py
│   ├── pca_stat_arb.py
│   └── vol_managed.py
├── portfolio/           # Portfolio construction
│   ├── optimizer.py     # Risk parity
│   ├── adaptive_allocation.py  # Dynamic weights + tail risk
│   └── vol_targeting.py
├── regime/              # Regime detection
│   ├── volatility_regime.py   # Percentile-based vol regimes
│   └── hmm_regime.py          # Hidden Markov Model
├── risk/                # Risk analysis
│   └── monte_carlo.py   # Block bootstrap, permutation, correlation stress, leverage stress
├── alpha_engine/        # Automated alpha discovery
├── ml_meta/             # ML meta-model for strategy weighting
└── dashboard/           # HTML dashboard generation
```

### Strategy Interface
Every strategy implements:
```python
class Strategy(ABC):
    def generate_signals(prices, returns, **kwargs) -> DataFrame  # [-1, +1]
    def compute_weights(signals, **kwargs) -> DataFrame            # portfolio weights
    def backtest_summary(weights, returns) -> dict                 # performance metrics
    def apply_drawdown_cap(weights, returns, max_dd) -> DataFrame  # risk control
```

---

*Generated from the QRT Quantitative Research Platform, March 2026*
