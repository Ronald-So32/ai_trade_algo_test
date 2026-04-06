# Backtest Integrity Validation Report

**Overall Status: VALIDATED**
**Data Source: real**

| Metric | Count |
|--------|-------|
| Passed | 40 |
| Failed | 0 |
| Warnings | 1 |

## Data Source

- [PASS] **Data source type**: Platform configured with real market data (source='real')
- [PASS] **Strategy independence from synthetic data**: No strategy files directly reference synthetic data generators
- [PASS] **Real data APIs**: Real data ingestion uses: yfinance, yf.Ticker
  - File: `qrt/data/real_data.py`
- [PASS] **Universe coverage**: Real universe contains 52 unique tickers across multiple sectors
  - File: `qrt/data/real_data.py`
- [PASS] **Storage: market_data.parquet**: Found market_data.parquet (11.0 MB)
  - File: `data/parquet/market_data.parquet`
- [PASS] **Storage: returns.parquet**: Found returns.parquet (11.5 MB)
  - File: `data/parquet/returns.parquet`
- [PASS] **Storage: security_master.parquet**: Found security_master.parquet (0.0 MB)
  - File: `data/parquet/security_master.parquet`
- [PASS] **Price positivity**: All prices are positive (no data corruption)
- [PASS] **Timestamp format**: Proper DatetimeIndex, range: 2010-01-04 to 2026-03-17
- [PASS] **No weekend data**: No weekend dates in price data (trading days only)

## Execution Lag

- [PASS] **Base strategy execution lag**: backtest_summary() applies weights.shift(1) before multiplying by returns — signals at t are traded at t+1
  - File: `qrt/strategies/base.py`
- [PASS] **Pipeline execution lag**: run_research.py also applies weights.shift(1) when computing strategy returns
  - File: `run_research.py`

## Execution Realism

- [PASS] **Transaction cost components**: All cost components present: commission, spread, slippage, turnover penalty
  - File: `qrt/costs/transaction_costs.py`
- [PASS] **Market impact model**: Square-root market impact model implemented (institutional standard)
  - File: `qrt/costs/transaction_costs.py`

## Model Validation

- [PASS] **Walk-forward train/test split**: Walk-forward module implements train/test temporal split
  - File: `qrt/walkforward/walk_forward.py`
- [PASS] **Rolling window validation**: Walk-forward uses rolling windows (not single fixed split)
  - File: `qrt/walkforward/walk_forward.py`
- [WARN] **Purge gap**: No explicit purge gap found — potential information leakage at window boundaries
  - File: `qrt/walkforward/walk_forward.py`
- [PASS] **Time-series cross-validation**: ML meta-model uses time-series aware CV (not random k-fold)
  - File: `qrt/ml_meta/cross_validation.py`

## Research Grounding

- [PASS] **Research grounding: cross_sectional_momentum**: Strategy cross_sectional_momentum includes research grounding metadata
  - File: `qrt/strategies/cross_sectional_momentum.py`
- [PASS] **Research grounding: short_term_reversal**: Strategy short_term_reversal includes research grounding metadata
  - File: `qrt/strategies/short_term_reversal.py`
- [PASS] **Research grounding: vol_managed**: Strategy vol_managed includes research grounding metadata
  - File: `qrt/strategies/vol_managed.py`
- [PASS] **Research grounding: carry**: Strategy carry includes research grounding metadata
  - File: `qrt/strategies/carry.py`
- [PASS] **Research grounding: distance_pairs**: Strategy distance_pairs includes research grounding metadata
  - File: `qrt/strategies/distance_pairs.py`
- [PASS] **Research grounding: vol_risk_premium**: Strategy vol_risk_premium includes research grounding metadata
  - File: `qrt/strategies/vol_risk_premium.py`
- [PASS] **Research grounding: residual_momentum**: Strategy residual_momentum includes research grounding metadata
  - File: `qrt/strategies/residual_momentum.py`
- [PASS] **Research grounding: kalman_pairs**: Strategy kalman_pairs includes research grounding metadata
  - File: `qrt/strategies/kalman_pairs.py`
- [PASS] **Research grounding: pca_stat_arb**: Strategy pca_stat_arb includes research grounding metadata
  - File: `qrt/strategies/pca_stat_arb.py`
- [PASS] **Research grounding: ml_alpha_strategy**: Strategy ml_alpha_strategy includes research grounding metadata
  - File: `qrt/strategies/ml_alpha_strategy.py`
- [PASS] **Research grounding: time_series_momentum**: Strategy time_series_momentum includes research grounding metadata
  - File: `qrt/strategies/time_series_momentum.py`
- [PASS] **Research grounding: factor_momentum**: Strategy factor_momentum includes research grounding metadata
  - File: `qrt/strategies/factor_momentum.py`
- [PASS] **Research grounding: mean_reversion**: Strategy mean_reversion includes research grounding metadata
  - File: `qrt/strategies/mean_reversion.py`
- [PASS] **Research grounding: volatility_breakout**: Strategy volatility_breakout includes research grounding metadata
  - File: `qrt/strategies/volatility_breakout.py`
- [PASS] **Research grounding: pead**: Strategy pead includes research grounding metadata
  - File: `qrt/strategies/pead.py`
- [PASS] **Research grounding: low_risk_bab**: Strategy low_risk_bab includes research grounding metadata
  - File: `qrt/strategies/low_risk_bab.py`

## Signal Integrity

- [PASS] **ML target construction in ml_alpha_strategy.py**: .shift(-N) used in target construction method '_stack_features_panel()' — acceptable for ML training labels when walk-forward purge gap is enforced
  - File: `qrt/strategies/ml_alpha_strategy.py`:304
- [PASS] **Trailing lookback in cross_sectional_momentum.py**: Momentum signal uses trailing lookback window (point-in-time correct)
  - File: `qrt/strategies/cross_sectional_momentum.py`
- [PASS] **Trailing lookback in residual_momentum.py**: Momentum signal uses trailing lookback window (point-in-time correct)
  - File: `qrt/strategies/residual_momentum.py`
- [PASS] **Trailing lookback in time_series_momentum.py**: Momentum signal uses trailing lookback window (point-in-time correct)
  - File: `qrt/strategies/time_series_momentum.py`
- [PASS] **Trailing lookback in factor_momentum.py**: Momentum signal uses trailing lookback window (point-in-time correct)
  - File: `qrt/strategies/factor_momentum.py`
- [PASS] **Rolling statistics in mean_reversion**: Mean reversion uses rolling window statistics (no future leakage)
  - File: `qrt/strategies/mean_reversion.py`
- [PASS] **Lookahead bias scan**: No lookahead bias patterns found in strategy or signal generation code
