# FTMO 1-Step Research & Validation Protocol

## 1. Backtesting Requirements

### 1.1 Point-in-Time Data Discipline
Every dataset used in research and backtesting must enforce:
- **No future bars in features** — bar-close features usable only after bar closes
- **No future balance/equity/risk-budget state** visible to strategy at decision time
- **No post-midnight CEST rule reset** assumptions applied before reset occurs
- **No forward-filled external data** beyond what would have been known
- **No revised data** substituted for original vintages without documentation

### 1.2 Required Cost Model
No strategy may be approved on gross performance alone. Every backtest must include:
- **Spread** — symbol-specific, from MT5 `symbol_info().spread`
- **Commission** — per-lot, per-side
- **Slippage** — 1-2 pips for forex, 1-3 points for indices, 0.1-0.5% for equities
- **Swap/financing** — overnight positions incur daily swap (from MT5 `symbol_info().swap_long/swap_short`)
- **Rollover effects** — for commodity CFDs

### 1.3 Required Backtest Modes
- **Normal strategy backtest** — pure signal P&L
- **FTMO 1-Step challenge simulation** — with MDL, ML, best day rule overlays
- **Walk-forward** — rolling/expanding window
- **Rolling OOS** — non-overlapping test periods
- **Stress test** — wider spreads (2x), higher slippage (2x), weekend gaps

### 1.4 FTMO Challenge Simulation Overlay
Every backtest must also report performance under FTMO constraints:
- Track daily P&L and flag any day that would breach -3% ($3,000)
- Track cumulative drawdown and flag breach of -10% ($10,000)
- Track best day rule: `max_day_profit / sum(positive_day_profits) <= 0.50`
- Compute **FTMO pass rate** across rolling challenge windows
- Compute **average days to pass** when successful
- Compute **average days to fail** when unsuccessful

---

## 2. Out-of-Sample Requirements

### 2.1 OOS Must Dominate Narrative
No strategy may be approved based primarily on in-sample performance. The promotion memo must lead with:
- Walk-forward OOS
- Rolling OOS
- Challenge-mode OOS (with FTMO rule overlays)
- Net-of-cost OOS

### 2.2 Train / Validation / Test / Lockbox Structure
```
|---- Training ----|---- Validation ----|---- Research OOS ----|---- LOCKBOX ----|
     (tune)             (select)              (evaluate)          (UNTOUCHED)
```
- **Training**: fit model parameters
- **Validation**: select between model variants, tune hyperparameters
- **Research OOS**: evaluate final candidate, report headline metrics
- **Lockbox**: NEVER touch until final go/no-go decision. If opened, create new lockbox and log breach.

### 2.3 Purged/Embargoed Cross-Validation
For strategies with serial dependence (all of ours):
- Standard random K-fold is **forbidden**
- Use **purged and embargoed** splits with gap between train and test
- Minimum embargo: 5 trading days for reversal, 30 days for momentum
- Prefer expanding-window walk-forward over fixed-window

### 2.4 OOS Stability
A strategy must show acceptable stability across subperiods:
- Subperiod breakdown (quarterly or semi-annual)
- Volatility-regime breakdown (low vol vs high vol)
- Long-trade vs short-trade breakdown
- Pre/post major regime changes (COVID, rate hikes, etc.)

### 2.5 OOS Degradation Accounting
Report and justify:
- IS vs OOS return difference
- IS vs OOS Sharpe difference
- IS vs OOS max drawdown difference
- IS vs OOS FTMO pass-rate difference

---

## 3. Lookahead Bias Minimisation

### 3.1 Mandatory Anti-Lookahead Checks
Before any strategy is promoted, run a **point-in-time audit** that checks:

1. **No feature uses bars beyond decision time**
   - All trailing returns, vol estimates, and signals use data up to T-1 only
   - Bar at time T is not available until T+1

2. **No target leakage through shifted columns or merged joins**
   - Earnings dates known only after announcement
   - Price features use only pre-decision data

3. **No balance/equity state uses future updates**
   - FTMO rule engine state at time T uses only information available at T
   - No "if we will breach tomorrow, reduce today"

4. **No survivorship bias**
   - Don't filter universe based on future availability
   - Include delisted/removed instruments if they were tradeable at decision time

5. **No future FTMO challenge state visible to strategy**
   - Strategy cannot know "we are at +8%, reduce size" using future equity
   - Risk scaling uses only current (real-time) equity, not forecasted equity

6. **Feature availability timestamps**
   - Bar-close features: available after bar closes
   - Earnings features: available after earnings release + 1 hour
   - Macro features: available after release time + ingestion lag

### 3.2 Point-in-Time Audit Output
The audit produces:
- Machine-readable JSON report with pass/fail per check
- Human-readable Markdown report
- Strategy is **blocked from promotion** if any check fails

---

## 4. Statistical Validation Requirements

### 4.1 Metrics Always Reported
For every backtest, walk-forward, and OOS run:

| Metric | Formula/Description |
|--------|-------------------|
| CAGR | Annualised compound return |
| Cumulative P&L | Total $ profit/loss |
| Hit rate | % of winning trades |
| Payoff ratio | avg_win / avg_loss |
| Volatility | Annualised std of daily returns |
| Downside deviation | Std of negative returns only |
| Max drawdown | Peak-to-trough |
| Time under water | Longest drawdown duration |
| Turnover | Daily average portfolio turnover |
| Avg holding period | Days |
| Sharpe ratio | (mean - rf) / std, annualised |
| Sortino ratio | (mean - rf) / downside_dev |
| Calmar ratio | CAGR / max_drawdown |
| Profit factor | gross_wins / gross_losses |
| Skewness | 3rd moment of daily returns |
| Excess kurtosis | 4th moment - 3 |
| FTMO pass rate | % of simulated challenges that pass |
| Best-day concentration | max_day / sum(positive_days) |
| Near-breach count | Days within 20% of MDL or ML |

### 4.2 Sharpe Ratio Inference
- Do NOT annualise by naive sqrt(252) without checking serial independence
- Use Lo (2002) standard error: `SE(SR) = sqrt((1 + SR^2/2) / T)`
- Report 95% confidence intervals
- For strategy comparison: use Ledoit-Wolf robust bootstrap

### 4.3 Multiple Testing Corrections
When multiple strategies, parameters, or filters are tested:
- Record ALL variants tried (never delete failed experiments)
- Compute **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014)
- Compute **Probability of Backtest Overfitting** if >10 variants tested
- Report number of variants tried alongside headline metrics

---

## 5. Research Governance

### 5.1 Research Cards
Every strategy must have a Research Card at `research/cards/<strategy>.md` containing:
- Exact hypothesis
- Economic mechanism (why the edge exists)
- Target instruments and holding period
- At least 2 independent academic sources
- Known failure modes and risks
- FTMO-specific feasibility assessment
- What constitutes falsification

### 5.2 FTMO Feasibility Assessment
Every Research Card must address:
- Is the return path too bursty for the 50% Best Day Rule?
- Is the drawdown path too sharp for the 3% Max Daily Loss?
- Could open-loss behavior cause intraday MDL breaches?
- Do holding patterns create weekend/overnight gap vulnerability?
- Do spread/swap costs materially degrade the edge?
- How many trading days to reach target without concentration risk?

### 5.3 Economic Rationale Requirement
No strategy may be implemented unless it has a documented reason why the edge exists:
- Risk compensation
- Behavioral underreaction/overreaction
- Inventory or liquidity effects
- Event-information processing delay
- Cross-sectional mispricing
- Volatility-risk or tail-risk compensation

Reject strategies whose only rationale is "it backtested well."

---

## 6. Strategy-Specific Protocols

### 6.1 TSMOM (Time-Series Momentum)
- **Academic basis**: Moskowitz, Ooi & Pedersen (2012), JFE
- **Economic rationale**: Behavioral underreaction to trends; hedging pressure in futures
- **Lookback bias check**: Signal uses only past 63/126/252 days of returns
- **Backtest**: Minimum 3 years of daily data per instrument
- **Walk-forward**: 1-year training window, 3-month test, expanding
- **FTMO feasibility**: Convex payoff (Fung & Hsieh 2001) — many small losses, occasional large gains. Natural fit for challenge.

### 6.2 Residual Short-Term Reversal
- **Academic basis**: Blitz et al. (2013, 2023), JFM/JPM
- **Economic rationale**: Liquidity provision premium; overreaction to idiosyncratic news
- **Lookback bias check**: Uses only trailing 5-day returns and 21-day vol
- **Sector residuals**: Subtract same-day sector mean (not future sector mean)
- **Backtest**: Minimum 1 year of daily data for 53 equities
- **FTMO feasibility**: Generates consistent small daily gains — excellent for Best Day Rule compliance

### 6.3 Earnings Black Swan
- **Academic basis**: Custom ML model validated with 886 OOS trades (2015-2026)
- **Economic rationale**: Market underestimates tail risk around earnings
- **Lookback bias check**: Features use only pre-earnings data; model trained on walk-forward basis with 30-day embargo
- **Backtest**: Walk-forward with quarterly retraining folds
- **FTMO feasibility**: Short CFD with 3% stop limits max loss per trade. TP capped at $1,500 for Best Day Rule.

---

## 7. Stress Testing Requirements

Every candidate strategy must pass stress tests for:
1. **Wider spreads** (2x normal)
2. **Higher slippage** (2x normal)
3. **Weekend gap shocks** (±3% gap on Monday open)
4. **Volatility regime shift** (vol doubles for 1 month)
5. **Correlation spike** (all assets correlate to 0.8 for 1 week)
6. **Higher swap costs** (2x normal)
7. **Partial fills** (50% fill rate on large orders)

Strategy must remain net profitable under stress scenarios 1-3 at minimum.

---

## 8. Promotion Pipeline

```
Research → Sandbox Backtest → Walk-Forward → FTMO Sim → Paper (Demo) → Challenge
```

Each stage requires:
- Passing all relevant checks above
- Written report with metrics
- No skipping stages without explicit justification

---

## 9. Key Academic References

- Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum", JFE
- Blitz, Huij, Lansdorp & Verbeek (2013) — "Short-Term Residual Reversal", JFM
- Blitz, van der Grient & Honarvar (2023) — "Reversing the Trend of STR", JPM
- Fung & Hsieh (2001) — "Risk in Trend-Following", RFS
- Moreira & Muir (2017) — "Volatility-Managed Portfolios", JF
- Lo (2002) — "The Statistics of Sharpe Ratios", FAJ
- Bailey & Lopez de Prado (2014) — "The Deflated Sharpe Ratio"
- Grossman & Zhou (1993) — "Optimal Investment Under Drawdown Constraints", MF
- Browne (1999) — "Reaching Goals by a Deadline", AAP
- Frazzini, Israel, Moskowitz (2015) — "Trading Costs of Anomalies"
- Asness, Moskowitz & Pedersen (2013) — "Value and Momentum Everywhere", JF
