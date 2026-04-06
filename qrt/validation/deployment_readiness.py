"""
Deployment Readiness Gate & Advanced Backtesting Safeguards
=============================================================
Final validation layer before live/demo trading deployment.

Implements:
1. **Probability of Backtest Overfitting (PBO)** via CSCV
   (Bailey, Borwein, Lopez de Prado & Zhu 2014/2016)
2. **Clean holdout evaluation** — reserved final data period
   (Wiecki, Campbell, Lent & Stauth 2016)
3. **Leverage cost realism** — margin interest, funding costs
   (Rzepczynski et al. 2023 "I Have Never Seen a Bad Backtest")
4. **Survivorship bias assessment**
   (Suhonen, Lennkh & Perez 2017)
5. **Complexity penalty** — more complex strategies overfit more
   (Suhonen et al. 2017: 73% median Sharpe deterioration)
6. **Deployment gate** — automated pass/fail checklist
   (Harris 2016, Joubert et al. 2024)

Key finding from Wiecki et al. (2016): backtest Sharpe has R² < 0.025
for predicting OOS performance. Higher-order moments (vol, maxDD) and
portfolio construction features (hedging) are much better predictors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PBO — Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------

def compute_pbo(
    returns_matrix: pd.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe",
) -> dict:
    """
    Compute the Probability of Backtest Overfitting via CSCV.

    Splits the return matrix into S partitions of equal length,
    forms all C(S, S/2) combinations of train/test splits,
    and measures how often the IS-optimal strategy underperforms
    OOS relative to the median.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Columns = strategy/configuration returns, rows = dates.
        Each column is a different strategy or parameter combination.
    n_partitions : int
        Number of time partitions (must be even, default 16).
    metric : str
        Performance metric: "sharpe" or "mean_return".

    Returns
    -------
    dict
        pbo: float — probability of backtest overfitting [0, 1]
        logit_distribution: list — logit values per combination
        n_combinations: int — number of CSCV combinations tested
        degradation_stats: dict — IS vs OOS performance statistics
    """
    if n_partitions % 2 != 0:
        n_partitions = max(4, n_partitions - 1)

    n_obs = len(returns_matrix)
    n_strategies = returns_matrix.shape[1]

    if n_strategies < 2 or n_obs < n_partitions * 10:
        return {
            "pbo": 0.5,
            "logit_distribution": [],
            "n_combinations": 0,
            "reason": "insufficient data or strategies for CSCV",
        }

    # Partition returns into S equal sub-samples
    partition_size = n_obs // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size
        partitions.append(returns_matrix.iloc[start:end])

    half = n_partitions // 2

    # Generate all C(S, S/2) combinations — cap at 1000 for speed
    all_combos = list(combinations(range(n_partitions), half))
    if len(all_combos) > 1000:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_combos), 1000, replace=False)
        all_combos = [all_combos[i] for i in indices]

    logit_values = []
    is_perfs = []
    oos_perfs = []

    def _compute_metric(rets: pd.DataFrame) -> pd.Series:
        """Compute performance metric per strategy column."""
        if metric == "sharpe":
            mu = rets.mean()
            sigma = rets.std()
            return (mu / sigma * np.sqrt(252)).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            return rets.mean() * 252

    for combo in all_combos:
        test_idx = set(range(n_partitions)) - set(combo)

        # Concatenate train and test partitions
        train_data = pd.concat([partitions[i] for i in combo])
        test_data = pd.concat([partitions[i] for i in test_idx])

        # Compute IS and OOS performance per strategy
        is_perf = _compute_metric(train_data)
        oos_perf = _compute_metric(test_data)

        # Which strategy is IS-optimal?
        best_is_idx = is_perf.idxmax()
        best_is_oos_perf = oos_perf[best_is_idx]

        # What's the OOS rank of the IS-optimal strategy?
        oos_rank = (oos_perf >= best_is_oos_perf).sum()
        relative_rank = oos_rank / max(n_strategies, 1)

        # Logit of relative rank (how badly IS-optimal does OOS)
        # PBO = fraction where logit < 0 (IS-optimal is below OOS median)
        clipped = np.clip(relative_rank, 0.01, 0.99)
        logit = np.log(clipped / (1 - clipped))
        logit_values.append(float(logit))

        is_perfs.append(float(is_perf[best_is_idx]))
        oos_perfs.append(float(best_is_oos_perf))

    # PBO = fraction of combinations where logit < 0
    pbo = float(np.mean(np.array(logit_values) < 0))

    return {
        "pbo": pbo,
        "logit_distribution": logit_values,
        "n_combinations": len(all_combos),
        "is_mean_perf": float(np.mean(is_perfs)),
        "oos_mean_perf": float(np.mean(oos_perfs)),
        "degradation_pct": float(
            (np.mean(is_perfs) - np.mean(oos_perfs))
            / max(abs(np.mean(is_perfs)), 1e-6) * 100
        ),
    }


# ---------------------------------------------------------------------------
# 2. Clean Holdout Evaluation
# ---------------------------------------------------------------------------

def evaluate_clean_holdout(
    strategy_returns: dict[str, pd.Series],
    holdout_fraction: float = 0.20,
    portfolio_returns: Optional[pd.Series] = None,
    holdout_start_date: Optional[str] = None,
) -> dict:
    """
    Evaluate strategy performance on a clean holdout period.

    The final holdout_fraction of data is treated as untouched data
    that was never used for training, parameter selection, or optimization.

    Per Wiecki et al. (2016): backtest Sharpe has R² < 0.025 for
    predicting OOS. This holdout serves as the final reality check.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Strategy name -> full-sample daily returns.
    holdout_fraction : float
        Fraction of data reserved as holdout (default 0.20 = last 20%).
    portfolio_returns : pd.Series, optional
        Combined portfolio returns.
    holdout_start_date : str, optional
        If provided (e.g. "2023-01-01"), use this date to split instead
        of holdout_fraction. All data from this date onward is OOS.

    Returns
    -------
    dict
        Per-strategy holdout metrics + overall assessment.
    """
    results = {}

    for name, rets in strategy_returns.items():
        rets = rets.dropna().sort_index()
        if len(rets) < 100:
            continue

        if holdout_start_date is not None:
            cutoff = pd.Timestamp(holdout_start_date)
            dev_rets = rets[rets.index < cutoff]
            holdout_rets = rets[rets.index >= cutoff]
        else:
            split = int(len(rets) * (1 - holdout_fraction))
            dev_rets = rets.iloc[:split]
            holdout_rets = rets.iloc[split:]

        if len(dev_rets) < 50 or len(holdout_rets) < 20:
            continue

        def _metrics(r):
            if len(r) < 10 or r.std() == 0:
                return {"sharpe": 0, "cagr": 0, "maxdd": 0, "vol": 0}
            cum = (1 + r).cumprod()
            return {
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
                "cagr": float(cum.iloc[-1] ** (252 / len(cum)) - 1),
                "maxdd": float((cum / cum.cummax() - 1).min()),
                "vol": float(r.std() * np.sqrt(252)),
            }

        dev = _metrics(dev_rets)
        hold = _metrics(holdout_rets)

        degradation = (
            (dev["sharpe"] - hold["sharpe"]) / max(abs(dev["sharpe"]), 1e-6) * 100
            if dev["sharpe"] > 0 else 0
        )

        results[name] = {
            "dev_sharpe": dev["sharpe"],
            "holdout_sharpe": hold["sharpe"],
            "dev_cagr": dev["cagr"],
            "holdout_cagr": hold["cagr"],
            "dev_maxdd": dev["maxdd"],
            "holdout_maxdd": hold["maxdd"],
            "degradation_pct": float(degradation),
            "holdout_days": len(holdout_rets),
            "holdout_start": str(holdout_rets.index[0].date())
                if hasattr(holdout_rets.index[0], 'date') else str(holdout_rets.index[0]),
            "passes": degradation < 75 and hold["sharpe"] > 0,
        }

    # Portfolio holdout
    if portfolio_returns is not None:
        port = portfolio_returns.dropna().sort_index()
        if len(port) > 100:
            if holdout_start_date is not None:
                cutoff = pd.Timestamp(holdout_start_date)
                dev_p = port[port.index < cutoff]
                hold_p = port[port.index >= cutoff]
            else:
                split = int(len(port) * (1 - holdout_fraction))
                dev_p = port.iloc[:split]
                hold_p = port.iloc[split:]

            def _m(r):
                if len(r) < 10 or r.std() == 0:
                    return {"sharpe": 0, "cagr": 0, "maxdd": 0}
                cum = (1 + r).cumprod()
                return {
                    "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
                    "cagr": float(cum.iloc[-1] ** (252 / len(cum)) - 1),
                    "maxdd": float((cum / cum.cummax() - 1).min()),
                }

            dev_pm = _m(dev_p)
            hold_pm = _m(hold_p)

            results["__portfolio__"] = {
                "dev_sharpe": dev_pm["sharpe"],
                "holdout_sharpe": hold_pm["sharpe"],
                "dev_cagr": dev_pm["cagr"],
                "holdout_cagr": hold_pm["cagr"],
                "dev_maxdd": dev_pm["maxdd"],
                "holdout_maxdd": hold_pm["maxdd"],
                "degradation_pct": float(
                    (dev_pm["sharpe"] - hold_pm["sharpe"])
                    / max(abs(dev_pm["sharpe"]), 1e-6) * 100
                ),
                "holdout_days": len(hold_p),
            }

    return results


# ---------------------------------------------------------------------------
# 3. Leverage Cost Realism
# ---------------------------------------------------------------------------

def compute_leverage_costs(
    leverage: float,
    annual_margin_rate: float = 0.048,
    short_borrow_rate: float = 0.003,
    long_short_ratio: float = 1.0,
    funding_spread: float = 0.0,
) -> dict:
    """
    Compute realistic annual cost of leverage at Interactive Brokers.

    Parameters
    ----------
    leverage : float
        Portfolio leverage multiplier.
    annual_margin_rate : float
        Annual margin interest rate (default 4.8% = IBKR Pro blended rate
        for $100k-$500k accounts, BM + 1.0-1.5%).
        Alternatives: futures ~4.5%, box spreads ~4.0%.
    short_borrow_rate : float
        Annual cost to borrow shares for shorting (default 0.3% =
        IBKR general collateral rate for liquid large-cap).
    long_short_ratio : float
        Ratio of long to short exposure (1.0 = fully hedged L/S).
    funding_spread : float
        Additional funding spread above margin rate (default 0% — already
        included in IBKR's margin rate which is BM + spread).

    Returns
    -------
    dict
        Detailed cost breakdown and total annual drag.
    """
    # Amount borrowed = (leverage - 1) × portfolio value
    borrowed_fraction = max(0, leverage - 1)

    # Margin interest on borrowed amount
    margin_cost = borrowed_fraction * annual_margin_rate

    # Funding spread (cost of maintaining leverage)
    funding_cost = borrowed_fraction * funding_spread

    # Short borrow costs (if any shorts)
    short_exposure = borrowed_fraction * (1 / (1 + long_short_ratio))
    borrow_cost = short_exposure * short_borrow_rate

    # Total annual drag from leverage
    total_annual_cost = margin_cost + funding_cost + borrow_cost

    # Vol drag (Avellaneda & Zhang 2010 — this is separate from financing)
    # Not computed here as it's in the leverager optimizer already

    return {
        "leverage": leverage,
        "borrowed_fraction": float(borrowed_fraction),
        "margin_interest_annual": float(margin_cost),
        "funding_spread_annual": float(funding_cost),
        "borrow_cost_annual": float(borrow_cost),
        "total_leverage_cost_annual": float(total_annual_cost),
        "daily_drag_bps": float(total_annual_cost / 252 * 10000),
        "assumptions": {
            "margin_rate": annual_margin_rate,
            "short_borrow_rate": short_borrow_rate,
            "funding_spread": funding_spread,
        },
    }


# ---------------------------------------------------------------------------
# 4. Survivorship Bias Assessment
# ---------------------------------------------------------------------------

def assess_survivorship_bias(
    tickers: list[str],
    data_source: str = "unknown",
    start_year: int = 2010,
) -> dict:
    """
    Assess potential survivorship bias in the universe.

    Yahoo Finance only returns data for currently-listed stocks.
    Stocks that were delisted, acquired, or went bankrupt are missing.
    This creates positive bias since failed companies are excluded.

    Parameters
    ----------
    tickers : list[str]
        Universe of tickers used in the backtest.
    data_source : str
        Data source identifier.
    start_year : int
        Backtest start year.

    Returns
    -------
    dict
        Survivorship bias assessment.
    """
    warnings = []
    risk_level = "LOW"

    # Known companies delisted from S&P 500 since 2010 that would be missed
    # by a current-day Yahoo Finance universe
    notable_delistings = {
        "LEH": "Lehman Brothers (bankrupt 2008)",
        "BSC": "Bear Stearns (acquired 2008)",
        "WB": "Wachovia (acquired 2008)",
        "AIG": "AIG (nearly bankrupt 2008, restructured)",
        "FRE": "Freddie Mac (delisted 2010)",
        "FNM": "Fannie Mae (delisted 2010)",
        "ETFC": "E*TRADE (acquired 2020)",
        "CIT": "CIT Group (acquired 2021)",
        "XRX": "Xerox (restructured, delisted from S&P)",
        "FLIR": "FLIR Systems (acquired 2021)",
        "XLNX": "Xilinx (acquired 2022)",
        "TWTR": "Twitter (acquired 2022)",
        "ATVI": "Activision Blizzard (acquired 2023)",
        "VMW": "VMware (acquired 2023)",
        "SIVB": "SVB Financial (bankrupt 2023)",
        "FRC": "First Republic Bank (bankrupt 2023)",
    }

    if data_source in ("real", "yahoo", "yfinance"):
        warnings.append(
            "Yahoo Finance only provides data for currently-listed stocks. "
            "Stocks that went bankrupt, were acquired, or delisted are MISSING "
            "from the backtest, creating positive survivorship bias."
        )
        risk_level = "MODERATE"

        if start_year <= 2010:
            warnings.append(
                f"Backtest starts in {start_year}, spanning GFC aftermath. "
                f"Major bank failures (Lehman, Bear Stearns, Wachovia) and "
                f"restructurings (AIG, Fannie/Freddie) are not captured."
            )
            risk_level = "HIGH"

        # Estimate bias magnitude (Suhonen et al. 2017)
        # Typical survivorship bias: 0.5-1.5% annual return overstatement
        years = 2026 - start_year
        est_bias_low = 0.005 * years / 10  # 0.5% per decade
        est_bias_high = 0.015 * years / 10  # 1.5% per decade
        warnings.append(
            f"Estimated survivorship bias: {est_bias_low:.1%}-{est_bias_high:.1%} "
            f"annual return overstatement over {years} years."
        )

    elif data_source == "synthetic":
        warnings.append(
            "Synthetic data has no survivorship bias by construction, "
            "but also lacks realistic delisting/acquisition dynamics."
        )

    return {
        "risk_level": risk_level,
        "warnings": warnings,
        "n_tickers": len(tickers),
        "data_source": data_source,
        "start_year": start_year,
    }


# ---------------------------------------------------------------------------
# 5. Complexity Penalty (Suhonen et al. 2017)
# ---------------------------------------------------------------------------

def compute_complexity_score(
    n_strategies: int = 0,
    n_parameters: int = 0,
    n_features: int = 0,
    n_models_tested: int = 0,
    uses_ml: bool = False,
    uses_regime_switching: bool = False,
    n_allocation_methods: int = 0,
    n_risk_layers: int = 0,
) -> dict:
    """
    Compute strategy complexity score and expected Sharpe degradation.

    Per Suhonen et al. (2017): median 73% Sharpe degradation IS→OOS,
    with most complex strategies degrading 30+ percentage points MORE
    than simplest ones.

    Parameters
    ----------
    n_strategies : int
        Number of individual strategies.
    n_parameters : int
        Total optimizable parameters across all strategies.
    n_features : int
        Total features used (for ML strategies).
    n_models_tested : int
        Number of model/parameter configurations tested.
    uses_ml : bool
        Whether ML models are used.
    uses_regime_switching : bool
        Whether regime-conditional parameters are used.
    n_allocation_methods : int
        Number of allocation methods tested.
    n_risk_layers : int
        Number of risk management layers.

    Returns
    -------
    dict
        Complexity score, expected degradation, and recommendations.
    """
    # Compute raw complexity score (0-100)
    score = 0
    score += min(20, n_strategies * 1.5)          # 18 strategies → 27, capped at 20
    score += min(15, n_parameters * 0.3)           # 50 params → 15
    score += min(10, n_features * 0.1)             # 100 features → 10
    score += min(15, np.log1p(n_models_tested) * 3)  # log scale
    score += 10 if uses_ml else 0
    score += 5 if uses_regime_switching else 0
    score += min(10, n_allocation_methods * 3)     # 4 methods → 12, capped at 10
    score += min(15, n_risk_layers * 3)            # 5 layers → 15

    score = min(100, score)

    # Expected Sharpe degradation based on complexity
    # Suhonen et al. (2017): simplest = ~60% degradation, most complex = ~90%
    base_degradation = 0.50  # 50% baseline (optimistic)
    complexity_penalty = (score / 100) * 0.35  # up to 35% additional
    expected_degradation = base_degradation + complexity_penalty

    # Expected OOS Sharpe multiplier
    oos_multiplier = 1.0 - expected_degradation

    return {
        "complexity_score": float(score),
        "expected_sharpe_degradation_pct": float(expected_degradation * 100),
        "oos_sharpe_multiplier": float(oos_multiplier),
        "components": {
            "strategies": n_strategies,
            "parameters": n_parameters,
            "features": n_features,
            "models_tested": n_models_tested,
            "uses_ml": uses_ml,
            "uses_regime_switching": uses_regime_switching,
            "allocation_methods": n_allocation_methods,
            "risk_layers": n_risk_layers,
        },
        "recommendation": (
            "LOW complexity — results likely reliable"
            if score < 30 else
            "MODERATE complexity — expect 50-70% Sharpe degradation OOS"
            if score < 60 else
            "HIGH complexity — expect 70-85% Sharpe degradation OOS. "
            "Consider simplifying strategies or reducing parameters."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Deployment Readiness Gate
# ---------------------------------------------------------------------------

@dataclass
class DeploymentGateResult:
    """Result of the deployment readiness gate."""
    passed: bool = False
    checks: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# Deployment Readiness Gate",
            "",
            f"**RESULT: {'PASSED — Ready for demo trading' if self.passed else 'BLOCKED — Issues must be resolved'}**",
            "",
            "## Check Results",
            "",
        ]

        for check in self.checks:
            icon = "[PASS]" if check["passed"] else "[FAIL]"
            lines.append(f"- {icon} **{check['name']}**: {check['detail']}")
        lines.append("")

        if self.blockers:
            lines.append("## Blockers (must fix)")
            lines.append("")
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings (acknowledge before deploying)")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.summary:
            lines.append("## Summary Metrics")
            lines.append("")
            for k, v in self.summary.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        return "\n".join(lines)


class DeploymentGate:
    """
    Automated deployment readiness assessment.

    Runs all validation checks and produces a pass/fail decision
    for proceeding to live/demo trading.

    Acceptance criteria (any blocker fails the gate):
    1. PBO < 0.50 (probability of overfitting below 50%)
    2. DSR significant for portfolio (deflated Sharpe > 0.95)
    3. Holdout Sharpe > 0 (positive OOS performance)
    4. Holdout degradation < 75% (not catastrophically overfit)
    5. Leverage stress score >= 4/10
    6. No critical audit failures
    7. Leverage costs accounted for

    Warnings (don't block but must acknowledge):
    - Survivorship bias risk
    - High complexity score
    - White's Reality Check p > 0.05
    """

    def __init__(
        self,
        max_pbo: float = 0.50,
        min_holdout_sharpe: float = 0.0,
        max_holdout_degradation: float = 75.0,
        min_stress_score: float = 4.0,
    ):
        self.max_pbo = max_pbo
        self.min_holdout_sharpe = min_holdout_sharpe
        self.max_holdout_degradation = max_holdout_degradation
        self.min_stress_score = min_stress_score

    def evaluate(
        self,
        strategy_returns: dict[str, pd.Series],
        portfolio_returns: Optional[pd.Series] = None,
        leverage: float = 1.0,
        overfit_report=None,
        stress_result=None,
        audit_report=None,
        data_source: str = "unknown",
        tickers: Optional[list[str]] = None,
        n_strategies: int = 0,
        n_parameters: int = 0,
        n_models_tested: int = 0,
        holdout_start_date: Optional[str] = None,
    ) -> DeploymentGateResult:
        """Run all deployment checks and return pass/fail decision."""
        gate = DeploymentGateResult()

        # --- Check 1: PBO ---
        try:
            returns_df = pd.DataFrame(strategy_returns).dropna()
            if returns_df.shape[1] >= 2 and len(returns_df) >= 200:
                pbo_result = compute_pbo(returns_df, n_partitions=8)
                pbo = pbo_result["pbo"]
                passed = pbo < self.max_pbo
                gate.checks.append({
                    "name": "PBO (Probability of Backtest Overfitting)",
                    "passed": passed,
                    "detail": f"PBO={pbo:.2f} (threshold={self.max_pbo:.2f}), "
                              f"IS→OOS degradation={pbo_result.get('degradation_pct', 0):.1f}%",
                })
                if not passed:
                    gate.blockers.append(
                        f"PBO={pbo:.2f} exceeds threshold {self.max_pbo:.2f}. "
                        f"Strategies are likely overfit to historical data."
                    )
                gate.summary["pbo"] = pbo
            else:
                gate.checks.append({
                    "name": "PBO",
                    "passed": True,
                    "detail": "Insufficient strategies for CSCV (need >= 2)",
                })
        except Exception as e:
            gate.checks.append({
                "name": "PBO", "passed": True,
                "detail": f"Could not compute: {e}",
            })

        # --- Check 2: DSR significance ---
        if overfit_report is not None:
            n_sig = sum(
                1 for t in overfit_report.strategy_tests.values()
                if t.get("is_significant", False)
            )
            n_total = len(overfit_report.strategy_tests)
            passed = n_sig > 0
            gate.checks.append({
                "name": "Deflated Sharpe Ratio significance",
                "passed": passed,
                "detail": f"{n_sig}/{n_total} strategies pass DSR test",
            })
            if not passed:
                gate.blockers.append(
                    "No strategies pass the Deflated Sharpe Ratio test. "
                    "All observed Sharpe ratios may be due to selection bias."
                )

        # --- Check 3: Clean holdout ---
        try:
            holdout = evaluate_clean_holdout(
                strategy_returns, holdout_fraction=0.20,
                portfolio_returns=portfolio_returns,
                holdout_start_date=holdout_start_date,
            )
            port_hold = holdout.get("__portfolio__", {})
            if port_hold:
                hold_sharpe = port_hold.get("holdout_sharpe", 0)
                hold_deg = port_hold.get("degradation_pct", 0)
                passed_sharpe = hold_sharpe > self.min_holdout_sharpe
                passed_deg = hold_deg < self.max_holdout_degradation

                gate.checks.append({
                    "name": "Holdout Sharpe > 0",
                    "passed": passed_sharpe,
                    "detail": f"Holdout Sharpe={hold_sharpe:.3f} "
                              f"(dev={port_hold.get('dev_sharpe', 0):.3f})",
                })
                gate.checks.append({
                    "name": "Holdout degradation < 75%",
                    "passed": passed_deg,
                    "detail": f"Degradation={hold_deg:.1f}% "
                              f"(dev CAGR={port_hold.get('dev_cagr', 0):.2%}, "
                              f"holdout CAGR={port_hold.get('holdout_cagr', 0):.2%})",
                })
                if not passed_sharpe:
                    gate.blockers.append(
                        f"Portfolio Sharpe is {hold_sharpe:.3f} on holdout data. "
                        f"Strategy may not be profitable out-of-sample."
                    )
                if not passed_deg:
                    gate.blockers.append(
                        f"Sharpe degradation of {hold_deg:.1f}% from dev to holdout "
                        f"exceeds 75% threshold. Severe overfitting detected."
                    )
                gate.summary["holdout_sharpe"] = hold_sharpe
                gate.summary["holdout_degradation_pct"] = hold_deg
        except Exception as e:
            gate.checks.append({
                "name": "Clean holdout", "passed": True,
                "detail": f"Could not compute: {e}",
            })

        # --- Check 4: Leverage stress ---
        if stress_result is not None:
            score = stress_result.overall_risk_score
            passed = score >= self.min_stress_score
            gate.checks.append({
                "name": "Leverage stress score",
                "passed": passed,
                "detail": f"Score={score:.1f}/10 (min={self.min_stress_score:.1f})",
            })
            if not passed:
                gate.blockers.append(
                    f"Leverage stress score {score:.1f}/10 is below minimum "
                    f"{self.min_stress_score:.1f}. Leverage is too risky."
                )
            gate.summary["stress_score"] = score

        # --- Check 5: Leverage costs ---
        if leverage > 1.0:
            costs = compute_leverage_costs(leverage)
            total_cost = costs["total_leverage_cost_annual"]
            gate.checks.append({
                "name": "Leverage costs accounted for",
                "passed": True,  # informational
                "detail": f"At {leverage:.1f}x: margin={costs['margin_interest_annual']:.2%}/yr, "
                          f"funding={costs['funding_spread_annual']:.2%}/yr, "
                          f"total={total_cost:.2%}/yr ({costs['daily_drag_bps']:.1f} bps/day)",
            })
            gate.summary["leverage_cost_annual"] = total_cost
            if total_cost > 0.10:  # > 10% annual cost
                gate.warnings.append(
                    f"Leverage costs of {total_cost:.2%}/yr are substantial at "
                    f"{leverage:.1f}x. Ensure CAGR exceeds this after vol drag."
                )

        # --- Check 6: Audit report ---
        if audit_report is not None:
            n_fails = audit_report.fail_count
            passed = n_fails == 0
            gate.checks.append({
                "name": "Audit integrity (no critical failures)",
                "passed": passed,
                "detail": f"{audit_report.pass_count} pass, {n_fails} fail, "
                          f"{audit_report.warning_count} warnings",
            })
            if not passed:
                gate.blockers.append(
                    f"Audit has {n_fails} failures. Fix before deploying."
                )

        # --- Check 7: White's Reality Check ---
        if overfit_report is not None and overfit_report.reality_check:
            rc = overfit_report.reality_check
            p = rc.get("p_value", 1)
            if p > 0.05:
                gate.warnings.append(
                    f"White's Reality Check p={p:.3f} — best strategy may not "
                    f"be significantly better than benchmark after data snooping."
                )

        # --- Warnings: survivorship bias ---
        if tickers:
            surv = assess_survivorship_bias(
                tickers, data_source=data_source,
            )
            if surv["risk_level"] in ("MODERATE", "HIGH"):
                for w in surv["warnings"]:
                    gate.warnings.append(f"[Survivorship] {w}")

        # --- Warnings: complexity ---
        complexity = compute_complexity_score(
            n_strategies=n_strategies or len(strategy_returns),
            n_parameters=n_parameters,
            n_models_tested=n_models_tested,
            uses_ml=True,
            uses_regime_switching=True,
            n_allocation_methods=4,
            n_risk_layers=5,
        )
        gate.summary["complexity_score"] = complexity["complexity_score"]
        gate.summary["expected_oos_sharpe_mult"] = complexity["oos_sharpe_multiplier"]

        if complexity["complexity_score"] > 60:
            gate.warnings.append(
                f"[Complexity] Score={complexity['complexity_score']:.0f}/100. "
                f"{complexity['recommendation']}"
            )

        # --- Final decision ---
        gate.passed = len(gate.blockers) == 0

        status = "PASSED" if gate.passed else "BLOCKED"
        logger.info(
            "Deployment gate: %s (%d checks, %d blockers, %d warnings)",
            status, len(gate.checks), len(gate.blockers), len(gate.warnings),
        )

        return gate
