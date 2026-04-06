"""
Backtest Overfitting Detection & Statistical Validation
=========================================================
Implements rigorous statistical tests to detect overfitting and validate
that backtested performance is likely to persist out-of-sample.

Academic basis:
  - Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio" — corrects
    for selection bias, multiple testing, and non-normality
  - Bailey et al. (2014): "The Probability of Backtest Overfitting" —
    CSCV (Combinatorial Symmetric Cross-Validation)
  - Harvey, Liu & Zhu (2016): "...and the Cross-Section of Expected Returns"
    — t-ratio must exceed 3.0 after multiple testing correction
  - White (2000): "A Reality Check for Data Snooping" — bootstrap test
    for best strategy vs benchmark
  - Bailey & Lopez de Prado (2012): "The Sharpe Ratio Efficient Frontier"
    — minimum backtest length formula
  - Lopez de Prado (2018): "Advances in Financial Machine Learning" —
    combinatorial purged cross-validation

Key insight: with N strategies tested, the expected maximum Sharpe ratio
of a random set of N strategies with zero true Sharpe is approximately
√(2 * log(N)) — the "Sharpe haircut". A portfolio of 18 strategies
tested on the same data needs SR > ~2.4 to be statistically significant
at the 5% level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class OverfittingReport:
    """Comprehensive overfitting diagnostic report."""
    strategy_tests: dict = field(default_factory=dict)
    portfolio_tests: dict = field(default_factory=dict)
    multiple_testing: dict = field(default_factory=dict)
    is_vs_oos: dict = field(default_factory=dict)
    reality_check: dict = field(default_factory=dict)
    leverage_haircut: dict = field(default_factory=dict)
    overall_confidence: str = "UNKNOWN"
    warnings: list = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown diagnostic report."""
        lines = [
            "# Backtest Overfitting Diagnostic Report",
            "",
            f"**Overall Confidence: {self.overall_confidence}**",
            "",
        ]

        # Strategy-level tests
        if self.strategy_tests:
            lines.append("## Strategy-Level Statistical Tests")
            lines.append("")
            lines.append(
                f"| {'Strategy':<25} | {'Sharpe':>7} | {'DSR':>7} | "
                f"{'PSR':>7} | {'MinBTL':>7} | {'Signif':>7} |"
            )
            lines.append(f"|{'-'*27}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
            for name, tests in self.strategy_tests.items():
                sig = "YES" if tests.get("is_significant", False) else "NO"
                lines.append(
                    f"| {name:<25} | {tests.get('sharpe', 0):>7.3f} | "
                    f"{tests.get('deflated_sharpe', 0):>7.3f} | "
                    f"{tests.get('prob_sharpe', 0):>7.3f} | "
                    f"{tests.get('min_btl_years', 0):>7.1f} | "
                    f"{sig:>7} |"
                )
            lines.append("")

        # Multiple testing correction
        if self.multiple_testing:
            lines.append("## Multiple Testing Correction (Harvey et al. 2016)")
            lines.append("")
            mt = self.multiple_testing
            lines.append(f"- Number of strategies tested: {mt.get('n_strategies', 0)}")
            lines.append(f"- Required t-ratio (Bonferroni): {mt.get('bonferroni_threshold', 0):.2f}")
            lines.append(f"- Required t-ratio (BH-FDR 5%): {mt.get('bh_fdr_threshold', 0):.2f}")
            lines.append(f"- Expected max Sharpe of null: {mt.get('expected_max_null_sharpe', 0):.3f}")
            lines.append(f"- Strategies surviving Bonferroni: {mt.get('n_surviving_bonferroni', 0)}")
            lines.append(f"- Strategies surviving BH-FDR: {mt.get('n_surviving_bh', 0)}")
            lines.append("")

        # IS vs OOS degradation
        if self.is_vs_oos:
            lines.append("## In-Sample vs Out-of-Sample Degradation")
            lines.append("")
            for name, comp in self.is_vs_oos.items():
                deg = comp.get("degradation_pct", 0)
                flag = " [OVERFIT WARNING]" if deg > 50 else ""
                lines.append(
                    f"- **{name}**: IS Sharpe={comp.get('is_sharpe', 0):.3f}, "
                    f"OOS Sharpe={comp.get('oos_sharpe', 0):.3f}, "
                    f"Degradation={deg:.1f}%{flag}"
                )
            lines.append("")

        # Reality check
        if self.reality_check:
            rc = self.reality_check
            lines.append("## White's Reality Check (Bootstrap)")
            lines.append("")
            lines.append(f"- Best strategy: {rc.get('best_strategy', 'N/A')}")
            lines.append(f"- Bootstrap p-value: {rc.get('p_value', 1):.4f}")
            sig = "YES" if rc.get("p_value", 1) < 0.05 else "NO"
            lines.append(f"- Significant after data snooping correction: {sig}")
            lines.append("")

        # Leverage haircut
        if self.leverage_haircut:
            lh = self.leverage_haircut
            lines.append("## Leverage Risk Haircut")
            lines.append("")
            lines.append(f"- Applied leverage: {lh.get('leverage', 1):.1f}x")
            lines.append(f"- Raw Sharpe: {lh.get('raw_sharpe', 0):.3f}")
            lines.append(f"- Haircut Sharpe (OOS estimate): {lh.get('haircut_sharpe', 0):.3f}")
            lines.append(f"- Expected OOS MaxDD: {lh.get('expected_oos_maxdd', 0):.2%}")
            lines.append(f"- Vol drag at leverage: {lh.get('vol_drag', 0):.2%}")
            lines.append(f"- Leverage safety score: {lh.get('safety_score', 0):.1f}/10")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    Corrects the observed Sharpe for:
    1. Selection bias from testing multiple strategies (n_trials)
    2. Non-normality of returns (skewness, kurtosis)

    The DSR gives the probability that the observed Sharpe is greater than
    what we'd expect from the best of n_trials random strategies.

    Parameters
    ----------
    observed_sharpe : float
        Annualised Sharpe ratio of the selected strategy.
    n_trials : int
        Number of strategies/configurations tested.
    n_observations : int
        Number of return observations (trading days).
    skewness : float
        Skewness of returns (0 for normal).
    kurtosis : float
        Kurtosis of returns (3 for normal, excess kurtosis + 3).
    sharpe_benchmark : float
        Benchmark Sharpe to test against (default 0).

    Returns
    -------
    float
        DSR p-value. Values > 0.95 indicate the Sharpe is significant
        even after accounting for selection bias.
    """
    if n_observations < 2 or n_trials < 1:
        return 0.0

    # Expected maximum Sharpe from n_trials under null (Euler-Mascheroni)
    euler_mascheroni = 0.5772156649
    e_max_sharpe = (
        (1 - euler_mascheroni) * scipy_stats.norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    # Standard error of the Sharpe ratio (Lo 2002, corrected for non-normality)
    se_sharpe = np.sqrt(
        (1 + 0.25 * (kurtosis - 3) * observed_sharpe**2
         - skewness * observed_sharpe)
        / n_observations
    )

    if se_sharpe <= 0:
        return 0.0

    # Test statistic: is our Sharpe > expected max of null?
    test_stat = (observed_sharpe - e_max_sharpe) / se_sharpe

    # Return CDF (probability that true Sharpe > benchmark)
    return float(scipy_stats.norm.cdf(test_stat))


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012)
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (PSR).

    Tests whether the observed Sharpe is statistically greater than a
    benchmark Sharpe, accounting for non-normality.

    Parameters
    ----------
    observed_sharpe : float
        Annualised Sharpe of the strategy.
    benchmark_sharpe : float
        Sharpe to test against (default 0 = test if profitable).
    n_observations : int
        Number of return observations.
    skewness : float
        Return skewness.
    kurtosis : float
        Return kurtosis (not excess — 3 for normal).

    Returns
    -------
    float
        Probability that the true Sharpe > benchmark_sharpe.
    """
    if n_observations < 2:
        return 0.0

    se_sharpe = np.sqrt(
        (1 + 0.25 * (kurtosis - 3) * observed_sharpe**2
         - skewness * observed_sharpe)
        / n_observations
    )

    if se_sharpe <= 0:
        return 0.0

    test_stat = (observed_sharpe - benchmark_sharpe) / se_sharpe
    return float(scipy_stats.norm.cdf(test_stat))


# ---------------------------------------------------------------------------
# 3. Minimum Backtest Length (Bailey & Lopez de Prado 2012)
# ---------------------------------------------------------------------------

def minimum_backtest_length(
    observed_sharpe: float,
    n_trials: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """
    Compute the Minimum Backtest Length (MinBTL) in years.

    The minimum number of years of data needed for the observed Sharpe
    to be statistically significant at the given confidence level.

    Parameters
    ----------
    observed_sharpe : float
        Annualised Sharpe ratio.
    n_trials : int
        Number of strategies tested.
    skewness : float
        Return skewness.
    kurtosis : float
        Return kurtosis.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    float
        Minimum years of data needed. If > actual data length, the
        backtest is too short to be trusted.
    """
    if observed_sharpe <= 0:
        return float("inf")

    z = scipy_stats.norm.ppf(confidence)

    # Expected max Sharpe under null
    if n_trials > 1:
        euler_mascheroni = 0.5772156649
        e_max = (
            (1 - euler_mascheroni) * scipy_stats.norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        e_max = 0.0

    effective_sharpe = observed_sharpe - e_max
    if effective_sharpe <= 0:
        return float("inf")

    # MinBTL in trading days
    min_days = (
        (z / effective_sharpe) ** 2
        * (1 + 0.25 * (kurtosis - 3) * observed_sharpe**2
           - skewness * observed_sharpe)
    )

    return float(min_days / 252)


# ---------------------------------------------------------------------------
# 4. Multiple Hypothesis Testing (Harvey, Liu & Zhu 2016)
# ---------------------------------------------------------------------------

def multiple_testing_correction(
    sharpe_ratios: dict[str, float],
    n_observations: int,
    significance: float = 0.05,
) -> dict:
    """
    Apply multiple testing corrections to a set of strategy Sharpe ratios.

    Implements:
    1. Bonferroni correction (most conservative)
    2. Benjamini-Hochberg FDR control (more powerful)
    3. Harvey et al. (2016) t-ratio threshold of 3.0

    Parameters
    ----------
    sharpe_ratios : dict[str, float]
        Strategy name -> annualised Sharpe ratio.
    n_observations : int
        Number of return observations per strategy.
    significance : float
        Family-wise error rate (default 0.05).

    Returns
    -------
    dict
        Multiple testing results including surviving strategies.
    """
    n = len(sharpe_ratios)
    if n == 0:
        return {"n_strategies": 0}

    names = list(sharpe_ratios.keys())
    sharpes = np.array([sharpe_ratios[k] for k in names])

    # Convert Sharpe to t-statistics (t ≈ Sharpe × √(n_obs / 252))
    t_stats = sharpes * np.sqrt(n_observations / 252)

    # Two-sided p-values
    p_values = 2 * (1 - scipy_stats.norm.cdf(np.abs(t_stats)))

    # 1. Bonferroni correction
    bonferroni_threshold = significance / n
    bonferroni_t = scipy_stats.norm.ppf(1 - bonferroni_threshold / 2)
    surviving_bonferroni = [
        names[i] for i in range(n) if p_values[i] < bonferroni_threshold
    ]

    # 2. Benjamini-Hochberg FDR
    sorted_idx = np.argsort(p_values)
    bh_threshold = 0.0
    surviving_bh = []
    for rank, idx in enumerate(sorted_idx, 1):
        threshold = significance * rank / n
        if p_values[idx] <= threshold:
            bh_threshold = threshold
            surviving_bh.append(names[idx])

    # 3. Harvey et al. (2016) threshold: t > 3.0
    surviving_harvey = [
        names[i] for i in range(n) if abs(t_stats[i]) > 3.0
    ]

    # Expected max Sharpe under null
    euler_mascheroni = 0.5772156649
    expected_max_null = (
        (1 - euler_mascheroni) * scipy_stats.norm.ppf(1 - 1.0 / max(n, 1))
        + euler_mascheroni * scipy_stats.norm.ppf(1 - 1.0 / (max(n, 1) * np.e))
    ) / np.sqrt(n_observations / 252)

    return {
        "n_strategies": n,
        "bonferroni_threshold": float(bonferroni_t / np.sqrt(n_observations / 252)),
        "bh_fdr_threshold": float(bh_threshold),
        "harvey_threshold": 3.0 / np.sqrt(n_observations / 252),
        "expected_max_null_sharpe": float(expected_max_null),
        "n_surviving_bonferroni": len(surviving_bonferroni),
        "n_surviving_bh": len(surviving_bh),
        "n_surviving_harvey": len(surviving_harvey),
        "surviving_bonferroni": surviving_bonferroni,
        "surviving_bh": surviving_bh,
        "surviving_harvey": surviving_harvey,
        "strategy_p_values": {names[i]: float(p_values[i]) for i in range(n)},
        "strategy_t_stats": {names[i]: float(t_stats[i]) for i in range(n)},
    }


# ---------------------------------------------------------------------------
# 5. White's Reality Check (Bootstrap, White 2000)
# ---------------------------------------------------------------------------

def whites_reality_check(
    strategy_returns: dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None,
    n_bootstrap: int = 5000,
    block_size: int = 5,
    random_state: int = 42,
) -> dict:
    """
    White's Reality Check for data snooping.

    Tests whether the best strategy's outperformance of a benchmark
    is statistically significant after accounting for the fact that
    we selected the best from N candidates.

    Uses stationary block bootstrap to preserve serial correlation.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Strategy name -> daily returns.
    benchmark_returns : pd.Series, optional
        Benchmark returns (default: zero = absolute performance).
    n_bootstrap : int
        Number of bootstrap replications.
    block_size : int
        Block size for block bootstrap.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Reality check results including p-value and best strategy.
    """
    rng = np.random.RandomState(random_state)

    # Align all series to common index
    names = list(strategy_returns.keys())
    df = pd.DataFrame(strategy_returns).dropna()
    if df.empty or len(df) < 50:
        return {"p_value": 1.0, "best_strategy": "N/A",
                "reason": "insufficient data"}

    n_obs = len(df)
    n_strats = len(names)
    ret_matrix = df.values  # (n_obs, n_strats)

    # Benchmark: default to zero (absolute performance test)
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(df.index).fillna(0).values
    else:
        bm = np.zeros(n_obs)

    # Excess returns over benchmark
    excess = ret_matrix - bm[:, np.newaxis]

    # Observed test statistic: mean excess return of best strategy
    mean_excess = excess.mean(axis=0)
    best_idx = np.argmax(mean_excess)
    observed_stat = mean_excess[best_idx]

    # Block bootstrap
    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = n_obs - block_size
    if max_start < 1:
        max_start = 1

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Sample block starts
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, min(s + block_size, n_obs)) for s in starts
        ])[:n_obs]

        # Bootstrap excess returns
        boot_excess = excess[indices]

        # Center bootstrap (under null: zero mean excess)
        boot_mean = boot_excess.mean(axis=0) - mean_excess
        boot_stats[b] = boot_mean.max()

    # p-value: fraction of bootstrap stats >= observed
    p_value = float((boot_stats >= observed_stat).mean())

    return {
        "p_value": p_value,
        "best_strategy": names[best_idx],
        "best_mean_excess": float(observed_stat * 252),  # annualised
        "n_strategies": n_strats,
        "n_bootstrap": n_bootstrap,
        "bootstrap_ci_95": (
            float(np.percentile(boot_stats, 2.5) * 252),
            float(np.percentile(boot_stats, 97.5) * 252),
        ),
    }


# ---------------------------------------------------------------------------
# 6. IS vs OOS Degradation Tracking
# ---------------------------------------------------------------------------

def compute_is_oos_degradation(
    strategy_returns: dict[str, pd.Series],
    split_ratio: float = 0.6,
) -> dict[str, dict]:
    """
    Compute in-sample vs out-of-sample Sharpe degradation.

    Splits each strategy's returns at split_ratio and compares
    IS vs OOS Sharpe ratios. Large degradation (>50%) flags overfitting.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Strategy name -> daily returns (full sample).
    split_ratio : float
        Fraction used as in-sample (default 0.6).

    Returns
    -------
    dict[str, dict]
        Per-strategy IS/OOS comparison.
    """
    results = {}
    for name, rets in strategy_returns.items():
        rets = rets.dropna()
        if len(rets) < 100:
            continue

        split_idx = int(len(rets) * split_ratio)
        is_rets = rets.iloc[:split_idx]
        oos_rets = rets.iloc[split_idx:]

        is_sharpe = (
            is_rets.mean() / is_rets.std() * np.sqrt(252)
            if is_rets.std() > 0 else 0.0
        )
        oos_sharpe = (
            oos_rets.mean() / oos_rets.std() * np.sqrt(252)
            if oos_rets.std() > 0 else 0.0
        )

        degradation = (
            (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 1e-6) * 100
            if is_sharpe > 0 else 0.0
        )

        # IS vs OOS MaxDD
        is_cum = (1 + is_rets).cumprod()
        oos_cum = (1 + oos_rets).cumprod()
        is_maxdd = float((is_cum / is_cum.cummax() - 1).min())
        oos_maxdd = float((oos_cum / oos_cum.cummax() - 1).min())

        results[name] = {
            "is_sharpe": float(is_sharpe),
            "oos_sharpe": float(oos_sharpe),
            "degradation_pct": float(degradation),
            "is_maxdd": is_maxdd,
            "oos_maxdd": oos_maxdd,
            "is_days": len(is_rets),
            "oos_days": len(oos_rets),
            "is_overfit": degradation > 50,
        }

    return results


# ---------------------------------------------------------------------------
# 7. Leverage Risk Haircut
# ---------------------------------------------------------------------------

def leverage_risk_haircut(
    base_sharpe: float,
    leverage: float,
    base_vol: float,
    base_maxdd: float,
    n_strategies: int = 1,
    n_observations: int = 2520,
) -> dict:
    """
    Compute expected OOS performance haircut for leveraged portfolio.

    Leverage amplifies both returns AND estimation errors. The expected
    OOS degradation is roughly proportional to leverage × sqrt(estimation_error).

    Parameters
    ----------
    base_sharpe : float
        Unlevered annualised Sharpe.
    leverage : float
        Applied leverage multiplier.
    base_vol : float
        Unlevered annualised volatility.
    base_maxdd : float
        Unlevered maximum drawdown (negative).
    n_strategies : int
        Number of strategies tested (selection bias correction).
    n_observations : int
        Number of observations in backtest.

    Returns
    -------
    dict
        Haircut metrics for leveraged portfolio.
    """
    # 1. Selection bias haircut (expected max null Sharpe)
    if n_strategies > 1:
        euler_mascheroni = 0.5772156649
        selection_haircut = (
            (1 - euler_mascheroni) * scipy_stats.norm.ppf(1 - 1.0 / n_strategies)
            + euler_mascheroni * scipy_stats.norm.ppf(1 - 1.0 / (n_strategies * np.e))
        ) / np.sqrt(n_observations / 252)
    else:
        selection_haircut = 0.0

    # 2. Estimation uncertainty haircut (~1/sqrt(T))
    estimation_haircut = 1.0 / np.sqrt(n_observations / 252)

    # 3. Vol drag at leverage
    vol_drag = leverage * (leverage - 1) / 2 * base_vol**2

    # 4. Leveraged Sharpe after haircuts
    # OOS Sharpe ≈ (IS Sharpe - selection_bias) × (1 - estimation_error)
    # Then leveraged Sharpe ≈ base_sharpe (leverage doesn't improve Sharpe)
    # BUT vol drag reduces it
    haircut_sharpe = max(0, base_sharpe - selection_haircut) * (1 - estimation_haircut)

    # 5. Expected OOS MaxDD (leverage × base MaxDD, with uncertainty)
    # MaxDD scales roughly linearly with leverage for moderate leverage,
    # but worse for high leverage due to compounding
    if leverage <= 3:
        dd_multiplier = leverage
    else:
        # Compounding effect: DD gets worse than linear at high leverage
        dd_multiplier = leverage * (1 + 0.1 * (leverage - 3))
    expected_oos_maxdd = base_maxdd * dd_multiplier

    # 6. Safety score (0-10)
    safety = 10.0
    # Penalise for leverage
    safety -= min(4, leverage * 0.5)
    # Penalise for low haircut Sharpe
    if haircut_sharpe < 0.5:
        safety -= 2
    elif haircut_sharpe < 1.0:
        safety -= 1
    # Penalise for expected deep drawdown
    if abs(expected_oos_maxdd) > 0.20:
        safety -= 2
    elif abs(expected_oos_maxdd) > 0.15:
        safety -= 1
    # Penalise for high vol drag
    if vol_drag > 0.10:
        safety -= 1
    safety = max(0, min(10, safety))

    return {
        "leverage": leverage,
        "raw_sharpe": base_sharpe,
        "selection_haircut": float(selection_haircut),
        "estimation_haircut": float(estimation_haircut),
        "haircut_sharpe": float(haircut_sharpe),
        "vol_drag": float(vol_drag),
        "expected_oos_maxdd": float(expected_oos_maxdd),
        "dd_multiplier": float(dd_multiplier),
        "safety_score": float(safety),
    }


# ---------------------------------------------------------------------------
# 8. Parameter Sensitivity Analysis
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    returns_func,
    param_name: str,
    param_values: list,
    base_value,
    **fixed_kwargs,
) -> dict:
    """
    Test sensitivity of strategy performance to a key parameter.

    Runs the strategy with each parameter value and checks if performance
    is robust (similar Sharpe across values) or fragile (highly sensitive).

    Parameters
    ----------
    returns_func : callable
        Function that takes param_name=value and returns pd.Series of returns.
    param_name : str
        Name of the parameter to vary.
    param_values : list
        Values to test.
    base_value : any
        The baseline parameter value used in the backtest.
    **fixed_kwargs
        Other fixed parameters passed to returns_func.

    Returns
    -------
    dict
        Sensitivity analysis results.
    """
    sharpes = []
    maxdds = []
    cagrs = []

    for val in param_values:
        try:
            kwargs = {**fixed_kwargs, param_name: val}
            rets = returns_func(**kwargs)
            if rets is None or len(rets) < 50:
                continue
            s = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
            cum = (1 + rets).cumprod()
            dd = float((cum / cum.cummax() - 1).min())
            cagr = float(cum.iloc[-1] ** (252 / len(cum)) - 1)
            sharpes.append(s)
            maxdds.append(dd)
            cagrs.append(cagr)
        except Exception:
            continue

    if not sharpes:
        return {"param_name": param_name, "robust": False, "reason": "no valid results"}

    sharpe_std = np.std(sharpes)
    sharpe_mean = np.mean(sharpes)
    sharpe_cv = sharpe_std / max(abs(sharpe_mean), 1e-6)

    # Robust if coefficient of variation < 0.5
    robust = sharpe_cv < 0.5

    return {
        "param_name": param_name,
        "param_values": param_values,
        "base_value": base_value,
        "sharpes": sharpes,
        "sharpe_mean": float(sharpe_mean),
        "sharpe_std": float(sharpe_std),
        "sharpe_cv": float(sharpe_cv),
        "maxdd_range": (float(min(maxdds)), float(max(maxdds))),
        "cagr_range": (float(min(cagrs)), float(max(cagrs))),
        "robust": robust,
    }


# ---------------------------------------------------------------------------
# 9. Comprehensive Overfitting Test Suite
# ---------------------------------------------------------------------------

class OverfittingTestSuite:
    """
    Run all overfitting detection tests on a set of strategy returns.

    Usage
    -----
    >>> suite = OverfittingTestSuite(n_strategies_tested=18)
    >>> report = suite.run_all(strategy_returns, portfolio_returns, leverage=7.1)
    >>> print(report.to_markdown())
    """

    def __init__(
        self,
        n_strategies_tested: int = 18,
        significance: float = 0.05,
        bootstrap_n: int = 5000,
    ) -> None:
        self.n_strategies_tested = n_strategies_tested
        self.significance = significance
        self.bootstrap_n = bootstrap_n

    def run_all(
        self,
        strategy_returns: dict[str, pd.Series],
        portfolio_returns: Optional[pd.Series] = None,
        leverage: float = 1.0,
        leverage_vol: float = 0.0,
        leverage_maxdd: float = 0.0,
        wf_results: Optional[dict] = None,
    ) -> OverfittingReport:
        """
        Run the complete overfitting test suite.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Per-strategy daily returns.
        portfolio_returns : pd.Series, optional
            Combined portfolio returns.
        leverage : float
            Applied leverage (for haircut calculation).
        leverage_vol : float
            Annualised vol of unlevered portfolio.
        leverage_maxdd : float
            MaxDD of unlevered portfolio (negative).
        wf_results : dict, optional
            Walk-forward results {strategy: {is_sharpe, oos_sharpe}}.

        Returns
        -------
        OverfittingReport
        """
        report = OverfittingReport()

        # 1. Per-strategy statistical tests
        for name, rets in strategy_returns.items():
            rets = rets.dropna()
            if len(rets) < 50:
                continue

            sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
            skew = float(scipy_stats.skew(rets, bias=False))
            kurt = float(scipy_stats.kurtosis(rets, bias=False) + 3)  # total kurtosis
            n_obs = len(rets)

            dsr = deflated_sharpe_ratio(
                sharpe, self.n_strategies_tested, n_obs, skew, kurt
            )
            psr = probabilistic_sharpe_ratio(sharpe, 0.0, n_obs, skew, kurt)
            min_btl = minimum_backtest_length(
                sharpe, self.n_strategies_tested, skew, kurt
            )

            actual_years = n_obs / 252
            is_significant = dsr > 0.95 and psr > 0.95 and min_btl < actual_years

            report.strategy_tests[name] = {
                "sharpe": sharpe,
                "deflated_sharpe": dsr,
                "prob_sharpe": psr,
                "min_btl_years": min_btl,
                "actual_years": actual_years,
                "skewness": skew,
                "kurtosis": kurt,
                "is_significant": is_significant,
            }

        # 2. Multiple testing correction
        sharpe_dict = {
            name: t["sharpe"] for name, t in report.strategy_tests.items()
        }
        if sharpe_dict:
            n_obs = max(
                len(rets.dropna()) for rets in strategy_returns.values()
            )
            report.multiple_testing = multiple_testing_correction(
                sharpe_dict, n_obs, self.significance
            )

        # 3. IS vs OOS degradation
        report.is_vs_oos = compute_is_oos_degradation(strategy_returns)

        # 4. White's Reality Check
        report.reality_check = whites_reality_check(
            strategy_returns,
            n_bootstrap=self.bootstrap_n,
        )

        # 5. Leverage haircut
        if leverage > 1.0 and portfolio_returns is not None:
            port_rets = portfolio_returns.dropna()
            base_sharpe = (
                float(port_rets.mean() / port_rets.std() * np.sqrt(252))
                if port_rets.std() > 0 else 0
            )
            base_vol = leverage_vol if leverage_vol > 0 else float(port_rets.std() * np.sqrt(252))
            base_maxdd = leverage_maxdd if leverage_maxdd != 0 else float(
                ((1 + port_rets).cumprod() / (1 + port_rets).cumprod().cummax() - 1).min()
            )
            report.leverage_haircut = leverage_risk_haircut(
                base_sharpe=base_sharpe / leverage,  # unlevered estimate
                leverage=leverage,
                base_vol=base_vol / leverage,  # unlevered vol
                base_maxdd=base_maxdd / leverage,  # unlevered dd estimate
                n_strategies=self.n_strategies_tested,
                n_observations=len(port_rets),
            )

        # 6. Determine overall confidence
        report = self._assess_confidence(report)

        return report

    def _assess_confidence(self, report: OverfittingReport) -> OverfittingReport:
        """Assess overall confidence level."""
        warnings = []
        score = 0
        max_score = 0

        # Check strategy significance
        if report.strategy_tests:
            n_sig = sum(
                1 for t in report.strategy_tests.values()
                if t.get("is_significant", False)
            )
            n_total = len(report.strategy_tests)
            max_score += 3
            if n_sig >= n_total * 0.5:
                score += 3
            elif n_sig >= n_total * 0.25:
                score += 2
            elif n_sig > 0:
                score += 1
            else:
                warnings.append(
                    f"No strategies pass DSR significance test "
                    f"(0/{n_total} significant)"
                )

        # Check multiple testing
        mt = report.multiple_testing
        if mt:
            max_score += 2
            if mt.get("n_surviving_bh", 0) > 0:
                score += 2
            elif mt.get("n_surviving_bonferroni", 0) > 0:
                score += 1
            else:
                warnings.append(
                    "No strategies survive multiple testing correction"
                )

        # Check IS vs OOS degradation
        if report.is_vs_oos:
            max_score += 2
            avg_deg = np.mean([
                v["degradation_pct"] for v in report.is_vs_oos.values()
            ])
            if avg_deg < 30:
                score += 2
            elif avg_deg < 50:
                score += 1
            else:
                warnings.append(
                    f"High average IS→OOS degradation: {avg_deg:.1f}%"
                )

        # Check Reality Check
        rc = report.reality_check
        if rc:
            max_score += 2
            if rc.get("p_value", 1) < 0.01:
                score += 2
            elif rc.get("p_value", 1) < 0.05:
                score += 1
            else:
                warnings.append(
                    f"White's Reality Check p-value={rc['p_value']:.3f} "
                    f"(not significant at 5%)"
                )

        # Check leverage haircut
        lh = report.leverage_haircut
        if lh:
            max_score += 1
            if lh.get("safety_score", 0) >= 6:
                score += 1
            else:
                warnings.append(
                    f"Leverage safety score only {lh['safety_score']:.1f}/10"
                )

        # Determine confidence
        if max_score == 0:
            confidence = "UNKNOWN"
        else:
            pct = score / max_score
            if pct >= 0.8:
                confidence = "HIGH"
            elif pct >= 0.5:
                confidence = "MODERATE"
            elif pct >= 0.25:
                confidence = "LOW"
            else:
                confidence = "VERY LOW — LIKELY OVERFIT"

        report.overall_confidence = confidence
        report.warnings = warnings

        logger.info(
            "Overfitting test suite: confidence=%s, score=%d/%d, warnings=%d",
            confidence, score, max_score, len(warnings),
        )

        return report
