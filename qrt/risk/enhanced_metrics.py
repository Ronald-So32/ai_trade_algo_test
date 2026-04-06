"""
Enhanced Risk Metrics
======================
Comprehensive risk/return metrics panel including drawdown-focused measures.

Beyond Sharpe, reports Sortino, Calmar/MAR, CVaR/ES, CDaR, skew, kurtosis,
time-under-water, and regime-conditional performance.

References
----------
- Chekhlov et al. (2005) — CDaR definition
- Rockafellar & Uryasev (2000) — CVaR optimization
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .drawdown_risk import compute_drawdown_series, compute_cdar, compute_cvar

logger = logging.getLogger(__name__)


def compute_full_metrics(
    returns: pd.Series,
    name: str = "portfolio",
    risk_free: float = 0.0,
    cvar_alpha: float = 0.95,
    cdar_alpha: float = 0.95,
) -> dict:
    """
    Compute a comprehensive set of risk/return metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    name : str
        Label for the return stream.
    risk_free : float
        Annualized risk-free rate (default 0).
    cvar_alpha : float
        CVaR confidence level (default 0.95).
    cdar_alpha : float
        CDaR confidence level (default 0.95).

    Returns
    -------
    dict
        Full metrics panel.
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return {"name": name, "n_obs": n, "error": "insufficient data"}

    ann = 252
    daily_rf = risk_free / ann

    # Basic return stats
    total_return = float((1 + returns).prod() - 1)
    cagr = float((1 + total_return) ** (ann / n) - 1) if n > 0 else 0.0
    ann_vol = float(returns.std() * np.sqrt(ann))
    excess = returns - daily_rf

    # Sharpe
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann)) if excess.std() > 0 else 0.0

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside ** 2).mean())) * np.sqrt(ann) if len(downside) > 0 else 1e-8
    sortino = float(excess.mean() * ann / downside_std) if downside_std > 0 else 0.0

    # Drawdown analysis
    dd = compute_drawdown_series(returns)
    max_dd = float(dd.min())
    max_dd_abs = abs(max_dd)

    # Calmar (CAGR / |MaxDD|)
    calmar = float(cagr / max_dd_abs) if max_dd_abs > 0 else 0.0

    # CVaR and CDaR
    cvar = compute_cvar(returns, alpha=cvar_alpha)
    cdar = compute_cdar(returns, alpha=cdar_alpha)

    # Higher moments
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())

    # Time under water
    in_dd = dd < -0.001  # at least 0.1% drawdown
    time_in_dd_pct = float(in_dd.sum() / n * 100) if n > 0 else 0.0

    # Drawdown duration analysis
    dd_episodes = _detect_dd_episodes(dd)
    avg_dd_duration = float(np.mean([e["duration"] for e in dd_episodes])) if dd_episodes else 0.0
    max_dd_duration = float(max([e["duration"] for e in dd_episodes])) if dd_episodes else 0.0

    # Turnover (if we have it, approximate from returns)
    avg_daily_return = float(returns.mean())
    hit_rate = float((returns > 0).sum() / n) if n > 0 else 0.5

    # Tail ratio (95th percentile gain / 5th percentile loss)
    p95 = float(returns.quantile(0.95))
    p05 = float(returns.quantile(0.05))
    tail_ratio = float(abs(p95 / p05)) if abs(p05) > 1e-10 else 0.0

    return {
        "name": name,
        "n_obs": n,
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_abs": max_dd_abs,
        "cvar_95": cvar,
        "cdar_95": cdar,
        "skewness": skew,
        "kurtosis": kurt,
        "time_in_drawdown_pct": time_in_dd_pct,
        "avg_dd_duration_days": avg_dd_duration,
        "max_dd_duration_days": max_dd_duration,
        "hit_rate": hit_rate,
        "tail_ratio": tail_ratio,
        "avg_daily_return": avg_daily_return,
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }


def _detect_dd_episodes(dd: pd.Series, threshold: float = -0.01) -> list[dict]:
    """Detect distinct drawdown episodes."""
    episodes = []
    in_dd = False
    start = None
    trough = 0.0
    trough_date = None

    for i, (date, val) in enumerate(dd.items()):
        if val < threshold and not in_dd:
            in_dd = True
            start = date
            trough = val
            trough_date = date
        elif in_dd:
            if val < trough:
                trough = val
                trough_date = date
            if val >= -0.001:  # recovered
                in_dd = False
                duration = (date - start).days if hasattr(date, 'days') else i
                try:
                    duration = (date - start).days
                except (TypeError, AttributeError):
                    duration = 0
                episodes.append({
                    "start": start,
                    "trough_date": trough_date,
                    "end": date,
                    "depth": float(trough),
                    "duration": duration,
                })

    return episodes


def compute_regime_metrics(
    returns: pd.Series,
    regime_labels: pd.Series,
    name: str = "portfolio",
) -> pd.DataFrame:
    """
    Compute metrics conditional on regime labels.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    regime_labels : pd.Series
        Regime label per date.
    name : str
        Portfolio label.

    Returns
    -------
    pd.DataFrame
        One row per regime with key metrics.
    """
    common = returns.index.intersection(regime_labels.index)
    returns = returns.loc[common]
    labels = regime_labels.loc[common]

    rows = []
    for regime in sorted(labels.unique(), key=str):
        mask = labels == regime
        r = returns[mask]
        if len(r) < 5:
            continue

        sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        dd = compute_drawdown_series(r)

        rows.append({
            "name": name,
            "regime": regime,
            "n_days": len(r),
            "ann_return": float(r.mean() * 252),
            "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": sharpe,
            "max_drawdown": float(dd.min()),
            "skewness": float(r.skew()),
            "hit_rate": float((r > 0).sum() / len(r)),
        })

    return pd.DataFrame(rows)


def metrics_comparison_table(
    strategy_returns: dict[str, pd.Series],
    portfolio_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build a comparison table of enhanced metrics across all strategies.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Per-strategy daily return series.
    portfolio_returns : pd.Series, optional
        Portfolio-level return series.

    Returns
    -------
    pd.DataFrame
        One row per strategy + portfolio with full metrics.
    """
    rows = []
    for name, rets in strategy_returns.items():
        m = compute_full_metrics(rets, name=name)
        rows.append(m)

    if portfolio_returns is not None:
        m = compute_full_metrics(portfolio_returns, name="PORTFOLIO")
        rows.append(m)

    return pd.DataFrame(rows).set_index("name")
