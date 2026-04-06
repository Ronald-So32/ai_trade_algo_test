"""
Benchmark Comparison Module
============================
Compare each strategy and factor sleeve against standard benchmarks:
  - Equal-weight benchmark
  - Market (cap-weighted) benchmark
  - Current strategy portfolio
  - Portfolio + factor sleeve (additive impact)

Produces comprehensive performance evaluation with all required metrics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _annualized_return(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def _sharpe(returns: pd.Series) -> float:
    vol = returns.std()
    if vol <= 0:
        return 0.0
    return float(returns.mean() / vol * np.sqrt(252))


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    downside_vol = downside.std() if len(downside) > 1 else 1e-10
    if downside_vol <= 0:
        return 0.0
    return float(returns.mean() / downside_vol * np.sqrt(252))


def _calmar(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    max_dd = abs(_max_drawdown(returns))
    if max_dd <= 0:
        return 0.0
    return float(ann_ret / max_dd)


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def _turnover(weights: pd.DataFrame | None) -> float:
    if weights is None:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).mean())


def _cost_drag(gross: pd.Series, net: pd.Series) -> float:
    drag = (gross - net).mean() * 252
    return float(drag)


def compute_metrics(
    returns: pd.Series,
    weights: pd.DataFrame | None = None,
    gross_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Compute full performance metric suite for a return series."""
    if len(returns) < 2:
        return {k: 0.0 for k in [
            "annualized_return", "volatility", "sharpe", "sortino",
            "calmar", "max_drawdown", "turnover", "cost_drag",
        ]}

    metrics = {
        "annualized_return": _annualized_return(returns),
        "volatility": _annualized_vol(returns),
        "sharpe": _sharpe(returns),
        "sortino": _sortino(returns),
        "calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "turnover": _turnover(weights),
        "cost_drag": _cost_drag(gross_returns, returns) if gross_returns is not None else 0.0,
    }
    return metrics


def drawdown_analysis(returns: pd.Series) -> dict:
    """Compute detailed drawdown analysis."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Find worst drawdown episodes
    in_drawdown = dd < 0
    episodes = []
    start = None
    for i, (date, val) in enumerate(dd.items()):
        if val < 0 and start is None:
            start = date
        elif val >= 0 and start is not None:
            episode_dd = dd.loc[start:date]
            trough_date = episode_dd.idxmin()
            episodes.append({
                "start": start,
                "trough": trough_date,
                "end": date,
                "depth": float(episode_dd.min()),
                "duration_days": (date - start).days,
                "recovery_days": (date - trough_date).days,
            })
            start = None

    # Sort by depth
    episodes.sort(key=lambda x: x["depth"])

    return {
        "max_drawdown": float(dd.min()),
        "drawdown_series": dd,
        "rolling_drawdown": dd,
        "worst_episodes": episodes[:5],
        "avg_drawdown": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "time_in_drawdown_pct": float((dd < 0).mean()),
    }


class BenchmarkComparison:
    """
    Compare strategies against standard benchmarks.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (dates x assets).
    returns : pd.DataFrame
        Return matrix (dates x assets).
    market_caps : pd.DataFrame, optional
        Market cap matrix for cap-weighted benchmark.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame | None = None,
    ) -> None:
        self.prices = prices
        self.returns = returns
        self.market_caps = market_caps
        self._benchmarks: dict[str, pd.Series] = {}
        self._build_benchmarks()

    def _build_benchmarks(self) -> None:
        """Build equal-weight and market benchmarks."""
        n_assets = self.returns.shape[1]

        # Equal-weight benchmark
        ew_returns = self.returns.mean(axis=1)
        self._benchmarks["equal_weight"] = ew_returns

        # Cap-weighted benchmark
        if self.market_caps is not None:
            caps_aligned = self.market_caps.reindex_like(self.returns).ffill().fillna(1e9)
            cap_weights = caps_aligned.div(caps_aligned.sum(axis=1), axis=0)
            cw_returns = (cap_weights.shift(1) * self.returns).sum(axis=1)
            self._benchmarks["market_cap_weighted"] = cw_returns
        else:
            self._benchmarks["market_cap_weighted"] = ew_returns

    def compare(
        self,
        strategy_returns: dict[str, pd.Series],
        strategy_weights: dict[str, pd.DataFrame] | None = None,
        portfolio_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compare each strategy against benchmarks.

        Returns DataFrame with rows = strategies + benchmarks,
        columns = performance metrics.
        """
        rows = {}

        # Benchmarks
        for bm_name, bm_ret in self._benchmarks.items():
            rows[bm_name] = compute_metrics(bm_ret)

        # Individual strategies
        for name, ret in strategy_returns.items():
            weights = strategy_weights.get(name) if strategy_weights else None
            rows[name] = compute_metrics(ret, weights=weights)

        # Current portfolio
        if portfolio_returns is not None:
            rows["portfolio"] = compute_metrics(portfolio_returns)

        return pd.DataFrame(rows).T

    def compare_with_factor(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.Series,
        factor_name: str,
        factor_weight: float = 0.10,
    ) -> dict:
        """
        Compare portfolio with and without a factor sleeve.

        Returns metrics for: portfolio_alone, factor_alone, portfolio+factor.
        """
        # Blend: (1 - w) * portfolio + w * factor
        blended = (1 - factor_weight) * portfolio_returns + factor_weight * factor_returns
        blended = blended.dropna()

        return {
            "portfolio_alone": compute_metrics(portfolio_returns),
            f"{factor_name}_alone": compute_metrics(factor_returns),
            f"portfolio_plus_{factor_name}": compute_metrics(blended),
            "blend_weight": factor_weight,
        }

    def regime_performance(
        self,
        strategy_returns: dict[str, pd.Series],
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Evaluate strategy performance across market regimes.

        Returns DataFrame indexed by (strategy, regime) with metrics.
        """
        rows = []
        for name, ret in strategy_returns.items():
            aligned_labels = regime_labels.reindex(ret.index).dropna()
            aligned_ret = ret.reindex(aligned_labels.index)

            for regime in aligned_labels.unique():
                mask = aligned_labels == regime
                regime_ret = aligned_ret[mask]
                if len(regime_ret) < 10:
                    continue
                metrics = compute_metrics(regime_ret)
                metrics["strategy"] = name
                metrics["regime"] = regime
                metrics["n_days"] = len(regime_ret)
                rows.append(metrics)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df.set_index(["strategy", "regime"])
