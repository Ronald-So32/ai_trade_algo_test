"""
Hierarchical Risk Allocation
==============================
Implements HERC (Hierarchical Equal Risk Contribution) with support for
downside risk measures (CVaR, CDaR) at the strategy level.

HERC extends HRP by:
1. Building a hierarchical cluster tree from the correlation matrix.
2. Allocating risk across clusters using an equal-risk-contribution
   principle.
3. Supporting downside risk measures (CVaR/CDaR) instead of just variance.

This is particularly suitable for strategy-level allocation where the
number of "assets" (strategies) is small relative to the sample size.

References
----------
- De Prado (2016), "Building Diversified Portfolios that Outperform
  Out-of-Sample" — original HRP.
- Thomas, Delcourt, et al. (2021), "The Hierarchical Equal Risk
  Contribution Portfolio" — HERC extension with downside risk.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

RiskMeasure = Literal["variance", "cvar", "cdar"]


def _compute_risk_measure(
    returns: pd.Series,
    measure: RiskMeasure,
    alpha: float = 0.95,
) -> float:
    """Compute a risk measure for a single return series."""
    if measure == "variance":
        return float(returns.std() ** 2) if len(returns) > 1 else 1e-8

    elif measure == "cvar":
        threshold = returns.quantile(1 - alpha)
        tail = returns[returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 1e-8

    elif measure == "cdar":
        cum = (1 + returns).cumprod()
        dd = (cum / cum.cummax() - 1).abs()
        threshold = dd.quantile(alpha)
        tail = dd[dd >= threshold]
        return float(tail.mean()) if len(tail) > 0 else 1e-8

    else:
        raise ValueError(f"Unknown risk measure: {measure}")


class HERCAllocator:
    """
    Hierarchical Equal Risk Contribution (HERC) allocator.

    Parameters
    ----------
    risk_measure : str
        Risk measure for allocation: "variance", "cvar", or "cdar".
    n_clusters : int or None
        Number of clusters. If None, auto-detect using inconsistency.
    alpha : float
        Confidence level for CVaR/CDaR (default 0.95).
    linkage_method : str
        Hierarchical clustering linkage method (default "ward").
    """

    def __init__(
        self,
        risk_measure: RiskMeasure = "cdar",
        n_clusters: Optional[int] = None,
        alpha: float = 0.95,
        linkage_method: str = "ward",
    ) -> None:
        self.risk_measure = risk_measure
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.linkage_method = linkage_method

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute HERC weights from strategy return streams.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x strategies).

        Returns
        -------
        pd.Series
            Strategy weights that sum to 1.
        """
        clean = returns.dropna(how="all").ffill().fillna(0.0)
        names = list(clean.columns)
        n = len(names)

        if n < 2:
            return pd.Series(1.0, index=names, name="weight")

        if n == 2:
            # Two strategies: inverse risk
            risks = [
                _compute_risk_measure(clean[c], self.risk_measure, self.alpha)
                for c in names
            ]
            inv = [1.0 / max(r, 1e-10) for r in risks]
            total = sum(inv)
            weights = [i / total for i in inv]
            return pd.Series(weights, index=names, name="weight")

        # Step 1: Correlation / distance matrix
        corr = clean.corr().values
        # Ensure valid correlation matrix
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        # Distance: d = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0.0)

        # Make symmetric
        dist = (dist + dist.T) / 2.0

        # Step 2: Hierarchical clustering
        condensed = squareform(dist, checks=False)
        # Handle potential NaN/inf in condensed distances
        condensed = np.nan_to_num(condensed, nan=1.0, posinf=2.0, neginf=0.0)
        Z = linkage(condensed, method=self.linkage_method)

        # Step 3: Determine clusters
        if self.n_clusters is None:
            # Auto: use sqrt(n) clusters, minimum 2
            k = max(2, min(n - 1, int(np.sqrt(n)) + 1))
        else:
            k = min(self.n_clusters, n - 1)
        k = max(2, k)

        cluster_labels = fcluster(Z, t=k, criterion="maxclust")

        # Step 4: HERC allocation
        # a) Compute risk for each cluster (as if equally weighted within)
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(names[i])

        cluster_risks = {}
        for label, members in clusters.items():
            cluster_ret = clean[members].mean(axis=1)
            cluster_risks[label] = _compute_risk_measure(
                cluster_ret, self.risk_measure, self.alpha
            )

        # b) Inter-cluster: inverse risk contribution
        inv_cluster_risk = {
            label: 1.0 / max(risk, 1e-10)
            for label, risk in cluster_risks.items()
        }
        total_inv = sum(inv_cluster_risk.values())
        cluster_weights = {
            label: inv / total_inv for label, inv in inv_cluster_risk.items()
        }

        # c) Intra-cluster: inverse risk within each cluster
        weights = {}
        for label, members in clusters.items():
            cw = cluster_weights[label]
            if len(members) == 1:
                weights[members[0]] = cw
            else:
                member_risks = {
                    m: _compute_risk_measure(
                        clean[m], self.risk_measure, self.alpha
                    )
                    for m in members
                }
                inv_risks = {
                    m: 1.0 / max(r, 1e-10) for m, r in member_risks.items()
                }
                total_member_inv = sum(inv_risks.values())
                for m in members:
                    weights[m] = cw * inv_risks[m] / total_member_inv

        # Normalize to sum to 1
        w = pd.Series(weights)
        w = w / w.sum()
        w.name = "weight"

        logger.info(
            "HERC allocation (%s, k=%d): %s",
            self.risk_measure,
            k,
            {n: f"{w[n]:.3f}" for n in names},
        )

        return w.reindex(names).fillna(0.0)
