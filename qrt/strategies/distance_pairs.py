"""
Distance-Based Pairs Trading Strategy
=======================================
Formation period: identify closest pairs by sum-of-squared-distances on
normalised price series.
Trading period: trade spread when it deviates > entry_z standard deviations;
exit at exit_z or after max_holding days.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .base import Strategy


PairList = List[Tuple[str, str]]


class DistancePairs(Strategy):
    """
    Distance-based pairs trading (Gatev et al. 2006 style).

    Parameters
    ----------
    formation_period : int
        Days used to identify pairs (default 252).
    n_pairs : int
        Number of pairs to trade simultaneously (default 20).
    entry_z : float
        Spread z-score required to open a trade (default 2.0).
    exit_z : float
        Spread z-score at which to close a trade (default 0.5).
    max_holding : int
        Maximum days to hold an open pair position (default 30).
    target_gross : float
        Target gross portfolio exposure (default 1.0).
    rebalance_freq : int
        How often (in days) to re-run pair formation (default 252).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Gatev, Goetzmann & Rouwenhorst (2006) — \"Pairs Trading\""
        ),
        "historical_evidence": (
            "Historically profitable; declining returns post-publication (2000s onward)"
        ),
        "implementation_risks": (
            "Structural breaks in pair relationships, convergence failure, execution costs"
        ),
        "realistic_expectations": (
            "Returns have declined as strategy became widely adopted; "
            "requires careful pair selection"
        ),
    }

    def __init__(
        self,
        formation_period: int = 252,
        n_pairs: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_holding: int = 30,
        target_gross: float = 1.0,
        rebalance_freq: int = 252,
    ) -> None:
        params = dict(
            formation_period=formation_period,
            n_pairs=n_pairs,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding=max_holding,
            target_gross=target_gross,
            rebalance_freq=rebalance_freq,
        )
        super().__init__(name="DistancePairs", params=params)

    # ------------------------------------------------------------------
    # Pair-formation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(prices: pd.DataFrame) -> pd.DataFrame:
        """Divide each series by its first value so all start at 1."""
        return prices / prices.iloc[0]

    def _find_pairs(self, formation_prices: pd.DataFrame) -> PairList:
        """
        Return the *n_pairs* asset pairs with smallest sum-of-squared
        distances on the normalised price series.
        """
        n_pairs: int = self.params["n_pairs"]
        norm = self._normalise(formation_prices)
        cols = list(norm.columns)

        # Pairwise SSD (sum of squared differences)
        price_matrix = norm.values.T  # shape (n_assets, n_dates)
        dist_matrix = cdist(price_matrix, price_matrix, metric="sqeuclidean")

        # Extract upper triangle pairs (avoid self and duplicates)
        n = len(cols)
        pair_distances: list[tuple[float, str, str]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_distances.append((dist_matrix[i, j], cols[i], cols[j]))

        pair_distances.sort(key=lambda x: x[0])
        return [(a, b) for _, a, b in pair_distances[:n_pairs]]

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    @staticmethod
    def _spread(norm_a: pd.Series, norm_b: pd.Series) -> pd.Series:
        """Spread defined as: norm_a - norm_b (simple distance spread)."""
        return norm_a - norm_b

    def _spread_zscore(
        self,
        spread: pd.Series,
        formation_spread: pd.Series,
    ) -> pd.Series:
        """Z-score of spread using formation-period mean/std."""
        mu = formation_spread.mean()
        sigma = formation_spread.std()
        if sigma < 1e-10:
            return pd.Series(0.0, index=spread.index)
        return (spread - mu) / sigma

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Produce per-asset signals based on pairs spread z-scores.

        For each active pair (A, B):
          - If spread z > +entry_z  → short A, long  B  (signal_A=-1, signal_B=+1)
          - If spread z < -entry_z  → long  A, short B  (signal_A=+1, signal_B=-1)
          - Exit when |z| < exit_z or after max_holding days

        Signals for each asset are summed across all pairs it participates in,
        then clipped to [-1, +1].

        Returns
        -------
        pd.DataFrame
            Signals clipped to [-1, +1], same shape as *prices*.
        """
        formation: int = self.params["formation_period"]
        entry_z: float = self.params["entry_z"]
        exit_z: float = self.params["exit_z"]
        max_hold: int = self.params["max_holding"]
        rebal_freq: int = self.params["rebalance_freq"]

        dates = prices.index
        n_dates = len(dates)
        signals_raw = pd.DataFrame(0.0, index=dates, columns=prices.columns)

        if n_dates <= formation:
            return signals_raw

        # State for each pair: position (-1 / 0 / +1 from A's perspective), holding days
        active_pairs: PairList = []
        pair_position: dict[tuple[str, str], float] = {}
        pair_hold: dict[tuple[str, str], int] = {}
        pair_mu: dict[tuple[str, str], float] = {}
        pair_sigma: dict[tuple[str, str], float] = {}
        last_rebal: int = -rebal_freq  # force formation at first opportunity

        for t in range(formation, n_dates):
            # ---- Re-form pairs if scheduled ----
            if t - last_rebal >= rebal_freq:
                form_prices = prices.iloc[t - formation: t]
                active_pairs = self._find_pairs(form_prices)
                last_rebal = t

                # Recompute formation stats for each pair
                norm_form = self._normalise(form_prices)
                for a, b in active_pairs:
                    if a not in norm_form.columns or b not in norm_form.columns:
                        continue
                    sp = self._spread(norm_form[a], norm_form[b])
                    pair_mu[(a, b)] = sp.mean()
                    pair_sigma[(a, b)] = max(sp.std(), 1e-10)
                    if (a, b) not in pair_position:
                        pair_position[(a, b)] = 0.0
                        pair_hold[(a, b)] = 0

            # ---- Normalise prices up to today ----
            # We use a rolling normalisation anchored at the formation start
            t_start = max(0, last_rebal - formation)
            anchor_prices = prices.iloc[t_start]
            norm_today = prices.iloc[t] / anchor_prices

            # ---- Update signals for each pair ----
            for a, b in active_pairs:
                if a not in prices.columns or b not in prices.columns:
                    continue
                if (a, b) not in pair_mu:
                    continue

                spread_val = norm_today[a] - norm_today[b]
                z = (spread_val - pair_mu[(a, b)]) / pair_sigma[(a, b)]

                pos = pair_position[(a, b)]

                # Manage existing position
                if pos != 0:
                    pair_hold[(a, b)] += 1
                    if (
                        pair_hold[(a, b)] >= max_hold
                        or abs(z) <= exit_z
                    ):
                        pair_position[(a, b)] = 0.0
                        pair_hold[(a, b)] = 0
                        pos = 0.0

                # Enter new position
                if pos == 0:
                    if z > entry_z:
                        pair_position[(a, b)] = -1.0  # spread too wide: short A, long B
                        pair_hold[(a, b)] = 0
                    elif z < -entry_z:
                        pair_position[(a, b)] = 1.0   # spread too narrow: long A, short B
                        pair_hold[(a, b)] = 0

                pos = pair_position[(a, b)]
                if pos != 0:
                    signals_raw.loc[dates[t], a] += pos
                    signals_raw.loc[dates[t], b] += -pos

        # Clip aggregate signals to [-1, 1]
        signals = signals_raw.clip(-1.0, 1.0)
        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Equal-weight within active positions, scaled to target gross exposure.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        target_gross: float = self.params["target_gross"]

        raw_weights = signals.copy()
        gross = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        # Regime filter (Endres & Stubinger 2019): pairs trading relies on
        # mean-reversion, which breaks down in crisis regimes where spreads
        # diverge.  Scale down when crisis probability is elevated.
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.4,  # pairs more tolerant than momentum
                floor=0.25,      # keep some exposure for recovery trades
            )

        return weights

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """End-to-end: signals → weights → backtest summary."""
        signals = self.generate_signals(prices, returns, **kwargs)
        weights = self.compute_weights(signals, **kwargs)
        summary = self.backtest_summary(weights, returns)
        return {"signals": signals, "weights": weights, "summary": summary}
