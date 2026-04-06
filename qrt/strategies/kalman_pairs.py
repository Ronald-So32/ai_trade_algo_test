"""
Kalman Filter Pairs Trading Strategy
======================================
Use a Kalman filter to estimate a dynamic hedge ratio between each pair of
assets.  Compute spread residuals from the filtered state.  Trade when the
spread z-score exceeds the entry threshold.

Dependency: pykalman  (pip install pykalman)
Falls back to a simple OLS rolling hedge ratio if pykalman is unavailable.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .base import Strategy

# Optional pykalman import with graceful fallback
try:
    from pykalman import KalmanFilter as _KF
    _PYKALMAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYKALMAN_AVAILABLE = False


PairList = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Kalman filter helper
# ---------------------------------------------------------------------------

class _KalmanHedgeRatio:
    """
    Online Kalman filter that estimates a dynamic hedge ratio β and intercept α
    for the spread:  y_t = α_t + β_t * x_t + ε_t

    State vector θ = [α, β]ᵀ, modelled as a random walk.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_cov: float = 1.0,
    ) -> None:
        self.delta = delta
        self.obs_cov = obs_cov
        self._reset()

    def _reset(self) -> None:
        self.theta = np.zeros(2)           # [α, β]
        self.P = np.eye(2) * 1.0           # state covariance
        self.Q = np.eye(2) * self.delta    # process noise

    def update(self, y: float, x: float) -> tuple[float, float]:
        """
        One-step Kalman update.

        Returns
        -------
        (spread_residual, innovation_variance)
        """
        H = np.array([[1.0, x]])  # observation matrix
        # Predict
        # (random-walk state model: F = I)
        P_pred = self.P + self.Q

        # Innovation
        y_hat = float(H @ self.theta)
        innov = y - y_hat
        S = float(H @ P_pred @ H.T) + self.obs_cov

        # Kalman gain
        K = (P_pred @ H.T) / S  # shape (2,)

        # Update
        self.theta = self.theta + K.flatten() * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return innov, S


class _OLSHedgeRatio:
    """Rolling OLS hedge ratio fallback (used when pykalman is not installed)."""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._y_buf: list[float] = []
        self._x_buf: list[float] = []

    def update(self, y: float, x: float) -> tuple[float, float]:
        self._y_buf.append(y)
        self._x_buf.append(x)
        if len(self._y_buf) > self.window:
            self._y_buf.pop(0)
            self._x_buf.pop(0)
        yarr = np.array(self._y_buf)
        xarr = np.array(self._x_buf)
        if len(yarr) < 2 or xarr.std() < 1e-10:
            return 0.0, 1.0
        # Simple OLS
        beta = np.cov(yarr, xarr)[0, 1] / np.var(xarr)
        alpha = yarr.mean() - beta * xarr.mean()
        residual = y - (alpha + beta * x)
        variance = max(np.var(yarr - (alpha + beta * xarr)), 1e-10)
        return residual, variance


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class KalmanPairs(Strategy):
    """
    Kalman-filter pairs trading strategy.

    For each pair (A, B) selected by cointegration screening:
      - Maintain a dynamic hedge ratio β_t via Kalman filter.
      - Spread residual: e_t = log(A_t) - α_t - β_t * log(B_t)
      - Z-score: z_t = e_t / sqrt(innovation_variance)
      - Enter when |z| > entry_z; exit when |z| < exit_z.

    Parameters
    ----------
    n_pairs : int
        Number of top-ranked pairs to trade (default 10).
    formation_period : int
        Days used for initial cointegration screening (default 252).
    entry_z : float
        Z-score entry threshold (default 2.0).
    exit_z : float
        Z-score exit threshold (default 0.5).
    max_holding : int
        Max days to hold a position (default 30).
    delta : float
        Kalman process-noise parameter (default 1e-4).
    obs_cov : float
        Kalman observation noise (default 1.0).
    rebalance_freq : int
        Days between pair re-screening (default 252).
    target_gross : float
        Target gross exposure (default 1.0).
    warmup : int
        Minimum Kalman filter steps before trading (default 30).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Elliott, van der Hoek & Malcolm (2005) — Kalman filter for spread modeling"
        ),
        "historical_evidence": (
            "Improved hedge ratios vs static OLS; modest academic evidence of improvement"
        ),
        "implementation_risks": (
            "Filter divergence, overfitting to in-sample dynamics, computational complexity"
        ),
        "realistic_expectations": (
            "Marginal improvement over static pairs; benefits from adaptive hedge ratio "
            "but not transformative"
        ),
    }

    def __init__(
        self,
        n_pairs: int = 10,
        formation_period: int = 252,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_holding: int = 30,
        delta: float = 1e-4,
        obs_cov: float = 1.0,
        rebalance_freq: int = 252,
        target_gross: float = 1.0,
        warmup: int = 30,
    ) -> None:
        params = dict(
            n_pairs=n_pairs,
            formation_period=formation_period,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding=max_holding,
            delta=delta,
            obs_cov=obs_cov,
            rebalance_freq=rebalance_freq,
            target_gross=target_gross,
            warmup=warmup,
        )
        super().__init__(name="KalmanPairs", params=params)

    # ------------------------------------------------------------------
    # Pair-screening helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cointegration_score(log_a: np.ndarray, log_b: np.ndarray) -> float:
        """
        Simple ADF-proxy: fit OLS spread and return -variance of residuals
        (lower residual variance → tighter pair).
        """
        if len(log_a) < 10:
            return np.inf
        beta = np.cov(log_a, log_b)[0, 1] / max(np.var(log_b), 1e-12)
        alpha = log_a.mean() - beta * log_b.mean()
        resid = log_a - (alpha + beta * log_b)
        return resid.std()  # lower is better

    def _select_pairs(self, log_prices: pd.DataFrame) -> PairList:
        """Select top *n_pairs* pairs by residual variance."""
        n_pairs: int = self.params["n_pairs"]
        cols = list(log_prices.columns)
        scores: list[tuple[float, str, str]] = []
        for a, b in [(c1, c2) for i, c1 in enumerate(cols) for c2 in cols[i + 1:]]:
            score = self._cointegration_score(
                log_prices[a].values,
                log_prices[b].values,
            )
            scores.append((score, a, b))
        scores.sort(key=lambda x: x[0])
        return [(a, b) for _, a, b in scores[:n_pairs]]

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
        Run Kalman filter on selected pairs and generate spread signals.

        Returns
        -------
        pd.DataFrame
            Signals in [-1, +1], same shape as *prices*.
        """
        formation: int = self.params["formation_period"]
        entry_z: float = self.params["entry_z"]
        exit_z: float = self.params["exit_z"]
        max_hold: int = self.params["max_holding"]
        delta: float = self.params["delta"]
        obs_cov: float = self.params["obs_cov"]
        rebal_freq: int = self.params["rebalance_freq"]
        warmup: int = self.params["warmup"]

        dates = prices.index
        n_dates = len(dates)
        signals_raw = pd.DataFrame(0.0, index=dates, columns=prices.columns)

        if n_dates <= formation:
            return signals_raw

        log_prices = np.log(prices.clip(lower=1e-8))

        active_pairs: PairList = []
        kf_estimators: dict[tuple[str, str], _KalmanHedgeRatio | _OLSHedgeRatio] = {}
        pair_position: dict[tuple[str, str], float] = {}
        pair_hold: dict[tuple[str, str], int] = {}
        pair_warmup_count: dict[tuple[str, str], int] = {}
        pair_resid_history: dict[tuple[str, str], list[float]] = {}
        pair_innov_var: dict[tuple[str, str], list[float]] = {}

        last_rebal: int = -rebal_freq

        for t in range(formation, n_dates):
            # ---- Re-screen pairs if scheduled ----
            if t - last_rebal >= rebal_freq:
                form_lp = log_prices.iloc[t - formation: t]
                active_pairs = self._select_pairs(form_lp)
                last_rebal = t

                # Initialise Kalman filters for new pairs
                for pair in active_pairs:
                    if pair not in kf_estimators:
                        if _PYKALMAN_AVAILABLE:
                            kf_estimators[pair] = _KalmanHedgeRatio(
                                delta=delta, obs_cov=obs_cov
                            )
                        else:
                            kf_estimators[pair] = _OLSHedgeRatio(window=60)
                        pair_position[pair] = 0.0
                        pair_hold[pair] = 0
                        pair_warmup_count[pair] = 0
                        pair_resid_history[pair] = []
                        pair_innov_var[pair] = []

            # ---- Step each Kalman filter and generate signals ----
            for pair in active_pairs:
                a, b = pair
                if a not in log_prices.columns or b not in log_prices.columns:
                    continue

                y_t = float(log_prices.iloc[t][a])
                x_t = float(log_prices.iloc[t][b])

                resid, innov_s = kf_estimators[pair].update(y_t, x_t)
                pair_resid_history[pair].append(resid)
                pair_innov_var[pair].append(innov_s)
                pair_warmup_count[pair] += 1

                if pair_warmup_count[pair] < warmup:
                    continue

                # Rolling z-score of residuals
                recent_resids = pair_resid_history[pair][-max(warmup, 30):]
                resid_std = max(float(np.std(recent_resids)), 1e-10)
                resid_mean = float(np.mean(recent_resids))
                z = (resid - resid_mean) / resid_std

                pos = pair_position[pair]

                # Manage existing position
                if pos != 0:
                    pair_hold[pair] += 1
                    if pair_hold[pair] >= max_hold or abs(z) <= exit_z:
                        pair_position[pair] = 0.0
                        pair_hold[pair] = 0
                        pos = 0.0

                # Entry
                if pos == 0:
                    if z > entry_z:
                        # Spread too high: y (asset A) overpriced vs x (asset B)
                        pair_position[pair] = -1.0  # short A, long B
                        pair_hold[pair] = 0
                    elif z < -entry_z:
                        pair_position[pair] = 1.0   # long A, short B
                        pair_hold[pair] = 0

                pos = pair_position[pair]
                if pos != 0:
                    signals_raw.loc[dates[t], a] += pos
                    signals_raw.loc[dates[t], b] += -pos

        signals = signals_raw.clip(-1.0, 1.0)
        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Gross-normalise signals to target_gross exposure.

        Returns
        -------
        pd.DataFrame
            Portfolio weights, same shape as *signals*.
        """
        target_gross: float = self.params["target_gross"]
        raw_weights = signals.copy()
        gross = raw_weights.abs().sum(axis=1).replace(0, np.nan)
        weights = raw_weights.div(gross, axis=0).mul(target_gross).fillna(0.0)

        # Regime filter: Kalman pairs rely on mean-reversion dynamics.
        # In crisis regimes, spreads diverge rather than revert.
        crisis_probs = kwargs.get("crisis_probs", None)
        if crisis_probs is not None:
            weights = self.apply_regime_scaling(
                weights, crisis_probs,
                soft_start=0.4,
                floor=0.25,
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
