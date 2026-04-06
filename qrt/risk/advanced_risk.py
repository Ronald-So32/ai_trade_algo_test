"""
Advanced Risk Management Module
================================
Research-backed risk reduction techniques that go beyond max-drawdown
minimisation to address tail risk, systemic fragility, and loss asymmetry.

Implements six mechanisms:

1. **Turbulence Index** — Mahalanobis distance of current returns from the
   historical distribution.  When turbulence spikes, exposure is reduced
   *before* drawdowns materialise.
   Reference: Kritzman & Li (2010), "Skulls, Financial Turbulence, and Risk
   Management", Journal of Portfolio Management.

2. **Absorption Ratio** — Fraction of total return variance explained by a
   fixed number of PCA eigenvectors.  High absorption signals tightly coupled
   (fragile) markets where diversification breaks down.
   Reference: Kritzman, Li, Page & Rigobon (2011), "Principal Components as a
   Measure of Systemic Risk", Journal of Portfolio Management.

3. **Regime-Aware Adaptive Stops** — Volatility-scaled trailing stops that
   widen in high-vol regimes and tighten in low-vol, avoiding premature exits
   during normal volatility while preserving crash protection.
   Reference: Kaminski & Lo (2014), "When Do Stop-Loss Rules Stop Losses?",
   Journal of Investment Management.

4. **Downside Risk Parity** — Strategy allocation that equalises *downside*
   risk contribution (semi-variance) rather than total variance, directing
   risk budget away from left-tail-heavy strategies.
   Reference: Sortino & van der Meer (1991), "Downside risk", Journal of
   Portfolio Management; Boudt, Carl & Peterson (2013).

5. **CVaR-Optimized Allocation** — Minimises portfolio Conditional
   Value-at-Risk (Expected Shortfall) subject to a minimum return target,
   providing coherent tail-risk control.
   Reference: Rockafellar & Uryasev (2000, 2002), "Optimization of
   Conditional Value-at-Risk", Journal of Risk.

6. **Maximum Diversification Overlay** — Maximises the diversification ratio
   (weighted average vol / portfolio vol), ensuring the portfolio extracts
   maximum diversification benefit from its constituents.
   Reference: Choueifaty & Coignard (2008), "Toward Maximum Diversification",
   Journal of Portfolio Management.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Turbulence Index (Kritzman & Li, 2010)
# ---------------------------------------------------------------------------

class TurbulenceIndex:
    """
    Mahalanobis-distance-based turbulence detector.

    Turbulence_t = (r_t - mu)' Sigma^{-1} (r_t - mu)

    where mu and Sigma are estimated from a rolling window of historical
    returns.  High turbulence days indicate unusual multivariate return
    behaviour — exactly the conditions where diversification fails.

    The scaling mechanism is continuous: exposure scales linearly from 1.0
    (at turbulence <= threshold) down to floor (at turbulence >= critical).

    Parameters
    ----------
    lookback : int
        Rolling window for estimating mu and Sigma (default 252).
    threshold_percentile : float
        Percentile of historical turbulence above which scaling begins
        (default 0.75 — top 25% of turbulence days).
    critical_percentile : float
        Percentile above which exposure hits the floor (default 0.95).
    floor : float
        Minimum exposure when turbulence is extreme (default 0.20).
    """

    def __init__(
        self,
        lookback: int = 252,
        threshold_percentile: float = 0.75,
        critical_percentile: float = 0.95,
        floor: float = 0.20,
    ) -> None:
        self.lookback = lookback
        self.threshold_pct = threshold_percentile
        self.critical_pct = critical_percentile
        self.floor = floor

    def compute_turbulence(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute daily turbulence index for a multivariate return series.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x assets/strategies).

        Returns
        -------
        pd.Series
            Turbulence index value per date.
        """
        n_dates, n_assets = returns.shape
        turbulence = pd.Series(np.nan, index=returns.index)

        for t in range(self.lookback, n_dates):
            window = returns.iloc[t - self.lookback : t].values
            r_t = returns.iloc[t].values

            mu = window.mean(axis=0)
            diff = r_t - mu

            cov = np.cov(window, rowvar=False)
            # Regularise for invertibility
            cov += np.eye(n_assets) * 1e-8

            try:
                cov_inv = np.linalg.inv(cov)
                turb = float(diff @ cov_inv @ diff)
            except np.linalg.LinAlgError:
                turb = 0.0

            turbulence.iloc[t] = turb

        return turbulence

    def compute_scaling(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute exposure scaling factors based on turbulence.

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0].
        """
        turb = self.compute_turbulence(returns)

        # Use expanding window for percentile thresholds (avoids lookahead)
        scaling = pd.Series(1.0, index=returns.index)

        for t in range(self.lookback + 1, len(returns)):
            hist_turb = turb.iloc[self.lookback : t]
            if len(hist_turb) < 10:
                continue

            threshold = hist_turb.quantile(self.threshold_pct)
            critical = hist_turb.quantile(self.critical_pct)
            current = turb.iloc[t]

            if np.isnan(current):
                continue

            if current <= threshold:
                scaling.iloc[t] = 1.0
            elif current >= critical:
                scaling.iloc[t] = self.floor
            else:
                # Linear interpolation between threshold and critical
                ramp = critical - threshold
                if ramp > 0:
                    progress = (current - threshold) / ramp
                    scaling.iloc[t] = 1.0 - progress * (1.0 - self.floor)
                else:
                    scaling.iloc[t] = self.floor

        n_scaled = (scaling < 1.0).sum()
        logger.info(
            "Turbulence index: %d/%d days with reduced exposure (%.1f%%)",
            n_scaled, len(scaling), 100 * n_scaled / max(1, len(scaling)),
        )

        return scaling


# ---------------------------------------------------------------------------
# 2. Absorption Ratio (Kritzman et al., 2011)
# ---------------------------------------------------------------------------

class AbsorptionRatio:
    """
    Systemic risk indicator based on PCA variance concentration.

    AR_t = sum(sigma^2 of top k eigenvectors) / sum(all sigma^2)

    High AR means returns are driven by few common factors — the market is
    tightly coupled and diversification is illusory.  We reduce exposure
    when the AR is elevated relative to its own history.

    Parameters
    ----------
    n_components : int
        Number of top eigenvectors (default 5).
    lookback : int
        Rolling window for PCA (default 252).
    ar_delta_lookback : int
        Window for computing AR change (delta AR, default 21).
    threshold_delta : float
        AR delta above which scaling kicks in (default 0.01).
    floor : float
        Minimum exposure (default 0.30).
    """

    def __init__(
        self,
        n_components: int = 5,
        lookback: int = 252,
        ar_delta_lookback: int = 21,
        threshold_delta: float = 0.01,
        floor: float = 0.30,
    ) -> None:
        self.n_components = n_components
        self.lookback = lookback
        self.ar_delta_lookback = ar_delta_lookback
        self.threshold_delta = threshold_delta
        self.floor = floor

    def compute_absorption_ratio(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute rolling absorption ratio.

        Returns
        -------
        pd.Series
            AR value per date (fraction in [0, 1]).
        """
        n_dates, n_assets = returns.shape
        k = min(self.n_components, n_assets)
        ar = pd.Series(np.nan, index=returns.index)

        for t in range(self.lookback, n_dates):
            window = returns.iloc[t - self.lookback : t].values
            cov = np.cov(window, rowvar=False)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]  # descending

            total_var = eigvals.sum()
            if total_var > 0:
                ar.iloc[t] = eigvals[:k].sum() / total_var
            else:
                ar.iloc[t] = 1.0

        return ar

    def compute_scaling(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute exposure scaling based on absorption ratio dynamics.

        Uses the *change* (delta) in AR, not the level, per the original
        paper's finding that rising AR predicts stress better than the level.

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0].
        """
        ar = self.compute_absorption_ratio(returns)

        # Delta AR: change over ar_delta_lookback
        delta_ar = ar - ar.shift(self.ar_delta_lookback)

        scaling = pd.Series(1.0, index=returns.index)

        for t in range(self.lookback + self.ar_delta_lookback, len(returns)):
            d = delta_ar.iloc[t]
            if np.isnan(d):
                continue

            if d <= self.threshold_delta:
                scaling.iloc[t] = 1.0
            else:
                # Scale down proportional to how far delta exceeds threshold
                # Cap at 3x threshold for full reduction
                progress = min(1.0, (d - self.threshold_delta) / (2 * self.threshold_delta))
                scaling.iloc[t] = 1.0 - progress * (1.0 - self.floor)

        n_scaled = (scaling < 1.0).sum()
        logger.info(
            "Absorption ratio: %d/%d days with reduced exposure (%.1f%%)",
            n_scaled, len(scaling), 100 * n_scaled / max(1, len(scaling)),
        )

        return scaling


# ---------------------------------------------------------------------------
# 3. Regime-Aware Adaptive Stops (Kaminski & Lo, 2014)
# ---------------------------------------------------------------------------

class AdaptiveStopLoss:
    """
    Volatility-scaled trailing stops that adapt to the current regime.

    Instead of fixed percentage stops (which trigger too often in high-vol
    and too late in low-vol), stop levels scale with realised volatility:

        stop_level_t = entry_price × (1 - k × sigma_t)

    where k is the stop multiplier (in units of vol) and sigma_t is the
    rolling realised volatility at time t.

    The key insight from Kaminski & Lo (2014): stop-loss rules are most
    effective when calibrated to the volatility regime, and trailing stops
    outperform fixed-level stops by locking in gains during trends.

    Parameters
    ----------
    vol_lookback : int
        Lookback for realised vol (default 21).
    stop_multiplier : float
        Stop distance in units of vol (default 2.0).
    trailing : bool
        If True, use trailing stop (ratchets up with HWM). Default True.
    cooldown_days : int
        Minimum days between stop-out and re-entry (default 10).
    """

    def __init__(
        self,
        vol_lookback: int = 21,
        stop_multiplier: float = 2.0,
        trailing: bool = True,
        cooldown_days: int = 10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.stop_multiplier = stop_multiplier
        self.trailing = trailing
        self.cooldown_days = cooldown_days

    def apply_adaptive_stops(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply adaptive volatility-scaled stops to strategy weights.

        When the strategy's cumulative return from its high-water mark
        exceeds k * sigma_t, weights are zeroed until cooldown expires.

        Parameters
        ----------
        weights : pd.DataFrame
            Strategy weights (dates x assets).
        returns : pd.DataFrame
            Asset returns.

        Returns
        -------
        pd.DataFrame
            Weights with adaptive stops applied.
        """
        # Strategy-level returns
        strat_returns = (weights.shift(1) * returns).sum(axis=1)
        cum = (1 + strat_returns).cumprod()

        # Rolling vol of strategy returns
        vol = strat_returns.rolling(
            self.vol_lookback, min_periods=max(5, self.vol_lookback // 2)
        ).std() * np.sqrt(252)
        vol = vol.clip(lower=1e-6)

        adjusted = weights.copy()
        hwm = cum.iloc[0] if len(cum) > 0 else 1.0
        stopped = False
        cooldown_remaining = 0

        for t in range(1, len(cum)):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                adjusted.iloc[t] = 0.0
                continue

            current = cum.iloc[t]

            if self.trailing:
                hwm = max(hwm, current)

            # Drawdown from HWM
            dd_pct = (current - hwm) / hwm if hwm > 0 else 0.0

            # Adaptive stop level: k * current annualised vol
            # Convert to same scale as dd_pct (daily → fraction)
            daily_vol = vol.iloc[t] / np.sqrt(252) if not np.isnan(vol.iloc[t]) else 0.02
            stop_distance = self.stop_multiplier * daily_vol * np.sqrt(self.vol_lookback)

            if dd_pct < -stop_distance:
                # Stop triggered
                adjusted.iloc[t] = 0.0
                stopped = True
                cooldown_remaining = self.cooldown_days
                hwm = current  # reset HWM on re-entry
            else:
                stopped = False

        n_stopped = (adjusted.abs().sum(axis=1) == 0).sum()
        total = len(adjusted)
        logger.info(
            "Adaptive stops: %d/%d days stopped out (%.1f%%), "
            "vol_lookback=%d, multiplier=%.1f",
            n_stopped, total, 100 * n_stopped / max(1, total),
            self.vol_lookback, self.stop_multiplier,
        )

        return adjusted


# ---------------------------------------------------------------------------
# 4. Downside Risk Parity (Sortino & van der Meer, 1991)
# ---------------------------------------------------------------------------

class DownsideRiskParity:
    """
    Allocation that equalises downside risk contribution across strategies.

    Unlike standard risk parity (which uses total variance), this uses
    semi-variance (variance of negative returns only) as the risk measure.
    This allocates more to strategies with favourable skew and penalises
    those with fat left tails.

    The optimisation minimises:
        sum_i [ (w_i * MRC_i^{down} / sigma_p^{down} - 1/n)^2 ]

    where MRC_i^{down} is the marginal contribution to downside risk.

    Parameters
    ----------
    min_weight : float
        Minimum per-strategy weight (default 0.02).
    max_weight : float
        Maximum per-strategy weight (default 0.50).
    target_return_threshold : float
        Returns below this are "downside" (default 0.0).
    """

    def __init__(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.50,
        target_return_threshold: float = 0.0,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target = target_return_threshold

    def _semi_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute the semi-covariance matrix (co-movements on negative days).

        Only observations where at least one return is below the target
        are included, preserving co-downside dependency structure.
        """
        below = returns - self.target
        below = np.where(below < 0, below, 0.0)
        n_obs = returns.shape[0]
        if n_obs < 2:
            return np.eye(returns.shape[1])
        semi_cov = (below.T @ below) / (n_obs - 1)
        # Regularise
        semi_cov += np.eye(semi_cov.shape[0]) * 1e-8
        return semi_cov

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute downside-risk-parity weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Strategy returns (dates x strategies).

        Returns
        -------
        pd.Series
            Strategy weights summing to 1.0.
        """
        ret_arr = returns.values
        n = returns.shape[1]
        semi_cov = self._semi_covariance(ret_arr)

        def objective(w):
            port_semi_var = w @ semi_cov @ w
            if port_semi_var <= 0:
                return 0.0
            port_semi_vol = np.sqrt(port_semi_var)
            # Marginal contribution to downside risk
            mcr = (semi_cov @ w) / port_semi_vol
            # Risk contribution
            rc = w * mcr
            total_rc = rc.sum()
            if total_rc <= 0:
                return 0.0
            # Target: equal risk contribution
            rc_pct = rc / total_rc
            target_pct = np.ones(n) / n
            return float(np.sum((rc_pct - target_pct) ** 2))

        # Initial guess: inverse semi-vol
        semi_vols = np.sqrt(np.diag(semi_cov))
        w0 = (1.0 / semi_vols)
        w0 = w0 / w0.sum()

        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if result.success:
            w = result.x
        else:
            logger.warning(
                "Downside risk parity optimisation did not converge: %s. "
                "Using inverse semi-vol fallback.",
                result.message,
            )
            w = w0

        # Ensure constraints
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()

        weights = pd.Series(w, index=returns.columns)

        logger.info(
            "Downside risk parity: %s",
            {col: f"{v:.3f}" for col, v in weights.items()},
        )

        return weights


# ---------------------------------------------------------------------------
# 5. CVaR-Optimized Allocation (Rockafellar & Uryasev, 2000)
# ---------------------------------------------------------------------------

class CVaROptimizer:
    """
    Minimise portfolio CVaR (Expected Shortfall) via linear programming
    reformulation.

    The key insight: CVaR is a coherent risk measure (unlike VaR) and can
    be optimised efficiently.  The Rockafellar-Uryasev formulation converts
    CVaR minimisation into a linear program:

        min_{w, alpha}  alpha + (1/(T*(1-beta))) * sum_t max(0, -r_t'w - alpha)

    subject to:
        w'mu >= target_return,  sum(w) = 1,  w >= 0

    Parameters
    ----------
    alpha : float
        CVaR confidence level (default 0.95 → minimise mean of worst 5%).
    min_weight : float
        Minimum strategy weight (default 0.02).
    max_weight : float
        Maximum strategy weight (default 0.50).
    target_return_quantile : float
        Target minimum return as quantile of strategy means (default 0.25).
    """

    def __init__(
        self,
        alpha: float = 0.95,
        min_weight: float = 0.02,
        max_weight: float = 0.50,
        target_return_quantile: float = 0.25,
    ) -> None:
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return_quantile = target_return_quantile

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute CVaR-minimising strategy weights.

        Uses scipy SLSQP as a fallback since we avoid requiring CVXPY.

        Parameters
        ----------
        returns : pd.DataFrame
            Strategy returns (dates x strategies).

        Returns
        -------
        pd.Series
            Strategy weights summing to 1.0.
        """
        ret_arr = returns.values  # (T, n)
        T, n = ret_arr.shape

        # Target return: 25th percentile of individual strategy means
        strat_means = ret_arr.mean(axis=0)
        target_ret = np.quantile(strat_means, self.target_return_quantile)

        def cvar_objective(params):
            """CVaR via Rockafellar-Uryasev auxiliary variable formulation."""
            w = params[:n]
            zeta = params[n]  # VaR auxiliary

            portfolio_returns = ret_arr @ w
            losses = -portfolio_returns - zeta
            excess_losses = np.maximum(losses, 0.0)

            cvar = zeta + excess_losses.mean() / (1 - self.alpha)
            return cvar

        # Constraints
        def weight_sum(params):
            return params[:n].sum() - 1.0

        def min_return(params):
            w = params[:n]
            return (ret_arr @ w).mean() - target_ret

        constraints = [
            {"type": "eq", "fun": weight_sum},
            {"type": "ineq", "fun": min_return},
        ]

        # Bounds: weights in [min, max], zeta unbounded
        bounds = [(self.min_weight, self.max_weight)] * n + [(-1.0, 1.0)]

        # Initial guess: equal weight + VaR estimate
        w0 = np.ones(n) / n
        port_ret_0 = ret_arr @ w0
        zeta0 = -np.quantile(port_ret_0, 1 - self.alpha)
        x0 = np.append(w0, zeta0)

        result = minimize(
            cvar_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            w = result.x[:n]
        else:
            logger.warning(
                "CVaR optimisation did not converge: %s. Using equal weight.",
                result.message,
            )
            w = np.ones(n) / n

        # Ensure constraints
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()

        weights = pd.Series(w, index=returns.columns)

        # Log the achieved CVaR
        port_ret = ret_arr @ w
        achieved_cvar = -np.mean(np.sort(port_ret)[: int(T * (1 - self.alpha))])

        logger.info(
            "CVaR-optimized allocation (alpha=%.2f): CVaR=%.4f, weights=%s",
            self.alpha,
            achieved_cvar,
            {col: f"{v:.3f}" for col, v in weights.items()},
        )

        return weights


# ---------------------------------------------------------------------------
# 6. Maximum Diversification Overlay (Choueifaty & Coignard, 2008)
# ---------------------------------------------------------------------------

class MaxDiversification:
    """
    Maximise the diversification ratio:

        DR(w) = w' sigma / sqrt(w' Sigma w)

    where sigma is the vector of individual volatilities and Sigma is the
    covariance matrix.  DR >=1 always, with equality iff all pairwise
    correlations are 1 (no diversification).

    Maximising DR is equivalent to minimising portfolio variance per unit
    of weighted-average vol, extracting maximum diversification benefit.

    Parameters
    ----------
    min_weight : float
        Minimum strategy weight (default 0.02).
    max_weight : float
        Maximum strategy weight (default 0.50).
    use_shrinkage : bool
        Apply Ledoit-Wolf shrinkage to covariance (default True).
    """

    def __init__(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.50,
        use_shrinkage: bool = True,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_shrinkage = use_shrinkage

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute maximum-diversification weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Strategy returns (dates x strategies).

        Returns
        -------
        pd.Series
            Strategy weights summing to 1.0.
        """
        n = returns.shape[1]
        cov = returns.cov().values

        if self.use_shrinkage:
            cov = self._shrink_cov(returns.values, cov)

        # Regularise
        cov += np.eye(n) * 1e-8
        vols = np.sqrt(np.diag(cov))

        def neg_diversification_ratio(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            weighted_vol = w @ vols
            return -weighted_vol / port_vol  # negative because we minimise

        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        # Start from inverse-vol
        w0 = 1.0 / vols
        w0 = w0 / w0.sum()

        result = minimize(
            neg_diversification_ratio,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if result.success:
            w = result.x
        else:
            logger.warning(
                "Max diversification optimisation did not converge: %s",
                result.message,
            )
            w = w0

        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()

        weights = pd.Series(w, index=returns.columns)

        # Compute achieved DR
        port_vol = np.sqrt(w @ cov @ w)
        dr = (w @ vols) / port_vol if port_vol > 0 else 1.0

        logger.info(
            "Max diversification: DR=%.2f, weights=%s",
            dr, {col: f"{v:.3f}" for col, v in weights.items()},
        )

        return weights

    @staticmethod
    def _shrink_cov(returns: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
        """Simple Ledoit-Wolf shrinkage toward diagonal."""
        n = sample_cov.shape[0]
        target = np.diag(np.diag(sample_cov))
        T = returns.shape[0]

        # Estimate optimal shrinkage intensity
        delta = sample_cov - target
        delta_sq_sum = np.sum(delta ** 2)
        sample_var_sum = np.sum(np.diag(sample_cov) ** 2)

        if delta_sq_sum > 0:
            intensity = min(1.0, max(0.0, sample_var_sum / (T * delta_sq_sum)))
        else:
            intensity = 0.0

        return (1 - intensity) * sample_cov + intensity * target


# ---------------------------------------------------------------------------
# Composite Risk Overlay — combines multiple risk signals
# ---------------------------------------------------------------------------

class CompositeRiskOverlay:
    """
    Combines Turbulence Index and Absorption Ratio into a single scaling
    factor, taking the *minimum* (most conservative) of both signals.

    This provides defence against two distinct failure modes:
    1. Turbulence: unusual return magnitudes/patterns (sudden events)
    2. Absorption: market fragility from correlated factor exposure (slow build)

    Parameters
    ----------
    turbulence_config : dict, optional
        kwargs for TurbulenceIndex.
    absorption_config : dict, optional
        kwargs for AbsorptionRatio.
    combination : str
        How to combine: "min" (most conservative), "mean", or "product".
    """

    def __init__(
        self,
        turbulence_config: Optional[dict] = None,
        absorption_config: Optional[dict] = None,
        combination: str = "min",
    ) -> None:
        self.turbulence = TurbulenceIndex(**(turbulence_config or {}))
        self.absorption = AbsorptionRatio(**(absorption_config or {}))
        self.combination = combination

    def compute_scaling(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute composite risk scaling factor.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns (dates x assets/strategies).

        Returns
        -------
        pd.Series
            Scaling factors in [floor, 1.0].
        """
        turb_scale = self.turbulence.compute_scaling(returns)
        abs_scale = self.absorption.compute_scaling(returns)

        if self.combination == "min":
            combined = pd.concat([turb_scale, abs_scale], axis=1).min(axis=1)
        elif self.combination == "mean":
            combined = (turb_scale + abs_scale) / 2
        elif self.combination == "product":
            combined = turb_scale * abs_scale
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")

        combined = combined.clip(lower=min(self.turbulence.floor, self.absorption.floor))

        n_scaled = (combined < 1.0).sum()
        logger.info(
            "Composite risk overlay (%s): %d/%d days with reduced exposure (%.1f%%)",
            self.combination, n_scaled, len(combined),
            100 * n_scaled / max(1, len(combined)),
        )

        return combined

    def apply_to_weights(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply composite risk scaling to portfolio weights.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates x assets).
        returns : pd.DataFrame
            Returns used for risk estimation.

        Returns
        -------
        pd.DataFrame
            Scaled weights.
        """
        scaling = self.compute_scaling(returns)
        return weights.mul(scaling, axis=0)
