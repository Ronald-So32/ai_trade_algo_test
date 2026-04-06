"""
signal_evaluator.py
-------------------
Evaluate each candidate alpha signal's quality through a comprehensive set
of performance and robustness metrics.

Metrics computed
~~~~~~~~~~~~~~~~
- Sharpe ratio          (annualised, from equal-weighted long-short portfolio)
- Sharpe stability      (rolling Sharpe standard deviation — lower is better)
- Information coefficient (IC) — Spearman rank correlation with forward returns
- IC information ratio  (mean IC / std IC, annualised)
- Max drawdown          (of the long-short cumulative P&L)
- Calmar ratio          (annualised return / max drawdown)
- Cost sensitivity      (net Sharpe at various cost levels in bps)
- Regime performance    (per-regime Sharpe, regime robustness score)
- Hit rate              (fraction of days the signal is directionally correct)
- Turnover              (average daily portfolio weight change)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_ROLLING_SHARPE_WINDOW: int = 63  # ~1 quarter
_COST_LEVELS_BPS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# SignalMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalMetrics:
    """
    Container for all quality metrics of a single alpha signal.

    Attributes
    ----------
    signal_name : str
        Identifier of the signal.
    sharpe : float
        Annualised Sharpe ratio of the long-short portfolio.
    sharpe_stability : float
        Standard deviation of rolling Sharpe ratios (lower = more stable).
    ic_mean : float
        Mean daily information coefficient (Spearman rank correlation with
        next-day forward returns).
    ic_std : float
        Standard deviation of daily IC.
    icir : float
        IC information ratio: ``ic_mean / ic_std * sqrt(252)``.
    max_drawdown : float
        Maximum peak-to-trough drawdown of the signal portfolio (as a
        positive fraction, e.g. 0.15 means 15% drawdown).
    calmar : float
        ``annualised_return / max_drawdown``.
    hit_rate : float
        Fraction of days the long-short portfolio generates a positive return.
    turnover : float
        Average daily one-way portfolio weight change (0 to 1 scale).
    cost_sensitivity : dict[float, float]
        ``{cost_bps: net_sharpe}`` mapping showing performance after costs.
    regime_sharpes : dict[str, float]
        Per-regime Sharpe ratios (keyed by regime label).  Empty if no
        regime labels were supplied.
    regime_robustness : float
        ``worst_regime_sharpe / best_regime_sharpe``.  Ranges from -inf to 1.
        Values close to 1 indicate consistent performance across regimes.
        NaN when regime labels are not supplied.
    annualised_return : float
        Annualised mean daily P&L of the long-short portfolio.
    annualised_vol : float
        Annualised volatility of daily P&L.
    skewness : float
        Skewness of daily P&L distribution.
    kurtosis : float
        Excess kurtosis of daily P&L distribution.
    """

    signal_name: str

    # Core performance
    sharpe: float = np.nan
    sharpe_stability: float = np.nan
    annualised_return: float = np.nan
    annualised_vol: float = np.nan

    # IC metrics
    ic_mean: float = np.nan
    ic_std: float = np.nan
    icir: float = np.nan

    # Drawdown / tail
    max_drawdown: float = np.nan
    calmar: float = np.nan
    hit_rate: float = np.nan
    skewness: float = np.nan
    kurtosis: float = np.nan

    # Capacity / trading
    turnover: float = np.nan

    # Cost-adjusted performance
    cost_sensitivity: dict[float, float] = field(default_factory=dict)

    # Regime analysis
    regime_sharpes: dict[str, float] = field(default_factory=dict)
    regime_robustness: float = np.nan

    # ------------------------------------------------------------------
    def to_series(self) -> pd.Series:
        """Flatten all scalar metrics (excluding dicts) into a Series."""
        scalars = {
            k: v
            for k, v in self.__dict__.items()
            if not isinstance(v, dict)
        }
        cost_flat = {
            f"net_sharpe_{c}bps": s
            for c, s in self.cost_sensitivity.items()
        }
        regime_flat = {
            f"regime_sharpe_{r}": s
            for r, s in self.regime_sharpes.items()
        }
        return pd.Series({**scalars, **cost_flat, **regime_flat})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_long_short_pnl(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Construct a daily long-short P&L series.

    At each date, go long (equal-weight) securities in the top quintile
    of the signal and short (equal-weight) those in the bottom quintile.
    Returns are sourced from ``forward_returns`` (already shifted).

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectionally z-scored signal values (dates x securities).
    forward_returns : pd.DataFrame
        Pre-aligned next-period returns (dates x securities).
    n_quantiles : int
        Number of quantiles used.  Top and bottom quantile form the
        long-short legs.

    Returns
    -------
    pd.Series
        Daily portfolio P&L.
    """
    common_idx = signal.index.intersection(forward_returns.index)
    sig = signal.loc[common_idx]
    fwd = forward_returns.loc[common_idx]

    daily_pnl = pd.Series(np.nan, index=common_idx)

    for date in common_idx:
        sig_row = sig.loc[date].dropna()
        fwd_row = fwd.loc[date].reindex(sig_row.index).dropna()
        sig_row = sig_row.reindex(fwd_row.index)

        n = len(sig_row)
        if n < n_quantiles * 2:
            continue

        cutoff = max(1, n // n_quantiles)
        sorted_idx = sig_row.sort_values().index

        short_idx = sorted_idx[:cutoff]
        long_idx = sorted_idx[-cutoff:]

        long_ret = fwd_row.loc[long_idx].mean()
        short_ret = fwd_row.loc[short_idx].mean()
        daily_pnl.loc[date] = long_ret - short_ret

    return daily_pnl.dropna()


def _max_drawdown(pnl: pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown of cumulative P&L."""
    if pnl.empty:
        return np.nan
    cumulative = (1.0 + pnl).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    return float(-drawdown.min())


def _rolling_sharpe(pnl: pd.Series, window: int) -> pd.Series:
    """Compute rolling annualised Sharpe over ``window`` trading days."""
    mu = pnl.rolling(window=window, min_periods=window // 2).mean()
    sigma = pnl.rolling(window=window, min_periods=window // 2).std()
    return (mu / sigma.replace(0, np.nan)) * np.sqrt(_TRADING_DAYS_PER_YEAR)


def _information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily cross-sectional Spearman IC between signal and forward returns.

    Returns
    -------
    pd.Series
        Daily IC values.
    """
    common_idx = signal.index.intersection(forward_returns.index)
    ic_values: list[float] = []
    idx_list: list[object] = []

    for date in common_idx:
        s_row = signal.loc[date].dropna()
        f_row = forward_returns.loc[date].reindex(s_row.index).dropna()
        s_row = s_row.reindex(f_row.index)

        if len(s_row) < 5:
            continue

        rho, _ = stats.spearmanr(s_row.values, f_row.values)
        ic_values.append(rho)
        idx_list.append(date)

    return pd.Series(ic_values, index=idx_list)


def _turnover(signal: pd.DataFrame) -> float:
    """
    Estimate average daily one-way turnover of a rank-weighted portfolio.

    Positions are derived by normalising the signal cross-sectionally to
    sum to zero (long-short dollar-neutral).  Turnover is the average
    absolute daily change in position weights.
    """
    # Normalise to unit L1 norm per row
    row_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    positions = signal.divide(row_sum, axis=0)
    position_change = positions.diff().abs().sum(axis=1)
    return float(position_change.mean())


# ---------------------------------------------------------------------------
# SignalEvaluator
# ---------------------------------------------------------------------------

class SignalEvaluator:
    """
    Evaluate candidate alpha signals across multiple quality dimensions.

    Parameters
    ----------
    n_quantiles : int
        Number of quantile buckets used when constructing the long-short
        portfolio.  Default 5 (quintiles).
    rolling_sharpe_window : int
        Look-back for rolling Sharpe stability calculation.
        Default 63 trading days (~1 quarter).
    cost_levels_bps : tuple[float, ...]
        Transaction cost levels (in basis points per side) at which to
        compute net-of-cost Sharpe ratios.
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        rolling_sharpe_window: int = _ROLLING_SHARPE_WINDOW,
        cost_levels_bps: tuple[float, ...] = _COST_LEVELS_BPS,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.rolling_sharpe_window = rolling_sharpe_window
        self.cost_levels_bps = cost_levels_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signal(
        self,
        signal: pd.DataFrame,
        forward_returns: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        signal_name: str = "unnamed",
    ) -> SignalMetrics:
        """
        Compute quality metrics for a single signal.

        Parameters
        ----------
        signal : pd.DataFrame
            Signal values, shape (dates, securities).
        forward_returns : pd.DataFrame
            Pre-shifted next-period returns, shape (dates, securities).
            Must be aligned to signal (i.e. ``forward_returns.loc[t]``
            contains the return *earned* when entering a position at the
            close of day ``t``).
        regime_labels : pd.Series, optional
            Categorical regime label for each date (e.g. "bull", "bear",
            "neutral").  Index must match ``signal.index``.
        signal_name : str
            Human-readable name stored in :class:`SignalMetrics`.

        Returns
        -------
        SignalMetrics
        """
        metrics = SignalMetrics(signal_name=signal_name)

        try:
            pnl = _build_long_short_pnl(signal, forward_returns, self.n_quantiles)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build P&L for '%s': %s", signal_name, exc)
            return metrics

        if pnl.empty or pnl.std() < 1e-12:
            logger.debug("Signal '%s' produced trivial P&L; skipping.", signal_name)
            return metrics

        # Core return / vol metrics
        ann_ret = float(pnl.mean() * _TRADING_DAYS_PER_YEAR)
        ann_vol = float(pnl.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
        metrics.annualised_return = ann_ret
        metrics.annualised_vol = ann_vol
        metrics.sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        # Sharpe stability
        rolling_sh = _rolling_sharpe(pnl, self.rolling_sharpe_window).dropna()
        metrics.sharpe_stability = float(rolling_sh.std()) if len(rolling_sh) > 1 else np.nan

        # IC metrics
        ic_series = _information_coefficient(signal, forward_returns)
        if not ic_series.empty:
            metrics.ic_mean = float(ic_series.mean())
            metrics.ic_std = float(ic_series.std())
            metrics.icir = (
                metrics.ic_mean / metrics.ic_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
                if metrics.ic_std > 0
                else np.nan
            )

        # Drawdown / tail metrics
        metrics.max_drawdown = _max_drawdown(pnl)
        metrics.calmar = (
            ann_ret / metrics.max_drawdown
            if metrics.max_drawdown and metrics.max_drawdown > 0
            else np.nan
        )
        metrics.hit_rate = float((pnl > 0).mean())
        metrics.skewness = float(stats.skew(pnl.values))
        metrics.kurtosis = float(stats.kurtosis(pnl.values))

        # Turnover
        try:
            metrics.turnover = _turnover(signal)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Turnover calc failed for '%s': %s", signal_name, exc)

        # Cost sensitivity: net Sharpe at different cost levels
        metrics.cost_sensitivity = self._cost_sensitivity(
            pnl, metrics.turnover, ann_vol
        )

        # Regime performance
        if regime_labels is not None:
            metrics.regime_sharpes, metrics.regime_robustness = self._regime_performance(
                pnl, regime_labels
            )

        return metrics

    def evaluate_all(
        self,
        signals_dict: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Evaluate every signal in ``signals_dict`` and return a tidy metrics table.

        Parameters
        ----------
        signals_dict : dict[str, pd.DataFrame]
            Mapping from signal name to signal DataFrame.
        forward_returns : pd.DataFrame
            Pre-shifted forward returns aligned to the signal dates.
        regime_labels : pd.Series, optional
            Regime labels indexed by date.

        Returns
        -------
        pd.DataFrame
            One row per signal; columns are all scalar metrics plus per-cost
            and per-regime Sharpe values.  Index is ``signal_name``.
        """
        rows: list[pd.Series] = []
        total = len(signals_dict)

        for i, (name, sig) in enumerate(signals_dict.items(), start=1):
            logger.info("Evaluating signal %d/%d: %s", i, total, name)
            try:
                m = self.evaluate_signal(
                    sig, forward_returns, regime_labels, signal_name=name
                )
                rows.append(m.to_series())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Evaluation failed for '%s': %s", name, exc)

        if not rows:
            logger.warning("No signals evaluated successfully.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("signal_name")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_sensitivity(
        self,
        pnl: pd.Series,
        turnover: float,
        ann_vol: float,
    ) -> dict[float, float]:
        """
        Compute net-of-cost Sharpe at each cost level.

        Cost is modelled as ``cost_bps * turnover`` per day subtracted
        from the daily P&L.
        """
        result: dict[float, float] = {}
        if np.isnan(turnover):
            turnover = 0.0

        for cost_bps in self.cost_levels_bps:
            daily_cost = (cost_bps / 10_000) * turnover
            net_pnl = pnl - daily_cost
            net_ann_ret = float(net_pnl.mean() * _TRADING_DAYS_PER_YEAR)
            result[cost_bps] = net_ann_ret / ann_vol if ann_vol > 0 else np.nan

        return result

    @staticmethod
    def _regime_performance(
        pnl: pd.Series,
        regime_labels: pd.Series,
    ) -> tuple[dict[str, float], float]:
        """
        Compute per-regime Sharpe ratios and regime robustness score.

        Parameters
        ----------
        pnl : pd.Series
            Daily long-short P&L.
        regime_labels : pd.Series
            Regime label for each date; must share index with ``pnl``.

        Returns
        -------
        regime_sharpes : dict[str, float]
        regime_robustness : float
            ``min_regime_sharpe / max_regime_sharpe`` (ratio closest to 1 is best).
        """
        common_idx = pnl.index.intersection(regime_labels.index)
        if common_idx.empty:
            return {}, np.nan

        aligned_pnl = pnl.loc[common_idx]
        aligned_labels = regime_labels.loc[common_idx]
        regimes = aligned_labels.unique()

        regime_sharpes: dict[str, float] = {}
        for regime in regimes:
            mask = aligned_labels == regime
            sub_pnl = aligned_pnl[mask]
            if len(sub_pnl) < 20 or sub_pnl.std() < 1e-12:
                continue
            sr = (
                sub_pnl.mean() / sub_pnl.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
            )
            regime_sharpes[str(regime)] = float(sr)

        if len(regime_sharpes) < 2:
            robustness = np.nan
        else:
            sharpe_values = list(regime_sharpes.values())
            max_sr = max(sharpe_values)
            min_sr = min(sharpe_values)
            # Robustness = worst / best; sensible only when best > 0
            if abs(max_sr) < 1e-6:
                robustness = np.nan
            else:
                robustness = min_sr / max_sr

        return regime_sharpes, robustness
