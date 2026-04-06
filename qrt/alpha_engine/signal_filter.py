"""
signal_filter.py
----------------
Filter candidate alpha signals by a set of configurable robustness criteria.

Filtering criteria
~~~~~~~~~~~~~~~~~~
- Minimum Sharpe ratio threshold (default 0.5)
- Maximum drawdown limit (default 20 %)
- Minimum regime robustness score (default 0.3) — ratio of worst-regime
  Sharpe to best-regime Sharpe; skipped when regime data is unavailable
- Maximum correlation with existing strategy returns (default 0.7)
- Minimum IC mean threshold (default 0.0 — any positive IC passes)
- Minimum IC information ratio threshold (default 0.0)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SignalFilter
# ---------------------------------------------------------------------------

class SignalFilter:
    """
    Apply a battery of robustness filters to a metrics table produced by
    :class:`~signal_evaluator.SignalEvaluator`.

    Parameters
    ----------
    min_sharpe : float
        Minimum annualised Sharpe ratio of the long-short portfolio.
        Default 0.5.
    max_drawdown : float
        Maximum allowable drawdown (expressed as a *positive* fraction,
        e.g. 0.20 = 20 %).  Default 0.20.
    min_regime_robustness : float
        Minimum regime robustness score (worst / best regime Sharpe ratio).
        Default 0.3.  Signals without regime labels are passed through
        (criterion not applied).
    max_correlation : float
        Maximum allowable Spearman rank correlation between the candidate
        signal's long-short returns and any of the existing strategy
        returns.  Default 0.7.
    min_ic_mean : float
        Minimum mean daily information coefficient.  Default 0.0 (any
        positive IC passes).
    min_icir : float
        Minimum IC information ratio.  Default 0.0.
    require_positive_ic : bool
        If True, signals with ``ic_mean <= 0`` are rejected outright,
        regardless of ``min_ic_mean``.  Default True.
    """

    def __init__(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.20,
        min_regime_robustness: float = 0.3,
        max_correlation: float = 0.7,
        min_ic_mean: float = 0.0,
        min_icir: float = 0.0,
        require_positive_ic: bool = True,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_regime_robustness = min_regime_robustness
        self.max_correlation = max_correlation
        self.min_ic_mean = min_ic_mean
        self.min_icir = min_icir
        self.require_positive_ic = require_positive_ic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signal_metrics: pd.DataFrame,
        existing_strategy_returns: Optional[pd.DataFrame] = None,
        signal_pnl_dict: Optional[dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Apply all filters and return the subset of signals that pass.

        Parameters
        ----------
        signal_metrics : pd.DataFrame
            Metrics table as produced by
            :meth:`~signal_evaluator.SignalEvaluator.evaluate_all`.
            Index must be ``signal_name``; expected columns include
            ``sharpe``, ``max_drawdown``, ``regime_robustness``,
            ``ic_mean``, ``icir``.
        existing_strategy_returns : pd.DataFrame, optional
            Daily returns of existing strategies, shape (dates, strategies).
            Used together with ``signal_pnl_dict`` to run the correlation
            check.  Ignored when ``signal_pnl_dict`` is None.
        signal_pnl_dict : dict[str, pd.Series], optional
            Mapping from signal name to its long-short daily P&L Series.
            Required to perform the correlation filter; if None the
            correlation filter is skipped.

        Returns
        -------
        pd.DataFrame
            Subset of ``signal_metrics`` whose signals passed all active
            filters.  Adds boolean columns ``passed_<criterion>`` for
            transparency.
        """
        if signal_metrics.empty:
            logger.warning("signal_metrics is empty; nothing to filter.")
            return signal_metrics.copy()

        df = signal_metrics.copy()

        # ---- Sharpe filter --------------------------------------------------
        df["passed_sharpe"] = self._apply_sharpe_filter(df)

        # ---- Drawdown filter ------------------------------------------------
        df["passed_drawdown"] = self._apply_drawdown_filter(df)

        # ---- Regime robustness filter ----------------------------------------
        df["passed_regime"] = self._apply_regime_filter(df)

        # ---- IC / ICIR filters -----------------------------------------------
        df["passed_ic"] = self._apply_ic_filter(df)

        # ---- Correlation filter (requires external data) --------------------
        if (
            signal_pnl_dict is not None
            and existing_strategy_returns is not None
            and not existing_strategy_returns.empty
        ):
            df["passed_correlation"] = self._apply_correlation_filter(
                df, signal_pnl_dict, existing_strategy_returns
            )
        else:
            df["passed_correlation"] = True

        # ---- Combine --------------------------------------------------------
        passed_cols = [c for c in df.columns if c.startswith("passed_")]
        df["passed_all"] = df[passed_cols].all(axis=1)

        passing = df[df["passed_all"]].copy()

        n_total = len(df)
        n_pass = len(passing)
        logger.info(
            "Filter results: %d / %d signals passed all criteria.", n_pass, n_total
        )
        self._log_filter_breakdown(df, passed_cols)

        return passing

    def correlation_check(
        self,
        candidate_returns: pd.Series,
        existing_returns: pd.DataFrame,
        max_corr: float = 0.7,
    ) -> bool:
        """
        Return True if the candidate signal's returns are sufficiently
        uncorrelated with *all* existing strategy returns.

        Parameters
        ----------
        candidate_returns : pd.Series
            Daily long-short P&L of the candidate signal.
        existing_returns : pd.DataFrame
            Daily returns of existing strategies (dates x strategies).
        max_corr : float
            Maximum absolute Spearman correlation allowed.  Default 0.7.

        Returns
        -------
        bool
            ``True`` if the candidate passes (low correlation with all
            existing strategies), ``False`` otherwise.
        """
        if existing_returns is None or existing_returns.empty:
            return True

        common_idx = candidate_returns.index.intersection(existing_returns.index)
        if len(common_idx) < 20:
            logger.debug(
                "Correlation check skipped: insufficient overlapping dates (%d).",
                len(common_idx),
            )
            return True

        cand_aligned = candidate_returns.loc[common_idx]
        exist_aligned = existing_returns.loc[common_idx]

        for col in exist_aligned.columns:
            strat_col = exist_aligned[col].dropna()
            common_sub = cand_aligned.index.intersection(strat_col.index)
            if len(common_sub) < 20:
                continue
            rho, _ = self._spearman_corr(
                cand_aligned.loc[common_sub].values,
                strat_col.loc[common_sub].values,
            )
            if abs(rho) > max_corr:
                logger.debug(
                    "Candidate fails correlation check vs strategy '%s': rho=%.3f > %.3f",
                    col,
                    rho,
                    max_corr,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _apply_sharpe_filter(self, df: pd.DataFrame) -> pd.Series:
        if "sharpe" not in df.columns:
            logger.debug("'sharpe' column missing; Sharpe filter skipped.")
            return pd.Series(True, index=df.index)
        return df["sharpe"].fillna(-np.inf) >= self.min_sharpe

    def _apply_drawdown_filter(self, df: pd.DataFrame) -> pd.Series:
        if "max_drawdown" not in df.columns:
            logger.debug("'max_drawdown' column missing; drawdown filter skipped.")
            return pd.Series(True, index=df.index)
        # max_drawdown stored as positive fraction (e.g. 0.15 = 15%)
        return df["max_drawdown"].fillna(np.inf) <= self.max_drawdown

    def _apply_regime_filter(self, df: pd.DataFrame) -> pd.Series:
        if "regime_robustness" not in df.columns:
            logger.debug("'regime_robustness' column missing; regime filter skipped.")
            return pd.Series(True, index=df.index)

        regime_col = df["regime_robustness"]
        # NaN means regime labels were not provided; pass through
        has_data = regime_col.notna()
        passes = pd.Series(True, index=df.index)
        passes[has_data] = regime_col[has_data] >= self.min_regime_robustness
        return passes

    def _apply_ic_filter(self, df: pd.DataFrame) -> pd.Series:
        passes = pd.Series(True, index=df.index)

        if "ic_mean" in df.columns:
            if self.require_positive_ic:
                passes &= df["ic_mean"].fillna(-np.inf) > 0
            passes &= df["ic_mean"].fillna(-np.inf) >= self.min_ic_mean

        if "icir" in df.columns and self.min_icir > 0:
            passes &= df["icir"].fillna(-np.inf) >= self.min_icir

        return passes

    def _apply_correlation_filter(
        self,
        df: pd.DataFrame,
        signal_pnl_dict: dict[str, pd.Series],
        existing_strategy_returns: pd.DataFrame,
    ) -> pd.Series:
        passes = pd.Series(True, index=df.index)

        for signal_name in df.index:
            if signal_name not in signal_pnl_dict:
                continue
            candidate_pnl = signal_pnl_dict[signal_name]
            passes.loc[signal_name] = self.correlation_check(
                candidate_pnl,
                existing_strategy_returns,
                max_corr=self.max_correlation,
            )

        return passes

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_filter_breakdown(df: pd.DataFrame, passed_cols: list[str]) -> None:
        n_total = len(df)
        for col in passed_cols:
            n_fail = int((~df[col]).sum())
            if n_fail:
                criterion = col.replace("passed_", "")
                logger.info("  [%s] rejected %d / %d signals.", criterion, n_fail, n_total)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman_corr(
        x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Spearman rank correlation between two 1-D arrays."""
        from scipy import stats  # local import to avoid top-level coupling

        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0, 1.0
        rho, pval = stats.spearmanr(x, y)
        return float(rho), float(pval)

    # ------------------------------------------------------------------
    # Convenience: build a ranked summary of passing signals
    # ------------------------------------------------------------------

    @staticmethod
    def rank_passing_signals(
        passing_metrics: pd.DataFrame,
        sort_by: str = "sharpe",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Sort and rank passing signals by a chosen metric.

        Parameters
        ----------
        passing_metrics : pd.DataFrame
            DataFrame of passing signals (output of :meth:`filter_signals`).
        sort_by : str
            Column to sort by.  Default ``"sharpe"``.
        ascending : bool
            Sort direction.  Default False (highest Sharpe first).

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame with an additional ``rank`` column (1 = best).
        """
        if passing_metrics.empty:
            return passing_metrics.copy()

        if sort_by not in passing_metrics.columns:
            logger.warning(
                "Sort column '%s' not found; defaulting to index order.", sort_by
            )
            ranked = passing_metrics.copy()
        else:
            ranked = passing_metrics.sort_values(sort_by, ascending=ascending)

        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        return ranked
