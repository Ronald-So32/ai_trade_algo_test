"""
alpha_research.py
-----------------
AlphaResearchEngine — end-to-end orchestrator for automated alpha discovery.

Workflow
~~~~~~~~
1.  **Generate** candidate signals via :class:`~signal_generator.SignalGenerator`.
2.  **Evaluate** every signal via :class:`~signal_evaluator.SignalEvaluator`.
3.  **Filter** signals via :class:`~signal_filter.SignalFilter`.
4.  **Compute** a signal correlation matrix for the surviving signals.
5.  Return a structured :class:`AlphaResearchResult`.

Usage
-----
    engine = AlphaResearchEngine()
    result = engine.run_discovery(
        prices=prices_df,
        returns=returns_df,
        volumes=volumes_df,
        existing_strategy_returns=existing_rets,
        regime_labels=regime_series,
    )
    result.summary()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from qrt.alpha_engine.signal_evaluator import SignalEvaluator, _build_long_short_pnl
from qrt.alpha_engine.signal_filter import SignalFilter
from qrt.alpha_engine.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlphaResearchResult
# ---------------------------------------------------------------------------

@dataclass
class AlphaResearchResult:
    """
    Container for the full output of a single alpha discovery run.

    Attributes
    ----------
    candidate_signal_library : dict[str, pd.DataFrame]
        All generated signals (name -> signal DataFrame, dates x securities).
    signal_performance : pd.DataFrame
        Metrics table with one row per signal and columns for every metric
        computed by :class:`~signal_evaluator.SignalEvaluator`.
    filtered_signals : list[str]
        Names of signals that passed all filter criteria.
    signal_correlation_matrix : pd.DataFrame
        Pairwise Spearman rank correlation of long-short P&L for *all*
        evaluated signals.  Shape: (n_signals, n_signals).
    signal_pnl_dict : dict[str, pd.Series]
        Daily long-short P&L for every evaluated signal.
    run_metadata : dict
        Diagnostic information about the run (timestamps, counts, etc.).
    """

    candidate_signal_library: dict[str, pd.DataFrame] = field(default_factory=dict)
    signal_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    filtered_signals: list[str] = field(default_factory=list)
    signal_correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_pnl_dict: dict[str, pd.Series] = field(default_factory=dict)
    run_metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self, top_n: int = 20) -> None:
        """
        Print a human-readable summary of the discovery run to stdout.

        Parameters
        ----------
        top_n : int
            Number of top signals (by Sharpe) to display in the table.
        """
        meta = self.run_metadata
        n_candidates = len(self.candidate_signal_library)
        n_evaluated = len(self.signal_performance)
        n_passed = len(self.filtered_signals)

        sep = "=" * 72
        print(sep)
        print("  Alpha Discovery Engine — Run Summary")
        print(sep)
        print(f"  Run timestamp   : {meta.get('timestamp', 'N/A')}")
        print(f"  Elapsed (s)     : {meta.get('elapsed_seconds', 'N/A'):.1f}")
        print(f"  Date range      : {meta.get('date_range', 'N/A')}")
        print(f"  Securities      : {meta.get('n_securities', 'N/A')}")
        print(f"  Candidates gen. : {n_candidates}")
        print(f"  Signals eval.   : {n_evaluated}")
        print(f"  Signals passed  : {n_passed}")
        print(sep)

        if self.signal_performance.empty:
            print("  No signal metrics available.")
            return

        # Top signals by Sharpe
        display_cols = [
            c for c in (
                "sharpe", "max_drawdown", "ic_mean", "icir", "regime_robustness",
                "turnover", "hit_rate", "annualised_return", "annualised_vol",
            )
            if c in self.signal_performance.columns
        ]

        perf = self.signal_performance[display_cols].copy()

        if "max_drawdown" in perf.columns:
            perf["max_drawdown"] = (perf["max_drawdown"] * 100).round(2).astype(str) + "%"

        print(f"\n  Top {top_n} signals by Sharpe ratio:")
        print("-" * 72)
        if "sharpe" in perf.columns:
            top = perf.sort_values("sharpe", ascending=False).head(top_n)
        else:
            top = perf.head(top_n)

        # Widen display for readability
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(top.to_string())

        print("\n  Signals that passed all filters:")
        if self.filtered_signals:
            for i, name in enumerate(self.filtered_signals, start=1):
                row = self.signal_performance.loc[name]
                sr = row.get("sharpe", np.nan)
                dd = row.get("max_drawdown", np.nan)
                ic = row.get("ic_mean", np.nan)
                print(f"    {i:>3}. {name:<45}  Sharpe={sr:+.3f}  DD={dd:.1%}  IC={ic:.4f}")
        else:
            print("    (none)")

        print(sep)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_signal(self, name: str) -> pd.DataFrame:
        """Return the signal DataFrame for the given name."""
        if name not in self.candidate_signal_library:
            raise KeyError(f"Signal '{name}' not found in the library.")
        return self.candidate_signal_library[name]

    def get_metrics(self, name: str) -> pd.Series:
        """Return the metrics row for a given signal name."""
        if name not in self.signal_performance.index:
            raise KeyError(f"Signal '{name}' not found in signal_performance.")
        return self.signal_performance.loc[name]

    def top_signals(self, n: int = 10, sort_by: str = "sharpe") -> pd.DataFrame:
        """
        Return top-N signals sorted by ``sort_by`` metric.

        Parameters
        ----------
        n : int
            Number of signals to return.
        sort_by : str
            Column in ``signal_performance`` to sort by.

        Returns
        -------
        pd.DataFrame
        """
        if self.signal_performance.empty:
            return pd.DataFrame()
        if sort_by not in self.signal_performance.columns:
            raise ValueError(f"Column '{sort_by}' not found in signal_performance.")
        return self.signal_performance.sort_values(sort_by, ascending=False).head(n)

    def filtered_performance(self) -> pd.DataFrame:
        """Return ``signal_performance`` subset to signals that passed all filters."""
        if not self.filtered_signals:
            return pd.DataFrame()
        mask = self.signal_performance.index.isin(self.filtered_signals)
        return self.signal_performance[mask]


# ---------------------------------------------------------------------------
# AlphaResearchEngine
# ---------------------------------------------------------------------------

class AlphaResearchEngine:
    """
    End-to-end alpha discovery pipeline.

    Parameters
    ----------
    generator : SignalGenerator, optional
        Custom signal generator.  If None, a default instance is created.
    evaluator : SignalEvaluator, optional
        Custom signal evaluator.  If None, a default instance is created.
    signal_filter : SignalFilter, optional
        Custom signal filter.  If None, a default instance is created.
    forward_return_horizon : int
        Number of periods ahead for the forward return used in IC
        calculation.  Default 1 (next-day return).
    min_history : int
        Minimum number of dates required for the run to proceed.
        Default 126 (~6 months of trading days).
    """

    def __init__(
        self,
        generator: Optional[SignalGenerator] = None,
        evaluator: Optional[SignalEvaluator] = None,
        signal_filter: Optional[SignalFilter] = None,
        forward_return_horizon: int = 1,
        min_history: int = 126,
    ) -> None:
        self.generator = generator or SignalGenerator()
        self.evaluator = evaluator or SignalEvaluator()
        self.signal_filter = signal_filter or SignalFilter()
        self.forward_return_horizon = forward_return_horizon
        self.min_history = min_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_discovery(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
        existing_strategy_returns: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
    ) -> AlphaResearchResult:
        """
        Execute the full alpha discovery pipeline.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (dates x securities).
        returns : pd.DataFrame
            Daily returns (dates x securities).  Should be aligned with
            ``prices`` (same index and columns).
        volumes : pd.DataFrame
            Share or dollar volumes (dates x securities).
        high : pd.DataFrame, optional
            Daily high prices.  Enables high-low range signals.
        low : pd.DataFrame, optional
            Daily low prices.  Enables high-low range signals.
        existing_strategy_returns : pd.DataFrame, optional
            Returns of existing strategies (dates x strategies).  Used in
            the correlation filter to ensure new signals are additive.
        regime_labels : pd.Series, optional
            Categorical regime label per date (e.g. "bull", "bear").
            Enables regime robustness metrics and filtering.

        Returns
        -------
        AlphaResearchResult
        """
        t_start = time.perf_counter()
        timestamp = pd.Timestamp.now().isoformat(timespec="seconds")

        logger.info("=== Alpha Discovery Run started at %s ===", timestamp)

        # ----------------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------------
        self._validate_inputs(prices, returns, volumes)

        # ----------------------------------------------------------
        # Build forward returns (shift by horizon)
        # ----------------------------------------------------------
        forward_returns = self._build_forward_returns(returns)

        # ----------------------------------------------------------
        # Step 1: Generate candidate signals
        # ----------------------------------------------------------
        logger.info("Step 1/4 — Generating candidate signals ...")
        signals = self.generator.generate_candidates(
            prices=prices,
            returns=returns,
            volumes=volumes,
            high=high,
            low=low,
        )

        # ----------------------------------------------------------
        # Step 2: Evaluate all signals
        # ----------------------------------------------------------
        logger.info("Step 2/4 — Evaluating %d signals ...", len(signals))
        signal_performance = self.evaluator.evaluate_all(
            signals_dict=signals,
            forward_returns=forward_returns,
            regime_labels=regime_labels,
        )

        # ----------------------------------------------------------
        # Step 3: Build signal P&L dict (needed for corr filter)
        # ----------------------------------------------------------
        logger.info("Step 3/4 — Building P&L dictionary ...")
        signal_pnl_dict = self._build_pnl_dict(signals, forward_returns)

        # ----------------------------------------------------------
        # Step 4: Filter signals
        # ----------------------------------------------------------
        logger.info("Step 4/4 — Filtering signals ...")
        passing_df = self.signal_filter.filter_signals(
            signal_metrics=signal_performance,
            existing_strategy_returns=existing_strategy_returns,
            signal_pnl_dict=signal_pnl_dict,
        )
        filtered_signals = list(passing_df.index)

        # ----------------------------------------------------------
        # Compute signal correlation matrix (all evaluated signals)
        # ----------------------------------------------------------
        corr_matrix = self._compute_correlation_matrix(signal_pnl_dict)

        # ----------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------
        elapsed = time.perf_counter() - t_start
        date_range = (
            f"{returns.index.min().date()} to {returns.index.max().date()}"
            if not returns.empty
            else "N/A"
        )

        metadata = {
            "timestamp": timestamp,
            "elapsed_seconds": elapsed,
            "date_range": date_range,
            "n_securities": returns.shape[1],
            "n_dates": returns.shape[0],
            "forward_return_horizon": self.forward_return_horizon,
            "n_candidates_generated": len(signals),
            "n_signals_evaluated": len(signal_performance),
            "n_signals_passed": len(filtered_signals),
        }

        result = AlphaResearchResult(
            candidate_signal_library=signals,
            signal_performance=signal_performance,
            filtered_signals=filtered_signals,
            signal_correlation_matrix=corr_matrix,
            signal_pnl_dict=signal_pnl_dict,
            run_metadata=metadata,
        )

        logger.info(
            "=== Alpha Discovery Run complete in %.1fs | "
            "%d/%d signals passed filters ===",
            elapsed,
            len(filtered_signals),
            len(signal_performance),
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> None:
        for name, df in [("prices", prices), ("returns", returns), ("volumes", volumes)]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"'{name}' must be a pd.DataFrame, got {type(df)}.")
            if df.empty:
                raise ValueError(f"'{name}' is empty.")

        if len(returns) < self.min_history:
            raise ValueError(
                f"Insufficient history: {len(returns)} dates < min_history={self.min_history}."
            )

        if not returns.index.equals(prices.index):
            logger.warning(
                "prices and returns indices do not match exactly; "
                "results will be aligned by intersection."
            )

    def _build_forward_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Shift returns by ``forward_return_horizon`` so that
        ``forward_returns.loc[t]`` = return earned by entering at close of ``t``.

        We use a simple shift, meaning the signal on day ``t`` is evaluated
        against the return of day ``t + horizon``.
        """
        return returns.shift(-self.forward_return_horizon)

    def _build_pnl_dict(
        self,
        signals: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Build a long-short daily P&L series for each signal in ``signals``.
        """
        pnl_dict: dict[str, pd.Series] = {}
        for name, sig in signals.items():
            try:
                pnl = _build_long_short_pnl(
                    sig, forward_returns, self.evaluator.n_quantiles
                )
                if not pnl.empty:
                    pnl_dict[name] = pnl
            except Exception as exc:  # noqa: BLE001
                logger.debug("P&L build failed for '%s': %s", name, exc)
        return pnl_dict

    @staticmethod
    def _compute_correlation_matrix(
        signal_pnl_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compute pairwise Spearman rank correlation of signal P&L series.

        Aligns all series to a common date index.  Missing values result
        in pairwise-complete correlations.

        Returns
        -------
        pd.DataFrame
            Square correlation matrix.  Returns empty DataFrame if fewer
            than two signals are available.
        """
        if len(signal_pnl_dict) < 2:
            return pd.DataFrame()

        # Combine all P&L series into a single DataFrame
        pnl_df = pd.DataFrame(signal_pnl_dict)

        # Spearman rank correlation: convert to ranks first, then Pearson
        ranked = pnl_df.rank()
        corr_matrix = ranked.corr(method="pearson", min_periods=20)

        return corr_matrix

    # ------------------------------------------------------------------
    # Convenience: incremental evaluation of a single new signal
    # ------------------------------------------------------------------

    def evaluate_new_signal(
        self,
        signal: pd.DataFrame,
        signal_name: str,
        returns: pd.DataFrame,
        result: AlphaResearchResult,
        regime_labels: Optional[pd.Series] = None,
        existing_strategy_returns: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Evaluate and filter a *single* new signal against an existing result.

        This is a lightweight method for interactive research — you can
        quickly check whether a hand-crafted signal meets the bar without
        re-running the full discovery pipeline.

        Parameters
        ----------
        signal : pd.DataFrame
            New signal values (dates x securities).
        signal_name : str
            Human-readable name.
        returns : pd.DataFrame
            Historical returns aligned to the signal.
        result : AlphaResearchResult
            Prior discovery result (used for the correlation check).
        regime_labels : pd.Series, optional
            Regime labels.
        existing_strategy_returns : pd.DataFrame, optional
            Returns of existing strategies for correlation check.

        Returns
        -------
        dict
            Keys: ``"metrics"`` (SignalMetrics), ``"passes_filter"`` (bool),
            ``"correlated_with"`` (list of signal names with high correlation).
        """
        forward_returns = self._build_forward_returns(returns)

        metrics = self.evaluator.evaluate_signal(
            signal=signal,
            forward_returns=forward_returns,
            regime_labels=regime_labels,
            signal_name=signal_name,
        )

        # Build P&L for correlation check
        pnl = _build_long_short_pnl(signal, forward_returns, self.evaluator.n_quantiles)

        # Check correlation against all existing signal P&Ls
        correlated_with: list[str] = []
        for existing_name, existing_pnl in result.signal_pnl_dict.items():
            passes = self.signal_filter.correlation_check(
                candidate_returns=pnl,
                existing_returns=existing_pnl.to_frame(existing_name),
                max_corr=self.signal_filter.max_correlation,
            )
            if not passes:
                correlated_with.append(existing_name)

        # Run the standard scalar filters on a one-row metrics DataFrame
        metrics_df = metrics.to_series().to_frame(signal_name).T
        metrics_df.index.name = "signal_name"

        pnl_dict = {signal_name: pnl}
        passing = self.signal_filter.filter_signals(
            signal_metrics=metrics_df,
            existing_strategy_returns=existing_strategy_returns,
            signal_pnl_dict=pnl_dict,
        )
        passes_filter = len(passing) > 0

        return {
            "metrics": metrics,
            "passes_filter": passes_filter,
            "correlated_with": correlated_with,
        }
