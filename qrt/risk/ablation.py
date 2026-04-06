"""
Ablation Testing Framework
============================
Systematic ablation tests for isolating the contribution of each component
to portfolio performance, with focus on drawdown reduction.

Runs each ablation at both strategy-sleeve and full-portfolio level, tracking:
- MaxDD, CDaR(95%), drawdown duration/time-under-water
- Sharpe, Sortino, Calmar
- Turnover, skew/kurtosis

References
----------
- The deep research report specifies a minimal ablation set ordered by
  expected drawdown impact.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .enhanced_metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class AblationTest:
    """
    A single ablation test: compares a baseline against one or more variants.

    Parameters
    ----------
    name : str
        Name of the ablation test.
    description : str
        What is being tested.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.results: list[dict] = []

    def add_variant(
        self,
        variant_name: str,
        returns: pd.Series,
        is_baseline: bool = False,
    ) -> None:
        """Add a variant's returns for comparison."""
        metrics = compute_full_metrics(returns, name=variant_name)
        metrics["variant"] = variant_name
        metrics["is_baseline"] = is_baseline
        metrics["ablation"] = self.name
        self.results.append(metrics)

    def summary(self) -> pd.DataFrame:
        """Return comparison table across all variants."""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        key_cols = [
            "ablation", "variant", "is_baseline", "sharpe", "sortino",
            "calmar", "max_drawdown", "cdar_95", "cvar_95",
            "skewness", "kurtosis", "time_in_drawdown_pct",
            "cagr", "ann_volatility",
        ]
        available = [c for c in key_cols if c in df.columns]
        return df[available]


class AblationFramework:
    """
    Run systematic ablation tests across the platform.

    Parameters
    ----------
    strategy_results : dict
        Dict of {name: {"weights": ..., "returns": ..., ...}}.
    returns_wide : pd.DataFrame
        Full asset return matrix.
    portfolio_returns : pd.Series
        Final portfolio returns.
    """

    def __init__(
        self,
        strategy_results: dict,
        returns_wide: pd.DataFrame,
        portfolio_returns: pd.Series,
    ) -> None:
        self.strategy_results = strategy_results
        self.returns_wide = returns_wide
        self.portfolio_returns = portfolio_returns
        self.tests: list[AblationTest] = []

    def run_drawdown_control_ablation(self) -> AblationTest:
        """
        Ablation 1: Drawdown control method.

        Compares:
        - Baseline: current binary circuit breaker
        - Variant 1: No drawdown control
        - Variant 2: Continuous drawdown scaling (CDaR-aware)
        """
        from qrt.strategies.base import Strategy
        from qrt.risk.drawdown_risk import ContinuousDrawdownScaler

        test = AblationTest(
            "drawdown_control",
            "Binary stop-out vs continuous drawdown scaling vs no control",
        )

        # For each momentum-like strategy, test all three variants
        for name, res in self.strategy_results.items():
            if "weights" not in res:
                continue

            raw_weights = res.get("_raw_weights", res["weights"])
            returns = self.returns_wide

            # Baseline (current binary breaker)
            breaker_weights = Strategy.apply_drawdown_cap(
                raw_weights, returns, max_dd=0.20, cooldown=21, reduction=1.0
            )
            breaker_returns = (breaker_weights.shift(1) * returns).sum(axis=1)
            test.add_variant(f"{name}_binary_breaker", breaker_returns, is_baseline=True)

            # No control
            no_control_returns = (raw_weights.shift(1) * returns).sum(axis=1)
            test.add_variant(f"{name}_no_control", no_control_returns)

            # Continuous CDaR scaler
            scaler = ContinuousDrawdownScaler(max_dd=0.20)
            cdar_weights = scaler.apply(raw_weights, returns)
            cdar_returns = (cdar_weights.shift(1) * returns).sum(axis=1)
            test.add_variant(f"{name}_continuous_cdar", cdar_returns)

        self.tests.append(test)
        return test

    def run_momentum_risk_ablation(self) -> AblationTest:
        """
        Ablation 2: Risk-managed momentum.

        Compares:
        - Baseline: raw momentum
        - Variant 1: constant-vol scaled
        - Variant 2: crash-gated
        """
        from qrt.portfolio.momentum_risk import MomentumRiskManager

        test = AblationTest(
            "momentum_risk_management",
            "Raw momentum vs constant-vol scaled vs crash-gated",
        )

        momentum_names = ["cross_sectional_momentum", "time_series_momentum",
                          "residual_momentum", "factor_momentum"]

        for name in momentum_names:
            if name not in self.strategy_results:
                continue
            res = self.strategy_results[name]
            if "weights" not in res:
                continue

            # Baseline (raw)
            raw_returns = (res["weights"].shift(1) * self.returns_wide).sum(axis=1)
            test.add_variant(f"{name}_raw", raw_returns, is_baseline=True)

            # Constant-vol scaled
            mgr = MomentumRiskManager(crash_gate=False)
            scaled_weights = mgr.risk_manage_sleeve(
                res["weights"], self.returns_wide, name
            )
            scaled_returns = (scaled_weights.shift(1) * self.returns_wide).sum(axis=1)
            test.add_variant(f"{name}_vol_scaled", scaled_returns)

            # Crash-gated
            mgr_crash = MomentumRiskManager(crash_gate=True)
            gated_weights = mgr_crash.risk_manage_sleeve(
                res["weights"], self.returns_wide, name
            )
            gated_returns = (gated_weights.shift(1) * self.returns_wide).sum(axis=1)
            test.add_variant(f"{name}_crash_gated", gated_returns)

        self.tests.append(test)
        return test

    def run_covariance_ablation(self) -> AblationTest:
        """
        Ablation 3: Covariance estimator.

        Compares:
        - Baseline: sample covariance
        - Variant: Ledoit-Wolf shrinkage
        """
        from qrt.portfolio.risk_parity import RiskParityAllocator
        from qrt.portfolio.shrinkage import ShrinkageEstimator

        test = AblationTest(
            "covariance_estimator",
            "Sample covariance vs Ledoit-Wolf shrinkage",
        )

        strat_returns = {
            name: res["returns"]
            for name, res in self.strategy_results.items()
        }
        returns_df = pd.DataFrame(strat_returns).dropna(how="all").fillna(0.0)

        if returns_df.shape[1] < 2:
            self.tests.append(test)
            return test

        # Baseline: sample covariance risk parity
        alloc = RiskParityAllocator()
        sample_weights = alloc.allocate(returns_df, method="covariance")
        sample_combined = (returns_df * sample_weights).sum(axis=1)
        test.add_variant("sample_cov_risk_parity", sample_combined, is_baseline=True)

        # Shrinkage covariance risk parity
        shrink = ShrinkageEstimator(target="constant_correlation")
        shrunk_cov = shrink.estimate(returns_df)
        shrunk_weights_arr = alloc.covariance_risk_parity(shrunk_cov)
        shrunk_weights = pd.Series(shrunk_weights_arr, index=returns_df.columns)
        shrunk_combined = (returns_df * shrunk_weights).sum(axis=1)
        test.add_variant("shrinkage_risk_parity", shrunk_combined)

        self.tests.append(test)
        return test

    def run_cost_stress_ablation(self) -> AblationTest:
        """
        Ablation 4: Cost model stress test.

        Compares:
        - Baseline: normal costs
        - Variant 1: 2× spreads/impact
        - Variant 2: 4× spreads/impact in stress
        """
        from qrt.costs.transaction_costs import TransactionCostModel

        test = AblationTest(
            "cost_stress",
            "Baseline costs vs 2× vs 4× spreads/impact",
        )

        # Use portfolio-level returns for cost impact
        test.add_variant("portfolio_baseline", self.portfolio_returns, is_baseline=True)

        # Approximate cost drag at different levels
        for mult, label in [(2, "2x_costs"), (4, "4x_costs")]:
            # Simple approximation: subtract additional cost drag
            # Daily cost drag scales roughly linearly with cost multiplier
            base_drag = 0.0002  # ~2bps base daily cost drag estimate
            additional_drag = base_drag * (mult - 1)
            stressed = self.portfolio_returns - additional_drag
            test.add_variant(f"portfolio_{label}", stressed)

        self.tests.append(test)
        return test

    def run_allocation_ablation(self) -> AblationTest:
        """
        Ablation 5: Allocation logic.

        Compares:
        - Static risk parity
        - Equal weight
        - HERC with CDaR
        """
        from qrt.portfolio.hierarchical import HERCAllocator

        test = AblationTest(
            "allocation_logic",
            "Static risk parity vs equal weight vs HERC-CDaR",
        )

        strat_returns = {
            name: res["returns"]
            for name, res in self.strategy_results.items()
        }
        returns_df = pd.DataFrame(strat_returns).dropna(how="all").fillna(0.0)

        if returns_df.shape[1] < 2:
            self.tests.append(test)
            return test

        # Equal weight
        ew_combined = returns_df.mean(axis=1)
        test.add_variant("equal_weight", ew_combined, is_baseline=True)

        # Risk parity (current baseline)
        from qrt.portfolio.risk_parity import RiskParityAllocator
        alloc = RiskParityAllocator()
        rp_weights = alloc.allocate(returns_df, method="covariance")
        rp_combined = (returns_df * rp_weights).sum(axis=1)
        test.add_variant("risk_parity", rp_combined)

        # HERC with CDaR
        try:
            herc = HERCAllocator(risk_measure="cdar")
            herc_weights = herc.allocate(returns_df)
            herc_combined = (returns_df * herc_weights).sum(axis=1)
            test.add_variant("herc_cdar", herc_combined)
        except Exception as e:
            logger.warning("HERC-CDaR ablation failed: %s", e)

        self.tests.append(test)
        return test

    def run_advanced_allocation_ablation(self) -> AblationTest:
        """
        Ablation 6: Advanced allocation techniques.

        Compares research-backed allocation methods against baseline:
        - Equal weight (baseline)
        - CVaR-optimized (Rockafellar & Uryasev, 2000)
        - Downside risk parity (Sortino & van der Meer, 1991)
        - Maximum diversification (Choueifaty & Coignard, 2008)
        - Blend of all three
        """
        from qrt.risk.advanced_risk import (
            CVaROptimizer,
            DownsideRiskParity,
            MaxDiversification,
        )

        test = AblationTest(
            "advanced_allocation",
            "CVaR-opt vs downside-RP vs max-div vs blend vs equal-weight",
        )

        strat_returns = {
            name: res["returns"]
            for name, res in self.strategy_results.items()
        }
        returns_df = pd.DataFrame(strat_returns).dropna(how="all").fillna(0.0)

        if returns_df.shape[1] < 2:
            self.tests.append(test)
            return test

        # Equal weight baseline
        ew_combined = returns_df.mean(axis=1)
        test.add_variant("equal_weight", ew_combined, is_baseline=True)

        # CVaR-optimized
        try:
            cvar_opt = CVaROptimizer(min_weight=0.02, max_weight=0.50)
            cvar_w = cvar_opt.allocate(returns_df)
            cvar_combined = (returns_df * cvar_w).sum(axis=1)
            test.add_variant("cvar_optimized", cvar_combined)
        except Exception as e:
            logger.warning("CVaR allocation ablation failed: %s", e)

        # Downside risk parity
        try:
            drp = DownsideRiskParity(min_weight=0.02, max_weight=0.50)
            drp_w = drp.allocate(returns_df)
            drp_combined = (returns_df * drp_w).sum(axis=1)
            test.add_variant("downside_risk_parity", drp_combined)
        except Exception as e:
            logger.warning("Downside RP ablation failed: %s", e)

        # Maximum diversification
        try:
            md = MaxDiversification(min_weight=0.02, max_weight=0.50)
            md_w = md.allocate(returns_df)
            md_combined = (returns_df * md_w).sum(axis=1)
            test.add_variant("max_diversification", md_combined)
        except Exception as e:
            logger.warning("Max diversification ablation failed: %s", e)

        # Blend of all three
        try:
            cvar_w2 = CVaROptimizer(min_weight=0.02, max_weight=0.50).allocate(returns_df)
            drp_w2 = DownsideRiskParity(min_weight=0.02, max_weight=0.50).allocate(returns_df)
            md_w2 = MaxDiversification(min_weight=0.02, max_weight=0.50).allocate(returns_df)
            blend_w = (cvar_w2 + drp_w2 + md_w2) / 3
            blend_w = blend_w / blend_w.sum()
            blend_combined = (returns_df * blend_w).sum(axis=1)
            test.add_variant("blend_cvar_drp_maxdiv", blend_combined)
        except Exception as e:
            logger.warning("Blend allocation ablation failed: %s", e)

        self.tests.append(test)
        return test

    def run_systemic_risk_overlay_ablation(self) -> AblationTest:
        """
        Ablation 7: Systemic risk overlay.

        Compares portfolio returns with and without:
        - No overlay (baseline)
        - Turbulence index only (Kritzman & Li, 2010)
        - Absorption ratio only (Kritzman et al., 2011)
        - Composite overlay (min of both)
        """
        from qrt.risk.advanced_risk import (
            TurbulenceIndex,
            AbsorptionRatio,
            CompositeRiskOverlay,
        )

        test = AblationTest(
            "systemic_risk_overlay",
            "No overlay vs turbulence vs absorption vs composite",
        )

        # Use portfolio returns as baseline
        test.add_variant("no_overlay", self.portfolio_returns, is_baseline=True)

        strat_returns = {
            name: res["returns"]
            for name, res in self.strategy_results.items()
        }
        returns_df = pd.DataFrame(strat_returns).dropna(how="all").fillna(0.0)

        if returns_df.shape[1] < 2:
            self.tests.append(test)
            return test

        # Compute equal-weight portfolio for comparison
        ew_weights = pd.DataFrame(
            1.0 / returns_df.shape[1],
            index=returns_df.index,
            columns=returns_df.columns,
        )

        # Turbulence index only
        try:
            ti = TurbulenceIndex(lookback=126, floor=0.20)
            ti_scaling = ti.compute_scaling(returns_df)
            ti_weights = ew_weights.mul(ti_scaling, axis=0)
            ti_returns = (returns_df * ti_weights.shift(1)).sum(axis=1)
            test.add_variant("turbulence_overlay", ti_returns)
        except Exception as e:
            logger.warning("Turbulence ablation failed: %s", e)

        # Absorption ratio only
        try:
            ar = AbsorptionRatio(lookback=126, floor=0.30)
            ar_scaling = ar.compute_scaling(returns_df)
            ar_weights = ew_weights.mul(ar_scaling, axis=0)
            ar_returns = (returns_df * ar_weights.shift(1)).sum(axis=1)
            test.add_variant("absorption_overlay", ar_returns)
        except Exception as e:
            logger.warning("Absorption ratio ablation failed: %s", e)

        # Composite (min of both)
        try:
            composite = CompositeRiskOverlay(
                turbulence_config={"lookback": 126, "floor": 0.20},
                absorption_config={"lookback": 126, "floor": 0.30},
                combination="min",
            )
            comp_scaling = composite.compute_scaling(returns_df)
            comp_weights = ew_weights.mul(comp_scaling, axis=0)
            comp_returns = (returns_df * comp_weights.shift(1)).sum(axis=1)
            test.add_variant("composite_overlay", comp_returns)
        except Exception as e:
            logger.warning("Composite overlay ablation failed: %s", e)

        self.tests.append(test)
        return test

    def run_adaptive_stops_ablation(self) -> AblationTest:
        """
        Ablation 8: Adaptive vs fixed stops.

        Compares:
        - No stops (baseline)
        - Fixed 3% stop-loss (current)
        - Adaptive volatility-scaled stop (Kaminski & Lo, 2014)
        """
        from qrt.risk.advanced_risk import AdaptiveStopLoss

        test = AblationTest(
            "adaptive_stops",
            "No stops vs fixed 3% vs adaptive vol-scaled stops",
        )

        for name, res in self.strategy_results.items():
            if "weights" not in res:
                continue

            weights = res["weights"]
            returns = self.returns_wide

            # No stops (baseline)
            raw_returns = (weights.shift(1) * returns).sum(axis=1)
            test.add_variant(f"{name}_no_stops", raw_returns, is_baseline=True)

            # Adaptive stops (tight)
            try:
                tight_stops = AdaptiveStopLoss(
                    vol_lookback=21, stop_multiplier=1.5, cooldown_days=10,
                )
                tight_w = tight_stops.apply_adaptive_stops(weights, returns)
                tight_ret = (tight_w.shift(1) * returns).sum(axis=1)
                test.add_variant(f"{name}_adaptive_tight", tight_ret)
            except Exception as e:
                logger.warning("Adaptive tight stops failed for %s: %s", name, e)

            # Adaptive stops (standard)
            try:
                std_stops = AdaptiveStopLoss(
                    vol_lookback=21, stop_multiplier=2.5, cooldown_days=10,
                )
                std_w = std_stops.apply_adaptive_stops(weights, returns)
                std_ret = (std_w.shift(1) * returns).sum(axis=1)
                test.add_variant(f"{name}_adaptive_standard", std_ret)
            except Exception as e:
                logger.warning("Adaptive standard stops failed for %s: %s", name, e)

        self.tests.append(test)
        return test

    def run_all(self) -> pd.DataFrame:
        """Run all ablation tests and return combined results."""
        logger.info("Running ablation tests...")

        self.run_drawdown_control_ablation()
        logger.info("  Completed: drawdown control ablation")

        self.run_momentum_risk_ablation()
        logger.info("  Completed: momentum risk ablation")

        self.run_covariance_ablation()
        logger.info("  Completed: covariance estimator ablation")

        self.run_cost_stress_ablation()
        logger.info("  Completed: cost stress ablation")

        self.run_allocation_ablation()
        logger.info("  Completed: allocation logic ablation")

        self.run_advanced_allocation_ablation()
        logger.info("  Completed: advanced allocation ablation")

        self.run_systemic_risk_overlay_ablation()
        logger.info("  Completed: systemic risk overlay ablation")

        self.run_adaptive_stops_ablation()
        logger.info("  Completed: adaptive stops ablation")

        # Combine all results
        all_results = []
        for t in self.tests:
            df = t.summary()
            if not df.empty:
                all_results.append(df)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            logger.info("  Total ablation variants tested: %d", len(combined))
            return combined

        return pd.DataFrame()

    def save_report(self, path: str, results: pd.DataFrame | None = None) -> None:
        """Save ablation results as markdown."""
        if results is None:
            results = self.run_all()
        if results.empty:
            return

        lines = [
            "# Ablation Test Results\n",
            f"**Total variants tested:** {len(results)}\n",
        ]

        for ablation_name in results["ablation"].unique():
            subset = results[results["ablation"] == ablation_name]
            lines.append(f"\n## {ablation_name.replace('_', ' ').title()}\n")

            # Format key metrics
            cols = ["variant", "is_baseline", "sharpe", "calmar",
                    "max_drawdown", "cdar_95", "skewness"]
            display_cols = [c for c in cols if c in subset.columns]
            try:
                lines.append(subset[display_cols].to_markdown(index=False))
            except ImportError:
                lines.append(subset[display_cols].to_string(index=False))
            lines.append("\n")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Ablation report saved to %s", path)
