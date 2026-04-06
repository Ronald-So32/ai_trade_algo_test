"""
Leverage-Specific Stress Testing & Overfitting Checks
=======================================================
Validates that leveraged portfolio results are robust and not artifacts
of overfitting, with stress scenarios specific to high-leverage strategies.

Academic basis:
  - Avellaneda & Zhang (2010): Vol drag formula for leveraged returns
  - Grossman & Zhou (1993): Optimal leverage under drawdown constraints
  - Kaminski & Lo (2014): Regime-dependent leverage risk

Key risks with leveraged backtests:
  1. Vol drag underestimation (historical vol ≠ future vol)
  2. Correlation breakdown (diversification evaporates in crises)
  3. Drawdown compounding (leverage × DD is non-linear at high L)
  4. Estimation error amplification (leverage × backtest error)
  5. Regime sensitivity (leverage optimal in bull, catastrophic in bear)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class LeverageStressResult:
    """Results from leverage stress testing."""
    base_metrics: dict = field(default_factory=dict)
    vol_shock_results: list = field(default_factory=list)
    correlation_shock_results: list = field(default_factory=list)
    regime_analysis: dict = field(default_factory=dict)
    drawdown_scenarios: list = field(default_factory=list)
    estimation_error: dict = field(default_factory=dict)
    overall_risk_score: float = 0.0
    recommendations: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Leverage Stress Test Report",
            "",
            f"**Overall Risk Score: {self.overall_risk_score:.1f}/10** "
            f"(lower is riskier)",
            "",
        ]

        if self.base_metrics:
            lines.append("## Base Metrics")
            for k, v in self.base_metrics.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        if self.vol_shock_results:
            lines.append("## Volatility Shock Scenarios")
            lines.append("")
            lines.append(
                f"| {'Scenario':<25} | {'Vol Mult':>10} | "
                f"{'CAGR':>10} | {'MaxDD':>10} | {'Sharpe':>8} |"
            )
            lines.append(
                f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|"
            )
            for s in self.vol_shock_results:
                lines.append(
                    f"| {s['scenario']:<25} | {s['vol_multiplier']:>10.2f} | "
                    f"{s['cagr']:>9.2%} | {s['maxdd']:>9.2%} | "
                    f"{s['sharpe']:>8.3f} |"
                )
            lines.append("")

        if self.drawdown_scenarios:
            lines.append("## Drawdown Stress Scenarios")
            lines.append("")
            for s in self.drawdown_scenarios:
                lines.append(
                    f"- **{s['scenario']}**: MaxDD={s['maxdd']:.2%}, "
                    f"Recovery={s.get('recovery_days', 'N/A')} days"
                )
            lines.append("")

        if self.estimation_error:
            ee = self.estimation_error
            lines.append("## Estimation Error Analysis")
            lines.append("")
            lines.append(f"- Sharpe estimation SE: {ee.get('sharpe_se', 0):.3f}")
            lines.append(f"- 95% CI for Sharpe: [{ee.get('sharpe_ci_low', 0):.3f}, {ee.get('sharpe_ci_high', 0):.3f}]")
            lines.append(f"- At leverage, error amplified by: {ee.get('error_amplification', 0):.1f}x")
            lines.append(f"- Expected OOS Sharpe range: [{ee.get('oos_sharpe_low', 0):.3f}, {ee.get('oos_sharpe_high', 0):.3f}]")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)


class LeverageStressTester:
    """
    Comprehensive stress testing for leveraged portfolios.

    Tests portfolio resilience under:
    1. Volatility regime shocks (vol increases 50%, 100%, 200%)
    2. Correlation breakdown (all correlations → 1)
    3. Historical drawdown replay at leverage
    4. Estimation error amplification
    5. Regime-conditional analysis
    """

    def __init__(
        self,
        leverage: float = 1.0,
        n_simulations: int = 2000,
        random_state: int = 42,
        shield=None,
    ):
        self.leverage = leverage
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_state)
        self.shield = shield  # Optional DrawdownShield for realistic stress testing

    def _apply_shield(self, leveraged_returns: pd.Series) -> pd.Series:
        """Apply DrawdownShield to leveraged returns if available."""
        if self.shield is None:
            return leveraged_returns
        try:
            pseudo_w = pd.DataFrame(
                {"portfolio": 1.0}, index=leveraged_returns.index,
            )
            pseudo_r = pd.DataFrame(
                {"portfolio": leveraged_returns.values},
                index=leveraged_returns.index,
            )
            shielded_w = self.shield.apply(pseudo_w, pseudo_r)
            return leveraged_returns * shielded_w["portfolio"]
        except Exception:
            return leveraged_returns

    def run_full_stress_test(
        self,
        base_returns: pd.Series,
        strategy_returns: dict[str, pd.Series] | None = None,
        regime_labels: pd.Series | None = None,
    ) -> LeverageStressResult:
        """Run all leverage stress tests."""
        result = LeverageStressResult()

        base = base_returns.dropna()
        if len(base) < 100:
            result.recommendations.append("Insufficient data for stress testing")
            return result

        # Base metrics
        result.base_metrics = self._compute_base_metrics(base)

        # 1. Volatility shock scenarios
        result.vol_shock_results = self._vol_shock_scenarios(base)

        # 2. Drawdown stress scenarios
        result.drawdown_scenarios = self._drawdown_stress(base)

        # 3. Estimation error analysis
        result.estimation_error = self._estimation_error_analysis(base)

        # 4. Regime analysis
        if regime_labels is not None:
            result.regime_analysis = self._regime_analysis(base, regime_labels)

        # 5. Correlation shock (if multi-strategy)
        if strategy_returns and len(strategy_returns) >= 2:
            result.correlation_shock_results = self._correlation_shock(
                strategy_returns
            )

        # Compute overall risk score
        result.overall_risk_score = self._compute_risk_score(result)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _compute_base_metrics(self, returns: pd.Series) -> dict:
        """Compute base portfolio metrics at current leverage."""
        lev_rets = self._apply_shield(returns * self.leverage)
        cum = (1 + lev_rets).cumprod()
        n = len(cum)
        cagr = float(cum.iloc[-1] ** (252 / n) - 1) if cum.iloc[-1] > 0 else -1
        maxdd = float((cum / cum.cummax() - 1).min())
        vol = float(lev_rets.std() * np.sqrt(252))
        sharpe = float(lev_rets.mean() / lev_rets.std() * np.sqrt(252)) if lev_rets.std() > 0 else 0
        base_vol = float(returns.std() * np.sqrt(252))
        vol_drag = self.leverage * (self.leverage - 1) / 2 * base_vol**2

        return {
            "leverage": self.leverage,
            "cagr": cagr,
            "maxdd": maxdd,
            "vol": vol,
            "sharpe": sharpe,
            "base_vol": base_vol,
            "vol_drag": vol_drag,
            "calmar": cagr / abs(maxdd) if maxdd != 0 else 0,
            "n_observations": n,
        }

    def _vol_shock_scenarios(self, returns: pd.Series) -> list[dict]:
        """Test leveraged portfolio under various vol multipliers."""
        scenarios = [
            ("Normal", 1.0),
            ("Mild stress (+50% vol)", 1.5),
            ("Moderate stress (+100% vol)", 2.0),
            ("Severe stress (+200% vol)", 3.0),
            ("Extreme stress (+300% vol)", 4.0),
        ]

        results = []
        base_mean = returns.mean()
        base_std = returns.std()

        for name, mult in scenarios:
            # Scale volatility while keeping mean (approximately)
            # New returns: mean + mult * (return - mean)
            shocked = base_mean + mult * (returns - base_mean)
            lev_shocked = shocked * self.leverage

            # Apply DrawdownShield if available — gives realistic stress results
            # that account for dynamic de-risking during crises
            lev_shocked = self._apply_shield(lev_shocked)

            cum = (1 + lev_shocked).cumprod()
            n = len(cum)

            if cum.iloc[-1] <= 0:
                cagr = -1.0
                sharpe = 0.0
            else:
                cagr = float(cum.iloc[-1] ** (252 / n) - 1)
                sharpe = (
                    float(lev_shocked.mean() / lev_shocked.std() * np.sqrt(252))
                    if lev_shocked.std() > 0 else 0
                )
            maxdd = float((cum / cum.cummax() - 1).min())

            results.append({
                "scenario": name,
                "vol_multiplier": mult,
                "cagr": cagr,
                "maxdd": maxdd,
                "sharpe": sharpe,
                "vol": float(lev_shocked.std() * np.sqrt(252)),
            })

        return results

    def _drawdown_stress(self, returns: pd.Series) -> list[dict]:
        """Replay worst historical drawdown periods at leverage."""
        cum = (1 + returns).cumprod()
        dd = cum / cum.cummax() - 1

        # Find worst drawdown episodes
        scenarios = []

        # Scenario 1: Worst historical DD at leverage
        worst_dd = dd.min()
        worst_date = dd.idxmin()
        # Find episode boundaries
        peak_before = cum.loc[:worst_date].idxmax()
        lev_dd = float(1 - (1 + worst_dd) ** self.leverage)  # approximate

        scenarios.append({
            "scenario": f"Worst historical DD at {self.leverage:.1f}x",
            "maxdd": -abs(lev_dd),
            "base_dd": float(worst_dd),
            "peak_date": str(peak_before.date()) if hasattr(peak_before, 'date') else str(peak_before),
            "trough_date": str(worst_date.date()) if hasattr(worst_date, 'date') else str(worst_date),
        })

        # Scenario 2: 2x worst DD (tail risk)
        tail_dd = worst_dd * 2
        lev_tail_dd = float(1 - (1 + tail_dd) ** min(self.leverage, 5))
        scenarios.append({
            "scenario": f"2x worst DD at {self.leverage:.1f}x",
            "maxdd": max(-1.0, -abs(lev_tail_dd)),
            "base_dd": float(tail_dd),
        })

        # Scenario 3: Flash crash (10% single-day drop)
        flash_dd = -0.10 * self.leverage
        scenarios.append({
            "scenario": f"10% flash crash at {self.leverage:.1f}x",
            "maxdd": max(-1.0, flash_dd),
            "base_dd": -0.10,
        })

        # Scenario 4: Sustained downturn (1% daily loss for 20 days)
        sustained_loss = (1 - 0.01 * self.leverage) ** 20 - 1
        scenarios.append({
            "scenario": f"1%/day × 20 days at {self.leverage:.1f}x",
            "maxdd": max(-1.0, sustained_loss),
            "base_dd": (0.99**20 - 1),
            "recovery_days": "N/A (scenario)",
        })

        return scenarios

    def _estimation_error_analysis(self, returns: pd.Series) -> dict:
        """Analyse how estimation errors are amplified by leverage."""
        n = len(returns)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        skew = float(scipy_stats.skew(returns, bias=False))
        kurt = float(scipy_stats.kurtosis(returns, bias=False) + 3)

        # Standard error of Sharpe (Lo 2002, corrected)
        se_sharpe = np.sqrt(
            (1 + 0.25 * (kurt - 3) * sharpe**2 - skew * sharpe) / n
        )

        # Confidence interval
        z = 1.96
        ci_low = sharpe - z * se_sharpe
        ci_high = sharpe + z * se_sharpe

        # At leverage, estimation error is amplified
        # Error in returns → leverage × error in returns
        # But Sharpe is scale-invariant, so leverage doesn't change Sharpe SE
        # HOWEVER: vol drag estimation error IS amplified
        base_vol = float(returns.std() * np.sqrt(252))
        vol_drag = self.leverage * (self.leverage - 1) / 2 * base_vol**2
        # Vol drag SE ≈ leverage^2 × vol_se^2 (quadratic amplification)
        vol_se = base_vol / np.sqrt(2 * n)  # SE of variance estimate
        vol_drag_se = self.leverage * (self.leverage - 1) * base_vol * vol_se

        return {
            "sharpe": sharpe,
            "sharpe_se": float(se_sharpe),
            "sharpe_ci_low": float(ci_low),
            "sharpe_ci_high": float(ci_high),
            "vol_drag": vol_drag,
            "vol_drag_se": float(vol_drag_se),
            "error_amplification": self.leverage,
            "oos_sharpe_low": float(ci_low),  # Sharpe is leverage-invariant
            "oos_sharpe_high": float(ci_high),
        }

    def _regime_analysis(
        self, returns: pd.Series, regime_labels: pd.Series,
    ) -> dict:
        """Analyse leveraged performance per regime."""
        aligned = pd.DataFrame({
            "returns": returns,
            "regime": regime_labels,
        }).dropna()

        results = {}
        for regime in aligned["regime"].unique():
            mask = aligned["regime"] == regime
            regime_rets = aligned.loc[mask, "returns"]
            if len(regime_rets) < 20:
                continue

            lev_rets = regime_rets * self.leverage
            cum = (1 + lev_rets).cumprod()
            n = len(cum)

            if cum.iloc[-1] <= 0:
                cagr = -1.0
                sharpe = 0.0
            else:
                cagr = float(cum.iloc[-1] ** (252 / n) - 1)
                sharpe = (
                    float(lev_rets.mean() / lev_rets.std() * np.sqrt(252))
                    if lev_rets.std() > 0 else 0
                )
            maxdd = float((cum / cum.cummax() - 1).min())

            results[str(regime)] = {
                "n_days": len(regime_rets),
                "cagr": cagr,
                "maxdd": maxdd,
                "sharpe": sharpe,
                "vol": float(lev_rets.std() * np.sqrt(252)),
            }

        return results

    def _correlation_shock(
        self, strategy_returns: dict[str, pd.Series],
    ) -> list[dict]:
        """Test portfolio under correlation breakdown scenarios."""
        df = pd.DataFrame(strategy_returns).dropna()
        if df.empty or df.shape[1] < 2:
            return []

        n_strats = df.shape[1]
        base_corr = df.corr().values
        base_avg_corr = (base_corr.sum() - n_strats) / (n_strats * (n_strats - 1))

        means = df.mean().values
        stds = df.std().values

        scenarios = [
            ("Normal correlations", base_corr.copy()),
            ("Moderate spike (+50%)", np.clip(base_corr * 1.5, -1, 1).copy()),
            ("Severe spike (all → 0.8)", np.full_like(base_corr, 0.8)),
            ("Complete breakdown (all → 1.0)", np.ones_like(base_corr)),
        ]
        # Fix diagonals
        for _, corr in scenarios:
            np.fill_diagonal(corr, 1.0)

        results = []
        for name, corr_matrix in scenarios:
            try:
                # Nearest PSD correlation
                eigvals = np.linalg.eigvalsh(corr_matrix)
                if eigvals.min() < 0:
                    eigvals_fixed, eigvecs = np.linalg.eigh(corr_matrix)
                    eigvals_fixed = np.maximum(eigvals_fixed, 1e-8)
                    corr_matrix = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
                    np.fill_diagonal(corr_matrix, 1.0)

                L = np.linalg.cholesky(corr_matrix)
                z = self.rng.standard_normal((len(df), n_strats))
                correlated = z @ L.T * stds + means

                # Equal weight portfolio
                port_rets = correlated.mean(axis=1) * self.leverage
                cum = np.cumprod(1 + port_rets)
                n = len(cum)
                cagr = float(cum[-1] ** (252 / n) - 1) if cum[-1] > 0 else -1
                maxdd = float((cum / np.maximum.accumulate(cum) - 1).min())
                sharpe = float(
                    np.mean(port_rets) / np.std(port_rets) * np.sqrt(252)
                ) if np.std(port_rets) > 0 else 0

                results.append({
                    "scenario": name,
                    "cagr": cagr,
                    "maxdd": maxdd,
                    "sharpe": sharpe,
                    "avg_corr": float(
                        (corr_matrix.sum() - n_strats) / (n_strats * (n_strats - 1))
                    ),
                })
            except Exception:
                continue

        return results

    def _compute_risk_score(self, result: LeverageStressResult) -> float:
        """Compute overall risk score (0=very risky, 10=safe)."""
        score = 10.0

        # Vol shock resilience
        if result.vol_shock_results:
            severe = [s for s in result.vol_shock_results
                      if s["vol_multiplier"] == 2.0]
            if severe:
                s = severe[0]
                if s["maxdd"] < -0.30:
                    score -= 3
                elif s["maxdd"] < -0.20:
                    score -= 2
                elif s["maxdd"] < -0.15:
                    score -= 1

        # Drawdown scenarios
        if result.drawdown_scenarios:
            worst = min(s["maxdd"] for s in result.drawdown_scenarios)
            if worst < -0.50:
                score -= 3
            elif worst < -0.30:
                score -= 2
            elif worst < -0.20:
                score -= 1

        # Estimation error
        ee = result.estimation_error
        if ee:
            if ee.get("sharpe_ci_low", 0) < 0:
                score -= 2
            elif ee.get("sharpe_ci_low", 0) < 0.5:
                score -= 1

        # Leverage penalty
        if self.leverage > 5:
            score -= 2
        elif self.leverage > 3:
            score -= 1

        return max(0, min(10, score))

    def _generate_recommendations(
        self, result: LeverageStressResult,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        if result.overall_risk_score < 4:
            recs.append(
                f"CRITICAL: Risk score {result.overall_risk_score:.1f}/10 is very low. "
                f"Consider reducing leverage from {self.leverage:.1f}x to "
                f"{max(1, self.leverage * 0.5):.1f}x."
            )

        if result.vol_shock_results:
            moderate = [s for s in result.vol_shock_results
                        if s["vol_multiplier"] == 2.0]
            if moderate and moderate[0]["maxdd"] < -0.25:
                recs.append(
                    f"WARNING: Under 2x vol stress, MaxDD = {moderate[0]['maxdd']:.2%}. "
                    f"This exceeds 25% drawdown threshold for most allocators."
                )

        ee = result.estimation_error
        if ee and ee.get("sharpe_ci_low", 0) < 0:
            recs.append(
                f"WARNING: Sharpe 95% CI includes zero [{ee['sharpe_ci_low']:.3f}, "
                f"{ee['sharpe_ci_high']:.3f}]. Strategy may not be profitable OOS."
            )

        if self.leverage > 3:
            recs.append(
                f"NOTE: At {self.leverage:.1f}x leverage, vol drag = "
                f"{result.base_metrics.get('vol_drag', 0):.2%} annually. "
                f"Ensure this is accounted for in return expectations."
            )

        if result.drawdown_scenarios:
            flash = [s for s in result.drawdown_scenarios
                     if "flash crash" in s["scenario"].lower()]
            if flash and flash[0]["maxdd"] < -0.50:
                recs.append(
                    f"CRITICAL: A 10% flash crash would cause {flash[0]['maxdd']:.2%} "
                    f"drawdown at current leverage. Ensure stop-losses or circuit "
                    f"breakers are in place."
                )

        if not recs:
            recs.append(
                f"Leverage of {self.leverage:.1f}x appears within acceptable "
                f"risk bounds based on stress testing."
            )

        return recs
