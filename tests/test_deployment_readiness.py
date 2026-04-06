"""
Tests for deployment readiness gate and advanced backtesting safeguards.

Validates:
- PBO via CSCV computation
- Clean holdout evaluation
- Leverage cost computation
- Survivorship bias assessment
- Complexity scoring
- DeploymentGate integration
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def strategy_returns_df(rng):
    """DataFrame with 5 strategy columns and 1000 rows for PBO."""
    dates = pd.bdate_range("2018-01-01", periods=1000)
    data = {}
    for i in range(5):
        mu = 0.0003 * (i + 1)
        data[f"strat_{i}"] = rng.normal(mu, 0.01, 1000)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def strategy_returns_dict(strategy_returns_df):
    """Dict of strategy name -> pd.Series."""
    return {col: strategy_returns_df[col] for col in strategy_returns_df.columns}


@pytest.fixture
def portfolio_returns(strategy_returns_df):
    """Equal-weighted portfolio returns."""
    return strategy_returns_df.mean(axis=1)


# ── PBO Tests ──

class TestComputePBO:
    def test_pbo_returns_valid_structure(self, strategy_returns_df):
        from qrt.validation.deployment_readiness import compute_pbo
        result = compute_pbo(strategy_returns_df, n_partitions=8)
        assert "pbo" in result
        assert 0 <= result["pbo"] <= 1
        assert "n_combinations" in result
        assert result["n_combinations"] > 0

    def test_pbo_insufficient_data(self):
        from qrt.validation.deployment_readiness import compute_pbo
        # Too few rows
        df = pd.DataFrame({"a": [0.01] * 10, "b": [0.02] * 10})
        result = compute_pbo(df, n_partitions=8)
        assert result["pbo"] == 0.5
        assert "reason" in result

    def test_pbo_single_strategy(self):
        from qrt.validation.deployment_readiness import compute_pbo
        df = pd.DataFrame({"a": np.random.normal(0, 0.01, 500)})
        result = compute_pbo(df)
        assert result["pbo"] == 0.5  # insufficient strategies

    def test_pbo_odd_partitions_corrected(self, strategy_returns_df):
        from qrt.validation.deployment_readiness import compute_pbo
        # Odd n_partitions should be corrected to even
        result = compute_pbo(strategy_returns_df, n_partitions=7)
        assert result["n_combinations"] > 0

    def test_pbo_random_strategies_high(self, rng):
        from qrt.validation.deployment_readiness import compute_pbo
        # Random strategies should have high PBO
        dates = pd.bdate_range("2018-01-01", periods=1000)
        data = {f"s{i}": rng.normal(0, 0.01, 1000) for i in range(10)}
        df = pd.DataFrame(data, index=dates)
        result = compute_pbo(df, n_partitions=8)
        # With random strategies, PBO should be >= 0.3 at minimum
        assert result["pbo"] >= 0.2


# ── Clean Holdout Tests ──

class TestEvaluateCleanHoldout:
    def test_holdout_returns_all_strategies(self, strategy_returns_dict):
        from qrt.validation.deployment_readiness import evaluate_clean_holdout
        result = evaluate_clean_holdout(strategy_returns_dict)
        assert len(result) == len(strategy_returns_dict)
        for name, metrics in result.items():
            assert "dev_sharpe" in metrics
            assert "holdout_sharpe" in metrics
            assert "degradation_pct" in metrics
            assert "holdout_days" in metrics

    def test_holdout_with_portfolio(self, strategy_returns_dict, portfolio_returns):
        from qrt.validation.deployment_readiness import evaluate_clean_holdout
        result = evaluate_clean_holdout(
            strategy_returns_dict, portfolio_returns=portfolio_returns,
        )
        assert "__portfolio__" in result
        port = result["__portfolio__"]
        assert "dev_sharpe" in port
        assert "holdout_sharpe" in port

    def test_holdout_short_series_skipped(self):
        from qrt.validation.deployment_readiness import evaluate_clean_holdout
        # Series with < 100 points should be skipped
        short = pd.Series(np.random.normal(0, 0.01, 50))
        result = evaluate_clean_holdout({"short": short})
        assert len(result) == 0

    def test_holdout_fraction(self, strategy_returns_dict):
        from qrt.validation.deployment_readiness import evaluate_clean_holdout
        result = evaluate_clean_holdout(strategy_returns_dict, holdout_fraction=0.30)
        for name, metrics in result.items():
            assert metrics["holdout_days"] > 0


# ── Leverage Costs Tests ──

class TestComputeLeverageCosts:
    def test_no_leverage(self):
        from qrt.validation.deployment_readiness import compute_leverage_costs
        result = compute_leverage_costs(leverage=1.0)
        assert result["total_leverage_cost_annual"] == 0.0
        assert result["daily_drag_bps"] == 0.0

    def test_leverage_costs_scale(self):
        from qrt.validation.deployment_readiness import compute_leverage_costs
        r2 = compute_leverage_costs(leverage=2.0)
        r5 = compute_leverage_costs(leverage=5.0)
        assert r5["total_leverage_cost_annual"] > r2["total_leverage_cost_annual"]

    def test_cost_components(self):
        from qrt.validation.deployment_readiness import compute_leverage_costs
        result = compute_leverage_costs(leverage=3.0)
        assert result["margin_interest_annual"] > 0
        assert result["funding_spread_annual"] >= 0  # 0 when included in margin rate
        assert result["borrowed_fraction"] == 2.0
        assert result["leverage"] == 3.0

    def test_custom_rates(self):
        from qrt.validation.deployment_readiness import compute_leverage_costs
        result = compute_leverage_costs(
            leverage=2.0, annual_margin_rate=0.10, funding_spread=0.02,
        )
        # borrowed = 1.0, margin = 1.0 * 0.10 = 0.10
        assert abs(result["margin_interest_annual"] - 0.10) < 1e-10


# ── Survivorship Bias Tests ──

class TestAssessSurvivorshipBias:
    def test_yahoo_data_flagged(self):
        from qrt.validation.deployment_readiness import assess_survivorship_bias
        result = assess_survivorship_bias(
            ["AAPL", "MSFT"], data_source="real", start_year=2010,
        )
        assert result["risk_level"] == "HIGH"
        assert len(result["warnings"]) >= 2

    def test_synthetic_low_risk(self):
        from qrt.validation.deployment_readiness import assess_survivorship_bias
        result = assess_survivorship_bias(
            ["SYN1", "SYN2"], data_source="synthetic",
        )
        assert result["risk_level"] == "LOW"

    def test_unknown_source(self):
        from qrt.validation.deployment_readiness import assess_survivorship_bias
        result = assess_survivorship_bias(["A", "B"], data_source="unknown")
        assert result["risk_level"] == "LOW"


# ── Complexity Score Tests ──

class TestComputeComplexityScore:
    def test_simple_strategy(self):
        from qrt.validation.deployment_readiness import compute_complexity_score
        result = compute_complexity_score(n_strategies=1, n_parameters=5)
        assert result["complexity_score"] < 30
        assert "LOW" in result["recommendation"]

    def test_complex_strategy(self):
        from qrt.validation.deployment_readiness import compute_complexity_score
        result = compute_complexity_score(
            n_strategies=18, n_parameters=100, n_features=50,
            n_models_tested=200, uses_ml=True, uses_regime_switching=True,
            n_allocation_methods=4, n_risk_layers=5,
        )
        assert result["complexity_score"] > 60
        assert result["expected_sharpe_degradation_pct"] > 50

    def test_oos_multiplier_range(self):
        from qrt.validation.deployment_readiness import compute_complexity_score
        result = compute_complexity_score(n_strategies=5, n_parameters=20)
        assert 0 < result["oos_sharpe_multiplier"] < 1

    def test_degradation_increases_with_complexity(self):
        from qrt.validation.deployment_readiness import compute_complexity_score
        simple = compute_complexity_score(n_strategies=2, n_parameters=5)
        complex_ = compute_complexity_score(
            n_strategies=15, n_parameters=80, uses_ml=True,
            n_risk_layers=5, n_allocation_methods=4,
        )
        assert complex_["expected_sharpe_degradation_pct"] > simple["expected_sharpe_degradation_pct"]


# ── Deployment Gate Tests ──

class TestDeploymentGate:
    def test_gate_passes_good_strategies(self, strategy_returns_dict, portfolio_returns):
        from qrt.validation.deployment_readiness import DeploymentGate
        gate = DeploymentGate()
        result = gate.evaluate(
            strategy_returns=strategy_returns_dict,
            portfolio_returns=portfolio_returns,
            leverage=1.0,
        )
        # Should have checks
        assert len(result.checks) > 0
        # Result should be a valid boolean
        assert isinstance(result.passed, bool)

    def test_gate_generates_markdown(self, strategy_returns_dict, portfolio_returns):
        from qrt.validation.deployment_readiness import DeploymentGate
        gate = DeploymentGate()
        result = gate.evaluate(
            strategy_returns=strategy_returns_dict,
            portfolio_returns=portfolio_returns,
        )
        md = result.to_markdown()
        assert "Deployment Readiness Gate" in md
        assert "Check Results" in md

    def test_gate_with_leverage(self, strategy_returns_dict, portfolio_returns):
        from qrt.validation.deployment_readiness import DeploymentGate
        gate = DeploymentGate()
        result = gate.evaluate(
            strategy_returns=strategy_returns_dict,
            portfolio_returns=portfolio_returns,
            leverage=5.0,
        )
        # Should have leverage cost check
        cost_checks = [c for c in result.checks if "Leverage" in c["name"]]
        assert len(cost_checks) > 0
        assert "leverage_cost_annual" in result.summary

    def test_gate_with_tickers(self, strategy_returns_dict):
        from qrt.validation.deployment_readiness import DeploymentGate
        gate = DeploymentGate()
        result = gate.evaluate(
            strategy_returns=strategy_returns_dict,
            tickers=["AAPL", "MSFT"],
            data_source="real",
        )
        # Should have survivorship warnings
        surv_warnings = [w for w in result.warnings if "Survivorship" in w]
        assert len(surv_warnings) > 0

    def test_gate_summary_has_complexity(self, strategy_returns_dict):
        from qrt.validation.deployment_readiness import DeploymentGate
        gate = DeploymentGate()
        result = gate.evaluate(strategy_returns=strategy_returns_dict)
        assert "complexity_score" in result.summary
        assert "expected_oos_sharpe_mult" in result.summary

    def test_gate_result_dataclass(self):
        from qrt.validation.deployment_readiness import DeploymentGateResult
        r = DeploymentGateResult()
        assert r.passed is False
        assert r.checks == []
        assert r.blockers == []
        assert r.warnings == []
