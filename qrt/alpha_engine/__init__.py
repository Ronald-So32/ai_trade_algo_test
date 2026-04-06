"""
Alpha Engine - Automated Alpha Discovery System
================================================

A systematic framework for generating, evaluating, and filtering candidate
alpha signals from price, return, and volume data.

Modules
-------
signal_generator
    Generates candidate alpha signals from combinations of features using
    price-based, momentum, volatility, mean-reversion, and cross-sectional
    techniques.

signal_evaluator
    Evaluates each candidate signal's quality via Sharpe ratio, drawdown,
    information coefficient, regime performance, and cost sensitivity.

signal_filter
    Filters signals by robustness criteria including Sharpe thresholds,
    drawdown limits, regime robustness, and correlation with existing
    strategies.

alpha_research
    Orchestrator (AlphaResearchEngine) that ties generation, evaluation,
    and filtering into a single discovery pipeline and returns a structured
    AlphaResearchResult.

Typical usage
-------------
    from alpha_engine import AlphaResearchEngine

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

from qrt.alpha_engine.signal_generator import SignalGenerator
from qrt.alpha_engine.signal_evaluator import SignalEvaluator, SignalMetrics
from qrt.alpha_engine.signal_filter import SignalFilter
from qrt.alpha_engine.alpha_research import AlphaResearchEngine, AlphaResearchResult

__all__ = [
    "SignalGenerator",
    "SignalEvaluator",
    "SignalMetrics",
    "SignalFilter",
    "AlphaResearchEngine",
    "AlphaResearchResult",
]

__version__ = "0.1.0"
