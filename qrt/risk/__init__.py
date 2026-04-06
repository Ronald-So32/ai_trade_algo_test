"""
Risk analysis sub-package.

Provides Monte Carlo simulation, drawdown risk management (CDaR/CVaR),
enhanced risk metrics, ablation testing tools, and advanced risk management
(turbulence index, absorption ratio, adaptive stops, CVaR/downside-RP/max-div
allocation, composite risk overlay).
"""

from qrt.risk.monte_carlo import MonteCarloRiskSimulator
from qrt.risk.drawdown_risk import (
    ContinuousDrawdownScaler,
    CDaRRiskBudget,
    compute_cdar,
    compute_cvar,
)
from qrt.risk.enhanced_metrics import compute_full_metrics, metrics_comparison_table
from qrt.risk.advanced_risk import (
    TurbulenceIndex,
    AbsorptionRatio,
    AdaptiveStopLoss,
    DownsideRiskParity,
    CVaROptimizer,
    MaxDiversification,
    CompositeRiskOverlay,
)

__all__ = [
    "MonteCarloRiskSimulator",
    "ContinuousDrawdownScaler",
    "CDaRRiskBudget",
    "compute_cdar",
    "compute_cvar",
    "compute_full_metrics",
    "metrics_comparison_table",
    "TurbulenceIndex",
    "AbsorptionRatio",
    "AdaptiveStopLoss",
    "DownsideRiskParity",
    "CVaROptimizer",
    "MaxDiversification",
    "CompositeRiskOverlay",
]
