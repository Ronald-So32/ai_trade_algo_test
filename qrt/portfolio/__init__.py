"""
Portfolio construction module.

Provides risk parity allocation (with Ledoit-Wolf shrinkage), volatility
targeting, HERC hierarchical allocation, momentum risk management,
combined portfolio optimization, and enhanced allocation with advanced
risk techniques (CVaR, downside RP, max diversification, systemic overlays).
"""

from .risk_parity import RiskParityAllocator
from .vol_targeting import VolatilityTargeter
from .optimizer import PortfolioOptimizer
from .shrinkage import ShrinkageEstimator
from .hierarchical import HERCAllocator
from .momentum_risk import MomentumRiskManager
from .enhanced_allocation import EnhancedAllocator, AllocationComparator

__all__ = [
    "RiskParityAllocator",
    "VolatilityTargeter",
    "PortfolioOptimizer",
    "ShrinkageEstimator",
    "HERCAllocator",
    "MomentumRiskManager",
    "EnhancedAllocator",
    "AllocationComparator",
]
