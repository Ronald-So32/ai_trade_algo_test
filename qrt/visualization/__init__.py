"""
Visualization and Risk Geometry
================================
Publication-quality plotting utilities for portfolio analytics, risk decomposition,
correlation analysis, risk geometry (PCA / efficient frontier), and regime overlays.
"""

from .performance import PerformanceVisualizer
from .risk_decomposition import RiskDecomposition
from .correlation import CorrelationVisualizer
from .risk_geometry import RiskGeometry
from .regime_plots import RegimeVisualizer

__all__ = [
    "PerformanceVisualizer",
    "RiskDecomposition",
    "CorrelationVisualizer",
    "RiskGeometry",
    "RegimeVisualizer",
]
