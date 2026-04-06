"""
Walk-forward testing module.

Provides rolling in-sample / out-of-sample testing infrastructure and
result aggregation for quantitative strategies.
"""

from .walk_forward import WalkForwardTester
from .result import WalkForwardResult

__all__ = [
    "WalkForwardTester",
    "WalkForwardResult",
]
