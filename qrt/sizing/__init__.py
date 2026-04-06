"""
Position Sizing Module
======================
Provides risk-aware position sizing tools that combine Bayesian estimation
of expected returns with the Kelly criterion and portfolio-level constraints.

Classes
-------
BayesianKellySizer
    Shrinkage-estimated Kelly sizer with fractional Kelly and exposure caps.
"""

from qrt.sizing.bayesian_kelly import BayesianKellySizer

__all__ = [
    "BayesianKellySizer",
]
