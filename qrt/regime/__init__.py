"""
Regime Detection Module
=======================
Provides market regime classification tools for identifying structural
shifts in volatility, correlation, and trend dynamics.

Classes
-------
VolatilityRegimeClassifier
    Percentile-based rolling volatility regime classifier with 4 states.
HMMRegimeDetector
    Hidden Markov Model regime detector using multi-feature Gaussian emissions.
"""

from qrt.regime.volatility_regime import VolatilityRegimeClassifier
from qrt.regime.hmm_regime import HMMRegimeDetector

__all__ = [
    "VolatilityRegimeClassifier",
    "HMMRegimeDetector",
]
