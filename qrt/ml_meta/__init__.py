"""
ML Meta-Model Package
=====================
Machine learning models that predict strategy performance and adjust
strategy weights dynamically based on market conditions.

Modules
-------
meta_model          : MetaModel — ensemble of classifiers and regressors
                      that produce expected return, outperformance
                      probability and weight-adjustment signals.
feature_engineering : MetaFeatureEngineer — rolling/lagged feature
                      construction with standardisation.
cross_validation    : TimeSeriesCV — purged time-series cross-validation
                      wrapper around sklearn TimeSeriesSplit.
"""

from __future__ import annotations

from qrt.ml_meta.meta_model import MetaModel
from qrt.ml_meta.feature_engineering import MetaFeatureEngineer
from qrt.ml_meta.cross_validation import TimeSeriesCV

__all__ = [
    "MetaModel",
    "MetaFeatureEngineer",
    "TimeSeriesCV",
]
