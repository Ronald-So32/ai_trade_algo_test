"""
qrt.strategies
==============
Strategy library for the QRT quantitative research platform.

All strategies subclass ``Strategy`` (defined in ``base.py``) and implement:
  - ``generate_signals(prices, returns, **kwargs) -> pd.DataFrame``
  - ``compute_weights(signals, **kwargs) -> pd.DataFrame``
  - ``backtest_summary(weights, returns) -> dict``  (inherited)

Quick reference
---------------
Strategy                  | File
--------------------------|------------------------------
Strategy (ABC)            | base.py
TimeSeriesMomentum        | time_series_momentum.py
CrossSectionalMomentum    | cross_sectional_momentum.py
MeanReversion             | mean_reversion.py
DistancePairs             | distance_pairs.py
KalmanPairs               | kalman_pairs.py
VolatilityBreakout        | volatility_breakout.py
CarryStrategy             | carry.py
FactorMomentum            | factor_momentum.py
PCAStatArb                | pca_stat_arb.py
VolatilityManagedOverlay  | vol_managed.py
"""

from .base import Strategy
from .time_series_momentum import TimeSeriesMomentum
from .cross_sectional_momentum import CrossSectionalMomentum
from .mean_reversion import MeanReversion
from .distance_pairs import DistancePairs
from .kalman_pairs import KalmanPairs
from .volatility_breakout import VolatilityBreakout
from .carry import CarryStrategy
from .factor_momentum import FactorMomentum
from .pca_stat_arb import PCAStatArb
from .vol_managed import VolatilityManagedOverlay
from .pead import PEAD
from .residual_momentum import ResidualMomentum
from .low_risk_bab import LowRiskBAB
from .ml_alpha_strategy import MLAlphaStrategy
from .short_term_reversal import ShortTermReversal
from .vol_risk_premium import VolRiskPremium
from .value_momentum import FiftyTwoWeekHigh
from .residual_reversal import ResidualReversal

__all__ = [
    "Strategy",
    "TimeSeriesMomentum",
    "CrossSectionalMomentum",
    "MeanReversion",
    "DistancePairs",
    "KalmanPairs",
    "VolatilityBreakout",
    "CarryStrategy",
    "FactorMomentum",
    "PCAStatArb",
    "VolatilityManagedOverlay",
    "PEAD",
    "ResidualMomentum",
    "LowRiskBAB",
    "MLAlphaStrategy",
    "ShortTermReversal",
    "VolRiskPremium",
]

# Registry: name → class, useful for config-driven instantiation
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "time_series_momentum": TimeSeriesMomentum,
    "cross_sectional_momentum": CrossSectionalMomentum,
    "mean_reversion": MeanReversion,
    "distance_pairs": DistancePairs,
    "kalman_pairs": KalmanPairs,
    "volatility_breakout": VolatilityBreakout,
    "carry": CarryStrategy,
    "factor_momentum": FactorMomentum,
    "pca_stat_arb": PCAStatArb,
    "vol_managed": VolatilityManagedOverlay,
    "pead": PEAD,
    "residual_momentum": ResidualMomentum,
    "low_risk_bab": LowRiskBAB,
    "ml_alpha": MLAlphaStrategy,
    "short_term_reversal": ShortTermReversal,
    "vol_risk_premium": VolRiskPremium,
    "fifty_two_week_high": FiftyTwoWeekHigh,
    "residual_reversal": ResidualReversal,
}


def get_strategy(name: str, **params) -> Strategy:
    """
    Factory function: instantiate a strategy by registry name.

    Parameters
    ----------
    name : str
        Registry key (see ``STRATEGY_REGISTRY``).
    **params
        Keyword arguments forwarded to the strategy constructor.

    Returns
    -------
    Strategy instance.

    Raises
    ------
    KeyError
        If *name* is not found in the registry.

    Examples
    --------
    >>> strat = get_strategy("time_series_momentum", lookback=126)
    >>> strat.name
    'TimeSeriesMomentum'
    """
    key = name.lower().replace("-", "_")
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(
            f"Strategy '{name}' not found. Available strategies: {available}"
        )
    return STRATEGY_REGISTRY[key](**params)
