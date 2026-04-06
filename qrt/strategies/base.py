from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement ``generate_signals`` and ``compute_weights``.
    Subclasses should also define a ``RESEARCH_GROUNDING`` class attribute
    documenting the academic basis, historical evidence, implementation risks,
    and realistic expectations for the strategy.
    """

    # Subclasses should override this with their research grounding.
    # This ensures no strategy is described as "guaranteed" or "proven".
    RESEARCH_GROUNDING: dict[str, str] = {
        "academic_basis": "Not specified",
        "historical_evidence": "Not specified",
        "implementation_risks": "Not specified",
        "realistic_expectations": (
            "This strategy represents a research-supported return premium. "
            "Past performance does not guarantee future results. "
            "Drawdowns, extended underperformance, and regime sensitivity are expected."
        ),
    }

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return DataFrame of signals, same shape as prices. Values in [-1, 1]."""
        pass

    @abstractmethod
    def compute_weights(self, signals: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Convert signals to target portfolio weights."""
        pass

    def backtest_summary(self, weights: pd.DataFrame, returns: pd.DataFrame) -> dict:
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1).mean()
        return {
            "strategy": self.name,
            "total_return": (1 + portfolio_returns).prod() - 1,
            "annualized_return": portfolio_returns.mean() * 252,
            "volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe": (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0,
            "max_drawdown": self._max_drawdown(portfolio_returns),
            "avg_turnover": turnover,
        }

    @staticmethod
    def apply_drawdown_cap(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        max_dd: float = 0.20,
        cooldown: int = 21,
        reduction: float = 1.0,
        method: str = "continuous",
    ) -> pd.DataFrame:
        """Scale weights to keep MaxDD within the target.

        Supports two methods:
        - "continuous" (default, recommended): smooth drawdown-aware scaling
          that reduces exposure proportionally as drawdown deepens, avoiding
          the whipsaw of hard stop-outs.  Based on CDaR research.
        - "binary": legacy circuit breaker with hard on/off switching.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates x assets).
        returns : pd.DataFrame
            Asset returns aligned with weights.
        max_dd : float
            Maximum tolerable drawdown (e.g. 0.20 = 20%).
        cooldown : int
            Days to keep reduced exposure after breaker triggers (binary only).
        reduction : float
            Fraction of exposure to cut (binary: 1.0 = full cut to zero).
        method : str
            "continuous" (CDaR-aware smooth scaling) or "binary" (legacy).

        Returns
        -------
        pd.DataFrame
            Adjusted weights with drawdown protection applied.
        """
        if method == "continuous":
            from qrt.risk.drawdown_risk import ContinuousDrawdownScaler
            scaler = ContinuousDrawdownScaler(max_dd=max_dd)
            return scaler.apply(weights, returns)

        # Legacy binary circuit breaker
        adjusted = weights.copy()
        trigger = max_dd * 0.75  # trigger early to absorb single-day shocks

        for _pass in range(5):
            strat_returns = (adjusted.shift(1) * returns).sum(axis=1)
            cum = (1 + strat_returns).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max

            actual_max_dd = abs(dd.min())
            if actual_max_dd <= max_dd:
                break

            scale = pd.Series(1.0, index=adjusted.index)
            days_since_trigger = cooldown + 1

            for i in range(len(dd)):
                if dd.iloc[i] < -trigger:
                    days_since_trigger = 0
                if days_since_trigger <= cooldown:
                    scale.iloc[i] = 1.0 - reduction
                days_since_trigger += 1

            adjusted = adjusted.mul(scale, axis=0)
            # Tighten trigger for next pass
            trigger = trigger * 0.8

        return adjusted

    @staticmethod
    def apply_regime_scaling(
        weights: pd.DataFrame,
        crisis_probs: pd.Series,
        soft_start: float = 0.3,
        floor: float = 0.2,
    ) -> pd.DataFrame:
        """
        Scale portfolio weights by (1 - crisis_probability).

        Based on Ang & Bekaert (2002): regime probabilities should
        continuously modulate exposure rather than using hard state switches.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights (dates x assets).
        crisis_probs : pd.Series
            Posterior probability of crisis state per date (0 to 1).
        soft_start : float
            Crisis probability above which scaling begins (default 0.3).
        floor : float
            Minimum exposure multiplier (default 0.2).

        Returns
        -------
        pd.DataFrame
            Regime-scaled weights.
        """
        crisis_probs = crisis_probs.reindex(weights.index).fillna(0.0)

        # Continuous scaling: 1.0 when P(crisis) < soft_start,
        # linearly decreasing to floor when P(crisis) = 1.0
        scale = pd.Series(1.0, index=weights.index)
        above_threshold = crisis_probs > soft_start
        if above_threshold.any():
            # Linear interpolation from 1.0 at soft_start to floor at 1.0
            scale[above_threshold] = 1.0 - (1.0 - floor) * (
                (crisis_probs[above_threshold] - soft_start) / (1.0 - soft_start)
            )
            scale = scale.clip(lower=floor, upper=1.0)

        return weights.mul(scale, axis=0)

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()
