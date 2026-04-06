"""
Event-Driven Backtest Engine.

Signal generation -> Order creation -> Execution simulation ->
Position update -> Portfolio update -> PnL calculation.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from .portfolio import Fill, PortfolioState
from .result import BacktestResult
from .trade_ledger import TradeLedger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols / duck-typing interfaces
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    """Any object that accepts price/return data and emits signal weights."""

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        universe: Optional[List[str]],
    ) -> Dict[str, float]:
        """
        Return a dict of {security_id: raw_signal} for the given date.
        Signals are then normalised into target weights by the engine.
        """
        ...

    @property
    def name(self) -> str: ...


class RegimeModel(Protocol):
    """Provides a scalar or dict regime label for a given date."""

    def predict(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Any:
        """Return a regime identifier (int, str, or dict of weights)."""
        ...


class PositionSizer(Protocol):
    """Converts raw signal weights into position-sized target weights."""

    def size(
        self,
        signals: Dict[str, float],
        prices_today: pd.Series,
        portfolio_value: float,
        current_weights: Dict[str, float],
        regime: Any,
    ) -> Dict[str, float]:
        """Return target portfolio weights {security_id: weight}."""
        ...


class CostModel(Protocol):
    """Computes transaction costs for a single fill."""

    def compute(
        self,
        security_id: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
    ) -> Tuple[float, float, float]:
        """Return (commission, slippage, spread_cost) in dollar terms."""
        ...


# ---------------------------------------------------------------------------
# Default cost model
# ---------------------------------------------------------------------------

class _DefaultCostModel:
    """
    Linear cost model parameterised by basis-point rates.
    """

    def __init__(
        self,
        commission_bps: float,
        spread_bps: float,
        slippage_bps: float,
    ) -> None:
        self.commission_rate = commission_bps / 10_000.0
        self.spread_rate = spread_bps / 10_000.0
        self.slippage_rate = slippage_bps / 10_000.0

    def compute(
        self,
        security_id: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
    ) -> Tuple[float, float, float]:
        notional = abs(shares * price)
        commission = notional * self.commission_rate
        spread_cost = notional * self.spread_rate
        slippage = notional * self.slippage_rate
        return commission, slippage, spread_cost


# ---------------------------------------------------------------------------
# Default signal -> weight normaliser
# ---------------------------------------------------------------------------

def _normalize_signals(signals: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw signals to long/short weights via z-score then renormalise
    so that the sum of absolute weights = 1.0.
    """
    if not signals:
        return {}

    values = np.array(list(signals.values()), dtype=float)
    std = values.std()
    if std < 1e-10:
        n = len(values)
        return {k: 1.0 / n for k in signals}

    z = (values - values.mean()) / std
    weights = dict(zip(signals.keys(), z))

    # Normalise so |w|.sum() == 1
    gross = sum(abs(w) for w in weights.values())
    if gross < 1e-10:
        return weights
    return {k: v / gross for k, v in weights.items()}


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven daily backtest engine.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in base currency.
    commission_bps  : float
        Round-trip commission in basis points.
    spread_bps      : float
        Half-spread cost in basis points applied per-side.
    slippage_bps    : float
        Market impact / slippage in basis points.
    max_leverage    : float
        Hard gross leverage ceiling; weights are scaled down if breached.
    rebalance_threshold : float
        Minimum absolute weight change before a rebalance order is generated.
        Set to 0.0 to always rebalance. Default 0.001 (10 bps).
    min_trade_notional : float
        Minimum dollar notional for a trade to be sent to execution.
        Filters tiny round-lots. Default 100.0.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 5.0,
        spread_bps: float = 2.0,
        slippage_bps: float = 3.0,
        max_leverage: float = 2.0,
        rebalance_threshold: float = 0.001,
        min_trade_notional: float = 100.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.max_leverage = max_leverage
        self.rebalance_threshold = rebalance_threshold
        self.min_trade_notional = min_trade_notional

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        strategy: Strategy,
        universe_members: Optional[pd.DataFrame] = None,
        regime_model: Optional[RegimeModel] = None,
        sizer: Optional[PositionSizer] = None,
        cost_model: Optional[CostModel] = None,
    ) -> BacktestResult:
        """
        Run full event-driven backtest.

        For each trading day:
          1. Determine the investable universe.
          2. Generate signals from strategy.
          3. Apply regime filter / scaling if provided.
          4. Compute target weights (via sizer or default normaliser).
          5. Apply leverage constraint.
          6. Generate orders (diff between current and target weights).
          7. Simulate execution with cost model.
          8. Update positions and portfolio state.
          9. Record PnL, weights and state.

        Parameters
        ----------
        prices           : DataFrame (DatetimeIndex x security_id) of close prices.
        returns          : DataFrame (DatetimeIndex x security_id) of daily returns.
        strategy         : strategy object implementing `generate_signals`.
        universe_members : optional boolean DataFrame same shape as prices.
                           True = security is eligible on that date.
        regime_model     : optional model for regime filtering.
        sizer            : optional position sizer; default uses z-score normalisation.
        cost_model       : optional cost model; default uses engine bps parameters.

        Returns
        -------
        BacktestResult
        """
        # Resolve cost model
        _cost_model: CostModel = cost_model or _DefaultCostModel(
            self.commission_bps, self.spread_bps, self.slippage_bps
        )

        # Align prices and returns
        prices = prices.sort_index()
        returns = returns.sort_index()
        common_dates = prices.index.intersection(returns.index)
        prices = prices.loc[common_dates]
        returns = returns.loc[common_dates]

        # Initialise portfolio and ledger
        portfolio = PortfolioState(
            initial_capital=self.initial_capital,
            max_leverage=self.max_leverage,
        )
        ledger = TradeLedger()

        # Output accumulators
        portfolio_values: List[float] = []
        daily_returns: List[float] = []
        weights_rows: List[Dict[str, float]] = []
        snapshots: List[dict] = []
        dates: List[pd.Timestamp] = []

        prev_value = self.initial_capital

        strategy_name = getattr(strategy, "name", type(strategy).__name__)

        for i, date in enumerate(prices.index):
            prices_today_series: pd.Series = prices.loc[date].dropna()
            prices_today: Dict[str, float] = prices_today_series.to_dict()

            # ----------------------------------------------------------------
            # 1. Determine eligible universe
            # ----------------------------------------------------------------
            if universe_members is not None and date in universe_members.index:
                mask = universe_members.loc[date]
                universe = [s for s in prices_today.keys() if mask.get(s, False)]
            else:
                universe = list(prices_today.keys())

            # ----------------------------------------------------------------
            # 2. Generate signals
            # ----------------------------------------------------------------
            try:
                raw_signals: Dict[str, float] = strategy.generate_signals(
                    prices=prices.iloc[: i + 1],
                    returns=returns.iloc[: i + 1],
                    date=date,
                    universe=universe,
                )
            except Exception as exc:
                logger.warning("Strategy signal generation failed on %s: %s", date, exc)
                raw_signals = {}

            # Keep only liquid / eligible securities
            raw_signals = {k: v for k, v in raw_signals.items() if k in universe and k in prices_today}

            # ----------------------------------------------------------------
            # 3. Regime filter
            # ----------------------------------------------------------------
            regime: Any = None
            if regime_model is not None:
                try:
                    regime = regime_model.predict(
                        date=date,
                        prices=prices.iloc[: i + 1],
                        returns=returns.iloc[: i + 1],
                    )
                    raw_signals = self._apply_regime_filter(raw_signals, regime)
                except Exception as exc:
                    logger.warning("Regime model failed on %s: %s", date, exc)

            # ----------------------------------------------------------------
            # 4. Compute target weights
            # ----------------------------------------------------------------
            portfolio_value = portfolio.mark_to_market(prices_today)
            current_weights = portfolio.get_weights(prices_today)

            if sizer is not None:
                try:
                    target_weights = sizer.size(
                        signals=raw_signals,
                        prices_today=prices_today_series,
                        portfolio_value=portfolio_value,
                        current_weights=current_weights,
                        regime=regime,
                    )
                except Exception as exc:
                    logger.warning("Position sizer failed on %s: %s", date, exc)
                    target_weights = _normalize_signals(raw_signals)
            else:
                target_weights = _normalize_signals(raw_signals)

            # ----------------------------------------------------------------
            # 5. Enforce leverage constraint
            # ----------------------------------------------------------------
            target_weights = portfolio.scale_to_leverage_limit(target_weights, prices_today)

            # ----------------------------------------------------------------
            # 6. Generate orders
            # ----------------------------------------------------------------
            orders = self._generate_orders(
                current_weights=current_weights,
                target_weights=target_weights,
                portfolio_value=portfolio_value,
                prices_today=prices_today,
            )

            # ----------------------------------------------------------------
            # 7. Execute orders
            # ----------------------------------------------------------------
            fills = self._execute_orders(
                orders=orders,
                prices_today=prices_today,
                cost_model=_cost_model,
                date=date,
                strategy_name=strategy_name,
                raw_signals=raw_signals,
            )

            # ----------------------------------------------------------------
            # 8. Update portfolio state
            # ----------------------------------------------------------------
            self._update_portfolio(
                portfolio=portfolio,
                fills=fills,
                ledger=ledger,
                prices_today=prices_today,
                date=date,
                strategy_name=strategy_name,
                raw_signals=raw_signals,
            )

            # ----------------------------------------------------------------
            # 9. Mark-to-market and record
            # ----------------------------------------------------------------
            pv = portfolio.mark_to_market(prices_today)
            daily_ret = (pv / prev_value - 1.0) if prev_value != 0 else 0.0
            prev_value = pv

            snap = portfolio.snapshot(prices_today, timestamp=date)
            snapshots.append(snap)

            portfolio_values.append(pv)
            daily_returns.append(daily_ret)
            weights_rows.append(portfolio.get_weights(prices_today))
            dates.append(date)

        # ------------------------------------------------------------------
        # Build result objects
        # ------------------------------------------------------------------
        pv_series = pd.Series(portfolio_values, index=dates, name="portfolio_value")
        ret_series = pd.Series(daily_returns, index=dates, name="returns")
        weights_df = pd.DataFrame(weights_rows, index=dates).fillna(0.0)

        return BacktestResult(
            portfolio_values=pv_series,
            returns=ret_series,
            weights_history=weights_df,
            trade_ledger=ledger,
            portfolio_snapshots=snapshots,
        )

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def _generate_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices_today: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Create a dict of {security_id: dollar_notional} orders representing the
        difference between current and target weights.

        Positive values = buy (increase exposure).
        Negative values = sell (decrease exposure).

        Small changes below `rebalance_threshold` or `min_trade_notional`
        are suppressed to avoid excessive churn.
        """
        orders: Dict[str, float] = {}

        all_securities = set(current_weights.keys()) | set(target_weights.keys())

        for sid in all_securities:
            current_w = current_weights.get(sid, 0.0)
            target_w = target_weights.get(sid, 0.0)
            delta_w = target_w - current_w

            # Threshold filter
            if abs(delta_w) < self.rebalance_threshold:
                continue

            dollar_notional = delta_w * portfolio_value
            if abs(dollar_notional) < self.min_trade_notional:
                continue

            orders[sid] = dollar_notional

        return orders

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_orders(
        self,
        orders: Dict[str, float],
        prices_today: Dict[str, float],
        cost_model: CostModel,
        date: pd.Timestamp,
        strategy_name: str,
        raw_signals: Dict[str, float],
    ) -> List[Fill]:
        """
        Simulate order execution with market impact.

        Slippage is applied directionally: buys fill slightly higher,
        sells fill slightly lower (adverse fill).
        """
        fills: List[Fill] = []
        slippage_rate = self.slippage_bps / 10_000.0

        for sid, dollar_notional in orders.items():
            price = prices_today.get(sid)
            if price is None or price <= 0:
                logger.debug("No price for %s on %s; skipping order.", sid, date)
                continue

            # Adverse slippage: buy at higher price, sell at lower
            direction = np.sign(dollar_notional)
            exec_price = price * (1.0 + direction * slippage_rate)

            shares = dollar_notional / exec_price  # signed shares

            commission, slippage_cost, spread_cost = cost_model.compute(
                security_id=sid,
                shares=shares,
                price=exec_price,
                date=date,
            )

            fill = Fill(
                security_id=sid,
                shares=shares,
                executed_price=exec_price,
                commission=commission,
                slippage=slippage_cost,
                spread_cost=spread_cost,
            )
            fills.append(fill)

        return fills

    # ------------------------------------------------------------------
    # Portfolio update
    # ------------------------------------------------------------------

    def _update_portfolio(
        self,
        portfolio: PortfolioState,
        fills: List[Fill],
        ledger: TradeLedger,
        prices_today: Dict[str, float],
        date: pd.Timestamp,
        strategy_name: str,
        raw_signals: Dict[str, float],
    ) -> None:
        """
        Apply fills to portfolio state and record each fill in the ledger.
        """
        batch_realised = portfolio.update(fills, prices_today)

        for fill in fills:
            sid = fill.security_id
            shares = fill.shares

            if shares > 0:
                order_type = "BUY"
                side = "OPEN_LONG" if portfolio.positions.get(sid, 0) >= 0 else "CLOSE_SHORT"
            else:
                order_type = "SELL"
                side = "OPEN_SHORT" if portfolio.positions.get(sid, 0) < 0 else "CLOSE_LONG"

            ledger.append_trade(
                timestamp=date,
                security_id=sid,
                strategy=strategy_name,
                order_type=order_type,
                signal_strength=raw_signals.get(sid, 0.0),
                executed_price=fill.executed_price,
                position_size=fill.shares,
                commission=fill.commission,
                slippage=fill.slippage,
                spread_cost=fill.spread_cost,
                pnl=batch_realised.get(sid, 0.0),
                side=side,
            )

    # ------------------------------------------------------------------
    # Regime helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_regime_filter(
        signals: Dict[str, float],
        regime: Any,
    ) -> Dict[str, float]:
        """
        Scale or zero-out signals based on the regime output.

        Accepted regime formats:
        - float / int : scalar multiplier applied to all signals
        - dict with key "scale" : multiplier
        - dict with key "allowed_sides" : "long_only", "short_only", "flat", or "both"
        - str "flat" : zero out all signals
        """
        if regime is None:
            return signals

        if isinstance(regime, (int, float)):
            scale = float(regime)
            return {k: v * scale for k, v in signals.items()}

        if isinstance(regime, str):
            if regime.lower() == "flat":
                return {}
            return signals  # unknown string; pass through

        if isinstance(regime, dict):
            scale = regime.get("scale", 1.0)
            allowed_sides = regime.get("allowed_sides", "both")

            scaled = {k: v * scale for k, v in signals.items()}

            if allowed_sides == "long_only":
                return {k: v for k, v in scaled.items() if v >= 0}
            if allowed_sides == "short_only":
                return {k: v for k, v in scaled.items() if v <= 0}
            if allowed_sides == "flat":
                return {}
            return scaled

        return signals
