"""Tests for the BacktestEngine and related portfolio/ledger components."""
import pytest
import numpy as np
import pandas as pd

from qrt.backtest.engine import BacktestEngine
from qrt.backtest.portfolio import PortfolioState, Fill
from qrt.backtest.trade_ledger import TradeLedger
from qrt.backtest.result import BacktestResult


# ---------------------------------------------------------------------------
# Minimal strategy stub
# ---------------------------------------------------------------------------

class _SimpleSignalStrategy:
    """Assigns a fixed signal proportional to trailing 5-day return rank."""

    name = "SimpleTestStrategy"

    def generate_signals(self, prices, returns, date, universe):
        if len(prices) < 6:
            return {}
        trailing = prices.iloc[-1] / prices.iloc[-6] - 1.0
        trailing = trailing.dropna()
        if trailing.empty:
            return {}
        return {col: float(trailing[col]) for col in universe if col in trailing.index}


class _ZeroSignalStrategy:
    """Always returns zero signals — used to test initial capital invariant."""

    name = "ZeroStrategy"

    def generate_signals(self, prices, returns, date, universe):
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return BacktestEngine(
        initial_capital=1_000_000.0,
        commission_bps=5.0,
        spread_bps=2.0,
        slippage_bps=3.0,
    )


@pytest.fixture
def engine_no_costs():
    return BacktestEngine(
        initial_capital=1_000_000.0,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        rebalance_threshold=0.0,
    )


# ---------------------------------------------------------------------------
# test_backtest_runs
# ---------------------------------------------------------------------------

def test_backtest_runs(engine, sample_prices, sample_returns):
    """BacktestEngine.run() completes without raising exceptions."""
    strategy = _SimpleSignalStrategy()
    result = engine.run(prices=sample_prices, returns=sample_returns, strategy=strategy)
    assert result is not None
    assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# test_initial_capital
# ---------------------------------------------------------------------------

def test_initial_capital(engine, sample_prices, sample_returns):
    """Portfolio value on day 0 should equal initial_capital (before any fills)."""
    strategy = _ZeroSignalStrategy()
    result = engine.run(prices=sample_prices, returns=sample_returns, strategy=strategy)

    first_value = result.portfolio_values.iloc[0]
    assert abs(first_value - engine.initial_capital) < 1.0, (
        f"First portfolio value {first_value:.2f} != initial_capital {engine.initial_capital:.2f}"
    )


# ---------------------------------------------------------------------------
# test_no_lookahead
# ---------------------------------------------------------------------------

def test_no_lookahead(sample_prices, sample_returns):
    """
    The strategy's generate_signals call at date t should only receive
    data up to index t+1 (prices.iloc[:i+1]).

    We verify this by using a spy strategy that records the maximum index
    length it ever sees and asserting it never exceeds the number of
    elapsed days + 1.
    """
    seen_lengths = []

    class _SpyStrategy:
        name = "SpyStrategy"

        def generate_signals(self, prices, returns, date, universe):
            seen_lengths.append(len(prices))
            return {}

    engine = BacktestEngine(initial_capital=1_000_000.0)
    engine.run(prices=sample_prices, returns=sample_returns, strategy=_SpyStrategy())

    n_dates = len(sample_prices)
    for step, length in enumerate(seen_lengths, start=1):
        assert length == step, (
            f"At step {step}, strategy received {length} rows — potential look-ahead"
        )


# ---------------------------------------------------------------------------
# test_deterministic
# ---------------------------------------------------------------------------

def test_deterministic(engine, sample_prices, sample_returns):
    """Running the backtest twice with identical inputs produces identical results."""
    strategy = _SimpleSignalStrategy()
    result_a = engine.run(prices=sample_prices, returns=sample_returns, strategy=strategy)
    result_b = engine.run(prices=sample_prices, returns=sample_returns, strategy=strategy)

    pd.testing.assert_series_equal(
        result_a.portfolio_values,
        result_b.portfolio_values,
        check_names=False,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# test_transaction_costs_reduce_returns
# ---------------------------------------------------------------------------

def test_transaction_costs_reduce_returns(sample_prices, sample_returns):
    """Net return (with costs) must be lower than gross return (zero costs)."""
    strategy = _SimpleSignalStrategy()

    engine_with_costs = BacktestEngine(
        initial_capital=1_000_000.0,
        commission_bps=10.0,
        spread_bps=5.0,
        slippage_bps=5.0,
        rebalance_threshold=0.0,
    )
    engine_no_costs = BacktestEngine(
        initial_capital=1_000_000.0,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        rebalance_threshold=0.0,
    )

    result_gross = engine_no_costs.run(prices=sample_prices, returns=sample_returns, strategy=strategy)
    result_net = engine_with_costs.run(prices=sample_prices, returns=sample_returns, strategy=strategy)

    gross_final = result_gross.portfolio_values.iloc[-1]
    net_final = result_net.portfolio_values.iloc[-1]

    # Net must be <= gross (costs always reduce portfolio value)
    assert net_final <= gross_final + 1.0, (
        f"Net final value {net_final:.2f} exceeds gross final value {gross_final:.2f} "
        f"— transaction costs should reduce returns"
    )


# ---------------------------------------------------------------------------
# test_trade_ledger_populated
# ---------------------------------------------------------------------------

def test_trade_ledger_populated(engine, sample_prices, sample_returns):
    """At least one trade must be recorded in the ledger after the backtest."""
    strategy = _SimpleSignalStrategy()
    result = engine.run(prices=sample_prices, returns=sample_returns, strategy=strategy)

    ledger_df = result.trade_ledger.to_dataframe()
    assert len(ledger_df) > 0, "Trade ledger is empty — no trades were recorded"

    # Verify all expected columns are present
    expected_cols = [
        "trade_id", "timestamp", "security_id", "strategy",
        "executed_price", "position_size", "notional",
        "commission", "slippage", "total_cost",
    ]
    for col in expected_cols:
        assert col in ledger_df.columns, f"Missing ledger column: {col}"


# ---------------------------------------------------------------------------
# test_portfolio_state_consistency
# ---------------------------------------------------------------------------

def test_portfolio_state_consistency():
    """
    PortfolioState invariant: cash + market_value_of_positions == mark_to_market().
    """
    portfolio = PortfolioState(initial_capital=100_000.0)

    # Simulate a buy
    fill = Fill(
        security_id="AAPL",
        shares=100.0,
        executed_price=150.0,
        commission=5.0,
        slippage=2.0,
        spread_cost=1.0,
    )
    prices = {"AAPL": 155.0}
    portfolio.update([fill], prices)

    mtm = portfolio.mark_to_market(prices)
    cash = portfolio.cash
    position_value = sum(
        shares * prices.get(sid, 0.0)
        for sid, shares in portfolio.positions.items()
    )

    assert abs(mtm - (cash + position_value)) < 1e-6, (
        f"Portfolio inconsistency: mtm={mtm:.4f}, cash+positions={cash+position_value:.4f}"
    )
