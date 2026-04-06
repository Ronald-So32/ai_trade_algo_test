"""Tests for transaction cost models."""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inline cost model (mirrors BacktestEngine._DefaultCostModel)
# ---------------------------------------------------------------------------

class LinearCostModel:
    """
    Linear basis-point cost model matching the engine's _DefaultCostModel.

    commission  = notional * commission_rate
    slippage    = notional * slippage_rate
    spread_cost = notional * spread_rate
    total       = sum of components
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        spread_bps: float = 2.0,
        slippage_bps: float = 3.0,
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
    ):
        """Return (commission, slippage, spread_cost)."""
        notional = abs(shares * price)
        commission = notional * self.commission_rate
        slippage = notional * self.slippage_rate
        spread_cost = notional * self.spread_rate
        return commission, slippage, spread_cost

    def total(self, security_id: str, shares: float, price: float, date: pd.Timestamp) -> float:
        c, sl, sp = self.compute(security_id, shares, price, date)
        return c + sl + sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cost_model():
    return LinearCostModel(commission_bps=5.0, spread_bps=2.0, slippage_bps=3.0)


@pytest.fixture
def trade_date():
    return pd.Timestamp("2022-06-15")


# ---------------------------------------------------------------------------
# test_cost_positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shares,price", [
    (100.0, 50.0),
    (-200.0, 75.0),   # short sale
    (1.0, 0.01),       # tiny trade
    (1_000_000.0, 1.0),  # large share count
])
def test_cost_positive(cost_model, trade_date, shares, price):
    """Transaction costs are always strictly positive for non-zero trades."""
    commission, slippage, spread_cost = cost_model.compute(
        "TEST", shares, price, trade_date
    )
    assert commission >= 0.0, f"commission={commission} is negative"
    assert slippage >= 0.0, f"slippage={slippage} is negative"
    assert spread_cost >= 0.0, f"spread_cost={spread_cost} is negative"
    total = commission + slippage + spread_cost
    assert total > 0.0, f"Total cost={total} is not positive for shares={shares}, price={price}"


# ---------------------------------------------------------------------------
# test_cost_proportional_to_size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("small_shares,large_shares", [
    (100.0, 1000.0),
    (50.0, 500.0),
    (1.0, 10.0),
])
def test_cost_proportional_to_size(cost_model, trade_date, small_shares, large_shares):
    """Larger trades must incur higher costs (linear cost model)."""
    price = 100.0

    small_total = cost_model.total("ASSET", small_shares, price, trade_date)
    large_total = cost_model.total("ASSET", large_shares, price, trade_date)

    assert large_total > small_total, (
        f"Larger trade ({large_shares} shares) did not cost more than "
        f"smaller trade ({small_shares} shares): "
        f"large={large_total:.4f}, small={small_total:.4f}"
    )

    # For a linear model the ratio of costs equals the ratio of notionals
    expected_ratio = large_shares / small_shares
    actual_ratio = large_total / small_total
    assert abs(actual_ratio - expected_ratio) < 1e-9, (
        f"Cost ratio {actual_ratio:.6f} != size ratio {expected_ratio:.6f} "
        f"(linear model expected)"
    )


# ---------------------------------------------------------------------------
# test_cost_breakdown_sums
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shares,price", [
    (500.0, 120.0),
    (-300.0, 80.0),
    (10_000.0, 5.0),
])
def test_cost_breakdown_sums(cost_model, trade_date, shares, price):
    """
    The sum of individual cost components (commission + slippage + spread_cost)
    must equal the separately computed total cost.
    """
    commission, slippage, spread_cost = cost_model.compute(
        "TEST", shares, price, trade_date
    )
    component_sum = commission + slippage + spread_cost
    direct_total = cost_model.total("TEST", shares, price, trade_date)

    assert abs(component_sum - direct_total) < 1e-10, (
        f"Component sum {component_sum:.8f} != total {direct_total:.8f}"
    )


# ---------------------------------------------------------------------------
# test_cost_zero_for_zero_notional (edge case)
# ---------------------------------------------------------------------------

def test_cost_zero_for_zero_shares(cost_model, trade_date):
    """Zero shares produce zero cost (no trade, no cost)."""
    commission, slippage, spread_cost = cost_model.compute("ASSET", 0.0, 100.0, trade_date)
    assert commission == 0.0
    assert slippage == 0.0
    assert spread_cost == 0.0


# ---------------------------------------------------------------------------
# test_cost_symmetry (buy vs. sell same size)
# ---------------------------------------------------------------------------

def test_cost_symmetry(cost_model, trade_date):
    """
    Cost of buying N shares equals cost of selling N shares at the same price
    (linear model is symmetric in |notional|).
    """
    price = 100.0
    shares = 250.0
    buy_cost = cost_model.total("ASSET", +shares, price, trade_date)
    sell_cost = cost_model.total("ASSET", -shares, price, trade_date)
    assert abs(buy_cost - sell_cost) < 1e-10, (
        f"Buy cost {buy_cost:.6f} != sell cost {sell_cost:.6f}"
    )


# ---------------------------------------------------------------------------
# test_cost_scales_with_bps
# ---------------------------------------------------------------------------

def test_cost_scales_with_bps(trade_date):
    """
    Doubling every bps rate should double the total cost.
    """
    base = LinearCostModel(commission_bps=5.0, spread_bps=2.0, slippage_bps=3.0)
    doubled = LinearCostModel(commission_bps=10.0, spread_bps=4.0, slippage_bps=6.0)

    shares, price = 1000.0, 50.0
    base_total = base.total("ASSET", shares, price, trade_date)
    doubled_total = doubled.total("ASSET", shares, price, trade_date)

    assert abs(doubled_total - 2 * base_total) < 1e-9, (
        f"Doubling bps should double cost: base={base_total:.6f}, doubled={doubled_total:.6f}"
    )
