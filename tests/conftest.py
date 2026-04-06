"""Shared test fixtures."""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def rng():
    return np.random.RandomState(42)

@pytest.fixture
def sample_prices(rng):
    """Generate simple price data for 10 assets over 500 days."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    n_assets = 10
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    returns = rng.normal(0.0003, 0.02, (500, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)

@pytest.fixture
def sample_returns(sample_prices):
    return sample_prices.pct_change().dropna()

@pytest.fixture
def sample_volumes(rng, sample_prices):
    dates = sample_prices.index
    return pd.DataFrame(
        rng.uniform(1e6, 1e7, sample_prices.shape),
        index=dates, columns=sample_prices.columns
    )
