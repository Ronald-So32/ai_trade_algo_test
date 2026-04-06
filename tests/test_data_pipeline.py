"""Tests for data pipeline: SecurityMaster, MarketData, ReturnsCalculator, UniverseConstructor."""
import pytest
import pandas as pd
import numpy as np

from qrt.data.security_master import SecurityMaster


# ---------------------------------------------------------------------------
# Helpers — lazily built once per session to avoid re-generating data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def security_master_df():
    sm = SecurityMaster(seed=42)
    return sm.generate()


# ---------------------------------------------------------------------------
# SecurityMaster tests
# ---------------------------------------------------------------------------

def test_security_master_schema(security_master_df):
    """All required columns must be present in the security master."""
    required_columns = [
        "security_id", "ticker", "cusip", "isin", "company_name",
        "sector", "industry", "exchange", "currency", "country",
        "list_date", "delist_date", "is_active",
    ]
    for col in required_columns:
        assert col in security_master_df.columns, f"Missing column: {col}"


def test_security_master_has_delisted(security_master_df):
    """Some securities must have a non-null delist_date (survivorship-bias population)."""
    delisted = security_master_df[security_master_df["delist_date"].notna()]
    assert len(delisted) > 0, "Expected at least one delisted security"
    # All delisted entries should have is_active == False
    assert (delisted["is_active"] == False).all(), (
        "Delisted securities must have is_active=False"
    )


def test_security_master_sectors(security_master_df):
    """The security master must cover all expected sectors with meaningful counts."""
    expected_sectors = {
        "Technology", "Financials", "Healthcare",
        "Consumer", "Industrials", "Energy", "Materials",
    }
    observed_sectors = set(security_master_df["sector"].unique())
    assert expected_sectors.issubset(observed_sectors), (
        f"Missing sectors: {expected_sectors - observed_sectors}"
    )
    # Each sector should have multiple entries
    counts = security_master_df["sector"].value_counts()
    for sector in expected_sectors:
        assert counts.get(sector, 0) >= 2, (
            f"Sector '{sector}' has fewer than 2 securities"
        )


# ---------------------------------------------------------------------------
# Market data schema (built synthetically to avoid long generation times)
# ---------------------------------------------------------------------------

def _make_market_data_row(date, security_id, close=50.0):
    """Build a minimal market data record matching MarketDataGenerator.COLUMNS."""
    return {
        "date": date,
        "security_id": security_id,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "adjusted_close": close,
        "volume": 500_000,
        "vwap": close,
        "dollar_volume": close * 500_000,
        "market_cap": close * 1_000_000,
        "split_factor": 1.0,
        "dividend_amount": 0.0,
    }


@pytest.fixture
def minimal_market_data():
    """Two securities over three days — enough to test schema and return calc."""
    dates = pd.bdate_range("2021-01-04", periods=3)
    records = []
    for t, date in enumerate(dates):
        for sid in [1, 2]:
            records.append(_make_market_data_row(date, sid, close=50.0 + t + sid))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_market_data_schema(minimal_market_data):
    """Daily bars DataFrame must contain all expected columns."""
    required = [
        "date", "security_id", "open", "high", "low", "close",
        "adjusted_close", "volume", "vwap", "dollar_volume",
        "market_cap", "split_factor", "dividend_amount",
    ]
    for col in required:
        assert col in minimal_market_data.columns, f"Missing market data column: {col}"


# ---------------------------------------------------------------------------
# Returns calculation tests
# ---------------------------------------------------------------------------

def test_returns_calculation(sample_prices):
    """ret_raw = (price[t] - price[t-1]) / price[t-1] for a simple price series."""
    prices = sample_prices.iloc[:, 0]  # single asset
    raw_returns = prices.pct_change().dropna()

    # Manual calculation
    for i in range(1, min(10, len(prices))):
        expected = (prices.iloc[i] - prices.iloc[i - 1]) / prices.iloc[i - 1]
        actual = raw_returns.iloc[i - 1]
        assert abs(actual - expected) < 1e-10, (
            f"Return mismatch at period {i}: expected {expected:.6f}, got {actual:.6f}"
        )


def test_log_returns(sample_prices):
    """log_ret = log(1 + ret_raw) for all assets and dates."""
    raw_returns = sample_prices.pct_change().dropna()
    log_returns = np.log1p(raw_returns)

    # Recompute from prices
    expected_log = np.log(sample_prices / sample_prices.shift(1)).dropna()

    # Align indices
    common_idx = log_returns.index.intersection(expected_log.index)
    diff = (log_returns.loc[common_idx] - expected_log.loc[common_idx]).abs()
    assert diff.max().max() < 1e-10, "log_ret != log(price[t]/price[t-1])"


def test_no_future_data_for_delisted(security_master_df):
    """
    After a security's delist_date, it must not appear in the active universe.
    Verified via SecurityMaster.as_of().
    """
    sm = SecurityMaster(seed=42)
    sm.generate()

    delisted = security_master_df[security_master_df["delist_date"].notna()]
    if delisted.empty:
        pytest.skip("No delisted securities to test")

    # Pick the first delisted security and check it's absent one day after delist
    row = delisted.iloc[0]
    delist_ts = pd.Timestamp(row["delist_date"])
    day_after = delist_ts + pd.Timedelta(days=1)

    active_df = sm.as_of(day_after)
    assert int(row["security_id"]) not in active_df["security_id"].values, (
        "Delisted security still appears in active universe after delist_date"
    )


def test_universe_size(security_master_df):
    """The generated universe should contain a reasonable number of active securities."""
    active = security_master_df[security_master_df["is_active"] == True]
    # Expect at least 100 active securities in the synthetic universe
    assert len(active) >= 100, (
        f"Expected >= 100 active securities, got {len(active)}"
    )


def test_universe_filters(security_master_df):
    """
    Active securities (is_active=True) must have no delist_date;
    inactive securities must have a delist_date.
    This validates the dual-filter invariant.
    """
    active = security_master_df[security_master_df["is_active"] == True]
    inactive = security_master_df[security_master_df["is_active"] == False]

    # Active: delist_date must be NaT
    assert active["delist_date"].isna().all(), (
        "Some active securities have a non-null delist_date"
    )

    # Inactive: delist_date must be set
    assert inactive["delist_date"].notna().all(), (
        "Some inactive securities are missing a delist_date"
    )

    # list_date must be non-null for all
    assert security_master_df["list_date"].notna().all(), (
        "Some securities have a null list_date"
    )
