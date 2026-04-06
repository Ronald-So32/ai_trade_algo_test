"""
qrt.data — Data pipeline module.

Exported classes
----------------
SecurityMaster          : Reference data (tickers, sectors, listing dates)
MarketDataGenerator     : Synthetic daily OHLCV bars with GBM + fat tails
ReturnsCalculator       : All return series (raw, adjusted, log, ex-div)
UniverseConstructor     : Monthly rebalanced, liquidity-filtered universes
DataPipeline            : Orchestrator — generate → persist → load

Typical usage
-------------
    from qrt.data import DataPipeline

    pipeline = DataPipeline(seed=42)
    pipeline.run()                       # generate + save to parquet
    # or, if parquet already exists:
    pipeline.load_dataset()

    sm   = pipeline.security_master     # pd.DataFrame
    md   = pipeline.market_data         # pd.DataFrame
    rets = pipeline.returns             # pd.DataFrame
    u150 = pipeline.get_universe(150, pd.Timestamp("2020-06-30"))  # list[int]
"""

from qrt.data.security_master import SecurityMaster
from qrt.data.market_data import MarketDataGenerator
from qrt.data.returns import ReturnsCalculator
from qrt.data.universe import UniverseConstructor
from qrt.data.pipeline import DataPipeline

__all__ = [
    "SecurityMaster",
    "MarketDataGenerator",
    "ReturnsCalculator",
    "UniverseConstructor",
    "DataPipeline",
]
