"""
qrt.backtest - Event-driven backtesting engine.

Public API
----------
BacktestEngine
    Main entry point. Instantiate with cost/leverage parameters, then
    call ``engine.run(prices, returns, strategy, ...)`` to produce a
    ``BacktestResult``.

BacktestResult
    Rich analytics container returned by ``BacktestEngine.run``.
    Exposes equity curve, drawdown, Sharpe, Sortino, Calmar, turnover,
    monthly returns, rolling metrics, and a ``summary()`` dict.

PortfolioState
    Internal portfolio tracker (positions, cash, leverage, mark-to-market).
    Exposed here for standalone use and testing.

Fill
    Dataclass representing a single executed order fill.

TradeLedger
    Append-only trade record.  Produces a tidy DataFrame and summary stats.

TradeRecord
    Individual trade row dataclass.
"""

from .engine import BacktestEngine
from .portfolio import Fill, PortfolioState
from .result import BacktestResult
from .trade_ledger import TradeLedger, TradeRecord

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PortfolioState",
    "Fill",
    "TradeLedger",
    "TradeRecord",
]
