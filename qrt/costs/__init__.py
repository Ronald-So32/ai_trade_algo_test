"""
Transaction Cost Module
=======================
Provides a realistic multi-component transaction cost model for
quantitative strategy evaluation.

Classes
-------
TransactionCostModel
    Models commission, bid-ask spread, market-impact slippage, and
    turnover penalties.  Computes per-trade costs and aggregate daily
    cost drag on a portfolio.
"""

from qrt.costs.transaction_costs import TransactionCostModel

__all__ = [
    "TransactionCostModel",
]
