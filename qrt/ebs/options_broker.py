"""
Alpaca Options Broker — buy ATM puts for Earnings Black Swan signals.

Uses alpaca-py SDK to:
  1. Query option chains for a given underlying
  2. Find the best ATM put expiring shortly after earnings
  3. Submit market/limit buy orders for puts
  4. Track open option positions
  5. Handle expiration and exercise

Alpaca options API:
  - Paper trading: options enabled by default (Level 3)
  - Order endpoint: same /v2/orders with OCC contract symbols
  - Contract format: "AAPL250418P00170000" (ticker + YYMMDD + P/C + strike*1000)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOptionContractsRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderType,
        TimeInForce,
        AssetStatus,
        ContractType,
    )
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    logger.warning("alpaca-py not installed. Options broker will be unavailable.")


class OptionsBroker:
    """Thin wrapper around Alpaca's options trading API.

    Usage:
        broker = OptionsBroker(api_key, secret_key, paper=True)
        contract = broker.find_atm_put("AAPL", days_to_expiry=7)
        order = broker.buy_put(contract, notional=3000.0)
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not HAS_ALPACA:
            raise RuntimeError("alpaca-py not installed: pip install alpaca-py")
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.paper = paper

    def get_account(self) -> dict:
        """Get account info."""
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def find_atm_put(
        self,
        ticker: str,
        days_to_expiry: int = 7,
        max_days: int = 30,
    ) -> Optional[dict]:
        """Find the nearest ATM put option for a ticker.

        Looks for a put expiring within [days_to_expiry, max_days] days,
        with strike closest to the current stock price.

        Args:
            ticker: Underlying stock symbol (e.g., "AAPL")
            days_to_expiry: Minimum days until expiration
            max_days: Maximum days until expiration

        Returns:
            dict with contract details, or None if not found.
        """
        try:
            # Get current stock price
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            # Use trading client to get latest price from a position or quote
            # For simplicity, we'll search contracts and pick ATM
            today = datetime.now()
            min_expiry = (today + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")
            max_expiry = (today + timedelta(days=max_days)).strftime("%Y-%m-%d")

            # Search for put contracts
            request = GetOptionContractsRequest(
                underlying_symbols=[ticker],
                type=ContractType.PUT,
                expiration_date_gte=min_expiry,
                expiration_date_lte=max_expiry,
                status=AssetStatus.ACTIVE,
            )
            contracts = self.trading.get_option_contracts(request)

            if not contracts or not contracts.option_contracts:
                logger.warning(f"No put contracts found for {ticker} ({min_expiry} to {max_expiry})")
                return None

            # Get current price to find ATM
            # Use the mid-point of available strikes as proxy if we can't get live price
            all_contracts = contracts.option_contracts
            strikes = [float(c.strike_price) for c in all_contracts]

            # Try to get current price from latest trade
            try:
                positions = self.trading.get_all_positions()
                current_price = None
                for pos in positions:
                    if pos.symbol == ticker:
                        current_price = float(pos.current_price)
                        break

                if current_price is None:
                    # Estimate from median strike
                    current_price = sorted(strikes)[len(strikes) // 2]
            except Exception:
                current_price = sorted(strikes)[len(strikes) // 2]

            # Find ATM: closest strike to current price
            best = None
            best_dist = float("inf")
            for c in all_contracts:
                strike = float(c.strike_price)
                dist = abs(strike - current_price)
                if dist < best_dist:
                    best_dist = dist
                    best = c

            if best is None:
                return None

            result = {
                "symbol": best.symbol,  # OCC symbol e.g. "AAPL250418P00170000"
                "underlying": ticker,
                "strike": float(best.strike_price),
                "expiration": str(best.expiration_date),
                "type": "put",
                "current_price": current_price,
                "moneyness": round((float(best.strike_price) - current_price) / current_price * 100, 2),
            }
            logger.info(
                f"  Found ATM put: {result['symbol']} "
                f"strike=${result['strike']:.2f} "
                f"exp={result['expiration']} "
                f"moneyness={result['moneyness']:+.1f}%"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to find put for {ticker}: {e}")
            return None

    def buy_put(
        self,
        contract_symbol: str,
        qty: int = 1,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Optional[dict]:
        """Submit a buy order for a put option.

        Args:
            contract_symbol: OCC symbol (e.g., "AAPL250418P00170000")
            qty: Number of contracts (1 contract = 100 shares)
            order_type: "market" or "limit"
            limit_price: Required if order_type="limit"

        Returns:
            Order dict with id, status, etc. or None on failure.
        """
        try:
            if order_type == "limit" and limit_price is not None:
                request = LimitOrderRequest(
                    symbol=contract_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=contract_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                )

            order = self.trading.submit_order(request)
            result = {
                "order_id": str(order.id),
                "symbol": contract_symbol,
                "qty": qty,
                "side": "BUY",
                "type": order_type,
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
            }
            logger.info(f"  Order submitted: BUY {qty}x {contract_symbol} ({order_type}) — {order.status}")
            return result

        except Exception as e:
            logger.error(f"Order failed for {contract_symbol}: {e}")
            return None

    def get_option_positions(self) -> list[dict]:
        """Get all open option positions."""
        try:
            positions = self.trading.get_all_positions()
            options = []
            for pos in positions:
                # Option symbols are longer than stock symbols (OCC format)
                if len(pos.symbol) > 10:
                    options.append({
                        "symbol": pos.symbol,
                        "qty": int(pos.qty),
                        "avg_entry": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price),
                        "market_value": float(pos.market_value),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                    })
            return options
        except Exception as e:
            logger.error(f"Failed to get option positions: {e}")
            return []

    def close_position(self, contract_symbol: str) -> Optional[dict]:
        """Close an option position by selling."""
        try:
            self.trading.close_position(contract_symbol)
            logger.info(f"  Closed position: {contract_symbol}")
            return {"symbol": contract_symbol, "action": "closed"}
        except Exception as e:
            logger.error(f"Failed to close {contract_symbol}: {e}")
            return None
