"""
Alpaca broker client for paper/live trading.

Wraps alpaca-py SDK for:
  - Account info & buying power
  - Fetching current positions
  - Submitting market orders (rebalance)
  - Fetching historical bars for signal generation
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestTradeRequest

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """Thin wrapper around Alpaca trading + data APIs."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    # ── Account ──────────────────────────────────────────────────────

    def get_account(self) -> dict:
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency": acct.currency,
            "status": acct.status.value if hasattr(acct.status, "value") else str(acct.status),
            "pattern_day_trader": acct.pattern_day_trader,
            "daytrade_count": acct.daytrade_count,
            "multiplier": acct.multiplier,
        }

    # ── Positions ────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, dict]:
        positions = self.trading.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry": float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
            }
        return result

    # ── Market Data ──────────────────────────────────────────────────

    def get_bars(
        self,
        symbols: list[str],
        days: int = 504,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars, return as (date, symbol) multi-index DataFrame."""
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.5))  # buffer for weekends/holidays

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,  # Free tier uses IEX, not SIP
        )
        bars = self.data.get_stock_bars(request)
        df = bars.df  # multi-index: (symbol, timestamp)
        if df.empty:
            return df

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_price_matrix(
        self, symbols: list[str], days: int = 504
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (prices_wide, returns_wide) matrices ready for strategy pipeline."""
        df = self.get_bars(symbols, days=days)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        prices_wide = df.pivot_table(
            index="date", columns="symbol", values="close"
        ).sort_index()
        prices_wide = prices_wide.ffill().dropna(how="all")

        returns_wide = prices_wide.pct_change().fillna(0)

        return prices_wide, returns_wide

    # ── Trading ──────────────────────────────────────────────────────

    def get_tradable_symbols(self) -> list[str]:
        """Get list of tradable US equity symbols."""
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self.trading.get_all_assets(request)
        return [a.symbol for a in assets if a.tradable and a.fractionable]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> dict:
        """Submit a market order. qty can be fractional."""
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC,
        )
        order = self.trading.submit_order(order_request)
        logger.info(f"Order submitted: {side} {qty:.4f} {symbol} -> {order.id}")
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
        }

    def close_position(self, symbol: str) -> dict:
        """Close entire position in a symbol."""
        try:
            result = self.trading.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return {"symbol": symbol, "status": "closed"}
        except Exception as e:
            logger.warning(f"Failed to close {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def close_all_positions(self) -> list[dict]:
        """Close all open positions."""
        results = self.trading.close_all_positions(cancel_orders=True)
        logger.info(f"Closed all positions: {len(results)} orders")
        return [{"status": "closed_all", "count": len(results)}]

    # ── Rebalance ────────────────────────────────────────────────────

    def rebalance(
        self,
        target_weights: dict[str, float],
        max_order_pct: float = 0.25,
    ) -> list[dict]:
        """
        Rebalance portfolio to target weights.

        Parameters
        ----------
        target_weights : dict
            {symbol: weight} where weight is fraction of portfolio (e.g. 0.05 = 5%).
            Negative weights = short positions.
        max_order_pct : float
            Max single order as fraction of portfolio (safety limit).

        Returns
        -------
        List of order results.
        """
        acct = self.get_account()
        equity = acct["equity"]
        if equity <= 0:
            logger.error("Account equity <= 0, cannot rebalance")
            return []

        current_positions = self.get_positions()
        orders = []

        # Calculate target dollar values
        target_dollars = {sym: weight * equity for sym, weight in target_weights.items()}

        # Calculate deltas
        all_symbols = set(list(target_dollars.keys()) + list(current_positions.keys()))

        # Pre-fetch latest prices for qty-based short orders
        short_symbols = [
            s for s in all_symbols
            if target_dollars.get(s, 0.0) - current_positions.get(s, {}).get("market_value", 0.0) < -equity * 0.001
        ]
        latest_prices = {}
        if short_symbols:
            try:
                request = StockLatestTradeRequest(symbol_or_symbols=short_symbols)
                trades = self.data.get_stock_latest_trade(request)
                for sym, trade in trades.items():
                    latest_prices[sym] = float(trade.price)
            except Exception as e:
                logger.warning(f"Failed to fetch latest prices: {e}")

        for symbol in all_symbols:
            target_val = target_dollars.get(symbol, 0.0)
            current_val = current_positions.get(symbol, {}).get("market_value", 0.0)
            delta_val = target_val - current_val

            # Skip tiny rebalances (< 0.1% of equity)
            if abs(delta_val) < equity * 0.001:
                continue

            # Safety: cap single order size
            if abs(delta_val) > equity * max_order_pct:
                logger.warning(
                    f"Capping order for {symbol}: ${delta_val:.0f} -> "
                    f"${equity * max_order_pct:.0f} (max_order_pct={max_order_pct})"
                )
                delta_val = equity * max_order_pct * (1 if delta_val > 0 else -1)

            if delta_val > 0:
                # Buy: use notional (fractional shares OK)
                try:
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        notional=round(abs(delta_val), 2),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    order = self.trading.submit_order(order_request)
                    orders.append({
                        "symbol": symbol,
                        "side": "buy",
                        "notional": round(abs(delta_val), 2),
                        "status": str(order.status),
                    })
                except Exception as e:
                    logger.warning(f"Buy order failed for {symbol}: {e}")
                    orders.append({"symbol": symbol, "side": "buy", "error": str(e)})

            elif delta_val < 0:
                # Sell/short: need whole shares for short sells
                is_short = symbol not in current_positions or target_val < 0
                try:
                    if is_short and symbol in latest_prices:
                        # Short sell: must use qty (whole shares), not notional
                        price = latest_prices[symbol]
                        qty = int(abs(delta_val) / price)  # floor to whole shares
                        if qty < 1:
                            logger.info(f"Skipping {symbol} short — less than 1 share (${abs(delta_val):.0f} / ${price:.2f})")
                            continue
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    elif symbol in current_positions and target_val >= 0:
                        # Reducing a long position: notional is fine
                        sell_val = min(abs(delta_val), abs(current_val))
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            notional=round(sell_val, 2),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    else:
                        # Fallback: try qty-based
                        price = latest_prices.get(symbol, 0)
                        if price <= 0:
                            logger.warning(f"No price for {symbol}, skipping")
                            continue
                        qty = int(abs(delta_val) / price)
                        if qty < 1:
                            continue
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )

                    order = self.trading.submit_order(order_request)
                    side_label = "sell" if symbol in current_positions and target_val >= 0 else "short"
                    orders.append({
                        "symbol": symbol,
                        "side": side_label,
                        "qty": qty if is_short else None,
                        "notional": round(abs(delta_val), 2),
                        "status": str(order.status),
                    })
                except Exception as e:
                    logger.warning(f"Sell/short order failed for {symbol}: {e}")
                    orders.append({"symbol": symbol, "side": "sell", "error": str(e)})

        logger.info(f"Rebalance complete: {len(orders)} orders submitted")
        return orders
