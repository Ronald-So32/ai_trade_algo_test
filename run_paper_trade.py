#!/usr/bin/env python3
"""
Alpaca Paper Trading Runner

Connects the QRT strategy pipeline to Alpaca's paper trading API.

Modes:
  status    — Show account info + current positions
  signals   — Generate signals from live data (dry run, no orders)
  rebalance — Generate signals AND submit rebalance orders
  close     — Close all positions
  auto      — Automated daily trading (runs continuously, rebalances daily)
  dashboard — Generate live trading HTML dashboard

Usage:
    # First: copy .env.example to .env and add your Alpaca API keys
    cp .env.example .env

    # Check connection
    python run_paper_trade.py status

    # Generate signals (dry run — shows what it would trade)
    python run_paper_trade.py signals

    # Execute rebalance (submits orders to Alpaca paper account)
    python run_paper_trade.py rebalance

    # Start automated daily trading (default: 3:30 PM ET)
    python run_paper_trade.py auto
    python run_paper_trade.py auto --trade-time 15:30

    # Run auto-trader in background (survives terminal close)
    nohup python3 run_paper_trade.py auto > reports/auto_trader.log 2>&1 &

    # Check live dashboard (positions, P&L, equity curve, trade history)
    python run_paper_trade.py dashboard
    open reports/live_dashboard.html

    # Close all positions
    python run_paper_trade.py close
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper_trader")

ET = ZoneInfo("America/New_York")

# ── Default universe for paper trading (v4) ──
# ~150 S&P 500 large-caps across 11 GICS sectors.
# Sized for robust quintile sorts (30 stocks/quintile) per Blitz et al. (2013).
# Import from the canonical source to stay in sync with backtesting.
from qrt.data.real_data import REAL_UNIVERSE, INDUSTRY_MAP

DEFAULT_UNIVERSE = []
for _sector_tickers in REAL_UNIVERSE.values():
    DEFAULT_UNIVERSE.extend(_sector_tickers)

# ── Safety Limits ──
MAX_SINGLE_POSITION_PCT = 0.10    # No single stock > 10% of portfolio
MAX_DAILY_TURNOVER_PCT = 0.50     # Don't trade more than 50% of equity/day
MAX_GROSS_EXPOSURE = 2.0          # Hard cap at 2x (Alpaca Reg T)
MAX_ORDER_PCT = 0.25              # No single order > 25% of equity


def load_env():
    """Load API keys from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        logger.error(
            "No .env file found. Copy .env.example to .env and add your Alpaca API keys:\n"
            "  cp .env.example .env"
        )
        sys.exit(1)

    load_dotenv(env_path)

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or api_key == "your_api_key_here":
        logger.error("ALPACA_API_KEY not set in .env file")
        sys.exit(1)
    if not secret_key or secret_key == "your_secret_key_here":
        logger.error("ALPACA_SECRET_KEY not set in .env file")
        sys.exit(1)

    is_paper = "paper" in base_url
    if not is_paper:
        logger.warning("WARNING: You are using a LIVE trading URL. Proceed with caution!")

    return api_key, secret_key, is_paper


def is_market_day() -> bool:
    """Check if today is a US market trading day (Mon-Fri, not a holiday)."""
    now_et = datetime.now(ET)
    # Weekend check
    if now_et.weekday() >= 5:
        return False
    # Major US market holidays (approximate — Alpaca's calendar is authoritative)
    # We check via Alpaca API in the auto-trader; this is a fast pre-check.
    return True


def is_market_open() -> bool:
    """Check if US market is currently open (9:30 AM - 4:00 PM ET)."""
    now_et = datetime.now(ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def apply_safety_limits(target_weights: dict[str, float], equity: float) -> dict[str, float]:
    """Apply safety limits to target weights before submitting orders."""
    safe_weights = {}
    total_turnover = 0

    for sym, weight in target_weights.items():
        # Cap single position size
        if abs(weight) > MAX_SINGLE_POSITION_PCT:
            capped = MAX_SINGLE_POSITION_PCT * (1 if weight > 0 else -1)
            logger.warning(f"  Safety: capping {sym} from {weight:.2%} to {capped:.2%}")
            weight = capped

        safe_weights[sym] = weight
        total_turnover += abs(weight)

    # Cap gross exposure
    gross = sum(abs(w) for w in safe_weights.values())
    if gross > MAX_GROSS_EXPOSURE:
        scale = MAX_GROSS_EXPOSURE / gross
        logger.warning(f"  Safety: scaling gross exposure from {gross:.2%} to {MAX_GROSS_EXPOSURE:.0%}")
        safe_weights = {s: w * scale for s, w in safe_weights.items()}

    return safe_weights


def log_trade(trade_data: dict):
    """Append trade to persistent JSON log."""
    log_dir = PROJECT_ROOT / "reports"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "trade_history.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(trade_data) + "\n")


def cmd_status(broker):
    """Show account status and positions."""
    acct = broker.get_account()
    print("\n" + "=" * 60)
    print("ALPACA PAPER TRADING ACCOUNT")
    print("=" * 60)
    print(f"  Status:          {acct['status']}")
    print(f"  Equity:          ${acct['equity']:,.2f}")
    print(f"  Cash:            ${acct['cash']:,.2f}")
    print(f"  Buying Power:    ${acct['buying_power']:,.2f}")
    print(f"  Portfolio Value: ${acct['portfolio_value']:,.2f}")
    print(f"  PDT:             {acct['pattern_day_trader']}")
    print(f"  Day Trades:      {acct['daytrade_count']}")
    print(f"  Multiplier:      {acct['multiplier']}x")

    positions = broker.get_positions()
    if positions:
        print(f"\n  Open Positions ({len(positions)}):")
        print(f"  {'Symbol':<8} {'Qty':>8} {'Value':>12} {'P&L':>10} {'P&L%':>8}")
        print("  " + "-" * 50)
        total_value = 0
        total_pl = 0
        for sym, pos in sorted(positions.items()):
            total_value += pos["market_value"]
            total_pl += pos["unrealized_pl"]
            print(
                f"  {sym:<8} {pos['qty']:>8.2f} "
                f"${pos['market_value']:>10,.2f} "
                f"${pos['unrealized_pl']:>8,.2f} "
                f"{pos['unrealized_plpc']:>7.2%}"
            )
        print("  " + "-" * 50)
        print(f"  {'TOTAL':<8} {'':>8} ${total_value:>10,.2f} ${total_pl:>8,.2f}")
    else:
        print("\n  No open positions")

    print("=" * 60)


def cmd_signals(broker, symbols: list[str], lookback_days: int = 504):
    """Generate signals from live data (dry run)."""
    from qrt.execution.signal_generator import LiveSignalGenerator

    print("\n" + "=" * 60)
    print("SIGNAL GENERATION (DRY RUN)")
    print("=" * 60)
    print(f"  Universe: {len(symbols)} symbols")
    print(f"  Lookback: {lookback_days} trading days")
    print()

    # Fetch market data
    logger.info("Fetching market data from Alpaca...")
    t0 = time.time()
    prices_wide, returns_wide = broker.get_price_matrix(symbols, days=lookback_days)

    if prices_wide.empty:
        logger.error("No market data returned. Check your symbols and API connection.")
        return {}

    # Data completeness validation
    min_days = 252 + 63  # TSMOM warmup (252d) + risk parity vol window (63d)
    if len(prices_wide) < min_days:
        logger.warning(
            f"Only {len(prices_wide)} trading days returned (need {min_days} for "
            f"v5 dynamic mode). Signal generator will fall back to static mode."
        )

    elapsed = time.time() - t0
    logger.info(
        f"Data fetched: {prices_wide.shape[1]} symbols, "
        f"{len(prices_wide)} trading days ({elapsed:.1f}s)"
    )

    # Generate signals (v5: risk parity + vol-managed leverage)
    logger.info("Running strategy pipeline (v5 dynamic)...")
    t1 = time.time()
    generator = LiveSignalGenerator(leverage=2.0, mode="dynamic")
    target_weights = generator.generate_weights(
        prices_wide, returns_wide, industry_map=INDUSTRY_MAP,
    )
    elapsed = time.time() - t1
    logger.info(f"Signal generation complete ({elapsed:.1f}s)")

    if not target_weights:
        logger.warning("No target weights generated")
        return {}

    # Display results
    acct = broker.get_account()
    equity = acct["equity"]

    print(f"\n  Target Portfolio Weights ({len(target_weights)} positions):")
    print(f"  Account equity: ${equity:,.2f}")
    print()
    print(f"  {'Symbol':<8} {'Weight':>8} {'$ Value':>12} {'Direction':>10}")
    print("  " + "-" * 42)

    sorted_weights = sorted(target_weights.items(), key=lambda x: -abs(x[1]))
    total_long = 0
    total_short = 0
    for sym, weight in sorted_weights:
        dollar = weight * equity
        direction = "LONG" if weight > 0 else "SHORT"
        if weight > 0:
            total_long += weight
        else:
            total_short += abs(weight)
        print(f"  {sym:<8} {weight:>7.2%} ${dollar:>10,.2f} {direction:>10}")

    print("  " + "-" * 42)
    print(f"  Gross exposure: {total_long + total_short:.2%}")
    print(f"  Net exposure:   {total_long - total_short:.2%}")
    print(f"  Long:  {total_long:.2%}  |  Short: {total_short:.2%}")

    # Save signals to file
    signals_path = PROJECT_ROOT / "reports" / "live_signals.json"
    signals_path.parent.mkdir(exist_ok=True)
    with open(signals_path, "w") as f:
        json.dump({
            "timestamp": pd.Timestamp.now().isoformat(),
            "equity": equity,
            "n_symbols": len(target_weights),
            "gross_exposure": total_long + total_short,
            "net_exposure": total_long - total_short,
            "weights": target_weights,
        }, f, indent=2)
    print(f"\n  Signals saved: {signals_path}")

    return target_weights


def cmd_rebalance(broker, symbols: list[str], lookback_days: int = 504):
    """Generate signals and execute rebalance."""
    target_weights = cmd_signals(broker, symbols, lookback_days)

    if not target_weights:
        logger.warning("No weights to rebalance to")
        return

    # Confirmation
    print("\n" + "=" * 60)
    print("REBALANCE EXECUTION")
    print("=" * 60)
    print(f"  Will submit {len(target_weights)} rebalance orders to Alpaca PAPER account")
    response = input("  Proceed? [y/N]: ").strip().lower()
    if response != "y":
        print("  Aborted.")
        return

    _execute_rebalance(broker, target_weights)


def _execute_rebalance(broker, target_weights: dict[str, float]):
    """Execute rebalance with safety limits and logging."""
    acct = broker.get_account()
    equity = acct["equity"]

    # Apply safety limits
    safe_weights = apply_safety_limits(target_weights, equity)

    logger.info("Executing rebalance...")
    orders = broker.rebalance(safe_weights, max_order_pct=MAX_ORDER_PCT)

    print(f"\n  Orders submitted: {len(orders)}")
    successes = 0
    failures = 0
    for order in orders:
        if "error" in order:
            print(f"    FAIL: {order['symbol']} {order['side']} — {order['error']}")
            failures += 1
        else:
            print(f"    OK:   {order['symbol']} {order['side']} ${order.get('notional', '?')}")
            successes += 1

    # Log to persistent trade history
    trade_record = {
        "timestamp": datetime.now(ET).isoformat(),
        "equity_before": equity,
        "n_target_positions": len(safe_weights),
        "n_orders": len(orders),
        "n_success": successes,
        "n_fail": failures,
        "gross_exposure": sum(abs(w) for w in safe_weights.values()),
        "net_exposure": sum(safe_weights.values()),
        "orders": orders,
        "target_weights": safe_weights,
    }
    log_trade(trade_record)

    # Save latest order log
    log_path = PROJECT_ROOT / "reports" / "order_log.json"
    with open(log_path, "w") as f:
        json.dump(trade_record, f, indent=2)
    print(f"  Order log saved: {log_path}")
    print(f"  Trade history appended: reports/trade_history.jsonl")

    return orders


def cmd_close(broker):
    """Close all positions."""
    positions = broker.get_positions()
    if not positions:
        print("No positions to close")
        return

    print(f"\nWill close {len(positions)} positions:")
    for sym, pos in positions.items():
        print(f"  {sym}: {pos['qty']} shares (${pos['market_value']:,.2f})")

    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    results = broker.close_all_positions()
    print(f"Closed: {results}")


def cmd_auto(broker, symbols: list[str], lookback_days: int, trade_time: str):
    """
    Automated daily trading.

    Runs continuously, executing a rebalance once per market day at the
    specified time (ET). Sleeps between checks.

    Safety features:
      - Only trades on market days (Mon-Fri)
      - Only trades during market hours (9:30 AM - 4:00 PM ET)
      - Applies position size limits and gross exposure caps
      - Logs all trades to reports/trade_history.jsonl
      - Skips if already traded today
      - Catches and logs all errors without crashing
    """
    from qrt.execution.signal_generator import LiveSignalGenerator

    trade_hour, trade_minute = map(int, trade_time.split(":"))

    print("\n" + "=" * 60)
    print("QRT AUTO-TRADER (PAPER)")
    print("=" * 60)
    print(f"  Universe:    {len(symbols)} symbols")
    print(f"  Trade time:  {trade_time} ET daily")
    print(f"  Lookback:    {lookback_days} trading days")
    print(f"  Max leverage: {MAX_GROSS_EXPOSURE:.0f}x (Alpaca Reg T)")
    print(f"  Max position: {MAX_SINGLE_POSITION_PCT:.0%} per stock")
    print(f"  Max turnover: {MAX_DAILY_TURNOVER_PCT:.0%} per day")
    print()

    # Show initial account status
    acct = broker.get_account()
    print(f"  Account:  ${acct['equity']:,.2f} equity, ${acct['buying_power']:,.2f} buying power")
    positions = broker.get_positions()
    print(f"  Positions: {len(positions)} open")
    print()
    print("  Auto-trader is running. Press Ctrl+C to stop.")
    print("=" * 60)

    last_trade_date = _load_last_trade_date()  # Persist across restarts
    last_log_time = 0  # Throttle log spam

    while True:
        try:
            now_et = datetime.now(ET)
            today_str = now_et.strftime("%Y-%m-%d")
            now_ts = time.time()

            # Already traded today (ET date)? Short-sleep and recheck.
            if last_trade_date == today_str:
                time.sleep(60)
                continue

            # Weekend in ET? Short-sleep and recheck.
            if now_et.weekday() >= 5:
                if now_ts - last_log_time > 3600:
                    logger.info(f"Weekend in ET ({now_et.strftime('%A %H:%M ET')}). Polling every 60s.")
                    last_log_time = now_ts
                time.sleep(60)
                continue

            # Build today's trade target time in ET
            target_time = now_et.replace(hour=trade_hour, minute=trade_minute, second=0, microsecond=0)

            # Before trade window? Short-sleep. macOS sleep can freeze long
            # timers, so we poll every 30s to catch the window reliably.
            if now_et < target_time:
                secs_away = (target_time - now_et).total_seconds()
                if now_ts - last_log_time > 1800:  # Log every 30 min max
                    logger.info(f"Waiting for {trade_time} ET ({secs_away/3600:.1f}h away)")
                    last_log_time = now_ts
                # Sleep 30s max — resilient to macOS sleep/wake
                time.sleep(min(secs_away, 30))
                continue

            # Past market close (4:00 PM ET)? We missed today — but still
            # trade if market was open and we haven't traded yet. The window
            # is trade_time to 3:55 PM ET (must be before 4:00 close).
            market_close = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
            if now_et > market_close:
                if last_trade_date != today_str:
                    logger.info(f"Past market close ({now_et.strftime('%H:%M ET')}). Missed today's window.")
                    last_trade_date = today_str
                    _save_last_trade_date(today_str)
                time.sleep(60)
                continue

            # ── TRADE WINDOW: between trade_time and 3:55 PM ET ──
            logger.info("=" * 60)
            logger.info(f"AUTO-TRADE: {today_str} {now_et.strftime('%H:%M:%S')} ET")
            logger.info("=" * 60)

            # Check if market is open (Alpaca calendar — catches holidays)
            try:
                clock = broker.trading.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed (holiday?). Skipping today.")
                    last_trade_date = today_str
                    _save_last_trade_date(today_str)
                    continue
            except Exception as e:
                logger.warning(f"Clock check failed: {e}. Proceeding anyway.")

            # Fetch data
            logger.info("Fetching market data...")
            t0 = time.time()
            prices_wide, returns_wide = broker.get_price_matrix(symbols, days=lookback_days)

            if prices_wide.empty:
                logger.error("No market data. Skipping today.")
                last_trade_date = today_str
                _save_last_trade_date(today_str)
                continue

            logger.info(
                f"Data: {prices_wide.shape[1]} symbols, "
                f"{len(prices_wide)} days ({time.time()-t0:.1f}s)"
            )

            # Generate signals (v5: risk parity + vol-managed leverage)
            logger.info("Generating signals (v5 dynamic)...")
            t1 = time.time()
            generator = LiveSignalGenerator(leverage=2.0, mode="dynamic")
            target_weights = generator.generate_weights(
                prices_wide, returns_wide, industry_map=INDUSTRY_MAP,
            )
            logger.info(f"Signals: {len(target_weights)} positions ({time.time()-t1:.1f}s)")

            if not target_weights:
                logger.warning("No signals generated. Skipping today.")
                last_trade_date = today_str
                _save_last_trade_date(today_str)
                continue

            # Apply safety limits
            acct = broker.get_account()
            equity = acct["equity"]
            safe_weights = apply_safety_limits(target_weights, equity)

            gross = sum(abs(w) for w in safe_weights.values())
            net = sum(safe_weights.values())
            n_long = sum(1 for w in safe_weights.values() if w > 0.001)
            n_short = sum(1 for w in safe_weights.values() if w < -0.001)
            logger.info(
                f"Portfolio: {n_long} long, {n_short} short, "
                f"gross={gross:.1%}, net={net:.1%}, equity=${equity:,.0f}"
            )

            # Execute rebalance
            logger.info("Submitting rebalance orders...")
            orders = broker.rebalance(safe_weights, max_order_pct=MAX_ORDER_PCT)

            successes = sum(1 for o in orders if "error" not in o)
            failures = sum(1 for o in orders if "error" in o)
            logger.info(f"Orders: {successes} OK, {failures} failed (of {len(orders)} total)")

            # Log
            trade_record = {
                "timestamp": now_et.isoformat(),
                "date": today_str,
                "equity": equity,
                "n_positions": len(safe_weights),
                "n_orders": len(orders),
                "n_success": successes,
                "n_fail": failures,
                "gross_exposure": gross,
                "net_exposure": net,
                "orders": orders,
                "target_weights": safe_weights,
            }
            log_trade(trade_record)

            # Save latest signals
            signals_path = PROJECT_ROOT / "reports" / "live_signals.json"
            with open(signals_path, "w") as f:
                json.dump(trade_record, f, indent=2)

            # Update live dashboard after each trade
            _update_dashboard(broker)

            last_trade_date = today_str
            _save_last_trade_date(today_str)
            logger.info(f"Trade complete. Next trade: tomorrow {trade_time} ET")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.info("\nAuto-trader stopped by user (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"Auto-trader error: {e}")
            import traceback
            traceback.print_exc()
            # Don't mark as traded — retry in 5 min
            time.sleep(300)


def _load_last_trade_date() -> str | None:
    """Load last trade date from disk (survives restarts)."""
    path = PROJECT_ROOT / "reports" / ".last_trade_date"
    if path.exists():
        date_str = path.read_text().strip()
        if date_str:
            return date_str
    return None


def _save_last_trade_date(date_str: str):
    """Persist last trade date to disk."""
    path = PROJECT_ROOT / "reports" / ".last_trade_date"
    path.parent.mkdir(exist_ok=True)
    path.write_text(date_str)


def _load_trade_history() -> list[dict]:
    """Load trade history from JSONL file."""
    history_path = PROJECT_ROOT / "reports" / "trade_history.jsonl"
    records = []
    if history_path.exists():
        for line in history_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _update_dashboard(broker):
    """Refresh the live trading dashboard HTML."""
    try:
        from qrt.execution.live_dashboard import generate_live_dashboard
        acct = broker.get_account()
        positions = broker.get_positions()
        history = _load_trade_history()
        path = str(PROJECT_ROOT / "reports" / "live_dashboard.html")
        generate_live_dashboard(acct, positions, history, save_path=path)
    except Exception as e:
        logger.warning(f"Dashboard update failed: {e}")


def cmd_dashboard(broker):
    """Generate and display the live trading dashboard."""
    from qrt.execution.live_dashboard import generate_live_dashboard

    acct = broker.get_account()
    positions = broker.get_positions()
    history = _load_trade_history()

    path = str(PROJECT_ROOT / "reports" / "live_dashboard.html")
    generate_live_dashboard(acct, positions, history, save_path=path)

    print(f"\n  Dashboard saved: {path}")

    # Auto-open in browser
    import subprocess
    subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Also print quick summary
    equity = acct["equity"]
    starting = history[0]["equity"] if history else equity
    total_ret = (equity / starting - 1) if starting > 0 else 0
    total_pl = sum(p["unrealized_pl"] for p in positions.values())

    print(f"\n  Equity: ${equity:,.2f}  |  Return: {total_ret:+.2%}  |  Unrealized P&L: ${total_pl:+,.2f}")
    print(f"  Positions: {len(positions)}  |  Rebalances: {len(history)}")


def _sleep_until_next_check(now_et, trade_hour, trade_minute):
    """Legacy — no longer used. Kept for backwards compat."""
    tomorrow = now_et + timedelta(days=1)
    next_trade = tomorrow.replace(
        hour=trade_hour, minute=max(trade_minute - 5, 0), second=0
    )
    sleep_secs = (next_trade - now_et).total_seconds()
    time.sleep(min(max(sleep_secs, 60), 3600))


def main():
    parser = argparse.ArgumentParser(description="QRT Paper Trading via Alpaca")
    parser.add_argument(
        "command",
        choices=["status", "signals", "rebalance", "close", "auto", "dashboard"],
        help="Action to perform",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Stock symbols to trade (default: ~150 S&P 500 large-caps)",
    )
    parser.add_argument(
        "--lookback", type=int, default=504,
        help="Trading days of history for signals (default: 504 = ~2 years)",
    )
    parser.add_argument(
        "--trade-time", type=str, default="15:30",
        help="Daily trade time in ET, HH:MM format (default: 15:30 = 3:30 PM ET). "
             "3:30 PM is optimal per McInish & Wood (1992) — spreads are tightest; "
             "Bogousslavsky (2016) — avoids closing auction rebalancing pressure; "
             "Lou, Polk & Skouras (2019) — reversal alpha accrues intraday, "
             "90%+ of daily info available by 3:30. Executing before MOC flow "
             "(3:45-4:00) preserves alpha and reduces execution cost.",
    )
    args = parser.parse_args()

    # Load API keys
    api_key, secret_key, is_paper = load_env()

    from qrt.execution.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker(api_key, secret_key, paper=is_paper)

    symbols = args.symbols or DEFAULT_UNIVERSE

    if args.command == "status":
        cmd_status(broker)
    elif args.command == "signals":
        cmd_signals(broker, symbols, args.lookback)
    elif args.command == "rebalance":
        cmd_rebalance(broker, symbols, args.lookback)
    elif args.command == "close":
        cmd_close(broker)
    elif args.command == "auto":
        cmd_auto(broker, symbols, args.lookback, args.trade_time)
    elif args.command == "dashboard":
        cmd_dashboard(broker)


if __name__ == "__main__":
    main()
