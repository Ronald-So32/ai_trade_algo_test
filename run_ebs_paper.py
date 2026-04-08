#!/usr/bin/env python3
"""
Earnings Black Swan — Alpaca Paper Trading Runner
===================================================
Scans upcoming earnings, generates put-buy signals using a walk-forward
logistic regression model, and executes via Alpaca's options API.

Modes:
  scan     — Show upcoming earnings and signals (dry run, no orders)
  trade    — Generate signals and submit put orders
  status   — Show account, open option positions, and trade history
  retrain  — Retrain model on latest historical data

Usage:
    # Set up credentials
    cp .env.example .env  # Add your Alpaca keys

    # Scan upcoming earnings (no orders)
    python run_ebs_paper.py scan

    # Execute trades
    python run_ebs_paper.py trade

    # Check positions
    python run_ebs_paper.py status

    # Retrain model on latest data
    python run_ebs_paper.py retrain
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from qrt.utils.logger import get_logger

logger = get_logger("ebs_paper")

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "data" / "ebs_model.pkl"
TRADE_LOG_PATH = PROJECT_ROOT / "reports" / "ebs_trades.jsonl"
EBS_DATASET_PATH = PROJECT_ROOT.parent / "earnings_black_swan" / "data" / "processed" / "earnings_dataset_v2.json"


def load_env():
    """Load Alpaca credentials from .env file."""
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        logger.error("ALPACA_API_KEY not set in .env file")
        return None, None
    if not secret_key or secret_key == "your_secret_key_here":
        logger.error("ALPACA_SECRET_KEY not set in .env file")
        return None, None
    return api_key, secret_key


def log_trade(trade: dict):
    """Append trade to JSONL log."""
    TRADE_LOG_PATH.parent.mkdir(exist_ok=True)
    trade["timestamp"] = datetime.now().isoformat()
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade) + "\n")


def cmd_retrain():
    """Retrain the EBS model on historical earnings dataset."""
    import numpy as np
    from qrt.ebs.signal_generator import EBSModel, FEATURE_NAMES, CAR_THRESHOLD

    logger.info("=" * 60)
    logger.info("RETRAIN EBS MODEL")
    logger.info("=" * 60)

    if not EBS_DATASET_PATH.exists():
        logger.error(f"Dataset not found: {EBS_DATASET_PATH}")
        logger.error("Run the EBS backtest first: cd ../earnings_black_swan && python -m ebs.backtest.engine_v2")
        return

    with open(EBS_DATASET_PATH) as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} events from {EBS_DATASET_PATH}")

    # Build feature matrix
    X_list, y_list = [], []
    for event in dataset:
        feats = event.get("features")
        if feats is None or len(feats) < len(FEATURE_NAMES):
            continue
        X_list.append(feats)
        car = event.get("car_pct", 0)
        y_list.append(1 if car <= CAR_THRESHOLD else 0)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Training on {len(X)} events ({y.sum()} catastrophic, {y.mean():.1%} base rate)")

    model = EBSModel()
    metrics = model.train(X, y)
    logger.info(f"Training complete: AUC={metrics['auc']:.3f}, features_used={metrics['n_features_used']}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


def cmd_scan(days_ahead: int = 7):
    """Scan upcoming earnings and show signals (no orders)."""
    from qrt.ebs.signal_generator import EBSModel, get_upcoming_earnings, generate_signals

    logger.info("=" * 60)
    logger.info(f"SCAN UPCOMING EARNINGS (next {days_ahead} days)")
    logger.info("=" * 60)

    # Load model
    model = EBSModel()
    if not model.load(MODEL_PATH):
        logger.error(f"No model found at {MODEL_PATH}. Run 'retrain' first.")
        return

    # Get upcoming earnings
    upcoming = get_upcoming_earnings(days_ahead=days_ahead)
    if not upcoming:
        logger.info("No upcoming earnings found in the next week.")
        return

    logger.info(f"\n{len(upcoming)} upcoming earnings events:")
    for event in upcoming:
        logger.info(f"  {event['ticker']:6s} — {event['earnings_date']}")

    # Generate signals
    logger.info("\nGenerating signals...")
    signals = generate_signals(model, upcoming, capital=100_000.0)

    if not signals:
        logger.info("No signals generated (all below threshold).")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"SIGNALS ({len(signals)} trades)")
    logger.info(f"{'='*60}")
    for sig in signals:
        logger.info(
            f"  {sig.ticker:6s} | "
            f"earnings={sig.earnings_date} | "
            f"P(crash)={sig.pred_prob:.3f} | "
            f"size={sig.position_pct:.2%} | "
            f"vol={sig.realized_vol:.0f}%"
        )


def cmd_trade(days_ahead: int = 7):
    """Generate signals and submit put orders to Alpaca."""
    from qrt.ebs.signal_generator import EBSModel, get_upcoming_earnings, generate_signals
    from qrt.ebs.options_broker import OptionsBroker
    from ebs.risk import check_concurrent_positions

    logger.info("=" * 60)
    logger.info("EBS PAPER TRADING — EXECUTE")
    logger.info("=" * 60)

    # Load credentials
    api_key, secret_key = load_env()
    if not api_key:
        return

    # Load model
    model = EBSModel()
    if not model.load(MODEL_PATH):
        logger.error(f"No model found. Run 'retrain' first.")
        return

    # Initialize broker
    broker = OptionsBroker(api_key, secret_key, paper=True)
    acct = broker.get_account()
    capital = acct["equity"]
    logger.info(f"Account equity: ${capital:,.2f}")

    # Check concurrent positions
    open_options = broker.get_option_positions()
    active = [{"ticker": p["symbol"][:4], "sector": "unknown"} for p in open_options]
    pos_check = check_concurrent_positions(active)
    if not pos_check["can_trade"]:
        logger.warning(f"Position limit reached: {pos_check['reason']}")
        return
    logger.info(f"Open option positions: {len(open_options)} (limit: 5)")

    # Get upcoming earnings
    upcoming = get_upcoming_earnings(days_ahead=days_ahead)
    if not upcoming:
        logger.info("No upcoming earnings.")
        return

    # Generate signals
    signals = generate_signals(
        model, upcoming, capital=capital,
        equity_curve=[capital],
    )
    if not signals:
        logger.info("No signals above threshold.")
        return

    logger.info(f"\n{len(signals)} signals generated. Executing trades...")

    for sig in signals:
        # Re-check position limits
        open_options = broker.get_option_positions()
        if len(open_options) >= 5:
            logger.warning("Hit 5-position limit. Stopping.")
            break

        # Find ATM put
        contract = broker.find_atm_put(
            sig.ticker,
            days_to_expiry=3,   # At least 3 days to capture [-1, +2] window
            max_days=21,        # Within 3 weeks
        )
        if contract is None:
            logger.warning(f"  No put contracts found for {sig.ticker}")
            continue

        # Calculate number of contracts
        # position_pct * capital = total notional for this trade
        # Each contract = 100 shares × premium
        # For simplicity, use notional-based sizing
        notional = capital * sig.position_pct
        # Estimate contract price as ~4% of stock price (earnings IV typical)
        est_premium = contract["current_price"] * 0.04 * 100  # per contract in $
        if est_premium <= 0:
            est_premium = 500  # fallback

        n_contracts = max(1, int(notional / est_premium))
        n_contracts = min(n_contracts, 10)  # hard cap

        logger.info(
            f"\n  TRADE: BUY {n_contracts}x PUT {contract['symbol']} "
            f"(${contract['strike']:.0f} strike, exp {contract['expiration']}) "
            f"for {sig.ticker} earnings {sig.earnings_date} "
            f"[P(crash)={sig.pred_prob:.3f}, size={sig.position_pct:.2%}]"
        )

        # Submit order
        order = broker.buy_put(contract["symbol"], qty=n_contracts, order_type="market")
        if order:
            trade_record = {
                "ticker": sig.ticker,
                "earnings_date": sig.earnings_date,
                "pred_prob": sig.pred_prob,
                "position_pct": sig.position_pct,
                "contract": contract["symbol"],
                "strike": contract["strike"],
                "expiration": contract["expiration"],
                "n_contracts": n_contracts,
                "order_id": order["order_id"],
                "order_status": order["status"],
                "features": sig.features,
            }
            log_trade(trade_record)
            logger.info(f"    Order {order['order_id']}: {order['status']}")


def cmd_status():
    """Show account status and open option positions."""
    from qrt.ebs.options_broker import OptionsBroker

    api_key, secret_key = load_env()
    if not api_key:
        return

    broker = OptionsBroker(api_key, secret_key, paper=True)

    # Account
    acct = broker.get_account()
    print("\nALPACA PAPER TRADING ACCOUNT (EBS)")
    print(f"  Equity:       ${acct['equity']:>12,.2f}")
    print(f"  Cash:         ${acct['cash']:>12,.2f}")
    print(f"  Buying power: ${acct['buying_power']:>12,.2f}")

    # Option positions
    options = broker.get_option_positions()
    if options:
        print(f"\nOPEN OPTION POSITIONS ({len(options)}):")
        for pos in options:
            print(
                f"  {pos['symbol']:25s} | "
                f"qty={pos['qty']} | "
                f"avg_entry=${pos['avg_entry']:.2f} | "
                f"current=${pos['current_price']:.2f} | "
                f"P&L=${pos['unrealized_pl']:+.2f} ({pos['unrealized_plpc']:+.1%})"
            )
    else:
        print("\nNo open option positions.")

    # Recent trades
    if TRADE_LOG_PATH.exists():
        print(f"\nRECENT TRADES (from {TRADE_LOG_PATH}):")
        with open(TRADE_LOG_PATH) as f:
            lines = f.readlines()
        for line in lines[-10:]:
            trade = json.loads(line)
            print(
                f"  {trade.get('timestamp', '?')[:19]} | "
                f"{trade.get('ticker', '?'):6s} | "
                f"P(crash)={trade.get('pred_prob', 0):.3f} | "
                f"{trade.get('n_contracts', 0)}x {trade.get('contract', '?')}"
            )


def main():
    parser = argparse.ArgumentParser(description="EBS Options Paper Trading")
    parser.add_argument("command", choices=["scan", "trade", "status", "retrain"],
                        help="Command to run")
    parser.add_argument("--days", type=int, default=7,
                        help="Days ahead to scan for earnings (default: 7)")
    args = parser.parse_args()

    if args.command == "retrain":
        cmd_retrain()
    elif args.command == "scan":
        cmd_scan(days_ahead=args.days)
    elif args.command == "trade":
        cmd_trade(days_ahead=args.days)
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
