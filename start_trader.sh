#!/bin/bash
# ── QRT Auto-Trader Background Runner ──
# Starts the auto-trader in the background so it survives terminal close.
#
# Usage:
#   ./start_trader.sh           # Start auto-trader (default 3:50 PM ET)
#   ./start_trader.sh 09:45     # Start with custom trade time
#   ./start_trader.sh stop      # Stop the auto-trader
#   ./start_trader.sh status    # Check if running + show account
#   ./start_trader.sh logs      # Tail the auto-trader logs

DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/reports/.trader.pid"
LOGFILE="$DIR/reports/auto_trader.log"

mkdir -p "$DIR/reports"

case "${1:-start}" in
    stop)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                echo "Auto-trader stopped (PID $PID)"
                rm -f "$PIDFILE"
            else
                echo "Auto-trader not running (stale PID file)"
                rm -f "$PIDFILE"
            fi
        else
            echo "Auto-trader not running (no PID file)"
        fi
        ;;

    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Auto-trader is RUNNING (PID $(cat "$PIDFILE"))"
            echo ""
            # Show account status
            cd "$DIR" && python3 run_paper_trade.py status
        else
            echo "Auto-trader is NOT RUNNING"
            [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
        fi
        ;;

    logs)
        if [ -f "$LOGFILE" ]; then
            tail -50 "$LOGFILE"
            echo ""
            echo "  (showing last 50 lines — full log: $LOGFILE)"
        else
            echo "No log file yet. Start the auto-trader first."
        fi
        ;;

    *)
        TRADE_TIME="${1:-15:50}"

        # Check if already running
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Auto-trader is already running (PID $(cat "$PIDFILE"))"
            echo "Run './start_trader.sh stop' first to restart."
            exit 1
        fi

        echo "Starting QRT auto-trader..."
        echo "  Trade time: $TRADE_TIME ET"
        echo "  Log file:   $LOGFILE"
        echo "  PID file:   $PIDFILE"
        echo ""

        # Start in background with nohup
        cd "$DIR"
        nohup python3 run_paper_trade.py auto --trade-time "$TRADE_TIME" >> "$LOGFILE" 2>&1 &
        TRADER_PID=$!
        echo "$TRADER_PID" > "$PIDFILE"

        echo "Auto-trader started (PID $TRADER_PID)"
        echo ""
        echo "Commands:"
        echo "  ./start_trader.sh status  — Check if running"
        echo "  ./start_trader.sh logs    — View recent logs"
        echo "  ./start_trader.sh stop    — Stop the auto-trader"
        echo "  python3 run_paper_trade.py dashboard  — View live dashboard"
        ;;
esac
