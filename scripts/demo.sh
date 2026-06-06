#!/usr/bin/env bash
# Start the SOC dashboard (single server: serves the page AND the websocket on one port).
# Usage:
#   scripts/demo.sh                                    # synthetic mechanics demo
#   scripts/demo.sh --feed replay --symbol AAPL        # single-stock real Alpaca replay
#   scripts/demo.sh --symbols AAPL,MSFT,NVDA --fps 30  # MULTI-STOCK universe monitor (minute bars)
set -e
cd "$(dirname "$0")/.."
echo "Open the printed URL in your browser. Ctrl-C to stop."
exec ./.venv/bin/python -m soc.server.run "$@"
