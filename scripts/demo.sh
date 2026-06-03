#!/usr/bin/env bash
# Start the SOC dashboard (single server: serves the page AND the websocket on one port).
# Usage:
#   scripts/demo.sh                              # synthetic mechanics demo
#   scripts/demo.sh --feed replay --symbol AAPL  # real Alpaca replay (after fetch_history)
#   scripts/demo.sh --feed replay --symbol AAPL --stride 25 --fps 30
set -e
cd "$(dirname "$0")/.."
echo "Open the printed URL in your browser. Ctrl-C to stop."
exec ./.venv/bin/python -m soc.server.run "$@"
