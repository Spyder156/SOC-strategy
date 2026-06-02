#!/usr/bin/env bash
# Start the SOC model websocket server + a static web server, then open the dashboard.
# Usage:
#   scripts/demo.sh                              # synthetic mechanics demo
#   scripts/demo.sh --feed replay --symbol AAPL  # real Alpaca replay (after fetch_history)
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
$PY -m soc.server.run "$@" &
WS=$!
$PY -m http.server 8080 --directory web >/dev/null 2>&1 &
HTTP=$!
trap "kill $WS $HTTP 2>/dev/null" EXIT
echo "SOC dashboard:  http://localhost:8080"
echo "(model ws on :8765, static on :8080 — Ctrl-C to stop)"
wait $WS
