"""Verify Alpaca keys work. Reads .env from the project root explicitly.

Run:  ./.venv/bin/python scripts/check_keys.py
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

key = os.environ.get("ALPACA_API_KEY_ID")
sec = os.environ.get("ALPACA_API_SECRET_KEY")

if not key or not sec or key.startswith("PK...") or sec.startswith("..."):
    raise SystemExit(
        "No real keys found in .env.\n"
        f"Create {ROOT/'.env'} with:\n"
        "  ALPACA_API_KEY_ID=PK...\n"
        "  ALPACA_API_SECRET_KEY=...\n"
    )

h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
print(f"Using key {key[:6]}…  (paper key starts PK, live starts AK)")

a = requests.get("https://paper-api.alpaca.markets/v2/account", headers=h, timeout=15)
print("account:", a.status_code, "| req-id:", a.headers.get("X-Request-ID"))
if a.ok:
    j = a.json()
    print("  ✓ account status:", j.get("status"), "| cash:", j.get("cash"))
else:
    print("  ✗", a.text[:200])

d = requests.get("https://data.alpaca.markets/v2/stocks/AAPL/quotes", headers=h,
                 params={"start": "2024-01-02", "end": "2024-01-03", "limit": 3, "feed": "iex"},
                 timeout=15)
print("data:   ", d.status_code, "| quotes returned:", len(d.json().get("quotes") or []) if d.ok else d.text[:200])

if a.ok and d.ok:
    print("\nAll good — fetch real data with:")
    print("  ./.venv/bin/python -m soc.data.fetch_history --symbol AAPL --start 2024-01-02 --end 2024-01-09")
