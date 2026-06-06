"""Fetch consolidated minute bars for a universe (clean, unlike raw IEX quotes).

Bars come from Alpaca's /bars endpoint, which is consolidated and free of the stale-quote
garbage that pollutes raw IEX quotes. Use this for multi-stock / cross-asset work where we
need clean, aligned series across symbols and a large sample.

Usage:
  python -m soc.data.fetch_bars --symbols AAPL,MSFT,NVDA --start 2024-06-03 --end 2024-12-31
Saves data_store/<SYM>_1min.parquet with columns [ts, c].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT / ".env")
except Exception:
    pass

DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
STORE = _ROOT / "data_store"


def _headers():
    k, s = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
    if not k or not s:
        sys.exit("Missing ALPACA keys in .env")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def fetch_bars(symbol, start, end, timeframe="1Min", feed="iex", adjustment="all"):
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    rows, tok = [], None
    while True:
        p = {"timeframe": timeframe, "start": start, "end": end, "limit": 10000,
             "feed": feed, "adjustment": adjustment}   # split/dividend adjusted -> continuous
        if tok:
            p["page_token"] = tok
        r = requests.get(url, headers=_headers(), params=p, timeout=30)
        if r.status_code != 200:
            sys.exit(f"Alpaca error {r.status_code}: {r.text[:200]}")
        d = r.json()
        for b in d.get("bars") or []:
            rows.append((b["t"], b["c"]))
        tok = d.get("next_page_token")
        print(f"  {symbol}: {len(rows)} bars", flush=True)
        if not tok:
            break
    df = pd.DataFrame(rows, columns=["t", "c"])
    # robust epoch-seconds (resolution-independent; .astype(int64) varies ns/us across versions)
    dt = pd.to_datetime(df["t"], format="ISO8601", utc=True)
    df["ts"] = (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return df[["ts", "c"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="comma-separated, e.g. AAPL,MSFT,NVDA")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--timeframe", default="1Min")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--adjustment", default="all", help="raw|split|dividend|all")
    args = ap.parse_args()
    STORE.mkdir(exist_ok=True)
    for sym in [s.strip().upper() for s in args.symbols.split(",")]:
        df = fetch_bars(sym, args.start, args.end, args.timeframe, args.feed, args.adjustment)
        out = STORE / f"{sym}_1min.parquet"
        df.to_parquet(out, index=False)
        print(f"saved {len(df):,} bars -> {out}")


if __name__ == "__main__":
    main()
