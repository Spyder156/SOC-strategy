"""Fetch historical Alpaca quotes for one symbol and store mid-prices locally.

We pull QUOTES (not trades) because the model's tick is a mid-price move, and
mid = (bid + ask) / 2 strips the bid-ask bounce that would otherwise masquerade as
avalanches. Free tier uses the IEX feed, which is plenty for one symbol in v1.

Usage:
    python -m soc.data.fetch_history --symbol AAPL --start 2024-01-02 --end 2024-02-01

Credentials come from the environment (.env): ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY.
Output: data_store/<SYMBOL>_quotes.parquet with columns [ts, bid, ask, mid]
(consecutive unchanged mids are dropped — they are not ticks).
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
    load_dotenv(dotenv_path=_ROOT / ".env")   # explicit path: works even when piped via stdin
except Exception:
    pass

DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
STORE = _ROOT / "data_store"


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        sys.exit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (put them in .env).")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def fetch_quotes(symbol: str, start: str, end: str, feed: str = "iex",
                 limit: int = 10_000) -> pd.DataFrame:
    url = f"{DATA_URL}/v2/stocks/{symbol}/quotes"
    headers = _headers()
    rows, token, page = [], None, 0
    while True:
        params = {"start": start, "end": end, "limit": limit, "feed": feed}
        if token:
            params["page_token"] = token
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            sys.exit(f"Alpaca error {r.status_code}: {r.text[:300]}")
        data = r.json()
        quotes = data.get("quotes") or []
        for q in quotes:
            bp, ap = q.get("bp", 0.0), q.get("ap", 0.0)
            if bp <= 0 or ap <= 0:           # skip one-sided / empty books
                continue
            rows.append((q["t"], bp, ap))
        page += 1
        print(f"  page {page}: +{len(quotes)} quotes (kept total {len(rows)})", flush=True)
        token = data.get("next_page_token")
        if not token:
            break

    if not rows:
        sys.exit("No quotes returned — check symbol, date range, and market hours.")

    df = pd.DataFrame(rows, columns=["t", "bid", "ask"])
    _dt = pd.to_datetime(df["t"], format="ISO8601", utc=True)
    df["ts"] = (_dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df[["ts", "bid", "ask", "mid"]].sort_values("ts").reset_index(drop=True)
    # drop consecutive unchanged mids — only mid *moves* are ticks
    df = df[df["mid"].ne(df["mid"].shift())].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or RFC3339")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or RFC3339")
    ap.add_argument("--feed", default="iex")
    args = ap.parse_args()

    print(f"Fetching {args.symbol} quotes {args.start}..{args.end} (feed={args.feed})")
    df = fetch_quotes(args.symbol, args.start, args.end, feed=args.feed)
    STORE.mkdir(exist_ok=True)
    out = STORE / f"{args.symbol}_quotes.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} mid-ticks -> {out}")


if __name__ == "__main__":
    main()
