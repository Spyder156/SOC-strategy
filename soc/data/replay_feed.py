"""Replay a locally-stored mid-price history as a tick stream.

This is the v1 dev workhorse: it replays REAL Alpaca mid-ticks (fetched by
fetch_history) as fast as we like, so the model can converge over months of data in
minutes. Replaying real data is genuine online learning, just time-compressed — not
synthetic, not pretraining. The same `Tick` interface is later fed by a live websocket.

`speed` optionally throttles to wall-clock-ish pacing for live watching; default is
unthrottled (max speed) for headless convergence runs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .feed import Tick

STORE = Path(__file__).resolve().parents[2] / "data_store"


class ReplayFeed:
    def __init__(self, symbol: str, path: Optional[str] = None,
                 speed: float = 0.0, max_ticks: Optional[int] = None):
        """
        symbol:   ticker (also used to locate data_store/<symbol>_quotes.parquet)
        speed:    ticks/second to emit; 0 = unthrottled (as fast as possible)
        max_ticks: cap for quick runs
        """
        self.symbol = symbol
        self.path = Path(path) if path else STORE / f"{symbol}_quotes.parquet"
        self.speed = speed
        self.max_ticks = max_ticks

    def __iter__(self) -> Iterator[Tick]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run: python -m soc.data.fetch_history "
                f"--symbol {self.symbol} --start <date> --end <date>"
            )
        df = pd.read_parquet(self.path)
        interval = 1.0 / self.speed if self.speed and self.speed > 0 else 0.0
        n = 0
        for ts, mid in zip(df["ts"].to_numpy(), df["mid"].to_numpy()):
            yield Tick(ts=float(ts), symbol=self.symbol, mid=float(mid))
            n += 1
            if self.max_ticks and n >= self.max_ticks:
                break
            if interval:
                time.sleep(interval)
