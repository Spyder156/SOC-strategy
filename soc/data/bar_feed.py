"""Aligned multi-symbol minute-bar replay (for the multi-stock live monitor).

Loads each symbol's clean, split-adjusted minute bars and aligns them on a common
timestamp grid, so every step advances the whole universe together — exactly what the
cross-asset coupling needs. Yields (ts, {symbol: price}) per minute.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

STORE = Path(__file__).resolve().parents[2] / "data_store"


class BarFeed:
    def __init__(self, symbols: List[str], speed: float = 0.0, max_bars: Optional[int] = None):
        self.symbols = [s.strip().upper() for s in symbols]
        self.speed = speed
        self.max_bars = max_bars

    def __iter__(self) -> Iterator[Tuple[float, Dict[str, float]]]:
        cols = {}
        for s in self.symbols:
            p = STORE / f"{s}_1min.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Fetch it: python -m soc.data.fetch_bars "
                    f"--symbols {s} --start <date> --end <date>")
            df = pd.read_parquet(p)
            cols[s] = pd.Series(df["c"].to_numpy(), index=df["ts"].to_numpy())
        al = pd.DataFrame(cols).dropna()
        interval = 1.0 / self.speed if self.speed and self.speed > 0 else 0.0
        n = 0
        for ts, row in al.iterrows():
            yield float(ts), {s: float(row[s]) for s in self.symbols}
            n += 1
            if self.max_bars and n >= self.max_bars:
                break
            if interval:
                time.sleep(interval)
