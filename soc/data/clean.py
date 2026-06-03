"""Quote cleaning — drop the wide/stale IEX quotes that poison the mid.

The free IEX feed frequently rests one side of its book far from the real price
(e.g. a stub bid at $170 while the ask is ~$188). The resulting mid spikes by several
dollars for a tick and snaps back. These are not real price moves; left in, they create
~10% of ticks as fake down-spikes and inflate any backtest.

We keep a quote only if its spread is plausible: positive, uncrossed, and no wider than
`max(min_abs, mult × median spread)`. The median spread is robust to the garbage tail,
so this adapts per symbol/session without hand-tuning. Returns a clean [ts, mid] frame
with consecutive unchanged mids dropped (only mid *moves* are ticks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_quotes(df: pd.DataFrame, mult: float = 5.0, min_abs: float = 0.05) -> pd.DataFrame:
    if "bid" not in df.columns or "ask" not in df.columns:
        # already a clean [ts, mid] frame
        out = df[["ts", "mid"]].copy()
        return out[out["mid"].ne(out["mid"].shift())].reset_index(drop=True)

    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    spread = ask - bid
    valid = spread[(spread > 0)]
    med = float(np.median(valid)) if valid.size else 0.0
    thr = max(min_abs, mult * med)

    keep = (bid > 0) & (ask > 0) & (bid < ask) & (spread <= thr)
    f = df[keep]
    out = pd.DataFrame({
        "ts": f["ts"].to_numpy(dtype=float),
        "mid": (f["bid"].to_numpy(dtype=float) + f["ask"].to_numpy(dtype=float)) / 2.0,
    }).sort_values("ts")
    out = out[out["mid"].ne(out["mid"].shift())].reset_index(drop=True)
    return out
