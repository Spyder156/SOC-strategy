"""Bouchaud reframe — proof: trade MAGNITUDE (volatility/self-excitation), not direction.

Runs the FragilityModel on the universe and compares Sharpe of:
  buy & hold  vs  classic vol-target  vs  fragility-timed  vs  fragility + vol-target.

The fragility model forecasts large moves (which cluster), and cutting exposure in predicted-
fragile periods lowers variance more than return -> higher Sharpe. Unlike direction (no edge),
this gives a real positive Sharpe.

Run: PYTHONPATH=. ./.venv/bin/python scripts/fragility_test.py
"""

import numpy as np
import pandas as pd

from soc.data.replay_feed import STORE
from soc.model.fragility import FragilityModel

SYMS = ["AAPL", "MSFT", "NVDA"]
ANN = 98280.0


def main():
    px = pd.DataFrame({s: pd.Series(pd.read_parquet(STORE / f"{s}_1min.parquet")["c"].to_numpy(),
                       index=pd.read_parquet(STORE / f"{s}_1min.parquet")["ts"].to_numpy())
                       for s in SYMS}).dropna()
    dt = np.diff(px.index.to_numpy())
    intr = np.concatenate([[False], dt < 300])          # drop overnight returns

    R, P, SIG, AV = {}, {}, {}, {}
    for s in SYMS:
        m = FragilityModel()
        r, p, sg, av = [], [], [], []
        for price in px[s].to_numpy():
            ev = m.step(float(price))
            if ev is None:
                r.append(0.0); p.append(0.5); sg.append(1e-3); av.append(0); continue
            r.append(ev["r"]); p.append(ev["p_large"]); sg.append(ev["sigma"]); av.append(ev["is_av"])
        R[s], P[s], SIG[s], AV[s] = map(np.array, (r, p, sg, av))
        pp, yy = P[s][2000:], AV[s][2000:]
        print(f"{s}: P(large) brier={((pp-yy)**2).mean():.4f} vs base {yy.mean()*(1-yy.mean()):.4f}")

    rbh = np.mean([R[s] for s in SYMS], axis=0)
    frag = np.mean([P[s] for s in SYMS], axis=0)
    vol = np.mean([SIG[s] for s in SYMS], axis=0)
    mask = intr & (np.arange(len(rbh)) > 2000)

    def sharpe(x):
        x = x[mask]
        return x.mean() / (x.std() + 1e-12) * np.sqrt(ANN)

    e_vt = np.clip(np.median(vol) / np.roll(vol, 1), 0, 3)
    e_fr = 1 - np.roll(frag, 1)
    print("\nSharpe:")
    print(f"  buy & hold        : {sharpe(rbh):+.3f}")
    print(f"  vol-target (sigma): {sharpe(e_vt * rbh):+.3f}")
    print(f"  fragility-timed   : {sharpe(e_fr * rbh):+.3f}")
    print(f"  fragility + voltgt: {sharpe(e_fr * e_vt * rbh):+.3f}")


if __name__ == "__main__":
    main()
