"""Visualise the cross-asset / loading analysis (clean minute bars + clean AAPL ticks).

Cross-asset panels use CONSOLIDATED minute bars (clean, 7 months) because raw IEX quotes
for names like MSFT are too corrupted to trust. The single-stock loading ramp uses clean
AAPL tick data. Falsification protocol: train 60% / test 40% by time, OOS comparison.

Panels:
  1. AAPL / MSFT / NVDA normalised price (7 months) — co-movement.
  2. Lead-lag: corr(j[t], AAPL[t+k]) for MSFT & NVDA — does anything lead AAPL?
  3. AAPL TICK loading ramp: P(downtick) as the pile loads, bounce-controlled.
  4. OOS skill: bounce vs lead-lag NULL vs SOC conditional.

Run: PYTHONPATH=. ./.venv/bin/python scripts/viz_crossasset.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soc.data.clean import clean_quotes
from soc.data.replay_feed import STORE

BG, FG, GRID, MUT = "#0d1117", "#e6edf3", "#283040", "#8b949e"
COL = {"AAPL": "#f0883e", "MSFT": "#58a6ff", "NVDA": "#3fb950"}


def bars(sym):
    df = pd.read_parquet(STORE / f"{sym}_1min.parquet")
    return pd.Series(df["c"].to_numpy(), index=df["ts"].to_numpy())


def style(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=10, loc="left")
    ax.tick_params(colors=MUT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)


def zcols(d, cols, mean, std):
    return np.column_stack([((d[c]-mean[c])/(std[c] or 1)).to_numpy() for c in cols] + [np.ones(len(d))])


def main():
    syms = ["AAPL", "MSFT", "NVDA"]
    px = pd.DataFrame({s: bars(s) for s in syms}).dropna()
    r = np.log(px).diff().dropna()
    r.columns = [f"r{s[0]}" for s in syms]   # rA, rM, rN
    # mask out overnight/gap bars (consecutive bars >120s apart) for the return-based panels
    dt = np.diff(px.index.to_numpy())
    intr = pd.Series(dt < 120, index=r.index)

    fig, ax = plt.subplots(2, 2, figsize=(15, 8.5)); fig.patch.set_facecolor(BG)
    a1, a2, a3, a4 = ax.flat
    xidx = np.arange(len(px))

    # Panel 1: normalised prices
    style(a1, "1  normalised price — 1-min bars, Jun–Dec 2024 (clean consolidated)")
    for s in syms:
        a1.plot(xidx, 100*px[s].to_numpy()/px[s].iloc[0], color=COL[s], lw=1.0, label=s)
    a1.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    a1.set_xlabel("minute index", color=MUT, fontsize=8)

    # Panel 2: lead-lag cross-corr
    style(a2, "2  lead-lag: corr(j[t], AAPL[t+k])   (k>0 ⇒ j leads AAPL)")
    ks = list(range(-3, 6))
    for s, rc in (("MSFT", "rM"), ("NVDA", "rN")):
        rj, rA = r[rc][intr].to_numpy(), r["rA"][intr].to_numpy(); n = len(rA); cc = []
        for k in ks:
            if k >= 0: cc.append(np.corrcoef(rj[:n-k or None], rA[k:])[0, 1] if k else np.corrcoef(rj, rA)[0, 1])
            else: cc.append(np.corrcoef(rj[-k:], rA[:n+k])[0, 1])
        a2.plot(ks, cc, "o-", color=COL[s], lw=1.3, label=s)
    a2.axhline(0, color=MUT, lw=0.8); a2.axvline(0, color=MUT, lw=0.5, ls=":")
    a2.set_xlabel("lag k (minutes)", color=MUT, fontsize=8)
    a2.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)

    # Panel 3: AAPL TICK loading ramp (clean ticks), bounce-controlled, no look-ahead
    m = clean_quotes(pd.read_parquet(STORE / "AAPL_quotes.parquet"))["mid"].to_numpy()
    mv = np.sign(np.diff(m)); yd = (mv < 0).astype(int)
    B = pd.Series(m).ewm(halflife=1000).mean().to_numpy()
    vol = pd.Series(np.abs(np.diff(m))).ewm(halflife=500).mean().shift(1).bfill().to_numpy()
    load = (m[:-1] - B[:-1]) / np.maximum(vol, 1e-9)
    prev = np.concatenate([[0.0], mv[:-1]])
    qs = np.quantile(load, np.linspace(0, 1, 9)); ctr, al, pu, pdn = [], [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (load >= lo) & (load < hi); ctr.append((lo+hi)/2)
        al.append(yd[s].mean()); pu.append(yd[s & (prev > 0)].mean()); pdn.append(yd[s & (prev < 0)].mean())
    style(a3, "3  AAPL tick loading ramp: P(downtick) as pile loads →")
    a3.plot(ctr, al, "o-", color=FG, lw=1.5, label="all ticks")
    a3.plot(ctr, pu, "o-", color=COL["AAPL"], lw=1.3, label="prev UP (bounce-controlled)")
    a3.plot(ctr, pdn, "o-", color=COL["MSFT"], lw=1.3, label="prev DOWN")
    a3.axhline(0.5, color=MUT, ls="--", lw=0.8)
    a3.set_xlabel("load (price above slow baseline, vol-scaled) → loaded toward x_c", color=MUT, fontsize=8)
    a3.set_ylabel("P(next tick down)", color=MUT, fontsize=8)
    a3.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)

    # Panel 4: OOS falsification on minute bars
    d = r.copy()
    d["rM_lag1"] = d["rM"].shift(1); d["rN_lag1"] = d["rN"].shift(1)
    base = px["AAPL"].ewm(halflife=60).mean().shift(1)
    vol2 = r["rA"].abs().ewm(halflife=60).mean().shift(1)
    d["load"] = (px["AAPL"].shift(1).reindex(d.index) - base.reindex(d.index)) / vol2.replace(0, np.nan)
    d["rM_x_load"] = d["rM"] * d["load"]; d["rN_x_load"] = d["rN"] * d["load"]
    d["target"] = d["rA"].shift(-1)
    d = d[intr.reindex(d.index).fillna(False)].dropna()   # intraday only
    allc = ["rA", "rM", "rN", "rM_lag1", "rN_lag1", "load", "rM_x_load", "rN_x_load"]
    cut = int(len(d)*0.6); tr, te = d.iloc[:cut], d.iloc[cut:]
    mean, std = tr[allc].mean(), tr[allc].std()
    def oos(fc):
        coef, *_ = np.linalg.lstsq(zcols(tr, fc, mean, std), tr["target"].to_numpy(), rcond=None)
        pred = zcols(te, fc, mean, std) @ coef
        return np.corrcoef(pred, te["target"].to_numpy())[0, 1]
    names = ["bounce\n(rA)", "lead-lag NULL\n(+rM,rN)", "SOC cond.\n(+load,+r·load)"]
    vals = [oos(["rA"]), oos(["rA", "rM", "rN", "rM_lag1", "rN_lag1"]), oos(allc)]
    style(a4, f"4  OUT-OF-SAMPLE skill (train {cut:,}/test {len(te):,} min) — SOC must beat lead-lag")
    a4.bar(names, vals, color=[MUT, COL["MSFT"], "#bc8cff"])
    for i, v in enumerate(vals): a4.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", color=FG, fontsize=9)
    a4.axhline(0, color=MUT, lw=0.8); a4.set_ylabel("OOS corr(pred, next-min return)", color=MUT, fontsize=8)

    fig.tight_layout()
    Path("research").mkdir(exist_ok=True)
    p = Path("research") / "crossasset.png"
    fig.savefig(p, dpi=110, facecolor=BG); print(f"saved {p}")


if __name__ == "__main__":
    main()
