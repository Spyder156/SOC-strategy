"""Live producer for the Bouchaud fragility model.

Per stock: realized vol, large-move (avalanche) detection, self-exciting intensity, a learned
P(large move) and a branching-ratio proxy n. Strategy: hold the universe long but vol-target
AND cut exposure when fragility is high — compared live against buy-and-hold. Also streams the
avalanche-size histogram so the dashboard can show the power-law tail (and any Dragon Kings).
"""

from __future__ import annotations

import math

from ..data.bar_feed import BarFeed
from ..model.fragility import FragilityModel

ANN = 98280.0
SESSION_GAP_SEC = 300.0
HIST_LO, HIST_HI, HIST_NB = 2.5, 40.0, 12      # log-spaced avalanche-size bins


def _hist(sizes):
    lo, hi = math.log(HIST_LO), math.log(HIST_HI)
    centers = [math.exp(lo + (hi - lo) * (b + 0.5) / HIST_NB) for b in range(HIST_NB)]
    counts = [0] * HIST_NB
    for s in sizes:
        if s <= 0:
            continue
        b = int((math.log(max(s, HIST_LO)) - lo) / (hi - lo) * HIST_NB)
        b = min(HIST_NB - 1, max(0, b))
        counts[b] += 1
    return {"centers": [round(c, 2) for c in centers], "counts": counts}


def _sharpe(n, sm, sq):
    if n < 30:
        return 0.0
    mu = sm / n
    var = sq / n - mu * mu
    return (mu / math.sqrt(var)) * math.sqrt(ANN) if var > 1e-18 else 0.0


async def producer_fragility(hub, args):
    syms = [s.strip().upper() for s in args.symbols.split(",")]
    feed = BarFeed(syms, max_bars=args.max_ticks)
    fm = {s: FragilityModel() for s in syms}
    await hub.broadcast({"type": "config", "mode": "fragility", "symbols": syms, "capital": args.capital})

    cap = args.capital
    eq = bheq = cap
    n_s = ssum = ssq = bhsum = bhsq = 0
    sizes, vols_seen = [], []
    prev_vol = prev_ts = None
    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    since = 0

    for ts, prices in feed:
        gap_bar = prev_ts is not None and (ts - prev_ts) > SESSION_GAP_SEC
        prev_ts = ts
        evs = {s: fm[s].step(prices[s]) for s in syms}
        if any(v is None for v in evs.values()):
            continue

        r = {s: evs[s]["r"] for s in syms}
        rbh = sum(r.values()) / len(syms)
        frag = sum(evs[s]["p_large"] for s in syms) / len(syms)
        vol = sum(evs[s]["sigma"] for s in syms) / len(syms)
        vols_seen.append(vol)
        win = vols_seen[-5000:]
        tvol = sorted(win)[len(win) // 2]                       # adaptive target = median vol

        e_vt = min(3.0, tvol / prev_vol) if prev_vol else 1.0   # vol-target (causal)
        e_fr = max(0.0, 1.0 - frag)                             # fragility dial (causal forecast)
        exposure = e_vt * e_fr

        if not gap_bar:
            fr_ret = exposure * rbh
            eq *= (1 + fr_ret); bheq *= (1 + rbh)
            n_s += 1; ssum += fr_ret; ssq += fr_ret * fr_ret; bhsum += rbh; bhsq += rbh * rbh
            for s in syms:
                if evs[s]["is_av"]:
                    sizes.append(evs[s]["size"])
            if len(sizes) > 8000:
                sizes = sizes[-8000:]
        prev_vol = vol

        since += 1
        if since >= args.stride:
            since = 0
            stocks = {s: {"price": prices[s], "sigma": evs[s]["sigma"], "k": fm[s].k_aval,
                          "p_large": evs[s]["p_large"], "n": evs[s]["n"], "is_av": evs[s]["is_av"],
                          "wS": round(evs[s]["wS"], 2)} for s in syms}
            await hub.broadcast({
                "type": "frag", "ts": ts, "symbols": syms, "stocks": stocks,
                "exposure": exposure, "frag": frag,
                "equity": eq, "bh_equity": bheq,
                "return_pct": 100 * (eq - cap) / cap, "bh_return_pct": 100 * (bheq - cap) / cap,
                "sharpe": _sharpe(n_s, ssum, ssq), "bh_sharpe": _sharpe(n_s, bhsum, bhsq),
                "size_hist": _hist(sizes), "n_aval": len(sizes)})
            await __import__("asyncio").sleep(frame_dt)
    print("Fragility feed exhausted.")
