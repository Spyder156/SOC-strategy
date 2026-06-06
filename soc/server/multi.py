"""Multi-stock producer: a universe of SOC engines with cross-asset coupling.

Each symbol runs its own latent-x_c hazard engine. After every aligned minute bar we:
  1. step every engine,
  2. update an online (EWMA) correlation matrix of returns,
  3. apply the CROSS-ASSET COUPLING — nudge each stock's x_c by its correlated peers'
     moves (the real engine: a peer dropping pulls your x_c down, shrinking your gap),
  4. size each position by edge and roll a portfolio.

Emits one "uni" snapshot per minute: every stock's state + portfolio + correlation matrix.
"""

from __future__ import annotations

import math

from ..data.bar_feed import BarFeed
from ..data.feed import Tick
from ..model.engine import Engine
from ..model.hazard import HazardModel
from ..metrics.metrics import Metrics
from ..strategy.allocate import UniverseStrategy

COUPLE_ETA = 1.0             # scales the LEARNING RATE of the coupling matrix (--couple)
CORR_HALFLIFE = 300.0        # bars (for the displayed return correlation)
SESSION_GAP_SEC = 300.0      # time jump > this = session boundary (overnight gap) -> reprice
# learned coupling matrix W_ij (peer j's move -> stock i's x_c), trained on prediction error:
ETA_W = 0.05                 # base learning rate for W
W_TAU = 4000.0               # decaying LR so W converges
W_FLOOR = 0.12               # ...but keeps adapting
WMAX = 5.0                   # clamp on each W_ij


def _a(halflife):
    return 1.0 - math.pow(0.5, 1.0 / halflife)


async def producer_multi(hub, args):
    syms = [s.strip().upper() for s in args.symbols.split(",")]
    feed = BarFeed(syms, max_bars=args.max_ticks)
    eng = {s: Engine(HazardModel(eta_theta=args.eta_theta), initial_gap=args.initial_gap) for s in syms}
    # parallel counterfactual engines: identical, but NEVER receive the cross-asset coupling.
    # Their x_c is "what x_c would be if this stock were alone" -> x_c_no_corr.
    eng_nc = {s: Engine(HazardModel(eta_theta=args.eta_theta), initial_gap=args.initial_gap) for s in syms}
    strat = UniverseStrategy(initial_capital=args.capital)   # ONE budget split across the universe
    met = {s: Metrics() for s in syms}

    await hub.broadcast({"type": "config", "mode": "multi", "symbols": syms,
                         "capital": args.capital})

    prev = {}                                   # last price per symbol
    prev_ret = None                             # previous bar's returns (drove last coupling)
    mean = {s: 0.0 for s in syms}               # EWMA mean return
    var = {s: 1e-8 for s in syms}               # EWMA variance
    cov = {(i, j): 0.0 for i in syms for j in syms if i < j}
    W = {i: {j: 0.0 for j in syms if j != i} for i in syms}   # LEARNED coupling matrix
    nbar = 0
    couple_eta = getattr(args, "couple", COUPLE_ETA)
    ar = _a(CORR_HALFLIFE)
    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    since = 0
    seen = 0

    def rho(i, j):
        if i == j:
            return 1.0
        c = cov[(i, j)] if i < j else cov[(j, i)]
        return c / math.sqrt(var[i] * var[j] + 1e-18)

    prev_ts = None
    for ts, prices in feed:
        # a large time jump = session boundary (overnight gap / halt): a DISCONTINUITY,
        # not continuous loading. Reprice x_c with the gap instead of treating it as a tick.
        gap_bar = prev_ts is not None and (ts - prev_ts) > SESSION_GAP_SEC
        prev_ts = ts

        evs = {}
        for s in syms:
            tk = Tick(ts, s, prices[s])
            ev = eng[s].reprice(tk) if gap_bar else eng[s].step(tk)
            # step the counterfactual engine identically; it just never gets coupled
            eng_nc[s].reprice(tk) if gap_bar else eng_nc[s].step(tk)
            if ev is not None:
                evs[s] = ev
        if len(evs) < len(syms):                # warm-up (first bar returns None)
            prev = dict(prices)
            strat.carry(prices)
            continue

        if not gap_bar:
            # correlate the stocks via their RETURNS (the real co-movement: ~0.4 for AAPL/MSFT;
            # the x_c's themselves move too slowly to correlate directly).
            ret = {s: (prices[s] - prev[s]) / prev[s] if prev.get(s) else 0.0 for s in syms}
            for s in syms:
                mean[s] = (1 - ar) * mean[s] + ar * ret[s]
                var[s] = (1 - ar) * var[s] + ar * (ret[s] - mean[s]) ** 2
            for (i, j) in cov:
                cov[(i, j)] = (1 - ar) * cov[(i, j)] + ar * (ret[i] - mean[i]) * (ret[j] - mean[j])

            # --- LEARN the coupling matrix W from prediction error ---
            # Last bar's coupling (driven by prev_ret) shifted x_c into THIS bar's prediction p_i.
            # Gradient of stock i's cross-entropy w.r.t. W_ij (through x_c -> log-gap -> p):
            #   dL/dW_ij = (p_i - y_i) * (-alpha_i / gap_i) * r_j(prev)
            # so descent gives:  W_ij += lr * (p_i - y_i) * (alpha_i / gap_i) * r_j(prev)
            nbar += 1
            lr = ETA_W * couple_eta * max(W_FLOOR, W_TAU / (W_TAU + nbar))
            if prev_ret is not None:
                for i in syms:
                    ev = evs[i]
                    resid = ev["p"] - ev["y"]
                    gi = max(math.log(ev["x_c"] / ev["x"]), 1e-4)
                    grad_i = resid * (ev["params"]["alpha"] / gi)
                    for j in syms:
                        if j == i:
                            continue
                        W[i][j] = max(-WMAX, min(WMAX, W[i][j] + lr * grad_i * prev_ret[j]))
            # --- APPLY: x_c_i *= exp( sum_j W_ij * r_j ) ---
            for i in syms:
                push = sum(W[i][j] * ret[j] for j in syms if j != i)
                st = eng[i].state
                m = eng[i].model
                st.x_c *= math.exp(push)
                floor = st.x * math.exp(max(m.eps, m.k_vol * st.vol))   # vol-scaled, no touch
                if st.x_c < floor:
                    st.x_c = floor
            prev_ret = dict(ret)
        prev = dict(prices)
        seen += 1
        training = seen <= args.warmup

        # WARM-UP: train and SHOW (charts + params converge live) but don't deploy capital or
        # count metrics. Trade + score only after warm-up. Gaps never trade.
        if training or gap_bar:
            strat.carry(prices)
            tr = {"exposure": {s: 0.0 for s in syms}, "weight": {s: 0.0 for s in syms},
                  "return_pct": 100 * (strat.equity - strat.initial_capital) / strat.initial_capital,
                  "sharpe": strat.sharpe()}
        else:
            tr = strat.step({s: evs[s]["p"] for s in syms}, prices,
                            {s: eng[s].state.vol for s in syms})
            for s in syms:
                met[s].update(evs[s]["p"], evs[s]["y"])

        stocks = {}
        for s in syms:
            ev = evs[s]
            ms = met[s].snapshot()
            x_c_nc = eng_nc[s].state.x_c if eng_nc[s].state else ev["x_c"]
            stocks[s] = {"x": ev["x"], "x_c": ev["x_c"], "x_c_no_corr": x_c_nc,
                         "x_bar": ev["x_bar"], "gap": ev["gap"],
                         "p": ev["p"], "y": ev["y"],
                         "exposure": tr["exposure"].get(s, 0.0), "weight": tr["weight"].get(s, 0.0),
                         "winrate": ms["winrate"], "brier": ms["brier"], "logloss": ms["logloss"],
                         "n": ms["n"], "reprice": gap_bar, "params": ev["params"]}

        since += 1
        if since >= args.stride:
            since = 0
            corr = [[round(rho(i, j), 3) for j in syms] for i in syms]
            couple = [[(0.0 if i == j else round(W[i][j], 3)) for j in syms] for i in syms]
            await hub.broadcast({
                "type": "uni", "ts": ts, "symbols": syms, "stocks": stocks,
                "equity": strat.equity, "return_pct": tr["return_pct"], "sharpe": tr["sharpe"],
                "training": training, "seen": seen, "warmup": args.warmup,
                "corr": corr, "couple": couple})
            await __import__("asyncio").sleep(frame_dt)

    print("Universe feed exhausted.")
