"""Run the SOC engine and stream events to the dashboards over a websocket.

Pipeline:  Feed -> Engine (predict+learn) -> Strategy (virtual money) -> Metrics
           -> Hub.broadcast -> browser.

Feeds:
  --feed synthetic   self-contained SOC sim (mechanics demo; no keys needed)
  --feed replay      real Alpaca mid-ticks from data_store/<symbol>_quotes.parquet

The model processes `stride` ticks per emitted frame and sends ~`fps` frames/sec, so it
converges fast while staying watchable. `--headless` skips the socket and just prints
progress (handy for validating on real data before opening the UI).

Examples:
  python -m soc.server.run --feed synthetic
  python -m soc.server.run --feed replay --symbol AAPL --stride 25 --fps 30
  python -m soc.server.run --feed replay --symbol AAPL --headless
"""

from __future__ import annotations

import argparse
import asyncio
import json

from ..data.replay_feed import ReplayFeed
from ..data.synthetic_feed import SyntheticMarket
from ..metrics.metrics import Metrics
from ..model.engine import Engine
from ..model.hazard import HazardModel
from ..strategy.allocate import Strategy
from .bus import Hub


def build_feed(args):
    if args.feed == "synthetic":
        return SyntheticMarket(symbol=args.symbol, n_ticks=args.n_ticks, seed=args.seed)
    return ReplayFeed(args.symbol, max_ticks=args.max_ticks)


def build_pipeline(args):
    engine = Engine(HazardModel(eta_theta=args.eta_theta, eta_xc=args.eta_xc),
                    initial_gap=args.initial_gap)
    strat = Strategy(initial_capital=args.capital)
    metrics = Metrics()
    return engine, strat, metrics


def view_frame(ev: dict, tr: dict) -> dict:
    """Compact frame for the charts (one per stride)."""
    return {
        "type": "view",
        "ts": ev["ts"],
        "symbol": ev["symbol"],
        "x": ev["x"],
        "x_c": ev["x_c"],
        "gap": ev["gap"],
        "p": ev["p"],
        "y": ev["y"],
        "params": ev["params"],
        "equity": tr["equity"],
        "exposure": tr["exposure"],
        "return_pct": tr["return_pct"],
        "traded": tr["traded"],
    }


async def producer(hub: Hub, args):
    feed = build_feed(args)
    engine, strat, metrics = build_pipeline(args)
    hub.config = {"type": "config", "symbol": args.symbol, "feed": args.feed,
                  "capital": args.capital}

    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    since = 0
    for tick in feed:
        ev = engine.step(tick)
        if ev is None:
            continue
        tr = strat.on_event(ev)
        metrics.update(ev["p"], ev["y"])
        since += 1
        if since >= args.stride:
            since = 0
            await hub.broadcast(view_frame(ev, tr))
            await hub.broadcast(metrics.snapshot())
            if frame_dt:
                await asyncio.sleep(frame_dt)
    # final snapshot
    await hub.broadcast(metrics.snapshot())
    print("Feed exhausted. Final:", json.dumps({k: metrics.snapshot()[k]
          for k in ("n", "logloss", "brier", "winrate", "base_rate")}, default=float))


async def serve(args):
    import websockets  # imported here so --headless works without the dep installed

    hub = Hub()

    async def handler(ws):
        await hub.register(ws)
        try:
            async for _ in ws:        # we don't expect client messages; just keep open
                pass
        finally:
            hub.unregister(ws)

    async with websockets.serve(handler, "localhost", args.port):
        print(f"SOC server on ws://localhost:{args.port}  feed={args.feed} symbol={args.symbol}")
        await producer(hub, args)
        await asyncio.Future()        # keep socket open after feed ends


async def headless(args):
    class _NullHub(Hub):
        async def broadcast(self, msg):
            if msg.get("type") == "metric" and msg["n"] % (args.stride * 200) == 0:
                print(f"n={msg['n']:>8}  loss={msg['logloss']:.4f}  brier={msg['brier']:.4f}"
                      f"  win={msg['winrate']:.4f}  base={msg['base_rate']:.4f}", flush=True)
    await producer(_NullHub(), args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", choices=["synthetic", "replay"], default="synthetic")
    ap.add_argument("--symbol", default="SYN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stride", type=int, default=20, help="ticks processed per emitted frame")
    ap.add_argument("--fps", type=float, default=30.0, help="frames/sec to client (0=unthrottled)")
    ap.add_argument("--headless", action="store_true")
    # model / sim knobs
    ap.add_argument("--eta-theta", type=float, default=1e-3)
    ap.add_argument("--eta-xc", type=float, default=5e-2)
    ap.add_argument("--initial-gap", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-ticks", type=int, default=200_000, help="synthetic feed length")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-ticks", type=int, default=None, help="replay cap")
    args = ap.parse_args()

    if args.headless:
        args.fps = 0.0            # never throttle a headless convergence run
        asyncio.run(headless(args))
    else:
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
