"""Run the SOC engine and serve the dashboard from a SINGLE websocket server.

One process, one port. The same `websockets` server serves the static dashboard files
(via process_request) AND the live websocket stream at path `/stream`. Because the page and
the socket share one origin and one port, there is no second server to land on by accident
and no cross-origin handshake to mangle — which is what broke the two-port setup.

Pipeline:  Feed -> Engine -> Strategy -> Metrics -> Hub -> ws(/stream) -> browser.

  python -m soc.server.run --feed synthetic
  python -m soc.server.run --feed replay --symbol AAPL --stride 25 --fps 30
  python -m soc.server.run --feed replay --symbol AAPL --headless
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from ..data.replay_feed import ReplayFeed
from ..data.synthetic_feed import SyntheticMarket
from ..metrics.metrics import Metrics
from ..model.engine import Engine
from ..model.hazard import HazardModel
from ..strategy.allocate import Strategy
from .bus import Hub
from .multi import producer_multi

WEB = Path(__file__).resolve().parents[2] / "web"
CTYPES = {".html": "text/html; charset=utf-8",
          ".js": "application/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8"}


# --------------------------------------------------------------------------- pipeline
def build_feed(args):
    if args.feed == "synthetic":
        return SyntheticMarket(symbol=args.symbol, n_ticks=args.n_ticks, seed=args.seed)
    return ReplayFeed(args.symbol, max_ticks=args.max_ticks)


def build_pipeline(args):
    engine = Engine(HazardModel(eta_theta=args.eta_theta), initial_gap=args.initial_gap)
    return engine, Strategy(initial_capital=args.capital), Metrics()


def view_frame(ev: dict, tr: dict) -> dict:
    return {"type": "view", "ts": ev["ts"], "symbol": ev["symbol"], "x": ev["x"],
            "x_c": ev["x_c"], "x_bar": ev["x_bar"], "gap": ev["gap"], "p": ev["p"], "y": ev["y"],
            "params": ev["params"], "equity": tr["equity"], "exposure": tr["exposure"],
            "return_pct": tr["return_pct"], "traded": tr["traded"]}


async def producer(hub, args):
    feed = build_feed(args)
    engine, strat, metrics = build_pipeline(args)
    await hub.broadcast({"type": "config", "symbol": args.symbol, "feed": args.feed,
                         "capital": args.capital})
    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    since = 0
    seen = 0
    for tick in feed:
        ev = engine.step(tick)
        if ev is None:
            continue
        seen += 1
        if seen <= args.warmup:                  # WARM-UP: train x_c/params silently, no trades/metrics
            if seen % max(1, args.warmup // 50) == 0:
                await hub.broadcast({"type": "status", "phase": "warmup", "seen": seen, "total": args.warmup})
                await asyncio.sleep(0)
            continue
        tr = strat.on_event(ev)
        metrics.update(ev["p"], ev["y"])
        since += 1
        if since >= args.stride:
            since = 0
            await hub.broadcast(view_frame(ev, tr))
            await hub.broadcast(metrics.snapshot())
            await asyncio.sleep(frame_dt)        # yield to the loop (serves ws clients)
        elif since % 256 == 0:
            await asyncio.sleep(0)               # keep loop responsive on unthrottled runs
    await hub.broadcast(metrics.snapshot())
    snap = metrics.snapshot()
    print("Feed exhausted. Final:", json.dumps({k: snap[k] for k in
          ("n", "logloss", "brier", "winrate", "base_rate")}, default=float))


# --------------------------------------------------------------------------- server
async def serve_http_ws(args):
    multi = bool(args.symbols and "," in args.symbols)
    index_file = "multi.html" if multi else "index.html"
    prod = producer_multi if multi else producer
    hub = Hub()

    async def process_request(connection, request):
        """Serve static files for normal GETs; let /stream proceed to a ws upgrade."""
        path = request.path.split("?")[0]
        if path == "/stream":
            return None                          # None => continue with websocket handshake
        if path in ("/", ""):
            path = "/" + index_file
        f = (WEB / path.lstrip("/")).resolve()
        if not str(f).startswith(str(WEB)) or not f.is_file():
            return connection.respond(404, "not found")
        body = f.read_bytes()
        headers = Headers()
        headers["Content-Type"] = CTYPES.get(f.suffix, "application/octet-stream")
        headers["Content-Length"] = str(len(body))
        return Response(200, "OK", headers, body)

    async def ws_handler(conn):
        if conn.request.path.split("?")[0] != "/stream":
            await conn.close()
            return
        await hub.register(conn)
        try:
            async for _ in conn:                 # we never expect client messages
                pass
        except Exception:
            pass
        finally:
            hub.unregister(conn)

    async with serve(ws_handler, "localhost", args.port, process_request=process_request):
        tag = f"universe={args.symbols}" if multi else f"feed={args.feed} symbol={args.symbol}"
        print(f"SOC dashboard:  http://localhost:{args.port}   ({tag})")
        print("Ctrl-C to stop.")
        await prod(hub, args)
        await asyncio.Future()                   # keep serving after the feed ends


# --------------------------------------------------------------------------- headless
async def run_headless(args):
    class _Print(Hub):
        async def broadcast(self, msg):
            if msg.get("type") == "metric" and msg["n"] % (args.stride * 200) == 0:
                print(f"n={msg['n']:>8}  loss={msg['logloss']:.4f}  brier={msg['brier']:.4f}"
                      f"  win={msg['winrate']:.4f}  base={msg['base_rate']:.4f}", flush=True)
    args.fps = 0.0
    await producer(_Print(), args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", choices=["synthetic", "replay"], default="synthetic")
    ap.add_argument("--symbol", default="SYN")
    ap.add_argument("--symbols", default="", help="comma list -> multi-stock universe view, e.g. AAPL,MSFT,NVDA")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--eta-theta", type=float, default=1e-3)
    ap.add_argument("--initial-gap", type=float, default=0.08, help="initial LOG gap (0.08 = 8%)")
    ap.add_argument("--warmup", type=int, default=5000, help="train this many ticks/bars before deploying capital")
    ap.add_argument("--couple", type=float, default=1.0, help="cross-asset coupling strength: correlated peer returns nudge x_c. 0=off")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-ticks", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-ticks", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_headless(args) if args.headless else serve_http_ws(args))


if __name__ == "__main__":
    main()
