"""Single-stock virtual-money strategy (v1).

Signal:   edge = 1 - 2p   (p = P(downtick)).  edge > 0 -> long, edge < 0 -> short.
Sizing:   target exposure = edge * max_notional  (dollar-notional, signed).
No-trade band:  only rebalance when |target - exposure| exceeds `band * max_notional`.
                This is the key cost control — it stops us churning on bid-ask-bounce
                noise in p and only pays the spread when conviction genuinely shifts.
Costs:    `cost_rate` fraction of the traded notional on every rebalance.

Timing is honest: the prediction p for the move (t-1 -> t) sets the exposure we hold
*through* that move, then the move is realized. No look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Strategy:
    initial_capital: float = 100_000.0
    max_notional: float = 100_000.0     # |exposure| cap (== 1x capital by default)
    band: float = 0.05                  # rebalance threshold, fraction of max_notional
    cost_rate: float = 5e-5             # cost per unit traded notional (5 bps round-ish)

    equity: float = 0.0
    exposure: float = 0.0               # signed dollar notional currently held
    prev_x: Optional[float] = None
    total_cost: float = 0.0

    def __post_init__(self):
        self.equity = self.initial_capital

    def on_event(self, ev: dict) -> dict:
        x = ev["x"]
        p = ev["p"]
        edge = 1.0 - 2.0 * p
        target = max(-self.max_notional, min(self.max_notional, edge * self.max_notional))

        traded = False
        cost = 0.0
        # rebalance (at prev price, before the move) only if conviction shift clears the band
        if abs(target - self.exposure) > self.band * self.max_notional:
            delta = target - self.exposure
            cost = self.cost_rate * abs(delta)
            self.equity -= cost
            self.total_cost += cost
            self.exposure = target
            traded = True

        # realize the move with the exposure we now hold
        pnl_step = 0.0
        if self.prev_x is not None and self.prev_x > 0:
            pnl_step = self.exposure * (x - self.prev_x) / self.prev_x
            self.equity += pnl_step
        self.prev_x = x

        return {
            "type": "trade",
            "ts": ev["ts"],
            "symbol": ev["symbol"],
            "edge": edge,
            "exposure": self.exposure,
            "target": target,
            "traded": traded,
            "cost": cost,
            "pnl_step": pnl_step,
            "equity": self.equity,
            "total_cost": self.total_cost,
            "return_pct": 100.0 * (self.equity - self.initial_capital) / self.initial_capital,
        }
