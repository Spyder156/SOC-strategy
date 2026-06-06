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


@dataclass
class UniverseStrategy:
    """Universe-level allocation: split the WHOLE budget across the universe by relative edge.

    Each tick:  edge_i = 1 - 2*p_i  (p_i = P(avalanche)).  We put capital where conviction is
    highest *across the universe*: w_i = edge_i / sum_j|edge_j|, so |weights| sum to the
    deployable budget. Long where p<0.5, short where p>0.5, sized by relative conviction.
    A no-trade band stops churn; P&L is one shared portfolio.
    """

    initial_capital: float = 100_000.0
    leverage: float = 1.0
    band: float = 0.08                      # wider no-trade band -> less turnover/cost
    cost_rate: float = 5e-5
    ann_bars: float = 98280.0               # ~252 trading days * 390 min, for annualised Sharpe
    # --- Sharpe-optimising knobs (optimize_sharpe branch) ---
    edge_min: float = 0.02                  # dead-zone: ignore |edge| below this (don't trade noise)
    vol_floor: float = 5e-4                 # min vol for inverse-vol sizing
    conviction_full: float = 0.12           # mean|edge| at which we fully deploy
    target_pvol: float = 2e-3               # portfolio per-bar vol target (vol-targeting)
    rev_hl: float = 50.0                    # cross-sectional reversal signal half-life (bars)

    def __post_init__(self):
        self.equity = self.initial_capital
        self.deployable = self.initial_capital * self.leverage
        self.exposure = {}                  # signed $ per symbol
        self.prev_px = {}
        self.total_cost = 0.0
        self._n = 0                         # per-bar return stats for Sharpe
        self._sum = 0.0
        self._sumsq = 0.0
        self._pvol = self.target_pvol       # EWMA of |portfolio return| (for vol-targeting)
        self._rev = {}                      # EWMA of each stock's return (reversal signal)

    def sharpe(self) -> float:
        if self._n < 30:
            return 0.0
        mean = self._sum / self._n
        var = self._sumsq / self._n - mean * mean
        if var <= 1e-18:
            return 0.0
        return (mean / (var ** 0.5)) * (self.ann_bars ** 0.5)

    def step(self, p: dict, price: dict, vol: dict) -> dict:
        syms = list(price.keys())
        # CROSS-SECTIONAL REVERSAL edge (the real gross alpha): long the relative loser, short
        # the relative winner. Uses PRIOR returns (self._rev, updated at the END of the bar) so
        # it is causal. The x_c direction signal (p) is unused here — it has no edge.
        mr = sum(self._rev.get(s, 0.0) for s in syms) / max(1, len(syms))
        edge = {s: -(self._rev.get(s, 0.0) - mr) for s in syms}
        # inverse-vol (risk-parity) raw weights: a volatile name gets a smaller position
        raw = {s: edge[s] / max(vol.get(s, self.vol_floor), self.vol_floor) for s in syms}
        gr = sum(abs(raw[s]) for s in syms)
        w = {s: (raw[s] / gr if gr > 1e-12 else 0.0) for s in syms}
        # gross deployment: vol-target the portfolio
        gross = min(1.0, self.target_pvol / max(self._pvol, 1e-9))
        target = {s: w[s] * gross * self.deployable for s in syms}

        # rebalance per stock past the band
        cost = 0.0
        for s in syms:
            cur = self.exposure.get(s, 0.0)
            if abs(target[s] - cur) > self.band * self.deployable:
                cost += self.cost_rate * abs(target[s] - cur)
                self.exposure[s] = target[s]
            else:
                self.exposure.setdefault(s, cur)
        self.equity -= cost
        self.total_cost += cost

        # realize the move; update the reversal EWMA with THIS bar's return (so it is causal
        # for the next bar's edge)
        a_rev = 1.0 - 0.5 ** (1.0 / self.rev_hl)
        pnl = 0.0
        for s in syms:
            pp = self.prev_px.get(s)
            if pp and pp > 0:
                ret = (price[s] - pp) / pp
                pnl += self.exposure[s] * ret
                self._rev[s] = (1.0 - a_rev) * self._rev.get(s, 0.0) + a_rev * ret
            self.prev_px[s] = price[s]
        self.equity += pnl

        r = (pnl - cost) / self.initial_capital         # NET per-bar return (after cost) for Sharpe
        self._n += 1; self._sum += r; self._sumsq += r * r
        self._pvol = 0.99 * self._pvol + 0.01 * abs(r)  # realized portfolio vol (vol-targeting)

        return {
            "equity": self.equity,
            "return_pct": 100.0 * (self.equity - self.initial_capital) / self.initial_capital,
            "sharpe": self.sharpe(),
            "pnl_step": pnl, "cost": cost, "total_cost": self.total_cost,
            "exposure": dict(self.exposure),
            "weight": {s: (self.exposure[s] / self.deployable) for s in syms},  # fraction of budget
            "edge": edge,
        }

    def carry(self, price: dict):
        """Carry prices across a gap without realising P&L (overnight reprice)."""
        for s, px in price.items():
            self.prev_px[s] = px
