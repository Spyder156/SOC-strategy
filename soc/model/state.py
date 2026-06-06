"""Per-stock running state and the derived input features.

These are the *running variables*: they never stop moving. They are updated every tick
from the raw mid-price stream and fed into the hazard function.

The x_c machinery runs in **log-price** space. A secular trend is a straight line in
log-price, so a slow EWMA baseline tracks it with bounded lag and the relative gap never
collapses — that is what keeps x_c smooth AND keeps price from ever overtaking it, across
trends and across very different price levels (AAPL ~$200 vs MSFT ~$400).

Feature notes:
- `velocity`        EWMA of returns (dx/x) — scale-free rate of "sand loading".
- `avalanche_rate`  EWMA of the downtick flag in [0,1] — recent empirical hazard.
- `Lbar`            slow low-pass of LOG price — the smooth anchor x_c relaxes toward.
- `surprise`        slow EWMA of the residual (y - p) — persistent mispricing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _ewma_alpha(halflife: float) -> float:
    return 1.0 - math.pow(0.5, 1.0 / halflife)


@dataclass
class RunningState:
    symbol: str
    x: float                     # current mid price
    x_c: float                   # estimated critical PRICE (latent)

    Lbar: float = float("nan")   # slow low-pass of LOG price (the smooth anchor)
    velocity: float = 0.0
    avalanche_rate: float = 0.5
    vol: float = 1e-3            # EWMA of |return| — realized volatility (sets the gap floor)
    surprise: float = 0.0
    t_since_avalanche: int = 0

    prev_x: float = float("nan")
    n_ticks: int = 0
    last_y: int = -1

    vel_halflife: float = 50.0
    rate_halflife: float = 100.0
    vol_halflife: float = 30.0          # vol for the gap floor: reacts fast enough to fast rises
    baseline_halflife: float = 1000.0   # Lbar: tracks price faster so x_c depends more on x
    surprise_halflife: float = 300.0

    @classmethod
    def initialize(cls, symbol: str, first_mid: float, initial_gap: float = 0.08) -> "RunningState":
        """Cold start. `initial_gap` is a LOG (relative) gap, e.g. 0.01 = 1% above price."""
        return cls(symbol=symbol, x=first_mid, x_c=first_mid * math.exp(initial_gap),
                   Lbar=math.log(first_mid), prev_x=first_mid)

    def observe(self, new_mid: float, eps: float) -> int:
        """Advance state to the new mid; return outcome y (1=downtick). Rolls features.
        Does NOT touch x_c or the convergence params — that is the hazard model's job."""
        prev = self.x
        y = 1 if new_mid < prev else 0

        ret = (new_mid - prev) / prev if prev > 0 else 0.0
        a_v = _ewma_alpha(self.vel_halflife)
        self.velocity = (1.0 - a_v) * self.velocity + a_v * ret
        a_vol = _ewma_alpha(self.vol_halflife)
        self.vol = (1.0 - a_vol) * self.vol + a_vol * abs(ret)   # realized vol (gap-floor scale)

        a_r = _ewma_alpha(self.rate_halflife)
        self.avalanche_rate = (1.0 - a_r) * self.avalanche_rate + a_r * y

        # slow LOG-price baseline (a trend is linear here, so the lag stays bounded)
        a_b = _ewma_alpha(self.baseline_halflife)
        self.Lbar = (1.0 - a_b) * self.Lbar + a_b * math.log(new_mid)

        self.t_since_avalanche = 0 if y == 1 else self.t_since_avalanche + 1
        self.prev_x = prev
        self.x = new_mid
        self.last_y = y
        self.n_ticks += 1

        # safety: x_c stays above price (relative floor)
        if self.x_c < self.x * math.exp(eps):
            self.x_c = self.x * math.exp(eps)
        return y

    @property
    def gap(self) -> float:
        """Relative (log) gap used by the hazard: log(x_c) - log(x)."""
        return math.log(self.x_c) - math.log(self.x)

    @property
    def x_bar(self) -> float:
        """Baseline in price space (for display)."""
        return math.exp(self.Lbar)
