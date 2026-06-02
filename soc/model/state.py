"""Per-stock running state and the derived input features.

These are the *running variables* in our taxonomy: they never stop moving. They are
updated every tick from the raw mid-price stream and fed into the hazard function.

Feature design notes:
- `velocity` is an EWMA of *returns* (dx / x), so it is scale-free across price levels
  and stays O(1e-3..1e-2) — the rate of "sand loading".
- `avalanche_rate` is an EWMA of the downtick indicator y in [0, 1] — the recent
  empirical hazard, a direct readout of how often the pile has been relaxing.
- `t_since_avalanche` is the survival clock (ticks since the last downtick).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _ewma_alpha(halflife: float) -> float:
    """Smoothing factor for an EWMA with the given half-life (in ticks)."""
    return 1.0 - math.pow(0.5, 1.0 / halflife)


@dataclass
class RunningState:
    """Mutable running state for one symbol."""

    symbol: str
    x: float                     # current mid-price
    x_c: float                   # estimated critical value (latent)

    # derived features
    velocity: float = 0.0        # EWMA of returns dx/x
    avalanche_rate: float = 0.5  # EWMA of downtick indicator
    t_since_avalanche: int = 0

    # bookkeeping
    prev_x: float = float("nan")
    n_ticks: int = 0
    last_y: int = -1             # last observed outcome (1=down, 0=up), -1 if none

    # EWMA half-lives (ticks)
    vel_halflife: float = 50.0
    rate_halflife: float = 100.0

    @classmethod
    def initialize(cls, symbol: str, first_mid: float, initial_gap: float = 1.0) -> "RunningState":
        """Cold start: place x_c a small gap above the first observed price.

        The model knows nothing yet; the online filter will move x_c to where the
        data says it belongs. `initial_gap` just keeps the first log(gap) finite.
        """
        return cls(symbol=symbol, x=first_mid, x_c=first_mid + initial_gap, prev_x=first_mid)

    def observe(self, new_mid: float, eps: float) -> int:
        """Advance state to the new mid and return the realized outcome y.

        y = 1 if this was a downtick (avalanche), else 0. Updates velocity,
        avalanche_rate, and the survival clock. Does NOT touch x_c or the
        convergence params — that is the hazard model's job (see hazard.update).
        """
        prev = self.x
        y = 1 if new_mid < prev else 0

        # return-based velocity (scale-free)
        ret = (new_mid - prev) / prev if prev > 0 else 0.0
        a_v = _ewma_alpha(self.vel_halflife)
        self.velocity = (1.0 - a_v) * self.velocity + a_v * ret

        # empirical avalanche rate
        a_r = _ewma_alpha(self.rate_halflife)
        self.avalanche_rate = (1.0 - a_r) * self.avalanche_rate + a_r * y

        # survival clock
        self.t_since_avalanche = 0 if y == 1 else self.t_since_avalanche + 1

        self.prev_x = prev
        self.x = new_mid
        self.last_y = y
        self.n_ticks += 1

        # never let the estimated x_c sit at/below price (keeps log(gap) finite)
        if self.x_c < self.x + eps:
            self.x_c = self.x + eps
        return y

    @property
    def gap(self) -> float:
        return self.x_c - self.x
