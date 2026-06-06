"""Bouchaud-style fragility model.

The order parameter is NOT a price ceiling — it's the system's SELF-EXCITATION (the Hawkes
branching ratio). An "avalanche" is a LARGE move (|r| > k*sigma), in either direction. Large
moves cluster in time (self-exciting), so the *probability of a large move* is forecastable
even though its *sign* is not. We:

  - track realized vol sigma (so the avalanche threshold is scale-free),
  - maintain a self-exciting intensity S (Hawkes EWMA of recent avalanches),
  - learn a calibrated hazard P(large move next) = sigmoid(w0 + wS * excess-excitation),
  - report a branching-ratio proxy n in [0,1): the fraction of avalanche intensity that is
    self-triggered rather than background. n -> 1 means critical (a burst is imminent).

The tradeable use is vol-targeting: cut exposure when fragility is high (predicted big move),
hold when calm — lowering variance more than return, hence raising Sharpe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


def _a(halflife: float) -> float:
    return 1.0 - math.pow(0.5, 1.0 / halflife)


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


@dataclass
class FragilityModel:
    k_aval: float = 2.5          # avalanche = |r| > k_aval * sigma
    vol_hl: float = 100.0        # realized-vol timescale
    excite_hl: float = 25.0      # Hawkes self-excitation kernel timescale (bars)
    base_hl: float = 3000.0      # background avalanche-rate timescale
    eta: float = 5e-3            # hazard learning rate

    w0: float = -2.5             # learned hazard bias
    wS: float = 3.0              # learned self-excitation weight

    sigma: float = 1e-3
    S: float = 0.06              # self-exciting intensity (recent avalanche rate)
    base_rate: float = 0.06      # background avalanche rate
    prev_price: float = float("nan")
    n_ticks: int = 0
    sizes: List[float] = field(default_factory=list)

    def step(self, price: float) -> Optional[dict]:
        if math.isnan(self.prev_price) or self.prev_price <= 0:
            self.prev_price = price
            return None
        r = math.log(price / self.prev_price)
        self.prev_price = price

        a_v = _a(self.vol_hl)
        self.sigma = (1.0 - a_v) * self.sigma + a_v * abs(r)
        z_size = abs(r) / max(self.sigma, 1e-12)
        is_av = 1 if z_size > self.k_aval else 0

        # predict P(large move this bar) from PRIOR self-excitation (before this event)
        excess = self.S - self.base_rate
        p_large = _sig(self.w0 + self.wS * excess)

        # learn the hazard online (cross-entropy)
        resid = is_av - p_large
        self.w0 += self.eta * resid
        self.wS += self.eta * resid * excess

        # branching-ratio proxy: fraction of avalanche intensity that is self-triggered
        intensity = self.base_rate + max(0.0, self.wS) * max(0.0, excess)
        n = (intensity - self.base_rate) / max(intensity, 1e-12)

        # roll the self-exciting intensity and the slow baseline
        self.S = (1.0 - _a(self.excite_hl)) * self.S + _a(self.excite_hl) * is_av
        self.base_rate = (1.0 - _a(self.base_hl)) * self.base_rate + _a(self.base_hl) * is_av
        if is_av:
            self.sizes.append(z_size)
        self.n_ticks += 1

        return {"r": r, "sigma": self.sigma, "is_av": is_av, "size": z_size,
                "p_large": p_large, "n": min(1.0, max(0.0, n)), "S": self.S,
                "w0": self.w0, "wS": self.wS}
