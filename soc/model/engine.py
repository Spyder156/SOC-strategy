"""Engine: drive one symbol's online model from a tick stream.

Per tick it does the honest out-of-sample loop:
  1. predict p = P(downtick) using info up to the PREVIOUS tick
  2. observe the new mid, derive outcome y
  3. update theta (slow) and x_c (fast) from (y - p)
  4. roll running features forward

It emits one event dict per tick. The engine is deliberately UI-agnostic: metrics,
strategy, and the websocket server all consume these events; none reach into the model.
"""

from __future__ import annotations

import math
from typing import Optional

from ..data.feed import Tick
from .hazard import HazardModel
from .state import RunningState


class Engine:
    def __init__(self, model: Optional[HazardModel] = None, initial_gap: float = 0.08):
        self.model = model or HazardModel()
        self.initial_gap = initial_gap
        self.state: Optional[RunningState] = None

    def step(self, tick: Tick) -> Optional[dict]:
        """Process one tick. Returns an event dict, or None for the very first tick."""
        m = self.model

        # cold start: no prediction possible yet, just seed the state
        if self.state is None:
            self.state = RunningState.initialize(tick.symbol, tick.mid, self.initial_gap)
            self.model.mu = self.initial_gap     # margin starts at the seed gap
            return None

        s = self.state

        # 1. predict for the incoming tick using prior info
        p = m.predict(s)

        # 2. outcome
        y = 1 if tick.mid < s.x else 0

        # 3. learn from (y - p) using the pre-observe gap/features
        m.update(s, y, p)

        # 4. roll features forward to the new mid
        s.observe(tick.mid, m.eps)
        # enforce the vol-scaled floor with the NOW-current price & vol: x can never touch x_c
        floor = s.x * math.exp(max(m.eps, m.k_vol * s.vol))
        if s.x_c < floor:
            s.x_c = floor

        return self._event(tick, p, y, reprice=False)

    def reprice(self, tick: Tick) -> Optional[dict]:
        """Exogenous reprice (overnight gap / halt): shift x_c and the baseline WITH the
        price so the relative gap is preserved. This is a discontinuity, NOT continuous
        loading, so it triggers no avalanche and no learning — x can never 'touch' x_c."""
        if self.state is None or self.state.x <= 0:
            return self.step(tick)
        s, m = self.state, self.model
        ratio = tick.mid / s.x
        s.x_c *= ratio                      # carry x_c across the gap (gap % preserved)
        s.Lbar += math.log(ratio)           # carry the log baseline too
        s.prev_x = tick.mid
        s.x = tick.mid
        s.last_y = -1
        s.n_ticks += 1
        return self._event(tick, p=0.5, y=0, reprice=True)

    def _event(self, tick: Tick, p: float, y: int, reprice: bool) -> dict:
        s, m = self.state, self.model
        return {
            "type": "tick",
            "reprice": reprice,
            "ts": tick.ts,
            "symbol": tick.symbol,
            "x": s.x,                       # new mid
            "x_c": s.x_c,                   # post-update critical-value estimate (smooth)
            "x_bar": s.x_bar,               # slow price baseline (the anchor, exp(Lbar))
            "gap": s.x_c - s.x,             # price gap for display (model uses log gap internally)
            "p": p,                         # out-of-sample prediction for this tick
            "y": y,                         # realized outcome (1=down)
            "velocity": s.velocity,
            "avalanche_rate": s.avalanche_rate,
            "surprise": s.surprise,
            "t_since_avalanche": s.t_since_avalanche,
            "params": {
                "alpha": m.alpha,
                "beta": m.beta,
                "gamma_v": m.gamma_v,
                "gamma_a": m.gamma_a,
                "mu": m.mu,
            },
        }
