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

from typing import Optional

from ..data.feed import Tick
from .hazard import HazardModel
from .state import RunningState


class Engine:
    def __init__(self, model: Optional[HazardModel] = None, initial_gap: float = 1.0):
        self.model = model or HazardModel()
        self.initial_gap = initial_gap
        self.state: Optional[RunningState] = None

    def step(self, tick: Tick) -> Optional[dict]:
        """Process one tick. Returns an event dict, or None for the very first tick."""
        m = self.model

        # cold start: no prediction possible yet, just seed the state
        if self.state is None:
            self.state = RunningState.initialize(tick.symbol, tick.mid, self.initial_gap)
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

        return {
            "type": "tick",
            "ts": tick.ts,
            "symbol": tick.symbol,
            "x": s.x,                       # new mid
            "x_c": s.x_c,                   # post-update critical-value estimate
            "gap": s.gap,
            "p": p,                         # out-of-sample prediction for this tick
            "y": y,                         # realized outcome (1=down)
            "velocity": s.velocity,
            "avalanche_rate": s.avalanche_rate,
            "t_since_avalanche": s.t_since_avalanche,
            "params": {
                "alpha": m.alpha,
                "beta": m.beta,
                "gamma_v": m.gamma_v,
                "gamma_a": m.gamma_a,
            },
        }
