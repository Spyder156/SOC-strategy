"""Synthetic SOC market — UNIT TEST / DEBUG ONLY.

This generator builds a tick stream from a *known* moving critical value `x_c_true`,
using exactly the hazard form the model assumes. Its only purpose is to answer one
question: *if an x_c genuinely exists and drives the data, can our estimator recover
it?* If the model cannot track x_c when it provably exists, our code/math is broken.

THIS IS NOT A STRATEGY TEST. We never train the deployed model on synthetic data and
never make a trading decision from it — doing so would just confirm our own
assumptions (garbage in). Real Alpaca replay is the only thing we judge the strategy on.

The generator exposes the hidden `x_c_true` trajectory alongside the ticks so a test
can compare estimate vs truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List

import numpy as np

from .feed import Tick


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class SyntheticMarket:
    """Generates ticks from a hidden, slowly-moving x_c using the model's hazard.

    The true x_c follows a slow sine (so a test has a clean, visible ground truth)
    plus a small random-walk wobble. Price relaxes downward on avalanches and loads
    upward otherwise, self-organising to sit below x_c.
    """

    symbol: str = "SYN"
    n_ticks: int = 20_000
    tick_size: float = 0.01
    # true hazard parameters the estimator should approach
    alpha_true: float = 1.2
    beta_true: float = 0.6
    # x_c trajectory: baseline + slow sine of given amplitude/period + RW wobble
    xc_base: float = 100.0
    xc_amplitude: float = 8.0
    xc_period: int = 6_000
    xc_wobble: float = 0.02
    seed: int = 7

    # filled during generation so tests can compare estimate vs truth
    xc_true_path: List[float] = field(default_factory=list)

    def __iter__(self) -> Iterator[Tick]:
        rng = np.random.default_rng(self.seed)
        wobble = 0.0                    # random-walk component of x_c
        x = self.xc_base - 3.0          # start a few dollars below criticality
        self.xc_true_path = []

        for t in range(self.n_ticks):
            # slow, smooth true x_c = baseline + deterministic sine + RW wobble
            wobble += self.xc_wobble * rng.standard_normal()
            xc = (
                self.xc_base
                + self.xc_amplitude * math.sin(2.0 * math.pi * t / self.xc_period)
                + wobble
            )
            # keep price strictly below criticality
            gap = max(xc - x, self.tick_size)

            z = self.alpha_true * (-math.log(gap)) + self.beta_true
            p = _sigmoid(z)
            y = 1 if rng.random() < p else 0            # 1 = avalanche / downtick

            self.xc_true_path.append(xc)
            yield Tick(ts=float(t), symbol=self.symbol, mid=round(x, 4))

            # evolve price: relax down on avalanche, load up otherwise
            x += -self.tick_size if y == 1 else self.tick_size
            # safety: never let price punch above x_c in the generator
            if x > xc - self.tick_size:
                x = xc - self.tick_size
