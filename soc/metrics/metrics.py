"""Streaming metrics — the numbers the model dashboard renders.

On REAL data x_c is unobservable, so we cannot score "did we recover x_c". Instead the
primary health signal is **calibration** (needs no ground truth): of the ticks where we
said ~p, did ~p actually go down? We also track cross-entropy, Brier score, and the
directional win rate, each both cumulatively and as an EWMA so convergence is visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


def _ewma_alpha(halflife: float) -> float:
    return 1.0 - math.pow(0.5, 1.0 / halflife)


@dataclass
class Metrics:
    n_bins: int = 10
    ewma_halflife: float = 2_000.0

    # cumulative
    n: int = 0
    sum_logloss: float = 0.0
    sum_brier: float = 0.0
    n_wins: int = 0
    sum_y: int = 0

    # rolling (EWMA)
    roll_logloss: float = float("nan")
    roll_brier: float = float("nan")
    roll_winrate: float = float("nan")

    # calibration reliability bins
    bin_count: List[int] = field(default_factory=list)
    bin_sum_y: List[float] = field(default_factory=list)
    bin_sum_p: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.bin_count = [0] * self.n_bins
        self.bin_sum_y = [0.0] * self.n_bins
        self.bin_sum_p = [0.0] * self.n_bins

    def update(self, p: float, y: int) -> None:
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        logloss = -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        brier = (p - y) ** 2
        # directional bet: down if p>0.5 else up; tie counts as no-win
        win = 1 if ((p > 0.5 and y == 1) or (p < 0.5 and y == 0)) else 0

        self.n += 1
        self.sum_logloss += logloss
        self.sum_brier += brier
        self.n_wins += win
        self.sum_y += y

        a = _ewma_alpha(self.ewma_halflife)
        self.roll_logloss = logloss if math.isnan(self.roll_logloss) else (1 - a) * self.roll_logloss + a * logloss
        self.roll_brier = brier if math.isnan(self.roll_brier) else (1 - a) * self.roll_brier + a * brier
        self.roll_winrate = win if math.isnan(self.roll_winrate) else (1 - a) * self.roll_winrate + a * win

        b = min(int(p * self.n_bins), self.n_bins - 1)
        self.bin_count[b] += 1
        self.bin_sum_y[b] += y
        self.bin_sum_p[b] += p

    def calibration(self) -> List[dict]:
        """Reliability curve: per bin, mean predicted p vs realized downtick freq."""
        out = []
        for i in range(self.n_bins):
            c = self.bin_count[i]
            if c == 0:
                continue
            out.append({
                "bin": i,
                "pred": self.bin_sum_p[i] / c,
                "realized": self.bin_sum_y[i] / c,
                "n": c,
            })
        return out

    def snapshot(self) -> dict:
        base = self.sum_y / self.n if self.n else 0.5
        return {
            "type": "metric",
            "n": self.n,
            "logloss": self.sum_logloss / self.n if self.n else None,
            "brier": self.sum_brier / self.n if self.n else None,
            "winrate": self.n_wins / self.n if self.n else None,
            "base_rate": base,
            "roll_logloss": None if math.isnan(self.roll_logloss) else self.roll_logloss,
            "roll_brier": None if math.isnan(self.roll_brier) else self.roll_brier,
            "roll_winrate": None if math.isnan(self.roll_winrate) else self.roll_winrate,
            "calibration": self.calibration(),
        }
