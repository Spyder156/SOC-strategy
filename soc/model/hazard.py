"""The hazard model: convergence variables + the online update rules.

Convergence variables (should stabilize):  theta = {alpha, beta, gamma_v, gamma_a}
Running variable updated here:             x_c (per-stock critical value)

Hazard (probability the next mid-tick is a downtick / avalanche):

    gap = x_c - x                      (floored at eps)
    z   = alpha * (-log gap) + beta + gamma_v * velocity + gamma_a * avalanche_rate
    p   = sigmoid(z)

Learning is gradient ascent on the Bernoulli log-likelihood (== descent on binary
cross-entropy). With L = -[y log p + (1-y) log(1-p)], we have dL/dz = (p - y), so
every parameter step is  param += eta * (y - p) * dz/dparam.

x_c update — the sign that matters:

    dz/dx_c = -alpha / gap
    x_c += eta_xc * (y - p) * (-alpha / gap)      # == eta_xc * (p - y) * (alpha / gap)

Sanity: no avalanche (y=0) despite high p  ->  (y-p)<0 and (-alpha/gap)<0  ->  x_c rises.
That is the correct behaviour, and it is the OPPOSITE of the formula written in
SOC_Trading_Strategy.md Section 3.3.2 (which dropped the minus sign). We implement the
verified version here and pin it with tests/test_xc_update_sign.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .state import RunningState


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class HazardModel:
    """Shared structural model. One instance per symbol in v1 (universal in spirit)."""

    alpha: float = 1.0
    beta: float = 0.0
    gamma_v: float = 0.0
    gamma_a: float = 0.0

    eta_theta: float = 1e-3      # slow: structural params
    eta_xc: float = 5e-2         # fast: critical value (eta_xc >> eta_theta)
    eps: float = 1e-2            # gap floor (also the x_c-above-price floor)

    # parameter clamps to keep the online system numerically sane
    alpha_lo: float = 0.1
    alpha_hi: float = 10.0
    coef_abs: float = 10.0       # |beta|, |gamma| bound

    def logit(self, state: RunningState) -> float:
        gap = max(state.gap, self.eps)
        return (
            self.alpha * (-math.log(gap))
            + self.beta
            + self.gamma_v * state.velocity
            + self.gamma_a * state.avalanche_rate
        )

    def predict(self, state: RunningState) -> float:
        """P(next mid-tick is a downtick) given the current running state."""
        return sigmoid(self.logit(state))

    def update(self, state: RunningState, y: int, p: float) -> None:
        """One online step on theta (slow) and x_c (fast) from outcome y vs prediction p.

        Must be called with the SAME state that produced p (pre-observe), so the gap
        and features match the prediction.
        """
        gap = max(state.gap, self.eps)
        neg_log_gap = -math.log(gap)
        resid = y - p                       # (y - p); positive => avalanche underestimated

        # --- convergence variables: slow SGD (gradient ascent on log-likelihood) ---
        self.alpha = _clamp(self.alpha + self.eta_theta * resid * neg_log_gap,
                            self.alpha_lo, self.alpha_hi)
        self.beta = _clamp(self.beta + self.eta_theta * resid,
                           -self.coef_abs, self.coef_abs)
        self.gamma_v = _clamp(self.gamma_v + self.eta_theta * resid * state.velocity,
                              -self.coef_abs, self.coef_abs)
        self.gamma_a = _clamp(self.gamma_a + self.eta_theta * resid * state.avalanche_rate,
                              -self.coef_abs, self.coef_abs)

        # --- running variable: fast filter on x_c (the verified sign) ---
        dz_dxc = -self.alpha / gap
        state.x_c = state.x_c + self.eta_xc * resid * dz_dxc
        # keep x_c strictly above price
        if state.x_c < state.x + self.eps:
            state.x_c = state.x + self.eps
