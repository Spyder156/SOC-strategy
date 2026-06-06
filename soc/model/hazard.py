"""The hazard model: convergence variables + the online update rules (log-price space).

Convergence variables (should stabilize):  theta = {alpha, beta, gamma_v, gamma_a, mu}
Running state updated here:                 x_c (smooth), surprise S

The gap is RELATIVE (in log-price):   g = log(x_c) - log(x)   (a small positive number).
Hazard:
    z = alpha * (-log g) + beta + gamma_v * velocity + gamma_a * avalanche_rate
    p = sigmoid(z)                       # P(next mid-tick is a downtick / avalanche)

x_c is a SMOOTH anchored state, not a per-tick filter. In log space:
    Lbar : slow low-pass of log price      (state.observe)
    S    : slow EWMA of residual (y - p)
    mu   : learned LOG margin (x_c ~ x_bar * e^mu); absorbs the 1/g signal, learned slowly
    log(x_c) += kappa*((Lbar + mu) - log(x_c))  -  eta_s * S

Working in log space means a secular trend (linear in log) is tracked by Lbar with bounded
lag, so the relative gap never collapses: x_c stays smooth and price never overtakes it,
across trends and across price levels. Sign check: persistent avalanches-more-than-expected
=> S>0 => x_c drifts DOWN (more fragile than price implies). Correct.
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
    alpha: float = 1.0
    beta: float = 0.0
    gamma_v: float = 0.0
    gamma_a: float = 0.0
    mu: float = 0.01             # CONVERGENCE VAR: learned LOG margin (x_c ~ x_bar * e^mu)

    eta_theta: float = 2e-3      # base LR for alpha/beta/gamma (decays -> they converge)
    eta_mu: float = 2e-4         # base LR for the margin
    lr_decay_tau: float = 4000.0 # LR decays as tau/(tau+n): params settle but don't freeze
    lr_floor: float = 0.12       # never below this fraction, so they keep adapting (visible)
    kappa: float = 0.02          # pull of log(x_c) toward its anchor (more x-responsive, still smooth)
    eta_s: float = 1e-3          # x_c drift on persistent surprise
    eps: float = 1e-4            # numerical log-gap floor
    k_vol: float = 20.0          # x_c stays >= k_vol * realized-vol above x (so x never touches)

    alpha_lo: float = 0.1
    alpha_hi: float = 10.0
    coef_abs: float = 10.0
    mu_lo: float = 1e-3
    mu_hi: float = 0.15          # margin capped ~15% so x_c stays a sensible distance above x

    def _loggap(self, state: RunningState) -> float:
        return max(math.log(state.x_c) - math.log(state.x), self.eps)

    def logit(self, state: RunningState) -> float:
        g = self._loggap(state)
        return (self.alpha * (-math.log(g)) + self.beta
                + self.gamma_v * state.velocity + self.gamma_a * state.avalanche_rate)

    def predict(self, state: RunningState) -> float:
        return sigmoid(self.logit(state))

    def update(self, state: RunningState, y: int, p: float) -> None:
        g = self._loggap(state)
        neg_log_gap = -math.log(g)
        resid = y - p
        dz_dLc = -self.alpha / g          # dz/d(log x_c) = dz/d(mu)

        # convergence variables (SGD with DECAYING learning rate so they actually converge)
        decay = max(self.lr_floor, self.lr_decay_tau / (self.lr_decay_tau + state.n_ticks))
        et, em = self.eta_theta * decay, self.eta_mu * decay
        self.alpha = _clamp(self.alpha + et * resid * neg_log_gap, self.alpha_lo, self.alpha_hi)
        self.beta = _clamp(self.beta + et * resid, -self.coef_abs, self.coef_abs)
        self.gamma_v = _clamp(self.gamma_v + et * resid * state.velocity, -self.coef_abs, self.coef_abs)
        self.gamma_a = _clamp(self.gamma_a + et * resid * state.avalanche_rate, -self.coef_abs, self.coef_abs)
        self.mu = _clamp(self.mu + em * resid * dz_dLc, self.mu_lo, self.mu_hi)

        # persistent surprise
        a_s = 1.0 - math.pow(0.5, 1.0 / state.surprise_halflife)
        state.surprise = (1.0 - a_s) * state.surprise + a_s * resid

        # smooth x_c in LOG space: pull toward anchor (Lbar + mu), drift on surprise
        Lc = math.log(state.x_c)
        anchor = state.Lbar + self.mu
        Lc = Lc + self.kappa * (anchor - Lc) - self.eta_s * state.surprise
        Lx = math.log(state.x)
        floor_gap = max(self.eps, self.k_vol * state.vol)   # vol-scaled: x can never touch x_c
        if Lc < Lx + floor_gap:
            Lc = Lx + floor_gap
        state.x_c = math.exp(Lc)
