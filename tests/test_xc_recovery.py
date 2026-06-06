"""Mechanics test: if a moving x_c provably drives the data, the estimator must track it.

This runs the synthetic SOC market (known hidden x_c) through the live engine and checks
that the online x_c estimate co-moves with the true x_c after a burn-in. This validates
the gradient math + filter wiring only. It is NOT a strategy/profit test and uses
synthetic data we would never train the deployed model on.
"""

import numpy as np

from soc.data.synthetic_feed import SyntheticMarket
from soc.model.engine import Engine
from soc.model.hazard import HazardModel


def _run():
    # redesigned x_c is a SLOW, smooth estimator -> test it on a slowly-moving true x_c
    market = SyntheticMarket(n_ticks=30_000, xc_period=30_000, seed=11)
    engine = Engine(HazardModel(alpha=1.0, beta=0.0, eta_theta=2e-3),
                    initial_gap=0.05)
    est, idx = [], []
    for i, tick in enumerate(market):
        ev = engine.step(tick)
        if ev is not None:
            est.append(ev["x_c"])
            idx.append(i)
    true = np.array([market.xc_true_path[i] for i in idx])
    return np.array(est), true, engine


def test_estimate_tracks_true_xc():
    est, true, _ = _run()
    # evaluate after burn-in (second half)
    half = len(est) // 2
    e, t = est[half:], true[half:]
    corr = np.corrcoef(e, t)[0, 1]
    assert corr > 0.6, f"x_c estimate should co-move with true x_c, corr={corr:.3f}"


def test_estimate_stays_in_sane_range():
    est, true, engine = _run()
    # estimate must not diverge: stay within a few amplitudes of the true band
    assert np.all(np.isfinite(est))
    assert est.min() > 50.0 and est.max() < 150.0
    # alpha stays positive and within clamp
    assert 0.1 <= engine.model.alpha <= 10.0
