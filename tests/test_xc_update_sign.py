"""Pin the DIRECTION and SMOOTHNESS of the redesigned x_c dynamics (log-price space).

x_c is a smooth anchored state: a single tick barely moves it; only *persistent* surprise
(and slow baseline moves) shift it. We test aggregate behaviour over many ticks, the safety
floor (x_c stays above price), and that the path is smooth. Gaps/margins are LOG (relative).
"""

from soc.data.synthetic_feed import SyntheticMarket
from soc.model.engine import Engine
from soc.model.hazard import HazardModel
from soc.model.state import RunningState


def _drive(m, s, y, n=3000):
    for _ in range(n):
        p = m.predict(s)
        m.update(s, y=y, p=p)


def test_persistent_no_avalanche_raises_xc_and_mu():
    m = HazardModel()
    s = RunningState.initialize("T", 50.0, initial_gap=0.05)
    m.mu = 0.05
    xc0, mu0 = s.x_c, m.mu
    _drive(m, s, y=0)                  # avalanche keeps NOT happening despite the hazard
    assert m.mu > mu0                  # learned margin widens
    assert s.x_c > xc0                 # critical value rises (we had more room than thought)
    assert s.surprise < 0             # persistent negative residual


def test_persistent_avalanche_lowers_xc():
    m = HazardModel()
    s = RunningState.initialize("T", 50.0, initial_gap=0.12)
    m.mu = 0.12
    xc0 = s.x_c
    _drive(m, s, y=1)                  # avalanches keep surprising us
    assert s.x_c < xc0
    assert s.surprise > 0


def test_xc_never_drops_to_or_below_price():
    m = HazardModel(eta_s=10.0)        # absurd surprise gain
    s = RunningState.initialize("T", 50.0, initial_gap=0.05)
    m.mu = 0.05
    _drive(m, s, y=1, n=200)
    assert s.x_c > s.x                 # relative floor keeps x_c strictly above price


def test_xc_is_smooth_vs_price():
    """The whole point: x_c must move far more smoothly than the price itself."""
    market = SyntheticMarket(n_ticks=20000, xc_period=30000, seed=3)   # slow x_c (the design regime)
    eng = Engine(HazardModel(), initial_gap=0.05)
    xs, xcs = [], []
    for tick in market:
        ev = eng.step(tick)
        if ev:
            xs.append(ev["x"]); xcs.append(ev["x_c"])
    import numpy as np
    dprice = np.abs(np.diff(xs)).mean()
    dxc = np.abs(np.diff(xcs)).mean()
    assert dxc < 0.5 * dprice, f"x_c not smooth enough: d_xc={dxc:.4f} d_price={dprice:.4f}"
