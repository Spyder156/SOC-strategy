"""Pin the verified direction of the x_c update (guards the doc's sign bug).

Rules we lock in:
  - No avalanche (y=0) while the model expected one (p high)  ->  x_c must INCREASE.
  - Avalanche (y=1) while the model did not expect it (p low)  ->  x_c must DECREASE.
"""

from soc.model.hazard import HazardModel
from soc.model.state import RunningState


def _state(x=50.0, x_c=51.0):
    # fresh state with a known gap; neutral features
    return RunningState(symbol="T", x=x, x_c=x_c, velocity=0.0, avalanche_rate=0.5,
                        prev_x=x)


def test_no_avalanche_when_expected_pushes_xc_up():
    m = HazardModel(alpha=2.0, eta_xc=0.1)
    s = _state(x_c=50.2)             # tiny gap => model predicts high p
    p = m.predict(s)
    assert p > 0.5
    xc_before = s.x_c
    m.update(s, y=0, p=p)           # avalanche did NOT happen
    assert s.x_c > xc_before


def test_surprise_avalanche_pushes_xc_down():
    m = HazardModel(alpha=2.0, eta_xc=0.1)
    s = _state(x_c=80.0)            # huge gap => model predicts low p
    p = m.predict(s)
    assert p < 0.5
    xc_before = s.x_c
    m.update(s, y=1, p=p)          # avalanche DID happen (surprise)
    assert s.x_c < xc_before


def test_xc_never_drops_to_or_below_price():
    m = HazardModel(alpha=5.0, eta_xc=10.0)   # absurdly large step
    s = _state(x_c=50.05)
    p = m.predict(s)
    m.update(s, y=1, p=p)          # strong downward push on x_c
    assert s.x_c >= s.x + m.eps
