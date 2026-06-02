# SOC Sandpile — Self-Organized-Criticality Trading Engine (v1)

An online-learning trading model that treats a stock as a sandpile: it does **not** predict
price direction blindly — it estimates each stock's latent **critical value `x_c`** (a slow
"fragility threshold") and, from the gap `(x_c − x)`, emits the probability that the next
mid-tick is a downtick (an "avalanche"). It learns **online, on real data**, starting cold and
converging in real time. Two live dashboards let you watch it converge.

Theory: `SOC_Trading_Strategy.md`. Build plan: `~/.claude/plans/i-want-to-research-misty-wolf.md`.

> **v1 scope:** single stock. Goal is to prove the machinery — that the convergence variables
> stabilise and the model becomes **calibrated** — not to make money yet. The real alpha (v2) is
> the cross-asset *fragility-gated* coupling, deliberately deferred.

## The model in one screen

```
gap = x_c − x                         (floored at ε)
z   = α·(−log gap) + β + γ_v·velocity + γ_a·avalanche_rate
p   = sigmoid(z)            = P(next mid-tick is down)
```
- **Convergence variables** (stabilise, slow SGD): `α, β, γ_v, γ_a`
- **Running variable** (fast filter): `x_c`, updated `x_c += η_xc·(y−p)·(−α/gap)`
- Learning is online binary cross-entropy. Primary health metric on real data = **calibration**
  (needs no ground truth — x_c is never observable).

> **Note:** this implements the *verified-correct* x_c update sign. `SOC_Trading_Strategy.md`
> §3.3.2 has a sign error (it drops the minus); pinned by `tests/test_xc_update_sign.py`.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Watch it now (no keys, synthetic mechanics demo)

```bash
scripts/demo.sh                       # then open http://localhost:8080
```
The synthetic feed is a self-contained SOC sim used **only** to show the wiring working — we never
train the real model on it.

## Run on real Alpaca data

1. Put your paper-trading keys in `.env` (copy from `.env.example`):
   ```
   ALPACA_API_KEY_ID=PK...
   ALPACA_API_SECRET_KEY=...
   ```
2. Fetch historical quotes (mid = (bid+ask)/2 strips bid-ask bounce):
   ```bash
   ./.venv/bin/python -m soc.data.fetch_history --symbol AAPL --start 2024-01-02 --end 2024-02-01
   ```
3. Replay it through the live dashboard:
   ```bash
   scripts/demo.sh --feed replay --symbol AAPL --stride 25 --fps 30
   ```
   Or headless (fast convergence, prints metrics):
   ```bash
   ./.venv/bin/python -m soc.server.run --feed replay --symbol AAPL --headless
   ```

## Tests

```bash
./.venv/bin/python -m pytest -q          # 5 tests: x_c recovery + update-sign + safety
```

## Architecture

```
soc/
  data/      feed (Tick interface) · synthetic_feed (test-only) · replay_feed · fetch_history (Alpaca)
  model/     state (running features) · hazard (z/p/grads + corrected x_c update) · engine
  strategy/  allocate (edge = 1−2p, no-trade band, virtual money + modeled cost)
  metrics/   cross-entropy · Brier · calibration bins · win rate
  server/    bus (websocket hub) · run (Feed→Engine→Strategy→Metrics→broadcast; synthetic|replay|headless)
web/         framework-free dashboard: Market view (price+x_c, P, equity) · Model view (params, loss, calibration)
tests/       mechanics + sign tests
```

The websocket event stream is the *only* coupling between backend and UI — the frontend never
reaches into the model. Swap synthetic → replay → (later) live feed without touching anything else.

## Roadmap
- **v1 (here):** single-stock convergence + calibration on real replay.
- **v2:** universe + cross-asset coupling (the fragility-gated lead-lag — the real edge).
- **v3+:** EKF/particle filter for x_c (self-scheduling learning rates), live paper trading, risk controls.
