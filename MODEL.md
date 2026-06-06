# MODEL.md — SOC Sandpile, exact math (as implemented)

Canonical reference for the equations actually in the code (v1). Where this differs from
`SOC_Trading_Strategy.md`, **this file is correct** — the strategy doc has a sign error in the
x_c update (noted below).

---

## 1. Variables by type

### A. Running variables — update every tick, per symbol  ([state.py](soc/model/state.py))

| Symbol | Code | Meaning | Update rule |
|---|---|---|---|
| `x` | `x` | current **mid price** | set from feed |
| `x_c` | `x_c` | **critical value** (latent) | filtered, see §4 |
| `v` | `velocity` | EWMA of returns | `v ← (1−a_v)·v + a_v·(Δx/x)` |
| `a` | `avalanche_rate` | EWMA of downtick flag | `a ← (1−a_r)·a + a_r·y` |
| `τ` | `t_since_avalanche` | ticks since last downtick | `0` if down else `+1` *(tracked, not in hazard)* |
| `x̄` | `x_bar` | slow **price baseline** (anchor) | `x̄ ← x̄ + a_b·(x − x̄)` |
| `S` | `surprise` | slow EWMA of residual `(y−p)` | `S ← S + a_s·((y−p) − S)` |
| `g` | `gap` | derived | `g = x_c − x` |

EWMA factor for half-life `H`:  `a_ewma = 1 − 0.5^(1/H)`.  Defaults `H_v = 50`, `H_a = 100`,
`H_baseline = 3000` (x̄, very slow), `H_surprise = 300` (S).

### B. Convergence variables — parameters `θ`, learned slowly, should stabilise  ([hazard.py](soc/model/hazard.py))

| Symbol | Code | Role | Clamp |
|---|---|---|---|
| `α` | `alpha` | hazard steepness as `x→x_c` | `[0.1, 10]` |
| `β` | `beta` | base-rate bias | `[−10, 10]` |
| `γ_v` | `gamma_v` | weight on velocity | `[−10, 10]` |
| `γ_a` | `gamma_a` | weight on avalanche-rate | `[−10, 10]` |
| `μ` | `mu` | **margin**: x_c sits ≈ `x̄ + μ` (the learned x→x_c relationship) | `[ε, 50]` |

### C. Constants / hyperparameters

| Symbol | Code | Default | Role |
|---|---|---|---|
| `η_θ` | `eta_theta` | `1e-3` | slow SGD on α,β,γ |
| `η_μ` | `eta_mu` | `1e-4` | very slow learning of the margin μ |
| `κ` | `kappa` | `0.02` | gentle pull of x_c to its anchor |
| `η_S` | `eta_s` | `0.01` | x_c drift on persistent surprise |
| `ε` | `eps` | `1e-2` | gap / margin floor |
| — | `initial_gap` | `1.0` | seed for `x_c` and `μ` at cold start |

### D. Observed / target
`y ∈ {0,1}` — `y = 1` if this tick's mid < previous mid (downtick / "avalanche"), else `0`.

---

## 2. Hazard (prediction)

```
g = max(x_c − x, ε)
z = α·(−ln g) + β + γ_v·v + γ_a·a
p = σ(z) = 1 / (1 + e^(−z))           # P(next tick is a downtick)
```
`x_c` enters only through `g`. As `x → x_c`, `g → 0`, `−ln g → ∞`, `p → 1`.

---

## 3. Loss

Binary cross-entropy; log-likelihood `ℓ = −L`:
```
L  = −[ y·ln p + (1−y)·ln(1−p) ]
dℓ/dz = (y − p)                       # the residual drives every update
```

---

## 4. Learning (gradient ascent on ℓ, same residual `(y−p)`)

Partials of the logit:
```
∂z/∂α = −ln g     ∂z/∂β = 1     ∂z/∂γ_v = v     ∂z/∂γ_a = a
∂z/∂x_c = ∂z/∂μ = −α/g          (g = x_c − x ⇒ ∂g/∂x_c = +1; anchor moves x_c ~1:1 in μ)
```

Convergence variables (slow), each clamped:
```
α   ← α   + η_θ·(y−p)·(−ln g)
β   ← β   + η_θ·(y−p)
γ_v ← γ_v + η_θ·(y−p)·v
γ_a ← γ_a + η_θ·(y−p)·a
μ   ← μ   + η_μ·(y−p)·(−α/g)          # the 1/g signal, learned VERY slowly so it averages out
```

Critical value — **smooth anchored state** (the redesign; replaces the old jittery `1/g` filter):
```
x̄  : slow low-pass of price            (state.observe)
S   ← S + a_s·((y−p) − S)               # persistent surprise
x_c ← x_c + κ·((x̄ + μ) − x_c)  −  η_S·S  # gentle pull to anchor + surprise drift
x_c ← max(x_c, x + ε)
```
Why this is smooth: a single tick moves x_c by only `κ·(anchor−x_c) − η_S·S`, both tiny; the
explosive `1/g` term lives in `μ`, learned slowly enough that its spikes cancel. Empirically x_c
moves ~170× less per tick than price. Sign check: persistent avalanches-more-than-expected ⇒ `S>0`
⇒ x_c drifts **down** (more fragile than price implies). ✓
(`SOC_Trading_Strategy.md` §3.3.2's per-tick `x_c` filter — and its sign — are both superseded here.)

---

## 5. Per-tick algorithm ([engine.py](soc/model/engine.py)) — no look-ahead

```
1. p = σ(z)              from PREVIOUS-tick state
2. y = 1 if new_mid < x  observe outcome
3. update θ and x_c      using (y−p) and pre-observe g, v, a
4. roll v, a, τ, x → new_mid
```

---

## 6. Trading layer ([allocate.py](soc/strategy/allocate.py))

```
edge   = 1 − 2p
target = clamp(edge·N, ±N)                       # N = max_notional ($100k)
if |target − exposure| > band·N:                 # no-trade band (band = 0.05)
    cost      = cost_rate·|target − exposure|     # cost_rate = 5e-5
    exposure  = target
pnl_step = exposure·(x − x_prev)/x_prev
equity  += pnl_step − cost
```

---

## 7. Metrics ([metrics.py](soc/metrics/metrics.py))

```
logloss = −[y·ln p + (1−y)·ln(1−p)]
brier   = (p − y)²
win     = 1 if (p>0.5 ∧ y=1) ∨ (p<0.5 ∧ y=0)
calibration: bin by p; compare mean(p) vs mean(y) per bin   # primary health metric on real data
```
All kept cumulative and as EWMA (half-life 2000).

---

## 8. Per-tick algorithm note
The smooth-x_c machinery slots into the existing loop: `S` and `x_c` update inside `hazard.update`
(step 3), `x̄` updates inside `state.observe` (step 4). No look-ahead changes.

## 9. Known issues / open work
- ✅ **x_c jitter — fixed.** x_c is now a smooth anchored state (this file §4); on real AAPL it moves
  ~170× less per tick than price. The old `1/g` per-tick filter is gone.
- `τ` is computed but not yet wired into the hazard.
- Anchoring x_c to `x̄` means on a *single* stock the gap ≈ `μ − (x − x̄)` (a mean-reversion oscillator);
  independent information is meant to come from the future cross-asset coupling term on x_c.
- Empirically, `y = downtick` has no tradeable edge (sign is a martingale; magnitude/volatility is the
  predictable SOC signal). The equations are correct; the **target** is the thing to reconsider.
```
