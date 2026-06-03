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
| `g` | `gap` | derived | `g = x_c − x` |

EWMA factor for half-life `H`:  `a_ewma = 1 − 0.5^(1/H)`.  Defaults `H_v = 50`, `H_a = 100`.

### B. Convergence variables — parameters `θ`, learned slowly, should stabilise  ([hazard.py](soc/model/hazard.py))

| Symbol | Code | Role | Clamp |
|---|---|---|---|
| `α` | `alpha` | hazard steepness as `x→x_c` | `[0.1, 10]` |
| `β` | `beta` | base-rate bias | `[−10, 10]` |
| `γ_v` | `gamma_v` | weight on velocity | `[−10, 10]` |
| `γ_a` | `gamma_a` | weight on avalanche-rate | `[−10, 10]` |

### C. Constants / hyperparameters

| Symbol | Code | Default |
|---|---|---|
| `η_θ` | `eta_theta` | `1e-3` (slow, structural) |
| `η_xc` | `eta_xc` | `5e-2` (fast state filter; `η_xc ≫ η_θ`) |
| `ε` | `eps` | `1e-2` (gap floor) |
| — | `initial_gap` | seed for `x_c` at cold start |

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
∂z/∂x_c = −α/g                  (g = x_c − x ⇒ ∂g/∂x_c = +1)
```

Convergence variables (slow, `η_θ`), each clamped:
```
α   ← α   + η_θ·(y−p)·(−ln g)
β   ← β   + η_θ·(y−p)
γ_v ← γ_v + η_θ·(y−p)·v
γ_a ← γ_a + η_θ·(y−p)·a
```

Critical value (fast, `η_xc`) — **verified sign**:
```
x_c ← x_c + η_xc·(y−p)·(−α/g)         # = η_xc·(p−y)·(α/g)
x_c ← max(x_c, x + ε)
```
Sanity: no avalanche (`y=0`) while `p` high ⇒ `(y−p)<0`, `(−α/g)<0` ⇒ `x_c` rises. ✓
(`SOC_Trading_Strategy.md` §3.3.2 drops the minus sign and is wrong.)

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

## 8. Known issues / open redesign
- The `x_c` filter uses `η_xc ≫ η_θ` **and** an `α/g` term that blows up as `g → 0`, making `x_c`
  jittery. A smooth-`x_c` redesign (anchor to a slow price baseline + a learned margin, surprise-driven
  drift) is under discussion — see the conversation / a future revision of this file.
- `τ` is computed but not yet wired into the hazard.
- Empirically, `y = downtick` has no tradeable edge (sign is a martingale; magnitude/volatility is the
  predictable SOC signal). The equations are correct; the **target** is the thing to reconsider.
```
