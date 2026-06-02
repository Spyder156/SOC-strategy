# The Sandpile Model
## A Self-Organized Criticality Approach to Quantitative Trading

**Research & Strategy Document**
Version 0.1 — April 2026
CONFIDENTIAL — DRAFT

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - 2.1 Self-Organized Criticality in Physics
   - 2.2 SOC in Financial Markets
   - 2.3 Why Not Predict Prices Directly?
3. [The Single-Stock Model](#3-the-single-stock-model)
   - 3.1 State Variables
   - 3.2 The Hazard Function
   - 3.3 The Learning System
4. [Multi-Stock Extension: Cross-Asset Inference](#4-multi-stock-extension-cross-asset-inference)
   - 4.1 The Correlation Hypothesis
   - 4.2 Modelling Cross-Asset Correlations
   - 4.3 Scope Limitations
5. [Trading Mechanics](#5-trading-mechanics)
   - 5.1 Signal Generation
   - 5.2 Capital Allocation
   - 5.3 Avalanche Definition
   - 5.4 Asymmetry
6. [Execution Architecture](#6-execution-architecture)
   - 6.1 Tick-by-Tick Loop
   - 6.2 Training vs. Inference
7. [Risks, Limitations, and Open Questions](#7-risks-limitations-and-open-questions)
8. [Development Roadmap](#8-development-roadmap)
9. [Mathematical Appendix](#9-mathematical-appendix)
   - 9.1 Full Gradient Derivations
   - 9.2 Connection to Survival Analysis
   - 9.3 Connection to Hawkes Processes
10. [References](#10-references)

---

## 1. Executive Summary

This document presents a novel quantitative trading strategy grounded in the physics of **Self-Organized Criticality (SOC)**. Rather than attempting to predict the exact direction or magnitude of stock price movements — a task that is both theoretically and practically impossible in chaotic, non-linear financial markets — we propose modelling the *critical value* of a stock: the latent threshold at which the probability of a sharp drawdown (an "avalanche") becomes high.

The core thesis is threefold:

1. Stock price dynamics resemble a sandpile system: prices drift upward under buying pressure (sand accumulation), and undergo sudden, sharp drawdowns (avalanches) when they approach an invisible critical threshold.

2. This critical threshold `x_c` is a smoother, slower-moving variable than the price itself, and is therefore more amenable to statistical estimation.

3. The critical values of different stocks are correlated — driven by shared market-level forces — and this correlation can be exploited for cross-asset inference and alpha generation.

The strategy operates at tick-level frequency, computing a real-time probability of avalanche for each stock in a universe, and allocating capital proportionally to the edge implied by that probability. The model learns online, updating both its structural parameters and its running estimates of each stock's critical value with every incoming tick.

---

## 2. Theoretical Foundations

### 2.1 Self-Organized Criticality in Physics

Self-Organized Criticality is a concept introduced by Per Bak, Chao Tang, and Kurt Wiesenfeld in 1987 [2] to describe complex systems that spontaneously evolve toward a critical state — a tipping point at the boundary between stability and chaos. The canonical example is the sandpile: grains of sand are dropped one at a time onto a pile. The pile grows gradually, but periodically undergoes avalanches of all sizes. The distribution of avalanche sizes follows a power law, meaning that while small avalanches are frequent, catastrophically large ones are rare but inevitable.

The key properties of SOC systems are:

- **(a)** They are driven slowly from outside (sand grains dropping).
- **(b)** They have a threshold or critical state beyond which the system becomes unstable.
- **(c)** They relax through avalanches whose size distribution is scale-free (power-law).
- **(d)** The system *spontaneously* returns to the critical state after each avalanche — no tuning is required.

### 2.2 SOC in Financial Markets

Jean-Philippe Bouchaud's 2024 review paper [1] argues that financial markets exhibit SOC-like behaviour. Asset prices frequently undergo large jumps for no particular external reason — the so-called "excess volatility puzzle." The distribution of price returns has power-law tails, characteristic of systems near a critical point. Volatility is intermittent and long-memory, similar to velocity fields in turbulence. Small perturbations can cascade into large disruptions, as seen in Black Monday (1987) and the Flash Crash (2010).

Bouchaud and collaborators have provided rigorous empirical evidence for this using Hawkes processes — self-exciting point processes originally developed for earthquake modelling [3]. Their analysis of E-mini S&P 500 futures from 1998 to 2011 showed that the Hawkes branching ratio — the average number of events triggered by each event — sits at approximately **n = 1** (the critical value), and has done so consistently for over a decade. This means the market perpetually sits at the edge of instability: each trade triggers, on average, exactly one more trade, creating the possibility of cascading avalanches of arbitrary size.

Separately, Fosset, Bouchaud, and Benzaquen [4] demonstrated that there exists a second-order phase transition in order book dynamics between a stable regime (weak feedback) and an unstable regime (strong feedback) in which liquidity crises arise with probability one. For real markets to be relevant, the system must sit just below this instability threshold — another manifestation of SOC.

### 2.3 Why Not Predict Prices Directly?

Financial markets are high-dimensional chaotic systems. The price at the next tick is the output of millions of interacting agents with heterogeneous information, strategies, and time horizons. Attempting to predict whether the next tick is up or down with a standard ML model (LSTMs, transformers, etc.) is analogous to trying to predict the exact trajectory of a turbulent fluid — it is a fool's errand. The Navier-Stokes equations, despite being fully deterministic, produce behaviour that is practically unpredictable at fine scales. Financial markets are far worse: they are not even governed by known equations.

What we *can* do is model the statistical envelope of the system. In turbulence, we cannot predict individual eddies, but we can predict the energy spectrum. In seismology, we cannot predict individual earthquakes, but we can model the hazard rate. In our sandpile analogy, we cannot predict the exact moment an avalanche occurs, but we can model the *probability* of one occurring given the current state. This probabilistic framing is the philosophical core of our strategy.

---

## 3. The Single-Stock Model

### 3.1 State Variables

For each stock, the model maintains two key quantities at every tick `t`:

- **`x(t)`** — the current observed price. This is given to us by the market data feed. It is the "height of the sandpile" at time t.

- **`x_c(t)`** — the estimated critical value. This is a *latent* (hidden) variable that the model must infer. It represents the height at which the sandpile becomes critically unstable — the threshold above which avalanche probability spikes. It is not directly observable.

The central assumption is that `x_c(t)` evolves *slowly* relative to `x(t)`. The price jitters tick-by-tick, but the critical value drifts on the timescale of fundamentals, sector rotation, and macro regimes. This separation of timescales is what makes x_c estimable: it is the slowly varying envelope that governs the fast, noisy price dynamics.

### 3.2 The Hazard Function

The model's core output is a probability: the chance that the next tick is a downtick (an "avalanche"), given the current price and the estimated critical value. We define this as a hazard function:

$$
p(t) = \sigma\Big( -\alpha \cdot \log\big(x_c(t) - x(t)\big) + \beta \Big)
$$

where:

- **σ(z) = 1 / (1 + exp(−z))** is the sigmoid function, mapping any real number to the interval (0, 1).

- **α > 0** is the **hazard shape parameter**. It controls how rapidly the avalanche probability increases as x approaches x_c. A larger α means a sharper ramp-up — the transition from "safe" to "dangerous" happens over a narrower gap.

- **β** is the **bias parameter**. It shifts the entire hazard curve up or down, controlling the base rate of avalanches even when x is far from x_c.

- **x_c(t) − x(t)** is the **gap** between the critical value and the current price. When this gap is large, the log term is large, the sigmoid input is very negative, and p(t) is low. As the gap shrinks toward zero, the log term diverges to −∞, the sigmoid input shoots to +∞, and p(t) → 1.

#### 3.2.1 Intuition

Think of it this way: when the stock price is far below x_c (lots of room in the sandpile), avalanches are unlikely — the probability is near the base rate (maybe 48–50%, since downticks happen naturally). As the price creeps up toward x_c, the probability smoothly increases. Very near x_c, it's almost certain that the next tick will be down. This smooth ramp is what gives us a tradeable signal: we don't need to know the *exact* moment of the avalanche; we just need to know that we're in a high-probability zone and position accordingly.

#### 3.2.2 Why This Functional Form?

We considered three candidate functional forms for the hazard function. Each encodes a different assumption about how avalanche probability relates to the gap:

| Form | Formula | Behaviour |
|------|---------|-----------|
| **Hard threshold** | `p = 0 if x < x_c, 1 if x ≥ x_c` | Binary. No gradient to trade on. By the time you know you're at x_c, the avalanche is happening. Useless for continuous trading. |
| **Exponential** | `p ∝ exp(−(x_c − x)/σ)` | Smooth, but decays too fast as x moves away from x_c. A stock 10% below x_c has essentially zero predicted risk, which is unrealistic. |
| **Power-law (chosen)** | `p = σ(−α·log(x_c−x) + β)` | Smooth ramp-up with fat tails. Always some baseline risk. Diverges as gap → 0. Physically motivated by SOC power-law statistics. Two learnable parameters. |

The power-law form was chosen because it naturally produces the fat-tailed avalanche statistics that Bouchaud's empirical work predicts, and because the logarithmic sensitivity provides a gradual, tradeable signal across a wide range of gap values.

### 3.3 The Learning System

The model has two interleaved learning processes running simultaneously. This dual-loop structure is crucial: the structural parameters α and β define the *shape* of the hazard curve (how steeply probability ramps up near x_c), while x_c defines *where* the curve is centred. Both must be learned, but on different timescales and with different update rules.

#### 3.3.1 Structural Parameters: θ = {α, β}

These define the universal relationship between the gap (x_c − x) and avalanche probability. They are learned slowly via stochastic gradient descent on binary cross-entropy loss. At each tick t, we observe the prediction p_t and the outcome y_t ∈ {0, 1} (where 1 = downtick). The loss is:

$$
\ell_t = y_t \cdot \log(p_t) + (1 - y_t) \cdot \log(1 - p_t)
$$

This is the standard binary cross-entropy. When the model predicts 95% chance of avalanche and the stock goes up, the loss is large (−log(0.05) ≈ 3.0). When the model predicts 50% and is wrong, the loss is moderate (−log(0.5) ≈ 0.69). This automatically scales the gradient: confident-and-wrong updates are large; uncertain-and-wrong updates are small.

The gradients with respect to α and β are:

$$
\frac{\partial \ell}{\partial \alpha} = (y_t - p_t) \cdot (-\log(x_c - x_t))
$$

$$
\frac{\partial \ell}{\partial \beta} = (y_t - p_t)
$$

And the update rule is standard SGD:

$$
\alpha \leftarrow \alpha + \eta_\theta \cdot \frac{\partial \ell}{\partial \alpha}
$$

$$
\beta \leftarrow \beta + \eta_\theta \cdot \frac{\partial \ell}{\partial \beta}
$$

where η_θ is a small learning rate (e.g., 0.001). We clamp α ∈ [0.1, 10] and β ∈ [−10, 10] to prevent divergence. These parameters should stabilise after thousands of ticks as the model learns the characteristic hazard shape of the asset class.

#### 3.3.2 Critical Value: x_c(t)

Unlike α and β, x_c is *not* a fixed parameter — it is a running state variable that changes continuously as market conditions evolve. Think of it as a Kalman-filter-like estimate of a latent state. The update rule is derived from the gradient of the same cross-entropy loss with respect to x_c:

$$
x_c(t+1) = x_c(t) + \eta_c \cdot (y_t - p_t) \cdot \frac{\alpha}{x_c(t) - x(t)}
$$

The intuition behind each term:

- **(y_t − p_t)** is the prediction error. If the model predicted 95% avalanche and it didn't happen, this is (0 − 0.95) = −0.95. Multiplied by the positive term α/(x_c − x), this gives a *positive* push to x_c: the critical value must be higher than we thought.

- Conversely, if the model predicted only 20% avalanche and one occurred, (1 − 0.2) = 0.8 gives a *negative* push: x_c is probably lower than estimated.

- **α / (x_c − x)** scales the update by how sensitive the prediction is to x_c at the current gap. When the price is close to x_c (small gap), the prediction is very sensitive to x_c, so the gradient is large. When the price is far below x_c, the prediction barely changes with x_c, so the update is small. This is the correct behaviour: information about x_c is most informative when x is near it.

We enforce the constraint **x_c(t) > x(t) + 0.5** at all times. If x_c ever falls to or below x, the log term in the hazard function becomes undefined. The floor of 0.5 (or some small ε) ensures numerical stability.

The learning rate η_c for x_c should be *faster* than η_θ. The structural shape of the hazard curve should be stable; the location of x_c should adapt to changing market conditions. A typical ratio might be η_c / η_θ ≈ 50.

#### 3.3.3 Forgetting and Non-Stationarity

Financial markets are non-stationary. The true x_c drifts with fundamentals; the true α and β may shift across regimes (bull markets vs. bear markets, low-vol vs. high-vol environments). To handle this, we incorporate a forgetting mechanism: rather than accumulating all history equally, we weight recent observations more heavily. Implementation options include:

1. **Exponential decay on gradients:** Use an exponential moving average of the gradient rather than the raw gradient. This is equivalent to an EWMA-weighted loss function.

2. **Sliding window:** Only compute the loss over the last N ticks (e.g., N = 5000). Simple and interpretable, but introduces a hard boundary.

3. **Adaptive learning rate:** Increase η_θ when prediction error is persistently high (indicating a regime change). This is analogous to the "surprise"-modulated learning rates in Bayesian filtering.

In the current implementation (v0.1), we use a constant learning rate with clamped parameters. Forgetting mechanisms will be added in v0.2 after establishing baseline performance.

---

## 4. Multi-Stock Extension: Cross-Asset Inference

### 4.1 The Correlation Hypothesis

The single-stock model trades on the gap between x and x_c for one instrument. The multi-stock extension exploits the fact that the critical values of related stocks are correlated. This correlation arises because stocks in the same sector, or in the same macro environment, share common drivers of their fundamental value — and therefore of their critical thresholds.

Consider two stocks, Nike (NKE) and Adidas (ADDYY). Both operate in sportswear. If Nike's price has been rising steadily without avalanches, the model infers that Nike's x_c must have also risen (otherwise, we would have seen avalanches as x approached x_c). Since Nike's and Adidas's critical values are correlated, the model can infer that Adidas's x_c has *also* likely risen — even if Adidas's own price hasn't moved yet. This means Adidas's current price x is now further below its (newly higher) x_c, implying lower avalanche risk and a higher probability of upward drift. **This is a buy signal.**

This is where the alpha comes from: the cross-asset inference provides a *lead-lag signal*. Stock A's behaviour reveals information about Stock B's latent state before Stock B's price has reflected it.

### 4.2 Modelling Cross-Asset Correlations

Let `x_c^i(t)` denote the critical value of stock i at time t. The single-stock update rule gives us a baseline estimate. The multi-stock extension adds a correction term based on the behaviour of correlated stocks:

$$
x_c^i(t+1) = x_c^i(t) + \eta_c \cdot G_i(t) + \lambda \cdot \sum_j \rho_{ij} \cdot \Delta x_c^j(t)
$$

where:

- **G_i(t)** is the single-stock gradient update (as derived in Section 3.3.2)
- **ρ_ij** is the running correlation between the x_c movements of stock i and stock j, estimated with an exponential moving average
- **Δx_c^j(t) = x_c^j(t) − x_c^j(t−1)** is the recent change in stock j's critical value
- **λ** is a coupling strength parameter that controls how much cross-asset information influences each stock's x_c estimate

The correlation matrix ρ is itself learned online. We maintain an exponentially weighted covariance matrix of the x_c update vectors across all stocks. This matrix captures not just pairwise correlations, but their evolution over time. Sector rotations, regime changes, and market structure shifts will naturally be reflected as the correlation matrix adapts.

### 4.3 Scope Limitations

Importantly, the cross-asset model does *not* attempt to predict large, systemic market movements — events that move all stocks simultaneously (e.g., a Fed rate decision, a geopolitical shock). These are exogenous shocks in the SOC framework: they change all x_c values simultaneously, and predicting them requires fundamentally different information (news, macro data) that is outside the scope of this model. What the cross-asset model *does* capture is the slower, endogenous co-movement of critical values driven by shared fundamentals within sectors and sub-sectors.

---

## 5. Trading Mechanics

### 5.1 Signal Generation

At each tick, the model computes `p_i(t)` = P(downtick for stock i | x_i(t), x_c^i(t), θ) for every stock in the universe. The trading signal for each stock is derived from the **edge**:

$$
\text{edge}_i = 1 - 2 \cdot p_i
$$

If p_i > 0.5 (avalanche more likely than not), the edge is negative: we want to be short. If p_i < 0.5, the edge is positive: we want to be long. If p_i ≈ 0.5, there is no edge and we hold.

### 5.2 Capital Allocation

Given edges for N stocks, we allocate capital proportionally to the magnitude of the edge, signed by its direction:

$$
w_i \propto |\text{edge}_i| \cdot \text{sign}(\text{edge}_i)
$$

The weights are then normalised so that the sum of absolute weights equals the total deployable capital. This ensures that a stock with a strong avalanche signal (e.g., p = 0.75, edge = −0.5) receives much more capital (short) than a stock with a weak signal (e.g., p = 0.52, edge = −0.04).

This is a simplified version of the Kelly criterion. The full Kelly criterion for correlated bets would require the covariance matrix of returns across all positions, which adds computational complexity. For the initial implementation, proportional-to-edge allocation captures 80% of Kelly's benefits with a fraction of the complexity. Full Kelly can be implemented in a later version once the single-stock model is validated.

### 5.3 Avalanche Definition

At tick-level resolution, we define an "avalanche" as any downtick — any tick where the new price is strictly lower than the previous price. This is the simplest possible definition. If the stock drops for three consecutive ticks, we treat this as three separate avalanches, each from the price at the start of that tick. The model does not attempt to predict avalanche *size*; it only predicts the *direction* of the next tick.

This simplification is justified at high frequencies, where individual ticks are approximately uniform in size (a single tick increment). At lower frequencies (daily bars), avalanche *size* becomes important, and the model would need to be extended to predict not just the probability of a down move, but also its expected magnitude. This is a natural extension for v0.2.

### 5.4 Asymmetry

The model is fundamentally asymmetric: it models *downward* avalanches only. Upward moves are treated as the "loading" phase — sand accumulating on the pile. This asymmetry is motivated by the empirical fact that financial drawdowns are sharper, faster, and more clustered than rallies. Stocks can't go below zero, but they can (in theory) go up indefinitely. The power-law tail of return distributions is fatter on the left (losses) than on the right (gains). Our model exploits this asymmetry by focusing its predictive power on the dangerous side.

---

## 6. Execution Architecture

### 6.1 Tick-by-Tick Loop

The execution loop at each tick is:

1. **Receive tick:** New price x(t) arrives from the market data feed for one or more stocks.
2. **Compute probabilities:** For each stock, compute p_i(t) = σ(−α·log(x_c^i − x_i) + β) using stored values of x_c, α, β. This is a single sigmoid evaluation per stock — microseconds.
3. **Compute allocation:** Derive edges and weights. Send orders.
4. **Observe outcome:** Wait for next tick. Determine y_t (was it a downtick?).
5. **Update model:** Update α, β, and x_c for each stock that ticked. Update cross-asset correlations on a slower schedule (every N ticks).
6. **Repeat.**

The computational cost per tick is trivial: a few log/exp evaluations and some multiplications per stock. The bottleneck will be network latency and order execution, not model computation. This makes the strategy viable for high-frequency deployment on modest hardware.

### 6.2 Training vs. Inference

Unlike deep learning models where training and inference are separate phases, this model trains and infers simultaneously. Every tick is both a prediction and a training example. However, the "slow LSTM" metaphor from our design discussions captures an important nuance: the parameters should learn on a timescale much longer than individual ticks. We don't want one bad tick to swing α by 20% — we want thousands of ticks to gradually shape the hazard curve. This is controlled by choosing small learning rates η_θ.

For the inference pass (steps 2–3 above), the computation is pure arithmetic. No neural network forward pass is required. This is by design: if we eventually train a more complex model (e.g., an LSTM that predicts x_c from a window of recent ticks and cross-asset features), we would train it offline on historical data and deploy its learned parameters as the initial conditions for the online model.

---

## 7. Risks, Limitations, and Open Questions

### 7.1 Identifiability of x_c

The critical value x_c is not directly observable. It must be inferred from the same price series it is supposed to explain. This creates a fundamental identifiability challenge: how do we distinguish "x_c has moved up" from "we just haven't had an avalanche yet by luck"? The gradient-based update rule partially addresses this (persistent no-avalanche outcomes push x_c up, while surprise avalanches push it down), but the estimates will always be noisy and lagged. Quantifying the uncertainty in x_c estimates is an important area for future work — a Bayesian treatment with a posterior distribution over x_c, rather than a point estimate, would be ideal.

### 7.2 Transaction Costs

At tick-level frequency, the edge per trade is tiny (a single tick's worth of price movement). Transaction costs — spreads, commissions, market impact — can easily exceed this edge. The strategy is only viable if: (a) the model achieves a win rate meaningfully above 50% (even 52–53% can be profitable at scale), and (b) the execution infrastructure minimises costs. Co-location, maker rebates, and smart order routing are not optional luxuries — they are necessary conditions for profitability.

### 7.3 Regime Changes

The SOC framework assumes the system is perpetually near criticality. But real markets occasionally undergo structural breaks — changes in market microstructure, regulatory shifts, liquidity regime changes — that may temporarily push the system away from criticality. During these periods, the model's assumptions break down and it may incur persistent losses. The forgetting mechanisms described in Section 3.3.3 provide partial protection, but a regime detection layer (e.g., monitoring the model's running win rate and pausing trading when it drops below a threshold) would be prudent.

### 7.4 The Cross-Asset Correlation Stability

The multi-stock model assumes that correlations between critical values are relatively stable. In practice, correlations are themselves non-stationary. Sector correlations spike during crises ("everything goes down together") and relax during calm periods. If the model's correlation estimates are stale during a regime shift, the cross-asset inference will generate false signals. Using exponentially weighted correlation estimates with a relatively short half-life (e.g., 1000–5000 ticks) mitigates this, but doesn't eliminate the risk.

### 7.5 Overfitting

With only two structural parameters (α, β) and one state variable (x_c) per stock, the model is remarkably parsimonious. This is a deliberate choice: in a noisy environment with limited edge, overfitting is the greatest risk. We resist the temptation to add more parameters (more complex hazard functions, higher-order features) until the base model demonstrably works. If the simplest possible SOC-inspired model cannot generate positive PnL in simulation, no amount of complexity will save it.

### 7.6 Model Validation and What Success Looks Like

Before deploying real capital, the model must pass the following validation gates:

1. **Synthetic data test:** Does the model recover the true x_c when run against a synthetic market with known SOC dynamics? (This is what the sandbox dashboard tests.)

2. **Historical backtest:** Run the model on real tick data (e.g., from LOBSTER or Algoseek) for at least 6 months across 10+ stocks. Measure Sharpe ratio, maximum drawdown, and win rate net of estimated transaction costs.

3. **Paper trading:** Deploy in a paper trading environment for 1–3 months with real-time data. Verify that live performance matches backtest performance (the gap between these is usually the most revealing signal of overfitting).

4. **Small capital deployment:** Deploy with minimal capital (≤10% of intended allocation) for 3 months. Monitor all dashboard metrics. Scale up only if performance is consistent.

---

## 8. Development Roadmap

| Phase | Milestone | Details |
|-------|-----------|---------|
| v0.1 | Sandbox simulation | Single-stock model on synthetic SOC market. Dashboard monitoring. Validate x_c recovery and parameter convergence. |
| v0.2 | Historical backtest | Plug in real tick data (LOBSTER/Algoseek). Add forgetting mechanism. Test on 10+ stocks. Measure Sharpe net of costs. |
| v0.3 | Multi-stock model | Add cross-asset correlation layer. Test on correlated pairs (e.g., NKE/ADDYY, AAPL/MSFT). Measure alpha from cross-inference. |
| v0.4 | Paper trading | Connect to broker API (IBKR, Alpaca). Live paper trading with real-time data. Compare to backtest. |
| v0.5 | Live deployment | Small capital. Full monitoring dashboard. Automated risk controls (max drawdown kill switch, regime detection pause). |
| v1.0 | Scale | Full capital deployment. Multiple sectors. Continuous model improvement. |

---

## 9. Mathematical Appendix

### 9.1 Full Gradient Derivations

Let `g = x_c − x` be the gap, and `z = −α · log(g) + β` be the logit. Then `p = σ(z)` and the loss is:

$$
\ell = y \cdot \log(\sigma(z)) + (1-y) \cdot \log(1-\sigma(z))
$$

Using the identity `dσ/dz = σ(z)(1−σ(z)) = p(1−p)`, and `dℓ/dp = y/p − (1−y)/(1−p)`, we get:

$$
\frac{d\ell}{dz} = (y - p)
$$

Then by the chain rule:

$$
\frac{d\ell}{d\alpha} = (y-p) \cdot \frac{dz}{d\alpha} = (y-p) \cdot (-\log(g))
$$

$$
\frac{d\ell}{d\beta} = (y-p) \cdot \frac{dz}{d\beta} = (y-p) \cdot 1
$$

$$
\frac{d\ell}{dx_c} = (y-p) \cdot \frac{dz}{dx_c} = (y-p) \cdot \left(\frac{-\alpha}{g}\right) \cdot (-1) = (y-p) \cdot \frac{\alpha}{g}
$$

Note the sign: since `dz/dg = −α/g` and `dg/dx_c = +1`, we get `dz/dx_c = −α/g`. The overall `dℓ/dx_c = (y−p) · (−α/g)`. For gradient *ascent* on the log-likelihood, the update adds this gradient to x_c, giving the update rule in Section 3.3.2. The sign convention works out: when y=0 (no avalanche) and p is high, (y−p) is negative, −α/g is negative, so the product is positive → x_c increases. Correct.

### 9.2 Connection to Survival Analysis

The model is formally equivalent to a discrete-time survival analysis with a time-varying hazard rate. Define the "survival function" as the probability that no avalanche has occurred after k ticks since the last avalanche:

$$
S(k) = \prod_{i=1}^{k} (1 - p(t_i))
$$

The expected number of ticks between avalanches is `E[T] = Σ_k S(k)`. This connects our model to the classical survival analysis literature and opens up possibilities for using Cox proportional hazards models, Weibull baseline hazards, or other semi-parametric methods as alternative or complementary modelling frameworks.

### 9.3 Connection to Hawkes Processes

Bouchaud's empirical work uses Hawkes processes [5] to model the self-exciting nature of market events. In a Hawkes process, the intensity (event arrival rate) is:

$$
\lambda(t) = \mu + \sum_{t_i < t} \varphi(t - t_i)
$$

where μ is the base rate and φ is the excitation kernel. The branching ratio `n = ∫φ(t)dt` determines criticality: n < 1 is subcritical (events die out), n = 1 is critical (barely self-sustaining), n > 1 is supercritical (explosive). Bouchaud's finding that n ≈ 1 in real markets is the empirical justification for the SOC paradigm.

Our model is *not* a Hawkes process. It is a latent-threshold hazard model inspired by the *philosophy* of SOC, not its exact mathematical formalism. The Hawkes framework models the feedback between individual events; our model abstracts this into a single latent variable x_c that captures the aggregate state of the system. A future extension could combine both: use a Hawkes process to model the short-term clustering of avalanches *within* an episode, while using the x_c framework to model the longer-term evolution of the system's critical state.

---

## 10. References

[1] Bouchaud, J.-P. (2024). *The Self-Organized Criticality Paradigm in Economics & Finance.* arXiv:2407.10284.

[2] Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: An explanation of 1/f noise. *Physical Review Letters*, 59(4), 381.

[3] Hardiman, S.J., Bercot, N., & Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *Eur. Phys. J. B*, 86, 442.

[4] Fosset, A., Bouchaud, J.-P., & Benzaquen, M. (2020). Endogenous Liquidity Crises. arXiv:1912.00359.

[5] Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83.

---

*End of Document*
