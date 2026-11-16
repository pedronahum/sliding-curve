# slidingcurve

Term premia and bond convexity under sliding-curve dynamics, in JAX.

An independent implementation of the one- and multi-factor quasi-Gaussian
(Cheyette) model of

> V. Piterbarg and W. Van Der Helm, *Term Premia and Bond Convexity: Sliding
> Curves, Trading Strategies, and Model Impact* (6 November 2026),

simulated in the physical measure defined by the sliding-curve condition.
Equation numbers in the docstrings refer to that paper. The code was written
from the paper; it is not the authors' code. It reproduces their plotted
results — see [docs/paper_results.md](docs/paper_results.md) for the
comparison and [docs/investigation.md](docs/investigation.md) for the log of
every issue found and closed along the way.

## Install

```bash
pip install -e ".[dev]"          # pytest, matplotlib, mypy, scipy-stubs
pytest                           # 109 tests
```

On a GPU, install a CUDA jaxlib (`pip install "jax[cuda13]"`); the same code
runs unchanged and reproduces the CPU results digit for digit, 10-27x faster
on a million paths. On a machine whose GPU shares host memory (NVIDIA DGX
Spark and similar), set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Quick start

```python
import jax, jax.numpy as jnp
from slidingcurve import Curve, Model, simulate, decompose, time_grid, volscale

curve = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
model = Model(kappa=0.03, delta=5.0, sigma=0.01, b=0.25)
t = time_grid(40.0)
ybar = volscale.ybar_zeroth(curve, model.delta, t)
prof = simulate(model, curve, t, ybar, jax.random.PRNGKey(0))
dec = decompose(prof, model, curve, jnp.array([10.0, 30.0]))
print(dec.theta, dec.gamma, dec.residual)
```

`simulate` returns the *path means* of the state on the grid, and because
`G(t, T)` is deterministic in this model those means are all the yield
decomposition `Y(0, T) = Y_eh(T) + theta(T) - gamma(T)` needs for every
maturity at once — one simulation, no per-maturity resampling. `residual` is
the identity's closure and is a genuine check of the engine in any parameter
regime.

## What is in the box

| Module | What it gives you |
|---|---|
| `curve` | A curve from one zero-yield function; forwards and discount factors by autodiff, so a new curve family is one function. Nelson-Siegel/Svensson, spline-through-nodes and a deposit/par-swap bootstrap (`bootstrap`). |
| `model` | Physical-measure simulator, one factor: the drift that keeps the benchmark yield an exact discrete martingale, optional front-end (35) and equilibrium (5.3) drift refinements, Andersen QE or smooth lognormal variance, frozen-vol or predictor-corrector steps. |
| `multifactor` | The d-factor model of Sections 5-6 with the general volatility-scaling bootstrap of Appendix C.3.2, plus its Gaussian closed forms. |
| `pricing` | Forward-measure simulator of the terminal state, affine zero-coupon bonds from the state, payer swaptions and caplets, the paper's yield option, a Gaussian-twin control variate, and path sharding across devices. |
| `gaussian` | Appendix D closed forms and Jamshidian's exact swaption price: the oracle every simulator is tested against. |
| `holee`, `forward` | Ho-Lee closed forms (Appendix E); the forward-rate decomposition, Eurodollar futures and Morton's construction (Appendix F). |
| `decomposition` | `Y = Y_eh + theta - gamma` from simulated profiles, with standard errors across batches. |
| `volscale` | The volatility scaling `Ybar_1(t)` of (79)-(81), differentiable in `sigma` so calibrations can carry its dependence inside the objective. |
| `calibration` | Least-squares fits of `sigma`, `b`, `eta` (and `kappa`) with a forward-mode autodiff Jacobian through the Monte-Carlo pricer. |
| `qmc` | Randomised quasi-Monte Carlo normals (scrambled Sobol, general Brownian bridge) for either simulator. |
| `bachelier` | Normal and displaced-lognormal quote conventions and implied vols. |
| `markets` | The paper's digitised curves and smile quotes, and Mark Joshi's displaced-diffusion LMM as a published test market. |

Everything is float64 and pure JAX: prices are differentiable in the model
parameters, and the simulators `jit`, `vmap` and shard across devices.

## Pricing a payoff of your own

`pricing.simulate_terminal` returns the model state `(x, y)` at one expiry,
in the forward measure of that expiry or of any `measure_maturity` you name,
as a `TerminalState`. Zero-coupon bonds are affine in that state, so every
rate, yield and swap rate at the expiry is a closed-form function of it:

```python
import jax
import jax.numpy as jnp
from slidingcurve import Curve, Model, pricing, volscale

curve = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
model = Model(kappa=0.03, delta=5.0, sigma=0.01, b=0.5)
key = jax.random.PRNGKey(0)
ybar = volscale.bootstrap_fn(curve, model, jnp.arange(1.0, 21.0))
state = pricing.simulate_terminal(model, curve, 10.0, ybar, key)

# bonds P(10, 15) and P(10, 20) on every path: [2, n_paths]
p = pricing.bond(curve, model.kappa, 10.0,
                 jnp.array([15.0, 20.0])[:, None], state.x[None, :], state.y[None, :])
payoff = jnp.maximum(p[0] - 0.80, 0.0)              # a call on the 15y zero
price = curve.discount(10.0) * payoff.mean()
```

Write your payoff in `jnp` over `state.x` and `state.y` and the whole thing
stays inside one traceable graph — `jax.grad` of a price with respect to
`sigma`, `b` or `eta` is the same call the calibrator uses. Built-in
instruments: `pricing.Swaption.regular` / `.caplet` with `swaption_prices`,
and `yield_option_prices` for options on the benchmark yield itself (with
`control_variate=True` for the Gaussian twin).

The boundary today is *European, one expiry*: the terminal-state simulator is
the only pricing path generator. Path-dependent and Bermudan payoffs need a
pathwise simulator, which is the next thing on the roadmap.

## Calibration

```python
from slidingcurve import calibration, markets

quotes = markets.paper_smile()                      # Swaption, strikes, prices, vols
fit = calibration.calibrate(Model(0.03, 5.0, 0.01), curve, quotes, free=("sigma", "b"))
print(fit.model, fit.vol_errors_bp)                 # errors in bp of implied normal vol
```

Residuals are price differences divided by each quote's normal vega, so the
objective is smooth without inverting Bachelier inside the optimiser, and the
volatility scaling is rebuilt from the trial `sigma` inside the objective.
`calibrate_yield_smile` fits the same way to vols of options on the benchmark
yield; with `eta` free, use `variance_scheme="lognormal"` to keep autodiff.

## Performance

- **GPU**: unchanged code, digit-for-digit the CPU results, 10-27x faster on
  a million paths.
- **Every core**: `XLA_FLAGS=--xla_force_host_platform_device_count=16` (or
  your core count) and `pricing.set_path_devices()`; a `(sigma, b)` smile fit
  then takes about 4s instead of 21s, with results unchanged within
  Monte-Carlo error.
- **Quasi-Monte Carlo**: scrambled Sobol with a Brownian bridge cuts variance
  23-290x with skew and 1-5x with stochastic volatility, passed to either
  simulator with `normals=`.
- **Control variate**: the Gaussian twin makes `sigma` for a given
  at-the-money vol exact at any seed and path count, and cuts variance 14-28x
  at the market skew.

## Replication of the paper

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/sliding-curve/blob/main/notebooks/replicate_paper.ipynb)

[`notebooks/replicate_paper.ipynb`](notebooks/replicate_paper.ipynb) rebuilds
every figure of Section 9 against the paper's values and ends with a
scorecard; it runs in a few minutes on a free Colab CPU runtime.

Figures 2, 3 and 5-13 are reproduced: the Gaussian decomposition to 0.24bp
(digitisation error), convexity to within Monte-Carlo error at low skew, the
market-curve decomposition to the printed basis point, the skew fit to
0.7bp, and the short-rate densities to the paper's own rounding. Two
residuals are closed as documented parameter choices rather than
discrepancies — convexity at high skew sits inside the sensitivity to the
volatility cap of Remark 8, which the paper leaves undefined, and Figure 9 at
`eta = 1.5` is the size of the paper's sampling noise. Figure 4 was a bug in
the paper's implied-volatility calculation: the authors confirmed it on
16 November 2026 and are posting a corrected version, and the `T / delta`
fingerprint this code measured is what located it.

The full comparison is in [docs/paper_results.md](docs/paper_results.md);
the investigation that produced and closed every item is in
[docs/investigation.md](docs/investigation.md); what was put to the authors
and their answer is in
[docs/questions_for_authors.md](docs/questions_for_authors.md).

## End to end on a published test market

`examples/joshi_calibration.py` runs the Section 9.4 workflow on Mark Joshi's
displaced-diffusion LMM example, with the 10y x 5y swaption at five strikes.
Fitting `sigma` and `b` matches all five normal volatilities to under 0.1bp
(`sigma = 1.83%`, `b = 0.53`).

![Calibration on Joshi's market](docs/joshi_calibration.png)

This is not the paper's data: the market is a model with a pure
displaced-diffusion skew, its curve runs from 0.8% to 8.8% and stops at 20.5
years, and the swaption quotes come from the frozen-weights approximation. It
exercises the machinery end to end on numbers published elsewhere.

## Repository layout

```
slidingcurve/     the library (typed; see slidingcurve/py.typed)
examples/         runnable demonstrations, one per paper section
experiments/      the investigation scripts behind docs/investigation.md
tests/            the regression suite (109 tests)
tools/            figure digitisation
docs/             this documentation, the paper comparison and the investigation log
```

## Choices the implementation makes

The paper leaves several things open; these are this implementation's
answers, each tested in `docs/investigation.md`:

- **Time stepping.** Monthly for the decomposition, 24 steps a year for
  pricing; `y` advances exactly for frozen volatility and the drift of `x`
  keeps the benchmark yield an exact discrete martingale. `Model(vol_scheme=
  "pc")` swaps in a predictor-corrector.
- **Variance process.** Andersen's quadratic-exponential scheme;
  `variance_scheme="lognormal"` is the smooth-in-`eta` alternative used when
  calibrating with `eta` free.
- **Volatility cap** (Remark 8): on `sigma_r` at 10x `sigma`, local vol
  floored at zero. At `b >= 0.5` the cap binds and convexity at 40 years
  depends on its level; the paper's definition is not precise enough to pin
  it down, so the sensitivity is tabulated instead.
- **Bootstrap grid** for the volatility scaling: monthly, where (81) is
  converged (annual is 3.4bp off on the steep curve at `b = 1`).
- **Synthetic curves**: splines through yields read off Figures 2 and 5,
  extended quadratically from 40 to 45 years.
- **Calibration objective**: price errors divided by normal vega, on fixed
  random numbers, with the at-the-money quote pinned.

## Licence

Apache-2.0. The digitised figure data in `slidingcurve/data/` was read off
the images of the paper cited above; it is a table of measured coordinates,
provided for reproducibility of the comparison in `docs/paper_results.md`.
