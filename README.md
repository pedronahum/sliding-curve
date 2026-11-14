# slidingcurve

Term premia and bond convexity under sliding-curve dynamics, in JAX.

A working replication-in-progress of

> V. Piterbarg and W. Van Der Helm, *Term Premia and Bond Convexity: Sliding
> Curves, Trading Strategies, and Model Impact* (6 November 2026).

Equation numbers in the code refer to that paper. This is an independent
implementation written from the paper; it is not the authors' code.

## What works today

| Piece | Module | Status |
|---|---|---|
| Initial curve from one zero-yield function, forwards by autodiff | `curve.py` | tested |
| Linear Gaussian closed forms, Appendix D | `gaussian.py` | tested against eqs. (93), (94), (96) and D.5 |
| One-factor quasi-Gaussian simulator in the physical measure, eq. (52) | `model.py` | Gaussian case matches closed forms |
| Yield decomposition `Y = Y_eh + theta - gamma`, eqs. (12), (54), (55) | `decomposition.py` | closes within Monte-Carlo noise |
| Volatility scaling: zeroth order and bootstrap, eqs. (79)-(81) | `volscale.py` | algebra tested; effect on smiles not yet |
| Skew `b` and stochastic volatility `eta` in the simulator | `model.py` | **implemented, validated only by model-free identities** |

Measured on the illustrative curve in `examples/` (Section 9.1 baseline:
`delta = 5`, `kappa = 0.03`, `sigma = 1%`, monthly steps, 65,536 paths):

- Gaussian case: Monte Carlo and closed form agree to 0.01bp for `Y_eh`,
  `theta` and `gamma` at every maturity from 1 to 40 years.
- With skew and/or stochastic volatility on, the three independently estimated
  terms sum to the market yield to within their standard error (under 1bp at
  40 years).

![Gaussian decomposition](docs/gaussian_decomposition.png)

## Not done yet

Nothing below exists, so **no figure of the paper beyond the Gaussian
decomposition can be reproduced yet**:

1. **Swaption/caplet pricer in the risk-neutral measure** and normal implied
   volatilities (Figures 4, 8, 11). Without it there is no way to check that
   the volatility scaling holds at-the-money volatility fixed across `b`, so
   the skew sweep in the example is labelled uncalibrated.
2. **Calibration** of `sigma`, `b`, `eta` (Sections 9.3-9.4). `jax.grad`
   through the pricer is the intended route.
3. **Paper inputs.** The synthetic curve is given only as a plot, and the
   21-Aug-2026 USD swap curve and five 20y x 5y smile quotes are not tabulated.
   `Curve.from_zero_nodes` is the hook for them once available.
4. **Market-curve bootstrap** from swap quotes (the spline constructor takes
   zero yields someone else has bootstrapped).
5. **Short-rate density** under P (Figure 13).
6. **Multi-factor model** (`d > 1`), the Ho-Lee checks of Appendix E, and the
   forward-rate decomposition of Appendix F.
7. Quasi-random numbers (Sobol with a Brownian bridge) and a GPU run.

## Choices the paper leaves open

These are assumptions of this implementation, to be revisited if the authors
clarify them:

- **Time stepping.** Monthly by default. `y` is advanced exactly for frozen
  volatility, and the drift of `x` is the increment that keeps the benchmark
  yield an exact discrete martingale; only the diffusion is Euler.
- **Variance process.** Full-truncation Euler.
- **Volatility cap** (Remark 8). Applied to `sigma_r` at 10x `sigma`; the
  local volatility is floored at zero.
- **Bootstrap grid** `T_m` for the volatility scaling is a user input.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # 28 tests, about a minute on CPU
python examples/gaussian_decomposition.py
```

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

## Design notes

- In the one-factor model `G(t, T)` is deterministic, so one simulation
  returns three mean profiles in time (`E[x]`, `E[mu^Y]`, `E[sigma_r^2]`) and
  every maturity's decomposition is a quadrature of those. No per-maturity
  resimulation.
- Paths are split into independent batches; standard errors of the integrated
  quantities are taken across batches.
- Float64 throughout: the quantities of interest are differences of yields at
  the 0.01bp level.

## Licence

None chosen yet. No code was copied from kooderive (GPL v3), so the choice is
open; importing its Sobol and Brownian-bridge modules later would make this
repository GPL v3 too.
