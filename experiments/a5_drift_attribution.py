"""A5: attribute the gap between the full model's uplift and the exact Jensen
ratio. Same variance paths (QE), same rate noise; the rate is first made
driftless (dx = -kappa x dt + sigma sqrt(z) dW, no y term), then the
forward-measure drift is restored. With no drift the model must reproduce
E(V)/E(sqrt V)^2 (14.52 / 16.27 / 17.51%)."""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, smile, save
from slidingcurve import Model, markets
from slidingcurve.gaussian import A, B
from slidingcurve.model import variance_step

T, SPY, N = 10.0, 24, 2**17
curve = markets.paper_curve("baseline")


def atm_vol(kappa, eta, drift, seed=0):
    m = Model(kappa=kappa, delta=5.0, sigma=0.01, eta=eta)
    n = int(T * SPY); dt = T / n
    t = jnp.arange(n) * dt
    Ad, Bd = A(kappa, 5.0), B(kappa, 5.0)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2 * kappa * dt), B(2 * kappa, dt)
    g = B(kappa, T + 5.0 - t)
    key = jax.random.PRNGKey(seed)

    def step(s, inp):
        x, y, z = s
        i, gi = inp
        vol = 0.01 * jnp.sqrt(jnp.maximum(z, 0.0))
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, N // 2))
        xi = jnp.concatenate([xi, -xi], 1)
        dr = (y - vol**2 * gi) * b1 if drift else 0.0
        return (x * e1 + dr + vol * jnp.sqrt(b2) * xi[0], y * e2 + vol**2 * b2, variance_step(m, z, xi[1], dt)), None

    (x, y, _), _ = jax.lax.scan(step, (jnp.zeros(N), jnp.zeros(N), jnp.ones(N)), (jnp.arange(n), g))
    y1 = np.asarray(Ad * x + (Bd**2 / 10.0 * y if drift else 0.0))
    return BP * smile(y1, y1.mean() + np.array([0.0]), y1.mean(), T)[0]


out = {}
for kappa in (0.0, 0.03, 0.05):
    row = {}
    for drift in (False, True):
        v0, v1 = atm_vol(kappa, 0.0, drift), atm_vol(kappa, 0.6, drift)
        row["with drift" if drift else "driftless"] = (v0 / v1) ** 2 - 1      # sigma recalibration is linear here
    out[kappa] = row
    print(f"kappa {kappa:4.2f}: driftless {100*row['driftless']:5.2f}%   forward-measure drift {100*row['with drift']:5.2f}%")
save("a5_drift_attribution", out)


# proper recalibration with the drift on: solve sigma so that the ATM vol at
# eta = 0.6 equals the eta = 0 one (common random numbers)
from scipy.optimize import brentq                                     # noqa: E402




def atm_sig(kappa, eta, sigma, seed=0):
    m = Model(kappa=kappa, delta=5.0, sigma=sigma, eta=eta)
    n = int(T * SPY); dt = T / n
    t = jnp.arange(n) * dt
    Ad, Bd = A(kappa, 5.0), B(kappa, 5.0)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2 * kappa * dt), B(2 * kappa, dt)
    g = B(kappa, T + 5.0 - t)
    key = jax.random.PRNGKey(seed)

    def step(s, inp):
        x, y, z = s
        i, gi = inp
        vol = sigma * jnp.sqrt(jnp.maximum(z, 0.0))
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, N // 2))
        xi = jnp.concatenate([xi, -xi], 1)
        return (x * e1 + (y - vol**2 * gi) * b1 + vol * jnp.sqrt(b2) * xi[0], y * e2 + vol**2 * b2,
                variance_step(m, z, xi[1], dt)), None

    (x, y, _), _ = jax.lax.scan(step, (jnp.zeros(N), jnp.zeros(N), jnp.ones(N)), (jnp.arange(n), g))
    y1 = np.asarray(Ad * x + Bd**2 / 10.0 * y)
    return BP * smile(y1, y1.mean() + np.array([0.0]), y1.mean(), T)[0]


for kappa in (0.0, 0.03, 0.05):
    target = atm_sig(kappa, 0.0, 0.01)
    s = brentq(lambda s_: atm_sig(kappa, 0.6, s_) - target, 0.0100, 0.0115, xtol=1e-8)
    out[kappa]["with drift, sigma solved"] = (s / 0.01) ** 2 - 1
    print(f"kappa {kappa:4.2f}: forward-measure drift, sigma solved: {100*((s/0.01)**2-1):5.2f}%")
save("a5_drift_attribution", out)
