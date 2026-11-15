"""The predictor-corrector volatility scheme (Model.vol_scheme = "pc")."""
import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, decompose, simulate, time_grid, volscale
from slidingcurve import pricing as pr
from slidingcurve.markets import paper_curve

BP = 1e4
CURVE = paper_curve("baseline")
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)


def _ybar(m, t):
    return volscale.bootstrap_fn(CURVE, m, volscale.bootstrap_grid(t[-1]))(t)


def test_pc_is_euler_in_the_gaussian_case():
    """Constant volatility: the corrector changes nothing. Not bit for bit: at
    eta = 0 the QE variance step returns z = 1 only to ~1e-6 (psi is floored
    at 1e-12), and the corrector reads the vol at the new z."""
    t = time_grid(10.0)
    key = jax.random.PRNGKey(3)
    a = simulate(BASE, CURVE, t, _ybar(BASE, t), key, n_paths=2**10)
    b = simulate(BASE._replace(vol_scheme="pc"), CURVE, t, _ybar(BASE, t), key, n_paths=2**10)
    np.testing.assert_allclose(np.asarray(a.x), np.asarray(b.x), rtol=1e-5, atol=1e-7)
    fn = volscale.bootstrap_fn(CURVE, BASE, np.arange(1.0, 11.0))
    xa, _ = pr.simulate_terminal(BASE, CURVE, 10.0, fn, key, 2**10)
    xb, _ = pr.simulate_terminal(BASE._replace(vol_scheme="pc"), CURVE, 10.0, fn, key, 2**10)
    np.testing.assert_allclose(np.asarray(xa), np.asarray(xb), rtol=1e-5, atol=1e-7)


def test_pc_keeps_the_decomposition_and_the_martingale():
    """With skew the identity Y = Y_eh + theta - gamma still closes, E[Y_1]
    stays at Y_1(0) within noise, and gamma agrees with the Euler scheme."""
    m = BASE._replace(b=1.0)
    t = time_grid(20.0)
    key = jax.random.PRNGKey(5)
    out = {}
    for scheme in ("euler", "pc"):
        mm = m._replace(vol_scheme=scheme)
        p = simulate(mm, CURVE, t, _ybar(m, t), key, n_paths=2**15)
        d = decompose(p, mm, CURVE, jnp.array([20.0]))
        assert abs(float(d.residual[0])) < 4 * float(d.residual_se[0]) + 1e-5
        y1 = BP * np.asarray(p.y1[:, -1])
        drift = y1.mean() - BP * float(CURVE.fwd_yield(0.0, 5.0))
        assert abs(drift) < 4 * y1.std(ddof=1) / np.sqrt(len(y1)) + 0.5
        out[scheme] = (BP * float(d.gamma[0]), BP * float(d.gamma_se[0]))
    diff = out["pc"][0] - out["euler"][0]
    assert abs(diff) < 1.0, out
