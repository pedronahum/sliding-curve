"""Randomised quasi-Monte Carlo normals and the `normals=` hooks."""
import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, simulate, time_grid, volscale, qmc
from slidingcurve import pricing as pr
from slidingcurve.gaussian import A, B
from slidingcurve.markets import paper_curve

CURVE = paper_curve("baseline")
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)


def test_bridge_plan_builds_every_point_once():
    for n in (1, 2, 7, 12, 480):
        idx = sorted(p[0] for p in qmc.bridge_plan(n))
        assert idx == list(range(1, n + 1))


def test_sobol_normals_are_standard_independent_increments():
    z = qmc.sobol_normals(24, 2, 2**12, seed=1)                    # [24, 2, 2, 4096]
    flat = z.transpose(1, 0, 2, 3).reshape(2 * 24, -1)             # rows: driver x step
    np.testing.assert_allclose(flat.mean(1), 0.0, atol=0.02)
    np.testing.assert_allclose(np.cov(flat), np.eye(48), atol=0.06)


def test_given_normals_reproduce_the_library_draws():
    """Passing the library's own draws reproduces the default bit for bit."""
    t = time_grid(5.0)
    m = BASE._replace(b=0.5, eta=0.5)
    ybar = volscale.bootstrap_fn(CURVE, m, np.arange(1.0, 6.0))(t)
    key = jax.random.PRNGKey(2)
    nb, n = 4, 2**9
    ref = simulate(m, CURVE, t, ybar, key, n_paths=n, n_batches=nb, antithetic=False)
    draws = jnp.stack([jax.random.normal(jax.random.fold_in(key, i), (2, nb, n // nb))
                       for i in range(t.shape[0])])
    got = simulate(m, CURVE, t, ybar, None, n_paths=n, n_batches=nb, normals=draws)
    np.testing.assert_array_equal(np.asarray(ref.x), np.asarray(got.x))


def test_rqmc_gaussian_price_is_unbiased_and_beats_mc():
    """ATM 10y x 5y yield option at b = 0: closed form within 4 s.e. across
    scrambles, and a smaller s.e. than pseudo-random MC at equal paths."""
    T, n_steps, P, R = 10.0, 60, 2**10, 8
    fn = volscale.bootstrap_fn(CURVE, BASE, np.arange(1.0, 11.0))
    s = float(A(0.03, 5.0) * 0.01 * np.sqrt(float(B(0.06, T))))
    exact = s / np.sqrt(2 * np.pi)
    z = qmc.sobol_normals(n_steps, R, P, seed=3)
    q = [float(pr.yield_option_prices(BASE, CURVE, T, np.array([0.0]), fn, None, n_paths=P,
                                      steps_per_year=6, normals=z[:, :, r, :])[1][0]) for r in range(R)]
    m = [float(pr.yield_option_prices(BASE, CURVE, T, np.array([0.0]), fn, jax.random.PRNGKey(r),
                                      n_paths=P, steps_per_year=6)[1][0]) for r in range(R)]
    se_q, se_m = np.std(q, ddof=1) / np.sqrt(R), np.std(m, ddof=1) / np.sqrt(R)
    # discretisation of the Gaussian scheme is exact in distribution, so only noise remains
    assert abs(np.mean(q) - exact) < 4 * se_q + 1e-7
    assert se_q < se_m
