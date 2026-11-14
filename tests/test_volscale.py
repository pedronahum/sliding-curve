import numpy as np
import jax.numpy as jnp

from slidingcurve import gaussian as g, volscale

EXP = np.arange(1.0, 31.0)


def test_bootstrap_first_bucket_and_zero_vol_limit(ns_curve):
    yb = volscale.ybar_bootstrap(ns_curve, 0.03, 5.0, 0.01, EXP)
    np.testing.assert_allclose(yb[0], g.yhat(ns_curve, 0.03, 5.0, 0.01, 1.0, 1.0), rtol=1e-12)
    # sigma -> 0: no convexity, the bootstrap telescopes to forwards at T_m
    yb0 = volscale.ybar_bootstrap(ns_curve, 0.03, 5.0, 0.0, EXP)
    fwd = np.asarray(ns_curve.fwd_yield(jnp.asarray(EXP), jnp.asarray(EXP) + 5.0))
    assert np.all(np.isfinite(yb0))
    np.testing.assert_allclose(yb0[0], fwd[0], rtol=1e-12)


def test_bootstrap_satisfies_eq80_rectangular(ns_curve):
    """Sum_k Ybar_k K_k = Sum_k Yhat(T_k, T_m) K_k for every m (unweighted)."""
    yb = volscale.ybar_bootstrap(ns_curve, 0.03, 5.0, 0.01, EXP)
    for m in (0, 4, 29):
        yh = np.asarray(g.yhat(ns_curve, 0.03, 5.0, 0.01, jnp.asarray(EXP[: m + 1]), EXP[m]))
        np.testing.assert_allclose(np.sum(yb[: m + 1]), np.sum(yh), rtol=1e-12)


def test_piecewise_constant_buckets():
    v = np.array([10.0, 20.0, 30.0])
    t = jnp.array([0.0, 0.5, 1.0, 1.01, 2.0, 2.5, 3.0, 9.0])
    np.testing.assert_array_equal(
        volscale.piecewise_constant([1.0, 2.0, 3.0], v, t),
        [10, 10, 10, 20, 20, 30, 30, 30])


def test_vectorised_bootstrap_equals_the_recursion_81():
    """B2: the telescoped bootstrap is the recursion (81) itself."""
    import numpy as np
    from slidingcurve import gaussian as g, markets, volscale
    curve = markets.paper_curve("market")
    for weighted in (False, True):
        for grid in (np.arange(1.0, 21.0), np.arange(1, 41) / 4.0):
            Tm = grid
            lo = np.concatenate([[0.0], Tm[:-1]])
            K = (np.exp(0.03 * Tm) - np.exp(0.03 * lo)) / 0.03 if weighted else Tm - lo
            ref = []
            for m in range(len(Tm)):
                yh = np.asarray(g.yhat(curve, 0.03, 5.0, 0.012, Tm[: m + 1], Tm[m]))
                ref.append(yh[m] + sum((yh[k] - ref[k]) * K[k] for k in range(m)) / K[m])
            fast = np.asarray(volscale.ybar_bootstrap(curve, 0.03, 5.0, 0.012, grid, weighted))
            np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-15)


def test_default_bootstrap_grid_is_converged():
    """A2: (81) discretises (80) with a first-order error in the grid
    spacing. On the steep curve at b = 1 the default (monthly) grid must agree
    with a weekly one; the annual grid the code used before does not (common
    random numbers: only the scaling differs)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale
    curve = markets.paper_curve("steep")
    m = Model(kappa=0.03, delta=5.0, sigma=0.01, b=1.0)
    t = time_grid(40.0, steps_per_year=24)

    def gamma(grid):
        ybar = volscale.bootstrap_fn(curve, m, grid)(t)
        d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**12), m, curve,
                      jnp.array([40.0]))
        return 1e4 * float(d.gamma[0])

    weekly = gamma(volscale.bootstrap_grid(40.0, per_year=52))
    assert abs(gamma(volscale.bootstrap_grid(40.0)) - weekly) < 0.5
    assert abs(gamma(np.arange(1.0, 41.0)) - weekly) > 2.0
