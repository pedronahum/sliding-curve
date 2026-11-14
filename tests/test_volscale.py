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
