"""C5: equilibrium-rate (5.3) and front-end (5.4) refinements of the P-drift."""
import numpy as np
import jax
import jax.numpy as jnp

from slidingcurve import Model, decompose, gaussian as g, markets, simulate, time_grid, volscale
from slidingcurve.model import front_end_path

CURVE = markets.paper_curve("market")
M = Model(kappa=0.03, delta=5.0, sigma=0.01157)
T_GRID = time_grid(30.0)
MATS = jnp.array([2.0, 5.0, 10.0, 30.0])


def _se(a):
    a = np.asarray(a)
    return a.std(0, ddof=1) / np.sqrt(a.shape[0])


def _run(model=M, **kw):
    ybar = volscale.bootstrap_fn(CURVE, model, np.arange(1.0, 31.0))(T_GRID)
    return simulate(model, CURVE, T_GRID, ybar, jax.random.PRNGKey(4), n_paths=2**14, **kw)


def test_default_is_unchanged():
    a = _run()
    b = _run(ey_path=np.full(T_GRID.shape, 0.0) + float(CURVE.fwd_yield(0.0, 5.0)))
    np.testing.assert_allclose(np.asarray(a.x), np.asarray(b.x), atol=1e-15)


def test_front_end_path_is_followed_and_decomposition_closes():
    """(35)/(37): E^P[Y_1(t)] follows the blend; the Gaussian E^P[x] is then
    exact, (m(t) - Y(0,t,t+delta) - c_y y(t)) / A, and Y_eh + theta - gamma = Y."""
    path = front_end_path(CURVE, 5.0, T_GRID, Tc=2.5, c=1.0)
    prof = _run(ey_path=path)
    assert np.all(np.abs(np.asarray(prof.y1.mean(0)) - np.asarray(path)) < 4 * _se(prof.y1) + 1e-7)
    A, cy = g.A(0.03, 5.0), g.B(0.03, 5.0) ** 2 / 10.0
    ex = (path - CURVE.fwd_yield(T_GRID, T_GRID + 5.0) - cy * g.y_det(0.03, M.sigma, T_GRID)) / A
    assert np.all(np.abs(np.asarray(prof.x.mean(0)) - np.asarray(ex)) < 4 * _se(prof.x) + 1e-7)
    d = decompose(prof, M, CURVE, MATS)
    assert np.all(np.abs(np.asarray(d.residual)) < 3 * np.asarray(d.residual_se) + 2e-6)
    # (36): for T << Tc, Y_eh is close to the market yield; the blend leaves
    # Y_eh above its martingale-condition value on this upward curve
    base = decompose(_run(), M, CURVE, MATS)
    assert abs(float(d.y_eh[0] - d.y[0])) < abs(float(base.y_eh[0] - base.y[0]))


def test_equilibrium_reversion_mean_and_closure():
    """5.3: E^P[Y_1(t)] = Y* + exp(-a t) (Y_1(0) - Y*), exactly, with skew too."""
    a, ystar = 0.2, 0.035
    m = M._replace(b=0.4)
    prof = _run(model=m, reversion=(a, ystar))
    y10 = float(CURVE.fwd_yield(0.0, 5.0))
    target = ystar + np.exp(-a * np.asarray(T_GRID)) * (y10 - ystar)
    assert np.all(np.abs(np.asarray(prof.y1.mean(0)) - target) < 4 * _se(prof.y1) + 1e-7)
    d = decompose(prof, m, CURVE, MATS)
    assert np.all(np.abs(np.asarray(d.residual)) < 3 * np.asarray(d.residual_se) + 2e-6)
    # Gaussian case: E^P[x(t)] is exact given the mean path, as for (35)
    prof0 = _run(reversion=(a, ystar))
    A, cy = g.A(0.03, 5.0), g.B(0.03, 5.0) ** 2 / 10.0
    ex = (target - np.asarray(CURVE.fwd_yield(T_GRID, T_GRID + 5.0))
          - cy * np.asarray(g.y_det(0.03, M.sigma, T_GRID))) / A
    assert np.all(np.abs(np.asarray(prof0.x.mean(0)) - ex) < 4 * _se(prof0.x) + 1e-7)
