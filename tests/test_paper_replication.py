"""Replication targets: numbers read off the paper's own figures
(markets.paper_figures, accurate to a few tenths of a basis point)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)       # Section 9.1
MATS = jnp.array([10.0, 20.0, 30.0, 40.0])


def at(fig, series, T):
    t = np.asarray(fig["T"])
    return np.array([fig[series][int(np.argmin(np.abs(t - x)))] for x in T], dtype=float)


def run(which, b, n_paths=2**15):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b)
    t = time_grid(40.0)
    ybar = volscale.bootstrap_fn(curve, m, np.arange(1.0, 41.0))(t)
    return decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=n_paths),
                     m, curve, MATS)


@pytest.mark.parametrize("which,fig", [("baseline", "fig2"), ("steep", "fig5")])
def test_expectation_hypothesis_yield_figures_2_and_5(which, fig):
    d = run(which, 0.0, n_paths=2**12)
    paper = at(markets.paper_figures()[fig], "Y_eh", np.asarray(MATS))
    np.testing.assert_allclose(1e4 * np.asarray(d.y_eh), paper, atol=0.5)


@pytest.mark.parametrize("which,fig,b", [("baseline", "fig3", 0.5), ("baseline", "fig3", 1.0),
                                         ("steep", "fig6", 1.0)])
def test_convexity_with_skew_figures_3_and_6(which, fig, b):
    d = run(which, b)
    paper = at(markets.paper_figures()[fig], f"b={b:.2f}", np.asarray(MATS))
    # within 4% or 1bp (short maturities, where plotted lines hide each other):
    # Monte-Carlo error plus a 1-3% shortfall at the long end that
    # is not yet explained (see README)
    np.testing.assert_allclose(1e4 * np.asarray(d.gamma), paper, rtol=0.04, atol=1.0)
    # and nowhere near the forward-centred scaling, which gives 185bp, not 320bp
    if which == "baseline" and b == 1.0:
        assert 1e4 * float(d.gamma[-1]) > 290
