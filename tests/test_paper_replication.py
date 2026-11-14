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
    ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
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


# --- Section 9.4: the market curve (Figures 10-12) ---------------------------

MKT_MATS = jnp.array([5.0, 10.0, 20.0, 30.0])
MKT_IDX = [19, 39, 79, 119]


def run_market(model, n_paths=2**14):
    curve = markets.paper_curve("market")
    t = time_grid(30.0)
    ybar = volscale.bootstrap_fn(curve, model, volscale.bootstrap_grid(30.0))(t)
    return decompose(simulate(model, curve, t, ybar, jax.random.PRNGKey(1), n_paths=n_paths),
                     model, curve, MKT_MATS)


def test_gaussian_yield_option_vol_gives_the_papers_sigma():
    """Section 9.4: an 82bp at-the-money vol on the 20y x 5y option gives
    sigma = 1.157%. In the Gaussian case the yield-option vol is
    A(delta) sigma sqrt(B_{2 kappa}(T) / T)."""
    from slidingcurve import bachelier as bl, gaussian as g, pricing as pr
    np.testing.assert_allclose(
        0.0082 / (g.A(0.03, 5.0) * np.sqrt(g.B(0.06, 20.0) / 20.0)), 0.01157, rtol=5e-4)
    curve = markets.paper_curve("market")
    m = BASE._replace(sigma=0.01157)
    off = np.array([-0.02, 0.0, 0.02])
    _, p, _ = pr.yield_option_prices(m, curve, 20.0, off, lambda t: volscale.ybar_zeroth(curve, 5.0, t),
                                     jax.random.PRNGKey(0), n_paths=2**15)
    np.testing.assert_allclose(bl.implied_normal_vol(np.asarray(p), 0.0, off, 20.0), 0.0082, rtol=0.01)


def test_market_decomposition_figure_10():
    d = run_market(BASE._replace(sigma=0.01157), n_paths=2**12)
    ref = markets.paper_figures()["fig10"]
    for name, mine in (("Y_eh", d.y_eh), ("theta", d.theta), ("gamma", d.gamma)):
        paper = np.array([ref[name][i] for i in MKT_IDX])
        # 0.5bp out to 20y; 2bp at 30y, where the curve beyond 30y is extrapolated
        np.testing.assert_allclose(1e4 * np.asarray(mine)[:3], paper[:3], atol=0.5)
        np.testing.assert_allclose(1e4 * np.asarray(mine)[3], paper[3], atol=2.0)


def test_market_decomposition_with_skew_figure_12():
    """At the parameters in the caption of Figure 13 (sigma = 1.1999%,
    b = 0.367). The 1.12% printed in the text of Section 9.4 does not
    reproduce the figure: it gives gamma(30y) = 115.6bp, not 140bp."""
    d = run_market(BASE._replace(sigma=0.011999, b=0.367), n_paths=2**15)
    ref = markets.paper_figures()["fig12"]
    np.testing.assert_allclose(1e4 * np.asarray(d.gamma), [ref["gamma"][i] for i in MKT_IDX],
                               rtol=0.02, atol=0.3)
    np.testing.assert_allclose(1e4 * np.asarray(d.theta), [ref["theta"][i] for i in MKT_IDX],
                               rtol=0.02, atol=0.7)


# --- Figure 8 (smiles by eta) and Figure 13 (short-rate densities) ------------

@pytest.mark.parametrize("eta", [0.5, 1.0, 1.5])
def test_atm_vol_falls_with_eta_figure_8(eta):
    """20y x 5y at-the-money yield-option vol at fixed sigma = 1%: the paper
    plots 66.1, 60.9 and 53.6bp."""
    from slidingcurve import bachelier as bl, pricing as pr
    curve = markets.paper_curve("baseline")
    m = BASE._replace(eta=eta)
    _, p, _ = pr.yield_option_prices(m, curve, 20.0, np.array([0.0]),
                                     lambda t: volscale.ybar_zeroth(curve, 5.0, t),
                                     jax.random.PRNGKey(0), n_paths=2**15)
    mine = 1e4 * bl.implied_normal_vol(np.asarray(p), 0.0, np.array([0.0]), 20.0)[0]
    paper = markets.paper_figures()["fig8"][f"eta={eta}"][5]
    assert abs(mine - paper) < 1.5


def test_short_rate_distribution_figure_13():
    curve = markets.paper_curve("market")
    ref = markets.paper_figures()["fig13"]
    t = time_grid(30.0)
    stats = {}
    for name, m in (("atm", BASE._replace(sigma=0.011572)),
                    ("skew", BASE._replace(sigma=0.011999, b=0.367))):
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(30.0))(t)
        _, r = simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**14,
                        snapshot_times=(20.0, 30.0))
        r = 1e4 * np.asarray(r)
        stats[name] = r
        for i, T in enumerate((20, 30)):
            assert abs(r[i].mean() - ref[name]["mean"][f"T={T}"]) < 12       # bp
    # Section 9.4: without skew the st.dev. of r(30) is "over 600bp" (it is
    # sigma sqrt(30): x has no mean reversion under P) and about one path in
    # four is negative; with skew the median is "around zero", about one in two
    np.testing.assert_allclose(stats["atm"][1].std(), 115.72 * np.sqrt(30.0), rtol=0.02)
    assert 0.24 < np.mean(stats["atm"][1] < 0) < 0.30
    assert abs(np.median(stats["skew"][1])) < 40
    assert 0.44 < np.mean(stats["skew"][1] < 0) < 0.52


def test_figure_8_smile_minimum_on_absolute_strikes():
    """A4: on the paper's absolute strike axis the eta = 1.5 smile bottoms out
    where Figure 8's does (about 400bp; the parabola through its three lowest
    points). Comparing offsets from different references (this code's mean,
    364bp; the paper's forward, 387bp) had put it 50bp to the right."""
    from slidingcurve import bachelier as bl, pricing as pr
    curve = markets.paper_curve("baseline")
    off = 1e-4 * np.arange(-60.0, 141.0, 10.0)
    mean, p, _ = pr.yield_option_prices(BASE._replace(eta=1.5), curve, 20.0, off,
                                        lambda t: volscale.ybar_zeroth(curve, 5.0, t),
                                        jax.random.PRNGKey(0), n_paths=2**16)
    v = bl.implied_normal_vol(np.asarray(p), 0.0, off, 20.0)
    lowest = 1e4 * (float(mean) + off[int(np.argmin(v))])
    ref = np.array(markets.paper_figures()["fig8"]["eta=1.5"])
    i = int(np.argmin(ref))
    paper = 387.0 + 50.0 * (i - 5) + 50.0 * (ref[i - 1] - ref[i + 1]) / (2 * (ref[i - 1] + ref[i + 1] - 2 * ref[i]))
    assert abs(lowest - paper) < 20.0
