"""A9: what the paper's numbers pin down about its undefined "yield option".
See docs/investigation.md, A9 and A10."""
import numpy as np
import jax
import pytest

from slidingcurve import Model, bachelier as bl, gaussian as g, markets, pricing as pr, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)


def vols(model, curve, expiry, offsets, ybar_fn, ref, n_paths=2**15, seed=0):
    mean, p, _ = pr.yield_option_prices(model, curve, expiry, offsets, ybar_fn, jax.random.PRNGKey(seed),
                                        n_paths, 24, ref)
    k = pr.yield_option_moneyness(curve, model, expiry, offsets, mean, ref)
    return 1e4 * bl.implied_normal_vol(np.asarray(p), 0.0, np.asarray(k), expiry)


@pytest.mark.parametrize("ref", ["mean", "forward"])
def test_gaussian_smile_is_flat_at_the_closed_form_for_either_strike_reference(ref):
    curve = markets.paper_curve("baseline")
    off = 1e-4 * np.array([-200.0, -50.0, 0.0, 50.0, 200.0])
    v = vols(BASE, curve, 20.0, off, lambda t: volscale.ybar_zeroth(curve, 5.0, t), ref, 2**16)
    exact = 1e4 * g.A(0.03, 5.0) * 0.01 * np.sqrt(float(g.B(0.06, 20.0)) / 20.0)     # 70.86bp
    np.testing.assert_allclose(v, exact, atol=0.4)
    assert np.ptp(v) < 0.2                                     # flat: same paths at every strike


def test_only_the_continuous_yield_gives_the_papers_sigma():
    """Section 9.4 / Figure 13: 82bp at-the-money gives sigma = 1.1572%. In the
    Gaussian model that pins the underlying: simple rates and par swap rates
    on the same 20y x 5y window need a different sigma."""
    curve = markets.paper_curve("market")
    T, d = 20.0, 5.0
    yield_sigma = 0.0082 / (g.A(0.03, d) * np.sqrt(float(g.B(0.06, T)) / T))
    assert abs(yield_sigma - 0.011572) < 5e-6

    def sigma_for_82bp(swp):
        F, ann = float(pr.forward_swap_rate(curve, swp)), float(pr.annuity(curve, swp))
        p = g.payer_swaption(curve, 0.03, 0.01, T, np.asarray(swp.pay_times), np.asarray(swp.accruals), F)
        return 0.01 * 0.0082 / float(bl.implied_normal_vol(p / ann, F, F, T))

    for swp in (pr.Swaption.caplet(T, d), pr.Swaption.regular(T, d, 1), pr.Swaption.regular(T, d, 2),
                pr.Swaption.regular(T, d, 4)):
        assert abs(sigma_for_82bp(swp) - 0.011572) > 1e-4


def test_papers_skew_parameters_reproduce_its_figure_11_fit_with_mean_referenced_strikes():
    """At the paper's stated sigma = 1.1999%, b = 0.367 the smile matches the
    fitted vols plotted in Figure 11 (to Monte-Carlo error, ~0.6bp) only when
    strikes are offsets from the model mean; offsets from the curve forward
    put the whole smile about 2.4bp high. sigma = 1.12% (Section 9.4 text)
    misses by 3-5bp either way."""
    curve = markets.paper_curve("market")
    offsets, _, paper_fit = markets.paper_smile()
    m = BASE._replace(sigma=0.011999, b=0.367)
    fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
    paper = 1e4 * paper_fit
    mean_ref = vols(m, curve, 20.0, offsets, fn, "mean", 2**16)
    fwd_ref = vols(m, curve, 20.0, offsets, fn, "forward", 2**16)
    assert np.max(np.abs(mean_ref - paper)) < 1.2
    assert np.mean(fwd_ref - paper) > 1.5
    text = vols(m._replace(sigma=0.0112), curve, 20.0, offsets, volscale.bootstrap_fn(
        curve, m._replace(sigma=0.0112), np.arange(1.0, 21.0)), "mean", 2**15)
    assert np.mean(text - paper) < -3.0
