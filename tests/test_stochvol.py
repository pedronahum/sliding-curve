"""Section 8 / 9.3: what stochastic volatility does to convexity."""
import numpy as np
import jax
import jax.numpy as jnp

from slidingcurve import (bachelier as bl, calibration, decompose, markets, pricing as pr,
                          simulate, time_grid, volscale)


def test_convexity_unchanged_by_eta_at_fixed_sigma(ns_curve, base_model, key):
    """Figure 7: with b = 0 and E^P[z] = 1, gamma does not depend on eta."""
    t = time_grid(30.0)
    ybar = volscale.ybar_zeroth(ns_curve, base_model.delta, t)
    mats = jnp.array([10.0, 20.0, 30.0])
    g = {}
    for eta in (0.0, 1.0):
        m = base_model._replace(eta=eta)
        g[eta] = decompose(simulate(m, ns_curve, t, ybar, key, n_paths=2**15), m, ns_curve, mats)
    assert np.all(np.abs(g[1.0].gamma - g[0.0].gamma) < 5 * g[1.0].gamma_se + 1e-7)


def test_recalibration_uplift_matches_section_8_2(ns_curve, base_model):
    """Section 8.2 quotes an uplift in convexity of 'about 16%' for eta = 0.6,
    kappa_z = 0.1, kappa = 0.03 over a ten-year horizon, at fixed at-the-money
    volatility. With b = 0 the uplift is (sigma_eta / sigma_0)^2."""
    key = jax.random.PRNGKey(0)
    ybar = lambda t: volscale.ybar_zeroth(ns_curve, base_model.delta, t)
    swp = pr.Swaption.caplet(10.0, base_model.delta)
    fwd, ann = float(pr.forward_swap_rate(ns_curve, swp)), float(pr.annuity(ns_curve, swp))
    k = np.array([fwd])
    p0 = np.asarray(pr.swaption_prices(base_model, ns_curve, swp, k, ybar, key, n_paths=2**15)[0])
    q = markets.Quotes(swp, fwd, ann, k, p0, bl.implied_normal_vol(p0 / ann, fwd, k, swp.expiry))
    fit = calibration.calibrate(base_model._replace(eta=0.6), ns_curve, q, free=("sigma",),
                                ybar_fn=ybar, key=key, n_paths=2**15)
    uplift = (fit.model.sigma / base_model.sigma) ** 2
    assert 1.12 < uplift < 1.19
