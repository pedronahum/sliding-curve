"""C4: curve bootstrap from deposit and swap quotes."""
import numpy as np
import pytest

from slidingcurve import Curve, bootstrap as bs


@pytest.mark.parametrize("freq", [1, 2])
def test_bootstrap_reprices_inputs_and_recovers_a_known_curve(freq):
    truth = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
    dep_t = [0.25, 0.5]
    sw_t = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50]
    dep_r = [float(bs.deposit_rate(truth, T)) for T in dep_t]
    sw_r = [float(bs.par_swap_rate(truth, T, freq)) for T in sw_t]
    c = bs.bootstrap(sw_t, sw_r, freq, dep_t, dep_r)
    np.testing.assert_allclose([float(bs.par_swap_rate(c, T, freq)) for T in sw_t], sw_r, atol=1e-12)
    np.testing.assert_allclose([float(bs.deposit_rate(c, T)) for T in dep_t], dep_r, atol=1e-12)
    grid = np.linspace(0.5, 50, 200)
    assert np.max(np.abs(np.asarray(c.zero(grid)) - np.asarray(truth.zero(grid)))) < 2e-5   # 0.2bp
    assert np.max(np.abs(np.asarray(c.forward(grid[grid > 1])) - np.asarray(truth.forward(grid[grid > 1])))) < 2e-4


def test_bootstrap_rejects_duplicate_maturities():
    with pytest.raises(ValueError):
        bs.bootstrap([1, 2], [0.03, 0.031], 1, [1.0], [0.03])
