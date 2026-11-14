"""C7: the second smile market is the published data, and consistent."""
import numpy as np

from slidingcurve import bachelier as bl, markets


def test_values_are_those_of_the_source():
    expiry, fwd, offsets, prices, vols = markets.choi_2017_10y10y()
    assert expiry == 10.0 and fwd == 0.030673
    np.testing.assert_array_equal(offsets, [-0.01, 0.0, 0.01])
    np.testing.assert_array_equal(prices, [1.506e-2, 9.083e-3, 5.108e-3])
    assert "1809.04035" in markets.CHOI_2017_SOURCE


def test_prices_are_arbitrage_free_and_invert():
    expiry, fwd, offsets, prices, vols = markets.choi_2017_10y10y()
    k = fwd + offsets
    assert np.all(prices > np.maximum(fwd - k, 0.0))                    # above intrinsic
    assert prices[0] - prices[1] < 0.01 and prices[1] - prices[2] < 0.01  # slope above -1
    assert prices[0] - 2 * prices[1] + prices[2] > 0                      # convex in strike
    np.testing.assert_allclose(bl.normal_price(fwd, k, vols, expiry), prices, rtol=1e-10)
    np.testing.assert_allclose(1e4 * vols, [73.00, 72.00, 73.42], atol=0.01)
