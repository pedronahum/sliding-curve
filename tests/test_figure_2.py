"""B3: the Appendix D closed forms on the paper's two synthetic curves against
all four plotted series of Figures 2 and 5 (b = 0, eta = 0, sigma = 1%)."""
import numpy as np
import pytest

from slidingcurve import gaussian as g, markets


@pytest.mark.parametrize("which,fig", [("baseline", "fig2"), ("steep", "fig5")])
def test_closed_forms_reproduce_figures_2_and_5(which, fig):
    curve = markets.paper_curve(which)
    ref = markets.paper_figures()[fig]
    t = np.asarray(ref["T"])
    for T in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0):
        i = int(np.argmin(np.abs(t - T)))
        mine = 1e4 * np.array([float(g.yield_eh(curve, 0.03, 5.0, 0.01, T)),
                               float(g.term_premium(curve, 0.03, 5.0, 0.01, T)),
                               float(g.convexity(0.03, 0.01, T))])
        paper = np.array([ref["Y_eh"][i], ref["theta"][i], ref["gamma"][i]], dtype=float)
        # digitisation is good to a few tenths of a basis point
        np.testing.assert_allclose(mine, paper, atol=0.5)
