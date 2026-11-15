"""D2: Figure 4's excess skew is a factor of T / delta, and this code's is not.

Pins both sides of the fingerprint (docs/investigation.md, D): the slope of
the b = 1 smile between -80 and +200bp does not grow with expiry in this
code (20y over 5y below 1.5, against the paper's 3.36), the 5y panel matches
the paper outright, and at 20y the paper's slope is over three times this
code's while its at-the-money vol is the higher one. A pricer change that
compounds the skew per benchmark period -- the only form found that reaches
the paper's slope -- lowers this code's at-the-money vol by tens of bp and
fails here; so does adopting a future release of the authors' Figure 4 code
without revisiting D.
"""
import numpy as np
import jax

from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01, b=1.0)


def smile(expiry, n_paths=2**15, seed=0):
    """(our slope, paper slope, our at-the-money vol, paper at-the-money vol),
    in bp, on the baseline curve with the bootstrap scaling."""
    curve = markets.paper_curve("baseline")
    fig4 = markets.paper_figures()["fig4"][str(int(expiry))]
    offsets = 1e-4 * np.asarray(fig4["offsets"])
    _, p, _ = pr.yield_option_prices(BASE, curve, expiry, offsets,
                                     volscale.bootstrap_fn(curve, BASE, volscale.bootstrap_grid(20.0)),
                                     jax.random.PRNGKey(seed), n_paths)
    v = 1e4 * bl.implied_normal_vol(np.asarray(p), 0.0, offsets, expiry)
    paper = np.asarray(fig4["b=1.00"], dtype=float)
    return v[10] - v[3], paper[10] - paper[3], v[5], paper[5]


def test_this_codes_skew_does_not_compound_with_expiry():
    s5, p5, a5, pa5 = smile(5.0)
    s20, p20, a20, pa20 = smile(20.0)
    assert abs(s5 - p5) < 3.0                    # the 5y panel matches outright
    assert abs(a5 - pa5) < 3.0
    assert s20 / s5 < 1.5                        # this code 0.8; the paper 3.36
    assert p20 / s20 > 3.0                       # the paper's slope: T / delta times ours
    assert pa20 - a20 > 8.0                      # and its level is the higher one


def test_the_fingerprint_is_stable_over_seeds():
    """The 20y slope is a property of the pricer, not of a draw: four seeds
    agree to well inside the gap to the paper's 109.3bp."""
    slopes = [smile(20.0, seed=k)[0] for k in range(4)]
    assert max(slopes) - min(slopes) < 3.0
    assert max(slopes) < 0.5 * 109.3
