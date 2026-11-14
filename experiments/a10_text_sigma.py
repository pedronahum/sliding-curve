"""A10: Figure 12's gamma(30y) at the two printed skew-fit sigmas, b = 0.367."""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

curve = markets.paper_curve("market")
t = time_grid(30.0)
out = {}
for sig in (0.011999, 0.0112):
    m = Model(0.03, 5.0, sig, b=0.367)
    d = decompose(simulate(m, curve, t, volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t),
                           jax.random.PRNGKey(1), n_paths=2**16), m, curve, jnp.array([30.0]))
    out[sig] = (BP * float(d.gamma[0]), BP * float(d.gamma_se[0]), BP * float(d.theta[0]), BP * float(d.y_eh[0]))
    print(f"sigma {100*sig:.4f}%: gamma(30y) {out[sig][0]:.1f} +- {out[sig][1]:.1f}, theta {out[sig][2]:.1f}, "
          f"Y_eh {out[sig][3]:.1f}  (paper Figure 12: gamma 140.3)")
save("a10_text_sigma", out)
