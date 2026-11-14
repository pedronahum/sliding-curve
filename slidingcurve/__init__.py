"""slidingcurve — term premia and bond convexity under sliding-curve dynamics.

One-factor quasi-Gaussian (Cheyette) model with local and stochastic
volatility, simulated in the physical measure defined by the sliding-curve
condition of Piterbarg & Van Der Helm (2026). Equation numbers in docstrings
refer to that paper.
"""
import jax

# Yield decompositions are differences of numbers that agree to ~1e-4, and the
# closed-form checks are at the 1e-6 level: work in float64 throughout.
jax.config.update("jax_enable_x64", True)

from .curve import Curve                                    # noqa: E402
from .model import Model, Profiles, front_end_path, simulate, time_grid     # noqa: E402
from .decomposition import Decomposition, decompose         # noqa: E402
from . import gaussian, volscale, bachelier, pricing, markets, calibration  # noqa: E402

__all__ = ["Curve", "Model", "Profiles", "simulate", "time_grid",
           "Decomposition", "decompose", "gaussian", "volscale", "bachelier",
           "pricing", "markets", "calibration"]
