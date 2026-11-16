"""slidingcurve — term premia and bond convexity under sliding-curve dynamics.

One-factor and multi-factor quasi-Gaussian (Cheyette) models with local and
stochastic volatility, simulated in the physical measure defined by the
sliding-curve condition of Piterbarg & Van Der Helm (2026). Equation numbers
in docstrings refer to that paper.
"""
import jax

# Yield decompositions are differences of numbers that agree to ~1e-4, and the
# closed-form checks are at the 1e-6 level: work in float64 throughout.
jax.config.update("jax_enable_x64", True)

from .curve import Curve                                    # noqa: E402
from .model import (Model, Profiles, VarianceModel, front_end_path, simulate,  # noqa: E402
                    time_grid)
from .decomposition import Decomposition, decompose         # noqa: E402
from .pricing import GaussianTwin, Swaption, TerminalState  # noqa: E402
from . import (bachelier, bootstrap, calibration, forward, gaussian, holee,  # noqa: E402
               markets, multifactor, pricing, qmc, volscale)

__all__ = ["Curve", "Model", "Profiles", "VarianceModel", "simulate", "time_grid",
           "front_end_path", "Decomposition", "decompose",
           "Swaption", "TerminalState", "GaussianTwin",
           "bachelier", "bootstrap", "calibration", "forward", "gaussian",
           "holee", "markets", "multifactor", "pricing", "qmc", "volscale"]
