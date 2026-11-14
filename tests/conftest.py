import jax
import pytest

from slidingcurve import Curve, Model


@pytest.fixture(scope="session")
def ns_curve():
    """Upward-sloping, humped Nelson-Siegel curve (illustrative, not the paper's)."""
    return Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)


@pytest.fixture(scope="session")
def base_model():
    """Section 9.1 baseline: delta = 5y, kappa = 0.03, Gaussian."""
    return Model(kappa=0.03, delta=5.0, sigma=0.01)


@pytest.fixture(scope="session")
def key():
    return jax.random.PRNGKey(20261106)
