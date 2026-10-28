import numpy as np
import pytest

from rbg.ModelSelection.crossvalidation import Resample


@pytest.fixture
def CoordinateDescent():
    """Load the solver expected by the regression branch's utility tests."""
    from rbg.Utilities import CoordinateDescent as solver

    return solver


def test_l1_shrinkage(CoordinateDescent):
    """Check a regular solution."""
    X = np.eye(3)
    y = np.array([3.0, 0.5, -2.0])

    beta = CoordinateDescent.solve(X, y, 1.0, False)

    assert isinstance(beta, np.ndarray)
    assert beta.shape == (3,)
    np.testing.assert_allclose(beta, [2.0, 0.0, -1.0], atol=1e-5)


def test_unpenalized_intercept(CoordinateDescent):
    """Check that the intercept is not penalized."""
    X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 4.0, 7.0])

    beta = CoordinateDescent.solve(X, y, 2.0, True)

    assert isinstance(beta, np.ndarray)
    assert beta.shape == (2,)
    np.testing.assert_allclose(beta, [4.0, 2.0], atol=1e-5)


class TestBootstrap:
    @pytest.mark.parametrize("data, size, qte, replace", [
        (np.array([1, 2, 3, 4, 5]), 5, 10, True),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8]), 3, 5, True),
        (np.array([10, 20, 30, 40]), 4, 2, False),
        (np.array([7, 8, 9]), 1, 3, True),
    ])
    def test_Resample_parametrizado(self, data, size, qte, replace):
        resultado = Resample(X=data, size=size, n_resamples=qte, replace=replace)

        assert isinstance(resultado, (np.ndarray, list))
        assert len(resultado) == qte
        for amostra in resultado:
            assert len(amostra) == size
