from rbg.Utilities import *
import rbg.Utilities as utilities
import numpy as np
import pytest

MSE(Y_predict,Y_train)

Matrix_design(X, intecept = True)


def test_l1_shrinkage(CoordinateDescent):
    """Check a regular solution."""
    X = np.eye(3)
    y = np.array([3.0, 0.5, -2.0])

    beta = CoordinateDescent.solve(X, y, 1.0, False)

    assert isinstance(beta, np.ndarray)
    assert beta.shape == (3,)
    np.testing.assert_allclose(beta, [2.0, 0.0, -1.0], atol=1e-5)


def test_unpenalized_intercept(CoordinateDescent):
    """Intercept test"""
    X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 4.0, 7.0])

    beta = CoordinateDescent.solve(X, y, 2.0, True)

    assert isinstance(beta, np.ndarray)
    assert beta.shape == (2,)
    np.testing.assert_allclose(beta, [4.0, 2.0], atol=1e-5)


