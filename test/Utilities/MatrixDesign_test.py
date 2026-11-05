import numpy as np
import pytest

from rbg.Utilities import MatrixDesign


def test_first_column_is_ones():
    result = MatrixDesign([[2.0, 5.0], [3.0, 6.0], [4.0, 7.0]])
    assert np.allclose(result[:, 0], 1.0)


def test_shape_gets_one_extra_column():
    X = np.zeros((4, 3))
    assert MatrixDesign(X).shape == (4, 4)


def test_original_values_are_kept():
    X = np.array([[2.0, 5.0], [3.0, 6.0]])
    result = MatrixDesign(X)
    assert np.allclose(result[:, 1:], X)


def test_1d_input_becomes_one_feature():
    result = MatrixDesign([2.0, 3.0, 4.0])
    expected = [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    assert np.allclose(result, expected)


def test_3d_input_raises():
    with pytest.raises(ValueError):
        MatrixDesign(np.zeros((2, 2, 2)))