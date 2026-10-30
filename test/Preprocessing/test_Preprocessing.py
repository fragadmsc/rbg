import numpy as np
import pandas as pd
import pytest

from rbg.Preprocessing import Standard


@pytest.mark.parametrize("data", [pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([1.0, 2.0, 3.0])])
def test_accepts_dataframe_and_numpy(data):
    assert isinstance(Standard(data), np.ndarray)


def test_standardizes_each_column():
    X = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    result = Standard(X)
    np.testing.assert_allclose(result[:, 0], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    np.testing.assert_array_equal(result[:, 1], 0.0)
    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)
