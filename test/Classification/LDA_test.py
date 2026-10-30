import numpy as np
import pandas as pd
import pytest

from rbg.Classification.LDA import LDA


@pytest.fixture
def samples():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [3, 3], [3, 4], [4, 3], [4, 4]], dtype=float)
    y = np.array(["contaminated"] * 4 + ["drinkable"] * 4)
    return X, y


def test_DataFrame(samples):
    X, y = samples
    model = LDA(labels=np.unique(y))
    model.fit(pd.DataFrame(X), pd.Series(y))
    assert isinstance(model.predict(pd.DataFrame(X[:3])), np.ndarray)


def test_Numpy(samples):
    X, y = samples
    model = LDA(labels=np.unique(y))
    model.fit(X, y)
    assert isinstance(model.predict(X[:3]), np.ndarray)


def test_MatchingDim(samples):
    X, y = samples
    model = LDA(labels=np.unique(y))
    with pytest.raises(ValueError, match="The data do not match"):
        model.fit(X, y[:-1])
    model.fit(X, y)
    assert model.predict(X[:4]).shape == (4,)


def test_attributes(samples):
    X, y = samples
    model = LDA(labels=np.unique(y))
    model.fit(X, y)
    np.testing.assert_array_equal(model.labels, np.unique(y))


def test_method(samples):
    X, y = samples
    model = LDA(labels=np.unique(y))
    model.fit(X, y)
    label = model.labels[0]
    assert model.cov_matrix().shape == (2, 2)
    assert model.mean_array(label).shape == (2,)
    assert isinstance(model.prior(label), np.number)
    assert isinstance(model.k_functions(label)(X[0]), (float, np.number))
