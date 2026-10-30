from rbg.Classification.Logistic import Logistic as LogReg
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        Data = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0, 4.0]})
        labels = pd.Series([0, 1, 0, 1])

        modelo_logreg = LogReg(labels=[0, 1])
        DtFrameResult = modelo_logreg.fit(Data, labels)

        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        Data = np.array([[1.0], [2.0], [3.0], [4.0]])
        labels = np.array([0, 1, 0, 1])

        modelo_logreg = LogReg(labels=[0, 1])
        NpResult = modelo_logreg.fit(Data, labels)

        assert isinstance(NpResult, np.ndarray)


class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3))
        Y = np.zeros(5)

        modelo_logreg = LogReg(labels=[0, 1])

        with pytest.raises(ValueError, match="same number of observations"):
            modelo_logreg.fit(X, Y)

        X_valid = np.array([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
            [1.0, 1.0], [1.0, 1.0],
        ])
        y_valid = np.array([0, 1] * 3)
        modelo_logreg.fit(X_valid, y_valid)

        Data = np.zeros((4, 2))

        resultado = modelo_logreg.predict(Data)

        assert resultado.shape == (4,)

    def test_params(self):
        pass

    def test_attributes(self):
        X_train = np.array([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
            [1.0, 1.0], [1.0, 1.0],
        ])
        Y_treino = np.array(['contaminated', 'drinkable'] * 3)

        modelo_logreg = LogReg(labels=['contaminated', 'drinkable'])
        modelo_logreg.fit(X_train, Y_treino)

        assert isinstance(modelo_logreg.labels, np.ndarray)
        assert np.array_equal(np.unique(modelo_logreg.labels), np.unique(Y_treino))
        assert np.array_equal(modelo_logreg.labels, np.array(['contaminated', 'drinkable']))

    def test_method(self):
        X_train = np.array([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
            [1.0, 1.0], [1.0, 1.0],
        ])
        Y_train = np.array([0, 1] * 3)

        n_features = X_train.shape[1]

        modelo_logreg = LogReg(labels=[0, 1])
        modelo_logreg.fit(X_train, Y_train)

        label = modelo_logreg.labels[0]

        func_k = modelo_logreg.k_function(k = label)
        assert callable(func_k)

        amostra_teste = X_train[0]
        resultado_func = func_k(amostra_teste)
        assert isinstance(resultado_func, (float, np.number))

        coefs = modelo_logreg.k_coef(k = label)
        assert isinstance(coefs, np.ndarray)
        assert coefs.shape == (n_features + 1,)
        assert 0.0 <= resultado_func <= 1.0
        assert np.isclose(
            sum(modelo_logreg.k_function(k)(amostra_teste) for k in modelo_logreg.labels),
            1.0,
        )
        assert np.array_equal(modelo_logreg.k_coef(modelo_logreg._reference_class), np.zeros(n_features + 1))

class TestRightResults():
    pass
