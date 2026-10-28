from rbg.Classification.Logistic import Logistic as LogReg
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        Data = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        
        modelo_logreg = LogReg()
        DtFrameResult = modelo_logreg.fit(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        Data = np.array([1.0, 2.0, 3.0])

        modelo_logreg = LogReg()
        NpResult = modelo_logreg.fit(Data)

        assert isinstance(NpResult, np.ndarray)


class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_logreg = LogReg()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_logreg.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_logreg.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_logreg.predict(Data)
        
        assert resultado.shape == (4,)

    def test_params(self):
        pass

    def test_attributes(self):
        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable'])
        
        modelo_logreg = LogReg()
        modelo_logreg.fit(X_train, Y_treino)
        
        assert isinstance(modelo_logreg.labels, np.ndarray)
        assert np.array_equal(np.unique(modelo_logreg.labels), np.unique(Y_treino))
        assert np.array_equal(modelo_logreg.labels, np.array(['contaminated', 'drinkable']))

    def test_method(self):
        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_train = np.array([0, 0, 1, 1])
        
        n_features = X_train.shape[1]

        modelo_logreg = LogReg()
        modelo_logreg.fit(X_train, Y_train)

        label = modelo_logreg.labels[0]

        func_k = modelo_logreg.k_function(k = label)
        assert callable(func_k)
        
        amostra_teste = X_train[0]
        resultado_func = func_k(amostra_teste)
        assert isinstance(resultado_func, (float, np.number))

        coefs = modelo_logreg.k_coef(k = label)
        assert isinstance(coefs, np.ndarray)

class TestRightResults():
    pass