from rbg.Classification.LDA import LDA
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.DataFrame({"coluna2": ['contaminated', 'drinkable', 'contaminated']})

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5]})

        modelo_lda = LDA()
        modelo_lda.fit(X, Y)

        DtFrameResult = modelo_lda.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([1.0, 2.0, 3.0])
        Y = pd.DataFrame({"coluna2": ['contaminated', 'drinkable', 'contaminated']})

        Data = np.array([1.5, 2.0, 0.5])

        modelo_lda = LDA()
        modelo_lda.fit(X, Y)

        NpResult = modelo_lda.predict(Data)

        assert isinstance(NpResult, np.ndarray)


class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_lda = LDA()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_lda.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_lda.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_lda.predict(Data)
        
        assert resultado.shape == (4,)


    def test_params(self):
        pass

    def test_attributes(self):
        X_treino = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable'])
        
        modelo_lda = LDA()
        modelo_lda.fit(X_treino, Y_treino)
        
        assert isinstance(modelo_lda.labels, np.ndarray)
        assert np.array_equal(np.unique(modelo_lda.labels), np.unique(Y_treino))
        assert np.array_equal(modelo_lda.labels, np.array(['contaminated', 'drinkable']))


    def test_method(self):

        X_treino = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array([0, 0, 1, 1])
        
        n_features = X_treino.shape[1]
        
        modelo_lda = LDA()
        modelo_lda.fit(X_treino, Y_treino)

        cov_matrix = modelo_lda.cov_matrix(k = modelo_lda.labels[0])
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)

        mean_array = modelo_lda.mean_array(k = modelo_lda.labels[0])
        assert isinstance(mean_array, np.ndarray)
        assert mean_array.shape == (n_features,)

        prior = modelo_lda.prior(k = modelo_lda.labels[0])
        assert isinstance(prior, np.number)

        k_function = modelo_lda.k_functions(k = modelo_lda.labels[0])

        assert callable(k_function)

        amostra_teste = X_treino[0] 
        resultado_func = k_function(amostra_teste)
        
        assert isinstance(resultado_func, (float, np.number))


#============= Fazer ainda =================
class TestRightResults():
    pass