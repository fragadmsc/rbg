import rbg.Classification.LDA as LDA
import numpy as np
import pandas as pd
import pytest

#=========================== fazer ainda ===================
#=========================== FAZ SENTIDO UTILIZAR OS VALORES DE CLASSIFICAÇÃO COMO INT OU STRING ? OU DEIXAR COMO APENAS INT?
class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.DataFrame({"coluna2": [1, 2, 1]})

        Data = pd.DataFrame({"coluna1": [1.5,2.0,0.5]})


        LDA.fit(X,Y)

        DtFrameResult = LDA.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.ndarray([1.0, 2.0, 3.0])
        Y = np.ndarray([1, 2, 1])

        Data = np.ndarray([1.5,2.0,0.5])

        LDA.fit(X,Y)

        NpResult = LDA.predict(Data)

        assert isinstance(NpResult, np.ndarray)


#========= Fazer ainda ================
class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            LDA.fit(X, Y)

        X_cut = X[:5, :] 
        
        LDA.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = LDA.predict(Data)
        
        assert resultado.shape == (4,)


    def test_params(self):
        pass

    def test_attributes(self):
        #======================= PADRONIZAR A INTERFAÇE POR K =============


        X_treino = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array([0, 0, 1, 1])
        
        LDA.fit(X_treino, Y_treino)

        
        assert isinstance(LDA.labels, np.ndarray)
        assert np.array_equal(np.unique(LDA.labels), np.unique(Y_treino))
        
    def test_method(self):

        X_treino = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array([0, 0, 1, 1])
        
        n_features = X_treino.shape[1]
        n_classes = len(np.unique(Y_treino))
        
        LDA.fit(X_treino, Y_treino)

        cov_matrix = LDA.cov_matrix(k = LDA.labels[0])
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)

        mean_array = LDA.mean_array(k = LDA.label[0])
        assert isinstance(mean_array, np.ndarray)
        assert mean_array.shape == (n_features,)

        prior = LDA.prior(k = LDA.label[0])
        assert isinstance(prior, np.number)

        #++++++++++++++++++++++++ decidir se guardamos em uma array ou se guardamos uma função que passamos o identificador da label como parametro
        LDA.k_functions
        #testa se é numpy array, testa se cada campo é uma função



#============= Fazer ainda =================
class TestRightResults():
    pass

