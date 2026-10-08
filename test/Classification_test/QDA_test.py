import rbg.Classification.QDA as QDA
import numpy as np
import pandas as pd
import pytest

#=========================== fazer ainda ===================
class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.DataFrame({"coluna2": [1, 2, 1]})

        Data = pd.DataFrame({"coluna1": [1.5,2.0,0.5]})


        QDA.fit(X,Y)

        DtFrameResult = QDA.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.ndarray([1.0, 2.0, 3.0])
        Y = np.ndarray([1, 2, 1])

        Data = np.ndarray([1.5,2.0,0.5])

        QDA.fit(X,Y)

        NpResult = QDA.predict(Data)

        assert isinstance(NpResult, np.ndarray)


#========= Fazer ainda ================
class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            QDA.fit(X, Y)

        X_cut = X[:5, :] 
        
        QDA.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = QDA.predict(Data)
        
        assert resultado.shape == (4,)


    def test_params(self):
        pass

    def test_attributes(self):
        
        #IMPORTANTE, decidir se cov_matriz guardará as matrizes como array ou como função que retorna a matriz
        #decidido por agora, será por função, e usaremos a label para iterar se necessário
        QDA.cov_matrix()
        #testa se a matrix de covariança é numpy, testa se tem a dimenção esperada 

        #==================== Verificar a implementacao do LDA============
        QDA.mean_array
        #testa se tem o tamanho correto, testa se é numpy

        
        QDA.prior
        #testa se é numpy array, testa se tem o tamanho correto
        

        X_treino = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array([0, 0, 1, 1])
        
        n_features = X_treino.shape[1]
        n_classes = len(np.unique(Y_treino))
        
        QDA.fit(X_treino, Y_treino)

        cov_matrix = QDA.cov_matrix(k = QDA.labels[0])
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)



        #++++++++++++++++++++++++ decidir se guardamos em uma array ou se guardamos uma função que passamos o identificador da label como parametro
        QDA.k_functions
        #testa se é numpy array, testa se cada campo é uma função


#============= Fazer ainda =================
class TestRightResults():
    pass

