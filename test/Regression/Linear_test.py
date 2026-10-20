import rbg.Regression.Linear as Linear
import numpy as np
import pandas as pd
import pytest

'''
#faça apenas um metodo para testa, mas de pass
Linear.set_params(intecept = False)
Linear.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Linear.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Linear.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Linear.coef


'''


class TestDataEntry():
    def test_DataFrame(self):
        # Utilizando valores contínuos para o Y já que se trata de Regressão
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0], "coluna2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5], "coluna2": [2.5, 3.0, 1.5]})

        Linear.fit(X, Y)

        DtFrameResult = Linear.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        Linear.fit(X, Y)

        NpResult = Linear.predict(Data)

        assert isinstance(NpResult, np.ndarray)

class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            Linear.fit(X, Y)

        X_cut = X[:5, :] 
        
        Linear.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = Linear.predict(Data)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    #=============Terminar-------------------
    def test_params(self):
        Linear.set_params(intecept = False)
        Linear.get_params()


    def test_attributes(self):
        xtrain = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        ytrain = np.array([10.0, 12.0, 20.0, 22.0])
        
        n_features = xtrain.shape[1]
        
        Linear.fit(xtrain, ytrain)

        assert isinstance(Linear.coef, np.ndarray)
        assert Linear.coef.shape == (n_features,)

        
#============= Fazer ainda =================
class TestRightResults():
    pass