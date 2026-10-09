import rbg.Classification.LogisticRegression as LogReg
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        Data = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        
        DtFrameResult =LogReg.fit(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        Data = np.ndarray([1.0,2.0,3.0])

        NpResult =LogReg.fit(Data)

        assert isinstance(NpResult, np.ndarray)


#========= Fazer ainda ================
class TestError():
    #não faça nada aqui
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            LogReg.fit(X, Y)

        X_cut = X[:5, :] 
        
        LogReg.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = LogReg.predict(Data)
        
        assert resultado.shape == (4,)

    #nem aqui
    def test_params(self):
        pass

    #nem aqui
    def test_attributes(self):
        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_treino = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable'])
        
        LogReg.fit(X_train, Y_treino)

        
        assert isinstance(LogReg.labels, np.ndarray)
        assert np.array_equal(np.unique(LogReg.labels), np.unique(Y_treino))
        assert np.array_equal(LogReg.labels,np.array(['contaminated', 'drinkable']))

    def test_method(self):

        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_train = np.array([0, 0, 1, 1])
        
        n_features = X_train.shape[1]

        # O treinamento deve ocorrer ANTES de extrair a label
        LogReg.fit(X_train, Y_train)

        # Selecionamos a primeira label (k) gerada após o fit, seguindo o padrão construído no LDA
        label = LogReg.labels[0]

        #retorna uma função testa igual o lda
        func_k = LogReg.k_function(k = label)
        assert callable(func_k)
        
        # Injeta uma amostra para garantir que a função gerada executa sem quebrar
        amostra_teste = X_train[0]
        resultado_func = func_k(amostra_teste)
        assert isinstance(resultado_func, (float, np.number))

        #aqui deve retornar uma array com coeficientes
        coefs = LogReg.k_coef(k = label)
        assert isinstance(coefs, np.ndarray)

#faz nada
class TestRightResults():
    pass

