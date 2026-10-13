import rbg.Classification.QDA as QDA
import numpy as np
import pandas as pd
import pytest

class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.DataFrame({"coluna2": ['contaminated', 'drinkable', 'contaminated']})

        Data = pd.DataFrame({"coluna1": [1.5,2.0,0.5]})


        QDA.fit(X,Y)

        DtFrameResult = QDA.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.ndarray([1.0, 2.0, 3.0])
        Y = pd.DataFrame({"coluna2": ['contaminated', 'drinkable', 'contaminated']})

        Data = np.ndarray([1.5,2.0,0.5])

        QDA.fit(X,Y)

        NpResult = QDA.predict(Data)

        assert isinstance(NpResult, np.ndarray)


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

        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_train = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable'])
        QDA.fit(X_train, Y_train)

        
        assert isinstance(QDA.labels, np.ndarray)
        assert np.array_equal(np.unique(QDA.labels), np.unique(Y_train))
    

        
    def test_method(self):
        X_train = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y_train = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable'])
        
        QDA.fit(X_train, Y_train)

        label = QDA.labels[0]

        n_features = X_train.size[0]

        cov_matrix = QDA.cov_matrix(k=label)
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)

        mean_array = QDA.mean_array(k=label)
        assert isinstance(mean_array, np.ndarray)
        assert mean_array.shape == (n_features,)

        prior = QDA.prior(k=label)
        assert isinstance(prior, (float, np.number))

        k_function = QDA.k_functions(k=label)
        assert callable(k_function)

        amostra_teste = X_train[0]
        resultado_func = k_function(amostra_teste)
        
        assert isinstance(resultado_func, (float, np.number))

#============= Fazer ainda =================
class TestRightResults():
    pass

