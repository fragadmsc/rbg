from rbg.Classification.KNN  import KNN
import numpy as np
import pandas as pd
import pytest

class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.Series([1, 2, 1], name="coluna2")

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5]})

        modelo_knn = KNN(classes=[1, 2])
        modelo_knn.fit(X, Y)

        DtFrameResult = modelo_knn.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([[1.0], [2.], [3.0]])
        Y = np.array([1, 2, 1])

        Data = np.array([[1.5], [2.0], [0.5]])

        modelo_knn = KNN(classes=[1, 2])
        modelo_knn.fit(X, Y)

        NpResult = modelo_knn.predict(Data)

        assert isinstance(NpResult, np.ndarray)


class TestError():

    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_knn = KNN(classes=[0])

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_knn.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_knn.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_knn.predict(Data)
        
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_knn = KNN(classes=[0])
        modelo_knn.set_params(k = 5)

        di = modelo_knn.get_params()

        assert type(di) == dict
        assert di['k'] == 5 


class TestRightResults():
    pass