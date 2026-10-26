import rbg.Classification.KNN as KNN
import numpy as np
import pandas as pd
import pytest

class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.DataFrame({"coluna2": [1, 2, 1]})

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5]})

        modelo_knn = KNN.KNN()
        modelo_knn.fit(X, Y)

        DtFrameResult = modelo_knn.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([1.0, 2.0, 3.0])
        Y = np.array([1, 2, 1])

        Data = np.array([1.5, 2.0, 0.5])

        modelo_knn = KNN.KNN()
        modelo_knn.fit(X, Y)

        NpResult = modelo_knn.predict(Data)

        assert isinstance(NpResult, np.ndarray)


class TestError():

    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_knn = KNN.KNN()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_knn.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_knn.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_knn.predict(Data)
        
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_knn = KNN.KNN()
        modelo_knn.set_params(k = 5)

        di = modelo_knn.get_params()

        assert type(di) == dict
        assert di['k'] == 5 


class TestRightResults():

    def test_resuts(self):
        X = np.array([[1.0, 1.5], [2.0, 2.5], [10.0, 10.5], [11.0, 11.5]])
        Y = np.array([0, 0, 1, 1])

        modelo_knn = KNN.KNN()
        modelo_knn.fit(X, Y)

        TrainPredict = modelo_knn.predict(X)
        
        np.testing.assert_allclose(
            TrainPredict, 
            Y, 
            atol=1e-9, 
            err_msg="The Predict do not match"
        )

        XTest = np.array([[1.2, 1.8], [10.5, 11.0]])
        YPred = np.array([0, 1])

        predict = modelo_knn.predict(XTest)

        np.testing.assert_allclose(
            predict, 
            YPred, 
            atol=1e-9, 
            err_msg="The predict do not match"
        )