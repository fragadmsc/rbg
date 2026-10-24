import rbg.DecisionTree.Bagging as Bagging
import rbg.DecisionTree.BaggingRegression as BaggingReg
import numpy as np
import pandas as pd
import pytest

'''
# para impurity: Gini, classification (por padrão), entropy
features = 100
Bagging.set_params(impurity = 'Gini', n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
Bagging.get_params()

Bagging.fit(X,Y)
Bagging.predict(Data)


#regressão, igualmente
features = 100
BaggingReg.set_params(n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
BaggingReg.get_params()

BaggingReg.fit(X,Y)
BaggingReg.predict(Data)
'''



class TestBagging_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series(['classA', 'classA', 'classB']) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        Bagging.fit(X, Y)
        DtFrameResult = Bagging.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array(['classA', 'classA', 'classB'])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        Bagging.fit(X, Y)
        NpResult = Bagging.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestBagging_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            Bagging.fit(X, Y)

        X_cut = X[:5, :] 
        Bagging.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = Bagging.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        Bagging.set_params(impurity='Gini', n_trees=100, min_inpurity_decrease=0.1, max_depth=30)
        
        params = Bagging.get_params()
        
        assert isinstance(params, dict)
        assert params.get('impurity') == 'Gini'
        assert params.get('n_trees') == 100
        assert params.get('min_inpurity_decrease') == 0.1
        assert params.get('max_depth') == 30

#teste regressão
class TestBaggingReg_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        BaggingReg.fit(X, Y)
        DtFrameResult = BaggingReg.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        BaggingReg.fit(X, Y)
        NpResult = BaggingReg.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestBaggingReg_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        with pytest.raises(ValueError, match="The data do not match"):
            BaggingReg.fit(X, Y)

        X_cut = X[:5, :] 
        BaggingReg.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = BaggingReg.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        BaggingReg.set_params(n_trees=100, min_inpurity_decrease=0.1, max_depth=30)
        
        params = BaggingReg.get_params()
        
        assert isinstance(params, dict)
        assert params.get('n_trees') == 100
        assert params.get('min_inpurity_decrease') == 0.1
        assert params.get('max_depth') == 30