from  decisiontree import RandomForest as RdForest
from decisiontree import RandomForestRegression as RdForestReg
import numpy as np
import pandas as pd
import pytest 

'''
# para impurity: Gini, classification (por padrão), entropy
# m representa o tamanho das divisões de features =============== talvez de pra pegar automaticamente, ou os dois 
features = 100
RdForest.set_params(impurity = 'Gini', m = features/3,n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForest.get_params()

RdForest.fit(X,Y)
RdForest.predict(Data)


#regressão, igualmente
features = 100
RdForestReg.set_params(m = features/3, n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForestReg.get_params()

RdForestReg.fit(X,Y)
RdForestReg.predict(Data)
'''
def divi_3(m):
    return m/3

class TestRdForest_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series(['classA', 'classA', 'classB']) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_rdforest = RdForest()
        modelo_rdforest.fit(X, Y)
        DtFrameResult = modelo_rdforest.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array(['classA', 'classA', 'classB'])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_rdforest = RdForest()
        modelo_rdforest.fit(X, Y)
        NpResult = modelo_rdforest.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestRdForest_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_rdforest = RdForest()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_rdforest.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_rdforest.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_rdforest.predict(Data)
        
        # Testa se retorna uma array numpy de tamanho compativel
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        features = 100
        modelo_rdforest = RdForest()
        modelo_rdforest.set_params(criterion_split='Gini', factor_selection = divi_3, n_trees=100, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_rdforest.get_params()
        
        assert isinstance(params, dict)
        assert params.get('criterion_split') == 'Gini'
        assert params.get('factor_selection') == divi_3
        assert params.get('n_trees') == 100
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30


#teste regressão
class TestRdForestReg_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.fit(X, Y)
        DtFrameResult = modelo_rdforest_reg.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.fit(X, Y)
        NpResult = modelo_rdforest_reg.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestRdForestReg_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_rdforest_reg = RdForestReg()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_rdforest_reg.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_rdforest_reg.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_rdforest_reg.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        features = 100
        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.set_params( n_trees=100, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_rdforest_reg.get_params()

        
        
        assert isinstance(params, dict)
        assert params.get('factor_selection') == np.sqrt
        assert params.get('n_trees') == 100
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30
