from decisiontree import ClassificationTree as Classification
from decisiontree import RegressionTree as Regression
import numpy as np
import pandas as pd
import pytest

'''
# para impurity: Gini, classification (por padrão), entropy
features = 100
#testa se consegue receber esses parametros
Classification.set_params(impurity = 'Gini', min_inpurity_decrease = 0.1 , max_depth = 30)
#verifica se retorna um dicionario com as coisas corretas
Classification.get_params()

#verifica se da certo e faz os testes de tamanho de array etc
Classification.fit(X,Y)
#verifica se retorna o tamanho correto etc
Classification.predict(Data)


#regressão, igualmente
#testes iguais
features = 100
Regression.set_params(min_inpurity_decrease = 0.1 , max_depth = 30)
Regression.get_params()

Regression.fit(X,Y)
Regression.predict(Data)
'''

class TestClassificationTree:
    
    def test_params(self):
        modelo_classificacao = Classification()
        modelo_classificacao.set_params(criterion_split='Gini', min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_classificacao.get_params()
        
        assert isinstance(params, dict)
        assert params.get('criterion_split') == 'Gini'
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_fit_and_predict(self):
        X = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y = np.array(['classeA', 'classeA', 'classeB', 'classeB'])
        
        X_errado = np.zeros((10, 2))
        Y_errado = np.zeros(5)
        
        modelo_classificacao = Classification()
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_classificacao.fit(X_errado, Y_errado)

        modelo_classificacao.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0]])
        resultado = modelo_classificacao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (2,)
     def test_iniciate_class(self):

        with pytest.raises(ValueError, match="The only split criterions avaiable for classification are Gini,classif_error,entropy, and Other"):
                  modelo_classificacao = Classification( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "MAE",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Same", custom_evaluation = None )

        modelo_classificacao = Classification(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "entropy", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Same", custom_evaluation = None )
        params = modelo_classificacao.get_params()
        assert params.get('criterion_split') == 'entropy'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == False
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "Standard"
        #assert params.get('custom_choose') == mode
        assert params.get('criterion_evaluation') == "Same"

    def test_fit_class(self):
        modelo_classificacao = Classification()
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

        with pytest.raises(ValueError, match="X must be two-dimensional"):
            modelo_classificacao.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
             modelo_classificacao.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            modelo_classificacao.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            modelo_classificacao.fit(X, Y)


        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_classificacao.fit(X, Y)
    def test_tree_args(self):
        modelo_classificacao = Classification()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        modelo_classificacao.fit(X, Y)

        dic = modelo_classificacao.get_tree_structure_info()
        assert isinstance(dic, dict)
        assert dic.get('n') == 7
        assert dic.get('children_left')[0] == 2
        assert dic.get('children_right')[0] == 1
        assert dic.get('children_left')[1] == -1
        assert dic.get('children_right')[2] == -1
        assert int(dic.get('impurity')[2]) == 0

    def test_predict(self):
        modelo_classificacao = Classification()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        modelo_classificacao.fit(X, Y)

        x_test = np.array([[3]])
        a = modelo_classificacao.predict(x_test)
        assert a == np.array([0])
        x_test = np.array([[3],[10]])
        a = modelo_classificacao.predict(x_test)
        assert np.array_equal(a, np.array([0, 1]))


class TestRegressionTree:
    
    def test_params(self):
        modelo_regressao = Regression(criterion_split = "MAE", criterion_choose = "median")
        modelo_regressao.set_params(min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_regressao.get_params()
        
        assert isinstance(params, dict)
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_fit_and_predict(self):
        X = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y = np.array([10.5, 12.0, 20.5, 22.0])
        
        X_errado = np.zeros((10, 2))
        Y_errado = np.zeros(5)
        
        modelo_regressao = Regression()
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_regressao.fit(X_errado, Y_errado)

        modelo_regressao.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0], [5.0, 5.0]])
        resultado = modelo_regressao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (3,)
        Data = np.array([[1.5, 2.0], [8.5, 9.0], [5.0, 5.0]])
        resultado = modelo_regressao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (3,)
