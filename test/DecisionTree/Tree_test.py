import rbg.DecisionTree.ClassificationTree as Classification
import rbg.DecisionTree.RegressionTree as Regression
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
        Classification.set_params(impurity='Gini', min_inpurity_decrease=0.1, max_depth=30)
        
        params = Classification.get_params()
        
        assert isinstance(params, dict)
        assert params.get('impurity') == 'Gini'
        assert params.get('min_inpurity_decrease') == 0.1
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
        
        with pytest.raises(ValueError, match="The data do not match"):
            Classification.fit(X_errado, Y_errado)

        Classification.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0]])
        resultado = Classification.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (2,)


class TestRegressionTree:
    
    def test_params(self):
        Regression.set_params(min_inpurity_decrease=0.1, max_depth=30)
        
        params = Regression.get_params()
        
        assert isinstance(params, dict)
        assert params.get('min_inpurity_decrease') == 0.1
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
        
        with pytest.raises(ValueError, match="The data do not match"):
            Regression.fit(X_errado, Y_errado)

        Regression.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0], [5.0, 5.0]])
        resultado = Regression.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (3,)