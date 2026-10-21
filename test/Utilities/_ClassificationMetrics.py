'''#é a quantidade total de acertos dividida pela tentativa, nao a diferença entre as classes em si
Acurracy(Y_predicted,Y_real)

#de tudo o que o modelo disse ser da classe A, o quanto realmente era.
Pressision(Y_predicted,Y_real)

#a proporcao entre o que o modelo disse que era classe A sobre tudo que era classe A.
Reccal(Y_predicted,Y_real)

#media harmonica entre recall e pressision
F1Score(Y_predicted,Y_real)

'''

import pytest
import numpy as np
from rbg.Utilities import Acurracy, Pressision,Reccal, F1Score

class TestMetrics:
    
    def test_Acurracy(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        resultado = Acurracy(Y_predicted, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert 0.0 <= resultado <= 1.0

    def test_Pressision(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        n_classes = len(np.unique(Y_real))
        resultado = Pressision(Y_predicted, Y_real)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (n_classes,)
        assert np.all((resultado >= 0.0) & (resultado <= 1.0))

    def test_Reccal(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        n_classes = len(np.unique(Y_real))
        
        resultado = Reccal(Y_predicted, Y_real)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (n_classes,)
        assert np.all((resultado >= 0.0) & (resultado <= 1.0))

    def test_F1Score(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        n_classes = len(np.unique(Y_real))
        
        resultado = F1Score(Y_predicted, Y_real)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (n_classes,)
        assert np.all((resultado >= 0.0) & (resultado <= 1.0))


    def test_Pressision_resiliencia_divisao_zero(self):
        #verifica se para a classe 1 por exemplo a funcao nao quebra, divisao por zero
        Y_real = np.array([0, 0, 1, 1])
        Y_predicted = np.array([0, 0, 0, 0]) 
        
        try:
            resultado = Pressision(Y_predicted, Y_real)
            assert isinstance(resultado, np.ndarray)
        except ZeroDivisionError as e:
            pytest.fail(f"A função de Precisão não tratou a divisão por zero. Erro: {e}")
        except Exception as e:
            pytest.fail(f"Erro genérico inesperado na Precisão. Erro: {e}")

    def test_Reccal_resiliencia_divisao_zero(self):
        #verifica se está tentando calcular um ponto que a classe não existe no mundo real e se a divisão por zero não falha
        Y_real = np.array([0, 0, 0, 0])
        Y_predicted = np.array([0, 1, 0, 0])
        
        try:
            resultado = Reccal(Y_predicted, Y_real)
            assert isinstance(resultado, np.ndarray)
        except ZeroDivisionError as e:
            pytest.fail(f"A função de Recall não tratou a divisão por zero. Erro: {e}")
