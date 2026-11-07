'''#é a quantidade total de acertos dividida pela tentativa, nao a diferença entre as classes em si
Accuracy(Y_predicted,Y_real)

#de tudo o que o modelo disse ser da classe A, o quanto realmente era.
Precision(Y_predicted,Y_real)

#a proporcao entre o que o modelo disse que era classe A sobre tudo que era classe A.
Recall(Y_predicted,Y_real)

#media harmonica entre recall e Precision
F1Score(Y_predicted,Y_real)

'''

import pytest
import numpy as np
from rbg.Utilities import Accuracy, Precision, Recall, F1Score
# nomes errados: from rbg.Utilities import Accuracy, Precision, Recall, F1Score


class TestMetrics:

    def test_Accuracy(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        resultado = Accuracy(Y_predicted, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert 0.0 <= resultado <= 1.0

    def test_Precision(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        n_classes = len(np.unique(Y_real))
        resultado = Precision(Y_predicted, Y_real)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (n_classes,)
        assert np.all((resultado >= 0.0) & (resultado <= 1.0))

    def test_Recall(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        n_classes = len(np.unique(Y_real))
        
        resultado = Recall(Y_predicted, Y_real)
        
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

    def test_Precision_resilience_divisao_zero(self):
        #verifica se para a classe 1 por exemplo a funcao nao quebra, divisao por zero
        Y_real = np.array([0, 0, 1, 1])
        Y_predicted = np.array([0, 0, 0, 0]) 

        try:
            resultado = Precision(Y_predicted, Y_real)
            assert isinstance(resultado, np.ndarray)
        except ZeroDivisionError as e:
            pytest.fail(f"A função de Precisão não tratou a divisão por zero. Erro: {e}")
        except Exception as e:
            pytest.fail(f"Erro genérico inesperado na Precisão. Erro: {e}")

    def test_Recall_resilience_divisao_zero(self):
        #verifica se está tentando calcular um ponto que a classe não existe no mundo real e se a divisão por zero não falha
        Y_real = np.array([0, 0, 0, 0])
        Y_predicted = np.array([0, 1, 0, 0])

        try:
            resultado = Recall(Y_predicted, Y_real)
            assert isinstance(resultado, np.ndarray)
        except ZeroDivisionError as e:
            pytest.fail(f"A função de Recall não tratou a divisão por zero. Erro: {e}")

    def test_error_rate(self):
        Y_real = np.array([0, 0, 1, 1, 2, 2])
        Y_predicted = np.array([0, 1, 1, 1, 0, 2])
        
        resultado = error_rate(Y_predicted, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert 0.0 <= resultado <= 1.0
        assert np.isclose(resultado, 1 - Accuracy(Y_predicted, Y_real))
