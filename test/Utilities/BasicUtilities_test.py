# from rbg.Utilities import Matrix_design, Resample
from rbg.ModelSelection.crossvalidation import Resample

'''
#a partir de uma array constroi uma matriz design
Matrix_design(X, intecept = True)


#Bootsratp
Resample(data,size = len(data),qte = 10 ,replace = True,)
#deve retornar uam array com tamanho qte em que cada indice representa uma nova reamostragem
'''



import pytest
import numpy as np


class TestBootstrap:
    
    @pytest.mark.parametrize("data, size, qte, replace", [
        #parametros pro teste
        (np.array([1, 2, 3, 4, 5]), 5, 10, True),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8]), 3, 5, True),
        (np.array([10, 20, 30, 40]), 4, 2, False),        
        (np.array([7, 8, 9]), 1, 3, True)
    ])
    def test_Resample_parametrizado(self, data, size, qte, replace):
        
        resultado = Resample(X = data, size=size, n_resamples=qte, replace=replace)
        
        assert isinstance(resultado, (np.ndarray, list))
        
        assert len(resultado) == qte
        
        for amostra in resultado:
            assert len(amostra) == size