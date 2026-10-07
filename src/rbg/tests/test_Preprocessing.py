import pytest
import rbg.Preprocessing as Prep
import numpy as np
import pandas as pd

#trocar as entradas Data por valores que faça sentido

#escolher o epsylon de maquina
MEpsylon = 10e-10

class TestDataEntry():
    @pytest.mark.parametrize("func", [Prep.Standard])
    def test_DataFrame(self,func):
        Data = pd.data_frame
        
        DtFrameResult = func(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    @pytest.mark.parametrize("func", [Prep.Standard])
    def test_Numpy(self,func):
        Data = np.ndarray()

        NpResult = func(Data)

        assert isinstance(NpResult, np.ndarray)

class TestFunctionality():

    def test_Standard(self):
        
        Data = np.ndarray()
        ExpectedResult = np.ndarray()

        #testa elemento por elemento da array
        np.testing.assert_allclose(
            Prep.Standard(Data), 
            ExpectedResult, 
            atol=MEpsylon,
            err_msg="O calculo divergiu da matriz esperada acima do limite de MEpsylon."
        )

