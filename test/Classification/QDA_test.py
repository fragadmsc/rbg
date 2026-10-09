import rbg.Classification as Cla
import numpy as np
import pandas as pd

#=========================== fazer ainda ===================
class TestDataEntry():
    def test_DataFrame(self):
        Data = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        
        DtFrameResult =(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        Data = np.ndarray([1.0,2.0,3.0])

        NpResult =(Data)

        assert isinstance(NpResult, np.ndarray)


#========= Fazer ainda ================
class TestError():
    pass

#============= Fazer ainda =================
class TestRightResults():
    pass

