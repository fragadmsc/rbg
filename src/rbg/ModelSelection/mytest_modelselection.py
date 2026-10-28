import numpy as np
import pytest
from rbg.ModelSelection.crossvalidation import GridSearch, KFold, LOOCV

class ToyModel:
    def __init__(self, param1=1,param2=10):
        self.param1 = 1
        self.param2 = 10
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.ones(X.shape[0])*self.param1
    
def SimpleMSE(y_real, y_pred):
    return float(np.mean(y_real - y_pred)**2)

def test_gera_combs():
    param_grid = {'param1':[1, 2, 3], 'param2':[10, 20]}
    grid = GridSearch(model=ToyModel, param_grid=param_grid, cv=KFold(data_size=10, k=2), metrics = SimpleMSE)
    combs = grid._gen_combs()
    assert len(combs) == 6
    assert {'param1': 1, 'param2': 10} in combs
    assert {'param1': 3, 'param2': 20} in combs
    