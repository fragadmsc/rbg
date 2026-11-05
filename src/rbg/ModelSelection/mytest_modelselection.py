import numpy as np
import pytest
from rbg.ModelSelection.KFold import KFold, StratifiedKFold, GroupKFold
from rbg.ModelSelection.LOOCV import LOOCV
from rbg.ModelSelection.HyperparameterSelection import GridSearch, RandomizedSearch
from rbg.Utilities.crossvalidation import train_test_split, train_val_test_split, cv_score
from rbg.Utilities.Resample import Resample

# imports adicionais:
# from rbg.Utilities

class ToyModel:
    def __init__(self, param1=1,param2=10):
        self.param1 = param1
        self.param2 = param2
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.ones(X.shape[0])*self.param1
    def __call__(self, **kwargs):
        return ToyModel(**kwargs)
    
def SimpleMSE(y_real, y_pred):
    return float(np.mean(y_real - y_pred)**2)



# ====TESTES GRID SEARCH====

# testa parametro greater is better funciona

def test_gera_combs():
    param_grid = {'param1':[1, 2, 3], 'param2':[10, 20]}
    grid = GridSearch(model=ToyModel, param_grid=param_grid, cv=KFold(k=2), metrics = SimpleMSE)
    combs = grid._gen_combs()
    assert len(combs) == 6
    assert {'param1': 1, 'param2': 10} in combs
    assert {'param1': 3, 'param2': 20} in combs
    
def test_grid_fit_best_hyperp():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    param_grid = {
        'param1': [1,2,3,4],
        'param2': [10]
    }
    kfold = KFold(k = 2)
    grid = GridSearch(ToyModel, param_grid, cv = kfold, metrics=SimpleMSE)
    grid.fit(X, y)
    assert grid.best_params_ == {'param1':3, 'param2':10}
    assert grid.best_score_ == 0.0 # n devia ser menor q um epsilon?
    assert grid.best_model_ != None
    predicoes = grid.predict(X)
    np.testing.assert_array_equal(predicoes, np.array([3.0, 3.0, 3.0, 3.0]))

# ====TESTES RANDOMIZED SEARCH===

# testa parametro greater is better funciona

def test_randomized_search():
    X = np.array([[1.0],[2.0],[3.0],[4.0]])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    param_dist = {'param1': [1, 2, 3, 4], 'param2': lambda rng: rng.uniform(0.0, 1.0) }
    kfold = KFold(k=2)
    seletor = RandomizedSearch(model=ToyModel,param_dist=param_dist,metrics=SimpleMSE,n_iter=5,cv=kfold,random_state=42)
    seletor.fit(X, y)
    
    assert seletor.best_params_ is not None
    assert 0.0 <= seletor.best_params_['param2'] <= 1.0  
    assert seletor.best_model_ is not None
    
def test_randomized_reprod():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    param_dist = {'param1': [1, 2, 3, 4],'param2': lambda rng: rng.uniform(0.0, 1.0)}
    kfold = KFold(k=2)
    rs1 = RandomizedSearch(
        model=ToyModel, 
        param_dist=param_dist, 
        metrics=SimpleMSE, 
        n_iter=3, 
        cv=kfold, 
        random_state=42
    )
    rs2 = RandomizedSearch(
        model=ToyModel, 
        param_dist=param_dist, 
        metrics=SimpleMSE, 
        n_iter=3, 
        cv=kfold, 
        random_state=42
    )
    rs1.fit(X, y)
    rs2.fit(X, y)
    
    assert rs1.best_params_ == rs2.best_params_
    assert rs1.best_score_ == rs2.best_score_
    
# ==== TESTES TRAIN TEST SPLIT====

# ========== TESTES TRAIN VAL TEST SPLIT ===

# ==== TESTES CV SCORE


# ===== TESTES RESAMPLE====