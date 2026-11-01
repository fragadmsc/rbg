import numpy as np
from .KFold import KFold
from rbg.Utilities import cv_score

class GridSearch:
    def __init__(self, model, param_grid, metrics, cv = KFold):
        self.model = model
        self.param_grid = param_grid 
        self.cv = cv
        self.metrics = metrics
        
        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None
        
    def _gen_combs(self):
        combinations = [{}]
        for key, values in self.param_grid.items():
            new_combs = []
            for comb in combinations:
                for val in values:
                    new_comb = comb.copy()
                    new_comb[key]=val
                    new_combs.append(new_comb)
            combinations = new_combs
        return combinations

    def fit(self, X, y):
        X = np.asarray(X)
        y =np.asarray(y)
        best_score = float('inf') # assumindo q queremos minimizar o erro
        for params in self._gen_combs():
            current_model = self.model(**params) 
            scores = cv_score(current_model, X, y, self.cv, self.metrics)
            total_score = np.mean(scores)
            if total_score < best_score:
                best_score = total_score
                self.best_params_ = params
        self.best_model_ = self.model(**self.best_params_)
        self.best_model_.fit(X, y)
        self.best_score_ = best_score
        
    def predict(self, X):
        return self.best_model_.predict(X)
    
class RandomizedSearch:
    def __init__(self, model, param_dist, metrics, n_iter : int = 10, cv = None, random_state = None): 
        self.model = model
        self.param_dist = param_dist
        self.metrics = metrics
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        
        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None
               
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        best_score = float('inf')
        for _ in range(self.n_iter):
            params ={}
            for key, value in self.param_dist.items():
                if callable(value):
                    params[key] = value(rng)
                elif isinstance(value, list):
                    params[key] = rng.choice(value)
                else:
                    raise TypeError(f"The parameter '{key}' must be a list or a function (callable).")
            model = self.model(**params)
            scores = cv_score(model, X, y, self.cv, self.metrics)
            total = np.mean(scores)
            if total < best_score:
                best_score = total
                self.best_params_ = params
                self.best_model_ = model(**params)        
        self.best_model_.fit(X, y)
        self.best_score_ = best_score
        
    def predict(self, X):
        return self.best_model_.predict(X)