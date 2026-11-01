import numpy as np
from .KFold import KFold
from rbg.Utilities import cv_score

class GridSearch:
    def __init__(self, model, param_grid, metrics, cv, greater_is_better = False):
        """Exhaustive search over specified parameter values for an estimator.
        
        Tests all possible mathematical combinations from a hyperparemeter dictionary
        to find the best configuration. After finding the best combination 
        through cross-validation, the winning estimator is refitted using the entire dataset.

        Args:
            model (class or object): The model to be optimized. It must have 'fit' and 'predict' methods.
            param_grid (dict): dictionary with parameter names (str) as keys and lists of parameter settings
            to try as values.
            metrics (callable): a scoring function that takes true targets and predictions and returns a single
            float score.
            cv (class or callable): Cross-Validation method used to split the data. KFold or LOOCV.
            greater_is_better (bool, optional): control parameter for the evaluation metric. Defaults to False.
            For metrics such as Accuracy, Recall, R², precision, F-1 score, it must be True.
            For metrics such as MSE, RMSE, MAE, it must be False.
        """
        self.model = model
        self.param_grid = param_grid 
        self.cv = cv
        self.metrics = metrics
        
        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None
        self.greater_is_better = greater_is_better
        
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
        y = np.asarray(y)
        if self.greater_is_better:
            best_score = float('-inf')
            for params in self._gen_combs:
                current_model = self.model(**params) 
                score = cv_score(current_model, X, y, self.cv, self.metrics)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params
        else:
            best_score = float('inf') # assumindo q queremos minimizar o erro
            for params in self._gen_combs():
                current_model = self.model(**params) 
                score = cv_score(current_model, X, y, self.cv, self.metrics)
                if score < best_score:
                    best_score = score
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