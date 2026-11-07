import numpy as np
from copy import deepcopy
from rbg.Utilities import cv_score

class RandomizedSearch:
    def __init__(self, model, param_dist, metrics, cv, n_iter: int = 10, random_state=None, greater_is_better=False):
        self.model = model
        self.param_dist = param_dist
        self.metrics = metrics
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.greater_is_better = greater_is_better

        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None

    def fit(self, X, y = None, groups = None):
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        if self.greater_is_better:
            best_score = float('-inf')
            for _ in range(self.n_iter):
                params = {}
                for key, value in self.param_dist.items():
                    if callable(value):
                        params[key] = value(rng)
                    elif isinstance(value, list):
                        params[key] = rng.choice(value)
                    else:
                        raise TypeError(
                            f"The parameter '{key}' must be a list or a function (callable)."
                        )
                current_model = deepcopy(self.model)
                current_model.set_params(**params)
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        else:
            best_score = float('inf')
            for _ in range(self.n_iter):
                params = {}
                for key, value in self.param_dist.items():
                    if callable(value):
                        params[key] = value(rng)
                    elif isinstance(value, list):
                        params[key] = rng.choice(value)
                    else:
                        raise TypeError(
                            f"The parameter '{key}' must be a list or a function (callable)."
                        )
                current_model = deepcopy(self.model)
                current_model.set_params(**params)
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score < best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        self.best_model_ = deepcopy(self.model)
        self.best_model_.set_params(**self.best_params_)
        if y is not None:
            self.best_model_.fit(X, y)
        else:
            self.best_model_.fit(X)
        self.best_score_ = best_score

        return self

    def predict(self, X):
        return self.best_model_.predict(X)