import numpy as np


def train_test_split(X : np.ndarray, y : np.ndarray, test_size : float = 0.2, shuffle = True, random_state = None):
    """
    Divides the dataset between a training set and a testing set.

    Args:
        X (np.ndarray): input values.
        y (np.ndarray): output values.
        test_size (float, optional): proportion of samples assigned to the test set. Defaults to 0.2.
        shuffle (bool, optional): if True, the data will be shuffled. Defaults to True.
        random_state (int, optional): as integer seeds that ensures the random shuffle produces the same split every time. Defaults to None.

    Raises:
        ValueError: if X and y dimensions don't match.
        ValueError: if the test_size is not a float between 0 and 1 neither a valid number of samples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y dimensions must match.")
    if isinstance(test_size, int):
        if test_size < 0 or test_size > len(X):
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the training set.")
    elif isinstance(test_size, float):
        if test_size < 0 or test_size > 1:
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the training set.")
    else:
        raise ValueError("'test_size' must represent the proportion or the exact number of samples on the training set.")

    n_samples = len(X)
    n_test = test_size * n_samples
    
    idxs = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.default_rng(random_state)
        idxs = rng.permutation()
        
    test_idxs = idxs[:n_test]
    train_idxs = idxs[n_test:]
    
    X_train = X[train_idxs]
    X_test = X[test_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]
    
    return X_train, X_test, y_train, y_test

def Resample(X : np.ndarray, size = None, n_resamples : int = 100, random_state = None, replace = True):
    """
    Returns resamples of the original dataset.
    If size = None and replace = True (by default), is a bootstrap.

    Args:
        X (np.ndarray): the original dataset
        size (optional): the size of each resample. Defaults to None.
        n_resamples (int, optional): the number of resamples. Defaults to 100.
        random_state (optional): seed for shuffling. Defaults to None.
        replace (bool, optional): if True, samples can be repeated. Defaults to True.

    Raises:
        ValueError: if 'n_resamples' is not an integer greater than 0.
        ValueError: if 'size' is not an integer greater than 0.
        ValueError: if replace = False and 'size' is greater than the size of X.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("'n_resamples' must be an integer greater than 0.")
    if size == None:
        size = n_samples
    else:
        if type(size) is not int or size<=0:
            raise ValueError("'size' must be an integer greater than 0.")
        if type(size) is int and replace == False and size > n_samples:
            raise ValueError("'size' must be at most the size of the dataset.")
    rng = np.random.default_rng(random_state)
    all_idxs = np.arange(n_samples)
    resamples = []
    for _ in range(n_resamples):
        train_idxs = rng.choice(all_idxs, size=size, replace=replace)
        test_idxs = np.setdiff1d(all_idxs, train_idxs)
        resamples.append((train_idxs)) # acho q o bootstrap devia retornar os testes tbm, se ele for usado como um metodo de cv
    return resamples # isso gasta mais memoria do que o yield, mas os testes pedem pra retornar uma lista ou array

class KFold:
    """K-Fold Cross Validation.
    
    Separates the dataset in k folds and uses k-1 folds as train and 1 fold as test.
    """
    def __init__(self, k : int = 5, shuffle : bool = False, random_state = None):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, data_size):
        if type(data_size) == np.ndarray or type(data_size) == list:
            n_samples = len(data_size)
        elif type(data_size) == int:
            n_samples = data_size
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if type(self.k) is not int or self.k <= 1 or self.k > n_samples:
            raise ValueError("'n_splits' must be an integer larger than 1 and less than the size of the dataset.")
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        idx_sets = np.array_split(idxs, self.k)
        for i in range(self.k):
            test_idxs = idx_sets[i]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            yield test_idxs, train_idxs 

class LOOCV:
    """Leave-One-Out Cross Validation
    
    Take one sample and uses it as the testing set. Is a great option if the dataset is small enough.
    """
    def __init__(self):
        pass
    
    def split(self, data_size):
        if type(data_size) == np.ndarray or type(data_size) == list:
            x = len(data_size)
        elif type(data_size) == int:
            x = data_size
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        return KFold(k=x).split(data_size = x)

def cv_score(model, X : np.ndarray, y : np.ndarray, cv_method, metrics):
    """Returns a list of scores given a machine learning model, a cross-validation method and a metric.
    
    Typical usage example:
        list_of_scores = cv_score(ex_model, X, y, KFold(X, 10), "MSE")
        list_of_scores = cv_score(ex_model2, X, y, LOOCV, "MSE")

    Args:
        model: a machine learning model with methods .fit and .predict
        X (np.ndarray): input of the dataset.
        y (np.ndarray): output of the dataset.
        cv_method: a cross-validation method, KFold or LOOCV.
        metrics: a metric.

    Returns:
        np.ndarray: a list of scores.
    """
    # conferir depois se o metrics(y_pred, y_test) ta certo
    scores = []
    for idxs_train, idxs_test in cv_method.split(X): 
        X_train = X[idxs_train]
        X_test = X[idxs_test]
        y_train = y[idxs_train]
        y_test = y[idxs_test]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = metrics(y_pred, y_test)
        scores.append(score)
    return np.array(scores)
        
        
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
            cur_model = self.model(**params) 
            scores = cv_score(cur_model, X, y, self.cv, self.metrics)
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