import numpy as np
from rbg.Utilities import MAE,RMSE,MSE,RSquare


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

def resample(X : np.ndarray, size = None, n_resamples : int = 100, random_state = None, replace = True):
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
    
    for _ in range(n_resamples):
        train_idxs = rng.choice(all_idxs, size=size, replace=replace)
        test_idxs = np.setdiff1d(all_idxs, train_idxs)
        yield train_idxs, test_idxs
        
class KFold:
    # nos testes, ao inicializar o obj, devemos passar o dataset,
    # ver depois qual jeito eh melhor
    
    """
    K-Fold Cross Validation.
    
    Splits the dataset in 'n_splits' consecutive parts (folds).
    Each fold is used once as a testing set, while the others
    are the training set.
    """
    def __init__(self, n_splits : int = 5, shuffle : bool = False, random_state = None):
        """Initializes a KFold object.

        Args:
            n_splits (int, optional): the number of splits. Defaults to 5.
            shuffle (bool, optional): if True, the dataset is shuffled. Defaults to False.
            random_state (optional): seed for shuffle. Defaults to None.

        Raises:
            ValueError: if 'n_splits' is an invalid entrance.
        """
        if type(n_splits) is not int or n_splits <= 1:
            raise ValueError("'n_splits' must be an integer greater than 1.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(n_samples)
        idx_sets = np.array_split(idxs, self.n_splits)
        
        for i in range(self.n_splits):
            test_idxs = idx_sets[i]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            yield train_idxs, test_idxs
            
class LOOCV:
    # eh melhor criar uma classe q herda as props de kfold?
    
    """
    Leave-One-Out Cross Validation.
    
    Uses one sample as the testing set.
    Is a particular case of K-Fold CV, with k = X.shape[0].
    """
    def __init__():
        """
        Initializes a LOOCV object.
        """
        pass
    def split(self, X):
        kfold = KFold(X.shape[0])
        yield from kfold.split(X)
        
def cv_score(model, X : np.ndarray, y : np.ndarray, cv_method, metrics):
    """Returns a list of scores given a machine learning model, a cross-validation method and a loss function.

    Args:
        model: a machine learning model with method .fit and .predict
        X (np.ndarray): input of the dataset.
        y (np.ndarray): output of the dataset.
        cv_method: a cross-validation method, KFold or LOOCV.
        metrics: a metrics such as MSE, imported from rbg.Utilities

    Returns:
        np.ndarray: a list of scores.
    """
    # conferir depois se o metrics(y_pred, y_test) ta certo (talvez o certo seja inverter)
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
        
        
# implementar busca de hiperparametros
