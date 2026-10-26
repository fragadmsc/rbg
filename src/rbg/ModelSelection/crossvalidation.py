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