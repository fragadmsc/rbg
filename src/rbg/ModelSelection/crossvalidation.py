import numpy as np

def train_test_split(X : np.ndarray, y : np.ndarray, test_size : float = 0.2, shuffle = True, random_state = None):
    """Divides the dataset between a training set and a testing set.

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