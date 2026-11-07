import numpy as np

class KFold:
    """K-Fold Cross Validation.
    
    Separates the dataset in k folds and uses k-1 folds as train and 1 fold as test.
    """
    def __init__(self, k : int = 5, shuffle : bool = False, random_state = 42):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 1:
            raise ValueError("The number of splits 'k' must be an integer greater than 1.")
        self.k = int(k)
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, data_size, y = None):
        if type(data_size) == np.ndarray or type(data_size) == list:
            n_samples = len(data_size)
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if type(self.k) is not int or self.k <= 1 or self.k > n_samples:
            raise ValueError("'k' must be an integer larger than 1 and at most the size of the dataset.")
        if y is not None:
            y = np.asarray(y)
            if n_samples != y.shape[0]:
                raise ValueError("X and y must have the same number of rows!")
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        idx_sets = np.array_split(idxs, self.k)
        for i in range(self.k):
            test_idxs = idx_sets[i]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            yield train_idxs, test_idxs
