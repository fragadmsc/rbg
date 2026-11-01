import numpy as np

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
            raise ValueError("'k' must be an integer larger than 1 and less than the size of the dataset.")
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        idx_sets = np.array_split(idxs, self.k)
        for i in range(self.k):
            test_idxs = idx_sets[i]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            yield train_idxs, test_idxs
