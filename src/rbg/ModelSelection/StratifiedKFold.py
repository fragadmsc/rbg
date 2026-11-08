import numpy as np

class StratifiedKFold:
    """
    Stratified K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets 
    while preserving the percentage of samples for each class.
    """
<<<<<<< HEAD
    def __init__(self, k: int = 5, shuffle: bool = False, random_state = 42):
=======
    def __init__(self, k: int = 5, shuffle: bool = False, random_state = None):
>>>>>>> f7f3b07 (fixes bugs and divides functions in different files)
        """Initializes a Stratified KFold instance.

        Args:
            k (int, optional): number of folds. Defaults to 5.
<<<<<<< HEAD
            shuffle (bool, optional): if true, the data will be shuffled. Defaults to False.
            random_state (int, optional): seed for shuffling. Defaults to 42.
        """
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 1:
=======
            shuffle (bool, optional): _description_. Defaults to False.
            random_state (_type_, optional): _description_. Defaults to None.
        """
        if type(k) is not int or k <= 1:
>>>>>>> f7f3b07 (fixes bugs and divides functions in different files)
            raise ValueError("The number of splits 'k' must be an integer greater than 1.")
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray = None):
        """Generates indices to split data into training and test sets.
        
        Args:
            X (np.ndarray): input data (features), used to determine the number of samples.
            y (np.ndarray): target labels, used to preserve class distribution.
        
        Returns:
        An iterator containing tuples of (train_indices, test_indices).
        """
        if y is None:
            raise ValueError("StratifiedKFold needs the vector of labels 'y' to keep proportion between classes.")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
<<<<<<< HEAD
            raise ValueError("X and y dimensions must match.")
        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError("Number of splits 'k' cannot be greater than the number of samples.")
        unique_classes, y_counts = np.unique(y, return_counts=True)
        min_samples = np.min(y_counts)
        if min_samples < self.k:
            raise ValueError("k can't be greater than the number of members in a class.")
        rng = np.random.default_rng(self.random_state)
        all_indices = np.arange(n_samples)
=======
            raise ValueError("X and y must have the same number of rows!")
        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError(f"Number of splits ({self.k}) cannot be greater than the number of samples ({n_samples}).")
        rng = np.random.default_rng(self.random_state)
        all_indices = np.arange(n_samples)
        unique_classes = np.unique(y)
>>>>>>> f7f3b07 (fixes bugs and divides functions in different files)
        class_folds = []
        for cls in unique_classes:
            cls_indices = all_indices[y == cls]
            if self.shuffle:
                cls_indices = rng.permutation(cls_indices)
            folds = np.array_split(cls_indices, self.k)
            class_folds.append(folds)
            
        for i in range(self.k):
            test_idxs_list = []
            for folds in class_folds:
                test_idxs_list.append(folds[i])
            test_idxs = np.concatenate(test_idxs_list)
            if self.shuffle:
                test_idxs = rng.permutation(test_idxs)
            train_idxs = np.setdiff1d(all_indices, test_idxs)
            if self.shuffle:
                train_idxs = rng.permutation(train_idxs)
            yield train_idxs, test_idxs
