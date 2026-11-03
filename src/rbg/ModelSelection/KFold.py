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
            raise ValueError("'k' must be an integer larger than 1 and at most the size of the dataset.")
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        idx_sets = np.array_split(idxs, self.k)
        for i in range(self.k):
            test_idxs = idx_sets[i]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            yield train_idxs, test_idxs

class StratifiedKFold:
    """
    Stratified K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets 
    while preserving the percentage of samples for each class.
    """
    def __init__(self, k: int = 5, shuffle: bool = False, random_state = None):
        """Initializes a Stratified KFold instance.

        Args:
            n_splits (int, optional): number of folds. Defaults to 5.
            shuffle (bool, optional): _description_. Defaults to False.
            random_state (_type_, optional): _description_. Defaults to None.
        """
        if type(k) is not int or k <= 1:
            raise ValueError("The number of splits 'k' must be an integer greater than 1.")
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray):
        """Generates indices to split data into training and test sets.
        
        Args:
            X (np.ndarray): input data (features), used to determine the number of samples.
            y (np.ndarray): target labels, used to preserve class distribution.
        
        Returns:
        An iterator containing tuples of (train_indices, test_indices).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows!")
        n_samples = X.shape[0]
        if self.n_splits > n_samples:
            raise ValueError(f"Number of splits ({self.n_splits}) cannot be greater than the number of samples ({n_samples}).")
        rng = np.random.default_rng(self.random_state)
        all_indices = np.arange(n_samples)
        unique_classes = np.unique(y)
        class_folds = []
        for cls in unique_classes:
            cls_indices = all_indices[y == cls] # pega todos os indices dessa classe
            # separamos os indices nos k folds, como se as amostras nessa classe correspondessem a um unico dataset:
            if self.shuffle:
                cls_indices = rng.permutation(cls_indices)
            folds = np.array_split(cls_indices, self.n_splits)
            class_folds.append(folds)
            
        for i in range(self.n_splits):
            test_idxs_list = []
            for folds in class_folds:
                test_idxs_list.append(folds[i])
            # concatena todos os "subfolds" de classes diferentes mas mesmo indice:
            test_idxs = np.concatenate(test_idxs_list)
            # o shuffle garante que as classes estão misturadas entre elas:
            if self.shuffle:
                test_idxs = rng.permutation(test_idxs)
            # cada fold é um teste, o treino é o resto:
            train_idxs = np.setdiff1d(all_indices, test_idxs)
            if self.shuffle:
                train_idxs = rng.permutation(train_idxs)
            yield train_idxs, test_idxs
            
# implementar group kfold!!!!