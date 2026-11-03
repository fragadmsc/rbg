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
            
class GroupKFold:
    """
    Group K-Fold cross-validation.

    Ensures that samples belonging to the same group
    are never separated between training and test sets.
    """

    def __init__(self, k: int = 5, shuffle: bool = False,random_state=None):

        if type(k) is not int or k <= 1:
            raise ValueError("'k' must be an integer larger than 1.")

        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, groups):
        """
        Generates train/test indices while keeping
        all samples from the same group together.

        Args:
            X (np.ndarray): Dataset used to determine the number of samples.
            groups (np.ndarray): Group identifier for each sample.

        Yields:
            Tuples of (train_indices, test_indices).
        """
        X = np.asarray(X)
        groups = np.asarray(groups)

        if X.shape[0] != groups.shape[0]:
            raise ValueError("X and groups must have the same number of samples.")
        if groups.ndim != 1:
            raise ValueError("'groups' must be a one-dimensional array.")

        unique_groups,group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.k>n_groups:
            raise ValueError(f"Number of splits ({self.k}) cannot be greater than the number of groups ({n_groups}).")

        rng = np.random.default_rng(self.random_state)
        group_sizes = np.bincount(group_indices) # qtde de amostras q pertencem a cada grupo
        group_order = np.arange(n_groups) # ordenação dos grupos do maior para o menor

        if self.shuffle:
            group_order = rng.permutation(group_order)

        group_order = group_order[np.argsort(-group_sizes[group_order], kind='stable') ]# stable faz com q grupos de mesmo tamanho mantem a ordem estabelecida pelo shuffle
        fold_groups = [[] for _ in range(self.k)]
        fold_sizes = np.zeros(self.k, dtype=int)

        # estrategia gulosa: botar o maior grupo atual no menor fold atual
        for group_idx in group_order:
            smallest_fold = np.argmin(fold_sizes)
            fold_groups[smallest_fold].append(group_idx)
            fold_sizes[smallest_fold] += group_sizes[group_idx]

        all_indices = np.arange(X.shape[0])
        for i in range(self.k):
            test_group_ids = fold_groups[i]
            test_mask = np.isin(group_indices, test_group_ids)

            test_idxs = all_indices[test_mask]
            train_idxs = all_indices[~test_mask]

            if self.shuffle:
                test_idxs = rng.permutation(test_idxs)
                train_idxs = rng.permutation(train_idxs)

            yield train_idxs, test_idxs
