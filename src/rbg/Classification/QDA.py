import numpy as np
from rbg.Classification.classifier import Classifier

class QDA(Classifier):
    def __init__(self, classes, reference_class=None):
        Classifier.__init__(self, classes, reference_class)

    def fit(self, X, y):
        self._fit_done = False
        
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if np.issubdtype(X.dtype, np.number) == False:
            raise ValueError("X must be a numerical array.")
        else:
            if np.isfinite(X).all() == False:
                raise ValueError("X must have only finite values.")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if not np.isin(y, self._classes).all():
            raise ValueError("y contains unknown class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match.")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")

        self.labels = np.unique(y)
        if self.labels.size < 2:
            raise ValueError("y must be at least two classes.")

        self._X_train = X
        self._y_train = y

        self._covariance = {}
        for k in self.labels:
            if np.sum(self._y_train == k) < 2:
                raise ValueError("Each class must have at least two samples.")
            self._covariance[k] = self.cov_matrix(k)
            if np.linalg.matrix_rank(self._covariance[k]) < self._covariance[k].shape[0]:
                raise ValueError("The covariance matrix is singular.")
        
        self._fit_done = True
    
    def mean_array(self, k):
        x_k = self._X_train[self._y_train == k]
        mean_k = np.mean(x_k, axis = 0)
        return mean_k
    
    def cov_matrix(self, k):
        x_k = self._X_train[self._y_train == k]
        detours = x_k - self.mean_array(k)
        matriz = detours.T @ detours / (x_k.shape[0] - 1)
        return matriz
    
    def prior(self, k):
        sum_k = (self._y_train == k).sum()
        prior_k = sum_k/self._y_train.size
        return prior_k
    
    def k_functions(self, k):
        def function_discriminant(x):
            covariance = self._covariance[k]
            mean = self.mean_array(k)
            difference = x - mean

            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("The covariance matrix is singular.")

            function = (-1/2 * logdet -1/2 * difference.T @ np.linalg.solve(covariance, difference)+ np.log(self.prior(k)))
            return function
        return function_discriminant
    
    def predict(self, X):
        X = np.asarray(X)
        if self._fit_done == False:
            raise ValueError("The model must be trained before prediction.")
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if np.issubdtype(X.dtype, np.number) == False:
            raise ValueError("X must be a numerical array.")
        else:
            if np.isfinite(X).all() == False:
                raise ValueError("X must have only finite values.")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must be the same number of features as the training X.") 
        
        functions_class = np.array([self.labels[np.argmax([self.k_functions(k)(x) for k in self.labels])]for x in X])
        return functions_class
