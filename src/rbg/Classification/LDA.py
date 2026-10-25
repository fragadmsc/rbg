import numpy as np
from rbg.Classification.classifier import Classifier

class LDA(Classifier):
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
        if X.shape[0] <= self.labels.size:
            raise ValueError("There must be more samples to estimate covariance.")
        
        self._X_train = X
        self._y_train = y

        self._covariance = self.cov_matrix()
        if np.linalg.matrix_rank(self._covariance) < self._covariance.shape[0]:
            raise ValueError("The covariance matrix is singular.")
        
        self._fit_done = True
    
    def mean_array(self, k):
        x_k = self._X_train[self._y_train == k]
        if x_k.shape[0] == 0:
            raise ValueError("The class must have training samples.")
        mean_k = np.mean(x_k, axis = 0)
        return mean_k
    
    def cov_matrix(self):
        matriz = np.zeros((self._X_train.shape[1], self._X_train.shape[1]))
        detours = [self._X_train[self._y_train == k] - self.mean_array(k) for k in self.labels]
        products = [detour.T @ detour for detour in detours]
        matriz = np.sum(np.stack(products), axis=0)/(self._y_train.size - self.labels.size)
        return matriz
    
    def prior(self, k):
        sum_k = (self._y_train == k).sum()
        prior_k = sum_k/self._y_train.size
        return prior_k
    
    def k_functions(self, k):
        def function_discriminant(x):
            function = (x.T @ np.linalg.inv(self._covariance) @ self.mean_array(k) 
                    - 1/2 * self.mean_array(k).T @ np.linalg.inv(self._covariance)
                    @ self.mean_array(k) + np.log(self.prior(k)))
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