import numpy as np
from .classifier import Classifier

class KNN(Classifier):

    def __init__(self, classes, k=3, reference_class = None):
        Classifier.__init__(self, classes, reference_class)
        self.set_params(k)

    def set_params(self, k):
        if isinstance(k, bool) == False:
            if isinstance(k, int) and k>0:
                self.k = k
            else:
                raise ValueError("k must be a positive integer.")
        else:
            raise ValueError("k must be a positive integer.")
        return self
    
    def get_params(self):
        return {"k": self.k}

    def fit(self, X,y):
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
        if X.shape[0] == 0 or y.shape[0] == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        
        self._X_train = X
        self._y_train = y
        self._fit_done = True

    def predict(self, X):
        if self._fit_done == False:
            raise ValueError("The model must be trained before prediction.")

        X = np.asarray(X)

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
        
        def min_distance(x, k):
            dist = [sum((x[i]-amostra[i])**2 for i in range(len(x)))**(1/2) for amostra in self._X_train]
            indice = np.argsort(dist)[:k]
            return indice
        
        def class_neighbors(indice):
            neighbors = self._y_train[indice]
            return neighbors
        
        prediction = [class_neighbors(min_distance(x, self.k)) for x in X]

        def final_class(prediction):
            classes, counts =  np.unique(prediction, return_counts=True)
            return classes[np.argmax(counts)]

        predicted_classes = np.array([final_class(neighbors) for neighbors in prediction])
        return predicted_classes