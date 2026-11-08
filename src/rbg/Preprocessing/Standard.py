import numpy as np

class Standard():
    def __init__(self):
        #para cada tipo de normalização salvo um tipo de dado
        #standard
        self._mean = None
        self._std =None
        self._ep = 1e-11


    def fit(self,X):
        """
        Standarize the data and saves mean and std to use late in Standard
        """
        self._mean = np.mean(X,axis = 0)
        self._std = np.std(X,axis = 0)

        return np.array((X - self._mean) / (self._std + self._ep))

    def transform(self,X):
        """
        Must be used before Standard_fit
        """
        return np.array((X - self._mean) / (self._std + self._ep))

    def untransform(self,X):
        """
        Untransform the Data
        """

        return X * (self._std + self._ep) + self._mean
