import numpy as np

class Scaler():
    def __init__(self):
        #para cada tipo de normalização salvo um tipo de dado
        #standard
        self._mean = None
        self._std =None



    def Standard_fit(self,X):
        """
        Standarize the data and saves mean and std to use late in Standard
        """
        self._mean = np.mean(X,axis = 0)
        self._std = np.std(X,axis = 0)

        ep = 1e-11
        return np.array((X - self._mean) / (self._std + ep))

    def Standanrd(self,X):
        """
        Must be used before Standard_fit
        """


        ep = 1e-11
        return np.array((X - self._mean) / (self._std + ep))