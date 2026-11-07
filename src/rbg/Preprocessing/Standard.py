import numpy as np

def Standard(Data):
    """
    Just normalize the Data array subtract the mean and divides by std
    """
    ep = 1e-11
    return np.array((Data - np.mean(Data,axis = 0)) / (np.std(Data,axis= 0)+ep))