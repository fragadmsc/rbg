import numpy as np

def Standard(Data):
    """
    Just normalize the Data array subtract the mean and divides by std
    """
    return np.array((Data - np.mean(Data)) / np.std(Data))