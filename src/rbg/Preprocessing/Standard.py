import numpy as np

def Standard(Data):
    return np.array((Data - np.mean(Data)) / np.std(Data))