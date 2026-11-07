from .KFold import KFold
import numpy as np

class LOOCV:
    """Leave-One-Out Cross Validation
    
    Take one sample and uses it as the testing set. Is a great option if the dataset is small enough.
    """
    def __init__(self):
        pass
    
    def split(self, data_size, y = None):
        if type(data_size) == np.ndarray or type(data_size) == list:
            x = len(data_size)
        elif type(data_size) == int:
            x = data_size
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if y is not None:
            y = np.asarray(y)
            if x != y.shape[0]:
                raise ValueError("X and y dimensions must match.")
        return KFold(k=x).split(data_size = data_size, y = y)