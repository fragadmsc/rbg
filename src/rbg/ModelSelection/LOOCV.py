class LOOCV:
    """Leave-One-Out Cross Validation
    
    Take one sample and uses it as the testing set. Is a great option if the dataset is small enough.
    """
    def __init__(self):
        pass
    
    def split(self, data_size):
        if type(data_size) == np.ndarray or type(data_size) == list:
            x = len(data_size)
        elif type(data_size) == int:
            x = data_size
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        return KFold(k=x).split(data_size = x)
