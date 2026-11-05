import numpy as np
from rbg.Preprocessing._base import check_array

def MatrixDesign(X: np.ndarray) -> np.ndarray:
    """Build the design matrix by adding an intercept column of ones.

    Args:
        X (np.ndarray): Data of shape (n_samples, n_features). A 1D array os treated as a single feature.

    Returns:
        Array of shhape (n_samples, n_features + 1) whose first column is all ones and whose remaining columns are the features of 'X'.
    
    Raises:
        ValueError: If 'X' is not a 1D or 2D array.
    """
    X = check_array(X)
    return np.column_stack([np.ones(X.shape[0]), X])