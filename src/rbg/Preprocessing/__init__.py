import numpy as np


def Standard(X):
    """Return column-wise z-scores, keeping constant columns at zero."""
    values = np.asarray(X, dtype=float)
    if values.ndim not in (1, 2) or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("X must be a nonempty finite vector or matrix.")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (values - mean) / scale
