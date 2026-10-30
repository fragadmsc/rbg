"""Construct polynomial feature design matrices."""

import numpy as np


def Matrix_design(X, degree=1, intercept=True):
    """Expand each feature through ``degree`` without interaction terms.

    Columns are ordered by feature and then by increasing power. When
    ``intercept`` is true, a column of ones is prepended.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix.")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer.")
    if not isinstance(intercept, bool):
        raise ValueError("intercept must be a boolean.")
    columns = [X[:, j] ** power for j in range(X.shape[1]) for power in range(1, degree + 1)]
    if intercept:
        columns.insert(0, np.ones(X.shape[0]))
    return np.column_stack(columns)
