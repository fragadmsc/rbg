import numpy as np

from rbg.Regression.LinearRegression import LinearRegression
from rbg.Regression.RidgeRegression import RidgeRegression


class PolynomialRegression():
    """Combine per-feature polynomial powers with a selected regression model.

    Expand each feature separately, without cross-feature interactions or a
    constant column. The internal estimator handles the intercept. For example,
    degrees [2, 3] produces columns x1, x1**2, x2, x2**2, x2**3.
    """

    def __init__(self, degrees, method="linear", fit_intercept=True):
        """Configure polynomial degrees and create the internal estimator.

        Args:
            degrees: Sequence of positive integer degrees, one per input feature.
            method: Estimator choice: 'linear', 'ridge', or 'lasso'.
            fit_intercept: Whether the internal estimator fits an intercept.

        Raises:
            ValueError: If degrees is not a nonempty one-dimensional sequence
                of positive integers, or method is not supported.
        """
        self._degrees = np.asarray(degrees)
        if (
            self._degrees.ndim != 1
            or self._degrees.size == 0
            or not np.issubdtype(self._degrees.dtype, np.integer)
            or np.any(self._degrees < 1)
        ):
            raise ValueError("degrees must be a nonempty sequence of positive integers.")

        models = {
            "linear": LinearRegression,
            "ridge": RidgeRegression,
        }

        if method == "lasso":
            from rbg.Regression.LassoRegression import LassoRegression

            models["lasso"] = LassoRegression

        if method not in models:
            raise ValueError("method must be 'linear', 'ridge', or 'lasso'.")

        self._model = models[method](fit_intercept=fit_intercept)

    def _transform(self, X):
        """Expand features into their individual polynomial powers.

        Args:
            X: Feature matrix of shape (n_samples, p_features).

        Returns:
            Matrix of shape (n_samples, sum(degrees)), ordered first by input
            feature and then by increasing power. No intercept column is added.

        Raises:
            ValueError: If X is not two-dimensional or its feature count does
                not match the number of configured degrees.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self._degrees):
            raise ValueError("Provide one degree per feature.")

        return np.column_stack([
            X[:, j] ** power
            for j, degree in enumerate(self._degrees)
            for power in range(1, degree + 1)
        ])

    def fit(self, X, y, **fit_options):
        """Transform training features and fit the internal estimator.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).
            **fit_options: Estimator-specific fitting options, such as alpha
                for Ridge or Lasso. Linear regression does not accept alpha.

        Returns:
            Coefficients for the expanded features, with the intercept first
            when enabled. Ridge and Lasso undo their internal scaling.
        """
        return self._model.fit(self._transform(X), y, **fit_options)

    def predict(self, X):
        """Transform new features and return the internal model's predictions.

        Args:
            X: Matrix of shape (n_new_samples, p_features), in training order.

        Returns:
            Predicted response vector of shape (n_new_samples,).

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.predict(self._transform(X))

    def residuals(self):
        """Return the internal estimator's training residuals.

        Returns:
            Observed targets minus predictions, with shape (n_samples,).

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.residuals()

    def inference(self, **options):
        """Delegate inference for the expanded coefficients to the estimator.

        Args:
            **options: Estimator-specific inference options

        Returns:
            Coefficients, estimated error variance, coefficient covariance,
            and training residuals, as returned by the internal estimator.

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.inference(**options)
