"""Check imports and working estimators after splitting the regression module."""
from importlib import import_module
import unittest

import numpy as np

from rbg.Classification.LogisticRegression import LogisticRegression
from rbg.Regression.baseestimator import BaseEstimator
from rbg.Regression.GLM import GLM
from rbg.Regression.LinearRegression import LinearRegression
from rbg.Regression.linearmodel import LinearModel
from rbg.Regression.PolynomialRegression import PolynomialRegression
from rbg.Regression.RidgeRegression import RidgeRegression


class TestRegressionModules(unittest.TestCase):
    """Use standard-library tests that can also be collected by pytest."""

    def test_legacy_imports_reference_the_same_classes(self) -> None:
        """Existing imports resolve to the newly extracted definitions."""
        legacy = import_module("rbg.Regression.regressao")
        for estimator in (BaseEstimator, LinearModel, LinearRegression,
                          RidgeRegression, PolynomialRegression, GLM):
            with self.subTest(estimator=estimator.__name__):
                self.assertIs(getattr(legacy, estimator.__name__), estimator)
        with self.assertRaises(AttributeError):
            getattr(legacy, "UnknownEstimator")

    def test_linear_fit_prediction_and_inference(self) -> None:
        """Preserve fitting, prediction and inference with or without intercept."""
        X = np.arange(1.0, 9.0).reshape(-1, 1)
        for fit_intercept in (True, False):
            with self.subTest(fit_intercept=fit_intercept):
                y = 3.0 * X[:, 0] + (2.0 if fit_intercept else 0.0)
                model = LinearRegression(fit_intercept=fit_intercept, inference=True)
                expected = [2.0, 3.0] if fit_intercept else [3.0]
                np.testing.assert_allclose(model.fit(X, y), expected, atol=1e-12)
                np.testing.assert_allclose(model.predict(X), y, atol=1e-12)
                np.testing.assert_allclose(model.residuals(), 0, atol=1e-12)
                beta, variance, covariance, residuals = model.inference()
                np.testing.assert_allclose(beta, expected, atol=1e-12)
                self.assertLess(variance, 1e-20)
                self.assertEqual(covariance.shape, (len(expected), len(expected)))
                self.assertEqual(residuals.shape, y.shape)

    def test_ridge_regularization(self) -> None:
        """Ridge still shrinks coefficients and preserves the intercept."""
        X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
        y = 2.0 + 3.0 * X[:, 0]
        model = RidgeRegression()
        # This implementation minimizes RSS + alpha*||scaled coefficients||².
        np.testing.assert_allclose(model.fit(X, y, alpha=4.0), [2.0, 1.5], atol=1e-12)
        np.testing.assert_allclose(model.predict([[0.0]]), [2.0], atol=1e-12)

    def test_polynomial_without_lasso_dependency(self) -> None:
        """Polynomial OLS/Ridge work without the unfinished CoordinateDescent."""
        X = np.linspace(-2.0, 2.0, 15).reshape(-1, 1)
        y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 0]**2
        for method in ("linear", "ridge"):
            with self.subTest(method=method):
                model = PolynomialRegression([2], method=method)
                options = {"alpha": 0.0} if method == "ridge" else {}
                np.testing.assert_allclose(model.fit(X, y, **options), [1, 2, -0.5], atol=1e-12)
                np.testing.assert_allclose(model.predict(X), y, atol=1e-12)

    def test_logistic_uses_extracted_glm(self) -> None:
        """Classification initializes against the same GLM base definition."""
        model = LogisticRegression(classes=[0, 1])
        self.assertIsInstance(model, GLM)
        self.assertIsInstance(model, BaseEstimator)
        self.assertFalse(model._fit_done)
        self.assertEqual(model._family.mean(np.zeros((2, 1))).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
