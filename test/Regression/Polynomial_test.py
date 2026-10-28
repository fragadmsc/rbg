"""Check exact polynomial expansions and dimensional contracts."""
import unittest

import numpy as np

from rbg.Regression.Polynomial import Polynomial


class TestPolynomialTransform(unittest.TestCase):
    def test_transform_known_combinations(self) -> None:
        cases = [
            ([1], [[2], [-3], [0]], [[2], [-3], [0]]),
            ([3], [[2], [-2], [0]], [[2, 4, 8], [-2, 4, -8], [0, 0, 0]]),
            ([2, 3], [[2, 3]], [[2, 4, 3, 9, 27]]),
            ([2, 1], [[-2, 0], [0.5, -3]], [[-2, 4, 0], [0.5, 0.25, -3]]),
            ([1, 2, 1], [[1, -2, 3], [0, 0, -1]], [[1, -2, 4, 3], [0, 0, 0, -1]]),
            ([2, 2], [[1, 2], [3, 4]], [[1, 1, 2, 4], [3, 9, 4, 16]]),
        ]
        for degrees, X, expected in cases:
            for intercept in (True, False):
                with self.subTest(degrees=degrees, X=X, intercept=intercept):
                    original = np.array(X, dtype=float)
                    result = Polynomial(degrees, fit_intercept=intercept)._transform(original)
                    self.assertEqual(result.shape, (len(X), sum(degrees)))
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(original, X)

    def test_transform_empty_batch_preserves_columns(self) -> None:
        result = Polynomial([2, 3])._transform(np.empty((0, 2)))
        self.assertEqual(result.shape, (0, 5))

    def test_transform_rejects_wrong_dimensions(self) -> None:
        model = Polynomial([2, 3])
        for X in (np.array(2), np.array([2, 3]), np.ones((2, 2, 1)),
                  np.ones((3, 1)), np.ones((3, 3))):
            with self.subTest(shape=X.shape):
                with self.assertRaises(ValueError):
                    model._transform(X)

    def test_polynomial_function_dimensions(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(24, 2))
        y = 1 + 2 * X[:, 0] + X[:, 0]**2 - X[:, 1]
        for method in ("linear", "ridge"):
            for intercept in (True, False):
                with self.subTest(method=method, intercept=intercept):
                    model = Polynomial([2, 1], method=method, fit_intercept=intercept)
                    p = 3 + int(intercept)
                    options = {"alpha": 0.1} if method == "ridge" else {}
                    beta = model.fit(X, y, **options)
                    self.assertEqual(beta.shape, (p,))
                    for n in (1, 5):
                        self.assertEqual(model.predict(X[:n]).shape, (n,))
                    self.assertEqual(model.residuals().shape, (len(X),))
                    inference_beta, variance, covariance, residuals = model.inference()
                    self.assertEqual(inference_beta.shape, (p,))
                    self.assertEqual(np.ndim(variance), 0)
                    self.assertEqual(covariance.shape, (p, p))
                    self.assertEqual(residuals.shape, (len(X),))


if __name__ == "__main__":
    unittest.main()
