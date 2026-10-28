import warnings 

import numpy as np

class ExponentialFamily():
    """Represent an exponential family through its cumulant and derivatives.

    Inputs use one row per observation and one column per response component.
    For scalar responses, retain a single component column. Family functions
    must support these batched inputs and correspond to a valid distribution.
    """

    def __init__(self, b, b_prime, b_second, h_inverse, phi=1.0):
        """Store the family functions and dispersion parameter.

        Args:
            b: Cumulant function returning an array of shape (n_samples,).
            b_prime: Gradient returning means of shape (n_samples, n_components).
            b_second: Hessian returning arrays of shape
                (n_samples, n_components, n_components).
            h_inverse: Inverse of b_prime on its mean domain. With a canonical
                link, map response means to natural parameters and predictors.
            phi: Finite, positive dispersion parameter; fixed at 1 for the
                standard categorical family.

        Raises:
            ValueError: If a family function is not callable or phi is invalid.
        """
        if not all(callable(f) for f in (b, b_prime, b_second, h_inverse)):
            raise ValueError("b, b_prime, b_second, and h_inverse must be callable.")
        if not isinstance(phi, (int, float, np.integer, np.floating)) or not np.isfinite(phi) or phi <= 0:
            raise ValueError("phi must be finite and positive.")
        self.b = b
        self.b_prime = b_prime
        self.b_second = b_second
        self.h_inverse = h_inverse
        self._phi = phi

    def log_likelihood(self, y, theta, weights=1.0):
        """Return the parameter-dependent part of the summed log-likelihood.

        Compute sum(weights / phi * (inner_product(y, theta) - b(theta))).
        The additive normalization term is omitted. This is the full
        log-likelihood only when that term is zero.

        Args:
            y: Response or sufficient-statistic matrix of shape
                (n_samples, n_components).
            theta: Natural-parameter matrix with the same shape as y.
            weights: Scalar weight or array of shape (n_samples,).

        Returns:
            Sum of the parameter-dependent log-likelihood contributions.

        Raises:
            ValueError: If input dimensions or weight dimensions are incompatible.
        """
        y = np.asarray(y)
        theta = np.asarray(theta)
        weights = np.asarray(weights)
        if y.ndim != 2 or theta.ndim != 2 or y.shape != theta.shape or y.size == 0:
            raise ValueError("y and theta must be nonempty matrices with the same shape.")
        if weights.ndim != 0 and weights.shape != (theta.shape[0],):
            raise ValueError("weights must be a scalar or one value per observation.")
        b_values = np.asarray(self.b(theta))
        terms = (weights / self._phi) * (np.sum(y*theta, axis=1) - b_values)
        return np.sum(terms)

    def mean(self,theta):
        """Evaluate the mean response as the gradient of the cumulant.

        Args:
            theta: Nonempty matrix of shape (n_samples, n_components).

        Returns:
            Mean responses returned by b_prime, expected to match theta's shape.

        Raises:
            ValueError: If theta is not a nonempty two-dimensional array.
        """
        theta = np.asarray(theta)
        if theta.ndim != 2 or theta.size == 0:
            raise ValueError("theta must be a nonempty matrix.")
        result = np.asarray(self.b_prime(theta))

        return result

    def covariance(self,theta, weights = 1.0):
        """Evaluate conditional response covariance as phi / weights * b_second.

        These are response covariance matrices, not coefficient covariances.

        Args:
            theta: Nonempty matrix of shape (n_samples, n_components).
            weights: Scalar weight or array of shape (n_samples,). The caller
                must supply positive weights.

        Returns:
            Array of shape (n_samples, n_components, n_components), assuming
            b_second returns a Hessian per observation.

        Raises:
            ValueError: If theta or weight dimensions are incompatible.
        """
        theta = np.asarray(theta)
        weights = np.asarray(weights)
        if theta.ndim != 2 or theta.size == 0:
            raise ValueError("theta must be a nonempty matrix.")
        if weights.ndim != 0 and weights.shape != (theta.shape[0],):
            raise ValueError("weights must be a scalar or one value per observation.")
        result = np.asarray(self.b_second(theta))
        scale = self._phi / weights
        if weights.ndim == 1:
            scale = scale[:, None, None]
        return scale * result


class FisherScoring:
    """Fisher scoring for GLM"""

    @staticmethod
    def solve(
        X: np.ndarray,
        y: np.ndarray,
        family: ExponentialFamily,
        beta0: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the coefficients that maximize the GLM likelihood.

        Args:
            X: Design matrix of shape (n, p), with the intercept column if used.
            y: Responses of shape (n,) or (n, K).
            family: Exponential family of the responses.
            beta0: Initial coefficients of shape (p,) or (p, K).
            tol: Stop when the size of the update is less than tol.
            max_iter: Maximum number of iterations.

        Returns:
            The coefficients, and the Fisher information at those coefficients, of shape (p*K, p*K).
        """
        n, p = X.shape

        is_vector = (y.ndim == 1)
        if is_vector:
            y = y.reshape(n, 1)

        K = y.shape[1]

        beta = np.array(beta0, dtype=float).reshape(p, K)

        converged = False
        for _ in range(max_iter):
            score, fisher = FisherScoring._working_values(X, y, family, beta)
            step = np.linalg.solve(fisher, score)

            step = step.reshape(K, p).T
            beta = beta + step

            if np.linalg.norm(step) < tol:
                converged = True
                break

        if not converged:
            warnings.warn("Did not converge")

        score, fisher = FisherScoring._working_values(X, y, family, beta)

        if is_vector:
            beta = beta[:, 0]

        return beta, fisher

    @staticmethod
    def _working_values(
        X: np.ndarray, y: np.ndarray, family: ExponentialFamily, beta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the score and the Fisher information at beta.

        Args:
            X: Design matrix of shape (n, p).
            y: Responses of shape (n, K).
            family: Exponential family of the responses.
            beta: Current coefficients of shape (p, K).

        Returns:
            The score, of shape (p*K,), and the Fisher information, of shape
            (p*K, p*K). Both have the coefficients of class 0 first, then
            class 1, and so on.
        """
        _ , p = X.shape
        K = beta.shape[1]
        phi = family._phi

        theta = X @ beta                   #(n, K)
        mu = family.mean(theta)            #(n, K)
        curvature = family.b_second(theta) #(n, K, K)

        score = np.zeros(K * p)
        for k in range(K):
            score[k * p:(k + 1) * p] = X.T @ (y[:, k] - mu[:, k]) / phi

        fisher = np.zeros((K * p, K * p))
        for k in range(K):
            for l in range(K):
                w = curvature[:, k, l]
                block = X.T @ (w[:, None] * X)
                fisher[k * p:(k + 1) * p, l * p:(l + 1) * p] = block / phi

        return score, fisher