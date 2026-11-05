import warnings
import numpy as np
from .ExponentialFamily import ExponentialFamily

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
