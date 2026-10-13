from abc import ABC, abstractmethod
import numpy as np
from rbg.Utilities import CoordinateDescent

class BaseEstimator(ABC):
    """Abstract base class with shared data storage and residual calculations."""

    def __init__(self, fit_intercept: bool = True, inference: bool = False) -> None:
        """Configure intercept fitting and inference.

        Args:
            fit_intercept: Whether to prepend a column of ones to X.
            inference: Whether to request inference calculations during fitting.
        """
        self._intercept = fit_intercept
        self._inference = inference
        self._inference_done = False
        self._residuals = None
        self._fit_done = False

    def _prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate and store training data, resetting previous fit results.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).

        Raises:
            ValueError: If inputs have invalid dimensions, incompatible sample
                counts, empty features, or nonnumeric or nonfinite values.
        """
        X = np.array(X)
        y = np.array(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be two-dimensional and y must be one-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")
        self._fit_done = False
        self._inference_done = False
        self._residuals = None
        self._X = X
        self._y = y
        self._X_input = self._X.copy()
        self._n, self._p = np.shape(self._X)
        if self._intercept:
                    self._X = np.column_stack((np.ones(self._n), self._X))
                    self._p += 1

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using the supplied training features and targets."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict responses for new observations using the fitted model.

        Args:
            X: Feature matrix without an automatically generated intercept column.
        """
        pass

    @abstractmethod
    def inference(self):
        """Return inference results specific to the fitted model."""
        pass

    @abstractmethod
    def _predict_mean(self, X):
        """Return predicted mean responses for residual calculations."""
        pass

    def residuals(self) -> np.ndarray:
        """Return observed targets minus predicted mean responses.

        Returns:
            Residual vector of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        if self._residuals is None:
            self._residuals = self._y - self._predict_mean(self._X_input)
        return self._residuals


class LinearModel(BaseEstimator):
    """Abstract base for linear predictors with model-specific fitting and inference.

    Subclasses store coefficients in the original feature scale, with the
    intercept first when enabled. Prediction and inference access are shared.
    """

    @abstractmethod
    def _compute_inference(self) -> None:
        """Compute the inference statistics specific to the subclass."""
        pass

    def inference(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Calculate and return the subclass's inference results.

        Returns:
            A tuple containing the coefficients, estimated error variance,
            estimated coefficient covariance matrix, and residual vector.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the subclass's inference requirements are not met.
            NotImplementedError: If the subclass does not implement inference.
        """
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")
        if not self._inference_done:
            self._inference = True
            self._compute_inference()
        return self._beta, self._sigma2_hat, self._cov_matrix_hat, self._residuals

    def _predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Return linear predictions as the estimated mean response."""
        return self.predict(X)

    def predict(self,X:np.ndarray) -> np.ndarray:
        """Predict responses using the fitted coefficients.

        Args:
            X: Matrix of shape (n_new_samples, p_features), with features in
                training order and without an automatically generated intercept
                column. The number of samples may differ from training.

        Returns:
            Predicted response vector of shape (n_new_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X is not two-dimensional, contains nonnumeric or
                nonfinite values, or has a different feature count from training.
        """
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")
    
        X = np.asarray(X)
        if 2 != X.ndim:
            raise ValueError("X must be a two-dimensional matrix.")
        if X.shape[1] != self._X_input.shape[1]:
            raise ValueError("X must contain the same number of features as the training data.")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        
        if self._intercept == True:
            X = np.column_stack((np.ones(np.shape(X)[0]), X))
        return  X @ self._beta





class LinearRegression(LinearModel):
    """Ordinary least-squares regression with optional classical inference.

    Inference assumes uncorrelated errors with constant variance.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit ordinary least squares, replacing previous training results.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Target vector of shape (n_samples,).

        Returns:
            Coefficient vector, with the intercept first when enabled.

        Raises:
            ValueError: If inputs are invalid or requested inference lacks
                full column rank or positive residual degrees of freedom.
        """
        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._beta, self._residuals_square_sum, self._rank,self._singularvaluesX = np.linalg.lstsq(self._X,self._y, rcond=None)
            self._fit_done = True
        if self._inference:
            self._compute_inference()
        return self._beta

    def _compute_inference(self) -> None:
        """Compute ordinary least-squares inference without refitting."""
        if not self._inference_done:
            if self._rank != self._p or self._n <= self._p:
                raise ValueError("X must have full column rank and n must exceed p to perform inference.")
            self._residuals = self.residuals()
            self._sigma2_hat = np.dot(self._residuals, self._residuals)/(self._n - self._p)
            self._cov_matrix_hat = self._sigma2_hat * np.linalg.inv(self._X.T@self._X)
            self._inference_done = True


class RidgeRegression(LinearModel):
    """Ridge regression with feature scaling and original-scale coefficients.

    Inference treats alpha as fixed. The residual variance estimate may be
    biased by shrinkage and does not establish ordinary t tests or intervals.
    """
        
        
    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float=1.0) -> np.ndarray:
        """Estimate coefficients and calculate inference statistics if requested.

        Each call replaces the previous fit and cached inference results.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Target vector of shape (n_samples,).
            alpha: Finite, nonnegative L2 penalty on scaled feature coefficients.

        Returns:
            Coefficient vector in the original feature scale, with the intercept
            first when enabled.

        Raises:
            ValueError: If alpha or training inputs are invalid, or requested
                inference fails its rank or residual degrees-of-freedom checks.
        """
        self._alpha = alpha
        if not isinstance(self._alpha, (int,float,np.integer,np.floating)):
            raise ValueError("Alpha must be a numeric scalar.")
        if not np.isfinite(self._alpha) or alpha < 0:
            raise ValueError("Alpha must be finite and nonnegative.")
        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._D = np.eye(self._p)
            if self._intercept == True:
                self._D[0,0] = 0
            self._means = self._X_input.mean(axis=0) if self._intercept else np.zeros(self._X_input.shape[1])
            self._stds = self._X_input.std(axis=0)
            self._stds = np.where(self._stds == 0, 1.0, self._stds)
            self._Z = (self._X_input - self._means) / self._stds
            if self._intercept:
                self._Z = np.column_stack((np.ones(self._n), self._Z))
            X_aug = np.vstack((self._Z, np.sqrt(self._alpha) * self._D))
            y_aug = np.concatenate((self._y, np.zeros(self._p)))
            self._beta, self._residuals_square_sum, _,_ = np.linalg.lstsq(X_aug,y_aug, rcond=None)
            # Convert coefficients back to the original feature scale.
            self._T = np.diag(1.0 / self._stds)
            if self._intercept:
                self._T = np.eye(self._p)
                self._T[0, 1:] = -self._means / self._stds
                self._T[1:, 1:] = np.diag(1.0 / self._stds)
            self._beta = self._T @ self._beta
            self._fit_done = True
        if self._inference:
            self._compute_inference()
        return self._beta

    def _compute_inference(self) -> None:
        """Estimate error variance and original-scale coefficient covariance.

        Use the standardized design and effective residual degrees of freedom.
        The current implementation requires full column rank in that design.

        Raises:
            ValueError: If the standardized design is rank deficient or the
                residual degrees of freedom are not positive.
        """
        if not self._inference_done:
            if np.linalg.matrix_rank(self._Z) < self._p:
                 raise ValueError("Z must have full column rank.")
            A = self._Z.T @ self._Z + self._alpha*self._D
            M, _,_,_ = np.linalg.lstsq(A,self._Z.T, rcond=None)
            H = self._Z@M
            df = self._n - 2*np.trace(H) + np.trace(H.T@H)
            if not df > 0:
                raise ValueError("Residual degrees of freedom must be positive.")

            e = self.residuals()

            self._sigma2_hat = np.dot(e,e)/df

            self._cov_matrix_hat_padronized = self._sigma2_hat * (M @ M.T)
            
            self._cov_matrix_hat = self._T @ self._cov_matrix_hat_padronized @ self._T.T
            self._inference_done = True


class LassoRegression(LinearModel, CoordinateDescent):
    """Lasso regression using the CoordinateDescent solver supplied externally.

    Assumed solver contract: CoordinateDescent.solve(X, y, alpha, fit_intercept)
    is a static or class method that returns a
    converged coefficient vector in the supplied feature scale, including the
    intercept first when enabled. It minimizes 0.5 * squared_error + alpha *
    sum(abs(feature_coefficients)), leaving the intercept unpenalized.
    CoordinateDescent must be imported before this class is defined; any
    required solver settings must be initialized according to its interface.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Prepare scaled features and delegate the L1 fit to CoordinateDescent.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).
            alpha: Finite, nonnegative penalty strength.

        Returns:
            Coefficients in the original feature scale,
            with the intercept first when enabled.

        Raises:
            ValueError: If inputs or alpha are invalid, or the solver returns
                coefficients with an invalid shape or nonfinite values.
            NotImplementedError: If inference is requested.
        """
        self._alpha = alpha
        if not isinstance(self._alpha, (int,float,np.integer,np.floating)):
            raise ValueError("Alpha must be a numeric scalar.")
        if not np.isfinite(self._alpha) or alpha < 0:
            raise ValueError("Alpha must be finite and nonnegative.")

        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._means = self._X_input.mean(axis=0) if self._intercept else np.zeros(self._X_input.shape[1])
            self._stds = self._X_input.std(axis=0)
            self._stds = np.where(self._stds == 0, 1.0, self._stds)
            self._Z = (self._X_input - self._means) / self._stds
            if self._intercept:
                self._Z = np.column_stack((np.ones(self._n), self._Z))
            beta_scaled = CoordinateDescent.solve(self._Z,self._y,self._alpha,self._intercept)
            self._T = np.diag(1.0 / self._stds)
            if self._intercept:
                self._T = np.eye(self._p)
                self._T[0, 1:] = -self._means / self._stds
                self._T[1:, 1:] = np.diag(1.0 / self._stds)
            self._beta = self._T @ beta_scaled
            self._fit_done = True
            e = self.residuals()
            self._residuals_square_sum = np.dot(e,e)
        if self._inference:
            self._compute_inference()
        return self._beta
    

    def inference(
        self, n_bootstrap: int = 1000, seed: int | None = None
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Configure bootstrap inference and use the inherited inference interface.

        Args:
            n_bootstrap: Number of bootstrap repetitions, at least two.
            seed: Nonnegative random seed, or None for unseeded sampling.

        Returns:
            Once bootstrap is complete, coefficients, estimated error variance,
            coefficient covariance, and training residuals.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the number of repetitions or seed is invalid.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
            raise ValueError("n_bootstrap must be an integer greater than or equal to two.")
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None.")
        if n_bootstrap != getattr(self, "_n_bootstrap", 1000) or seed != getattr(self, "_seed", None):
            self._inference_done = False
        self._n_bootstrap = n_bootstrap
        self._seed = seed
        return super().inference()

    def _compute_inference(self) -> None:
        """Estimate coefficient covariance by pairs bootstrap.

        Use fresh models with inference disabled to preserve the original fit.
        Estimate error variance from the original residuals and active parameter
        count. This approximation may be biased and does not guarantee valid
        confidence intervals for Lasso coefficients.
        """
        if self._inference_done:
            return
        self._seed = getattr(self, "_seed", None)
        self._n_bootstrap = getattr(self, "_n_bootstrap", 1000)
        features_beta = self._beta[1:] if self._intercept else self._beta
        self._d = np.count_nonzero(features_beta) + int(self._intercept)
        if self._n <= self._d:
            raise ValueError("Residual degrees of freedom must be positive.")
        rng = np.random.default_rng(self._seed)
        betas_boot = np.zeros((self._n_bootstrap, self._p))
        for i in range(self._n_bootstrap):
            indices = rng.choice(self._n, size=self._n, replace=True)
            X_boot = self._X_input[indices]
            y_boot = self._y[indices]
            model_boot = LassoRegression(fit_intercept=self._intercept, inference=False)
            betas_boot[i] = model_boot.fit(X_boot, y_boot, self._alpha)
        self._residuals = e = self.residuals()
        self._sigma2_hat = np.dot(e,e)/(self._n-self._d)
        self._cov_matrix_hat = np.atleast_2d(np.cov(betas_boot, rowvar=False, ddof=1))
        self._inference_done = True
        

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
            "lasso": LassoRegression,
        }

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
