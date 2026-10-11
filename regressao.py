from abc import ABC, abstractmethod
import numpy as np

# Abstract base class for regression models
class BaseEstimator(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self,X,y):
        pass


# Linear regression class
class LinearRegression(BaseEstimator):

    # Initialize the model with X and y and configure intercept fitting and inference.
    def __init__(self, X:np.ndarray,y:np.ndarray,fit_intercept=True, inference=False) -> None:
        self._intercept = fit_intercept
        self._X = np.array(X)   
        self._y = np.array(y)
        if  1 != self._y.ndim or 2 != self._X.ndim:
            raise ValueError("X must be two-dimensional and y must be one-dimensional.")
        self._n, self._p = np.shape(self._X)
        if self._intercept:
                    self._X = np.column_stack((np.ones(self._n), self._X))
                    self._p += 1
        self._inference = inference
        self._residuals = None
        self._fit_done = False
        if  self._n != np.shape(self._y)[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if self._n == 0 or self._p == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if not np.issubdtype(self._X.dtype, np.number) or not np.issubdtype(self._y.dtype, np.number):
            raise ValueError("X and y must contain numeric values.")
        
    # Estimate the regression coefficients and compute inference statistics if requested.
    def fit(self) -> np.ndarray:
        if self._fit_done == False:
            self._beta, self._residuals_square_sum, rank,self._singularvaluesX = np.linalg.lstsq(self._X,self._y, rcond=None)
            self._fit_done = True
        if self._inference==True:
            if rank != self._p or self._n <= self._p:
                raise ValueError("X must have full column rank and n must exceed p to perform inference.")
            self._residuals = self._y - self._X @ self._beta
            self._sigma2_hat = np.dot(self._residuals, self._residuals)/(self._n - self._p)
            self._cov_matrix_hat = self._sigma2_hat * np.linalg.inv(self._X.T@self._X)
        return self._beta

    # Return inference results for the fitted model.
    def inference(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")
        if self._inference==False:
            self._inference = True
            self.fit()
        return self._beta, self._sigma2_hat, self._cov_matrix_hat, self._residuals

    # Return the residuals.
    def residuals(self) -> np.ndarray:
        if self._residuals is None:
            if not self._fit_done:
                raise RuntimeError("The model has not been fitted yet.")
            self._residuals = self._y - self._X @ self._beta

        return self._residuals

    #Predict new output using a new X and the beta already fitted.
    def predict(self,X:np.ndarray, confidence_interval=False) -> np.ndarray:
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")
        if confidence_interval:
            if self._inference == False:
                self.inference()
            
        X = np.ndarray(X)
        if 2 != X.ndim:
            raise ValueError("X must be a two dimensional vector")
        
        if self._intercept == True:
            X = np.column_stack((np.ones(np.shape(X)[0]), X))
        return  X @ self._beta
