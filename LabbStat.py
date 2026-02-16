import numpy as np
from scipy import stats

class LinjearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self._n = None
        self._d = None
        self.beta_hat = None    
    
    @property
    def n(self):
        return self._n
    
    @property
    def d(self):
        return self._d

    #OLS, Ordinary Least Squares
    def fit(self, X, y):
 
        X = np.asarray(X)
        y = np.asarray(y)

        # Dimension handling
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError("X must be 1D or 2D")
        
        # NaN and shape missmatch handling
        if np.isnan(X).any():
            raise ValueError("X contains NaN errors") # Är detta rimligt agerande? Kanske ska man ta bort rader med NaN? Oklart...
        if np.isnan(y).any():
            raise ValueError("X contains NaN errors") # Är detta rimligt agerande? Kanske ska man ta bort rader med NaN? Oklart...
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of observations")

        # Calculating output
        self._n = X.shape[0]
        self._d = X.shape[1]
        self.X = np.column_stack([np.ones(self._n), X])
        self.y = y        
        self.beta_hat = np.linalg.pinv(self.X.T @ self.X) @self.X.T @ self.y
        return self


    def sse(self):
        # NaN and shape missmatch handling
        if (self.beta_hat is None) or (self.X is None):
            raise ValueError("Model has not been fitted yet")
        # Calculating output
        y_hat = self.X @ self.beta_hat
        return np.sum(np.square(self.y - y_hat)) #


    def variance(self):
        return self.sse() / (self.n - self.d -1)
    
    def standard_deviation(self):
        return np.sqrt(self.variance())

    def root_mean_squared_error(self):
        return np.sqrt(self.sse() / self.n)
