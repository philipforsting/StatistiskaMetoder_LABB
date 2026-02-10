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

    #OLS
    def fit(self, X, y):
 
        X = np.asarray(X)
        y = np.asarray(y)

        #Error handling
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError("X must be 1D or 2D")
        elif np.isnan(X).any():
            raise ValueError("X contains NaN errors?") # Är detta rimligt agerande? Kanske ska man ta bort rader med NaN? Oklart...

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of observations")

        #Calculating output
        self.X = np.column_stack([np.ones(n), X])
        self.y = y        
        self._n = n = X.shape[0]
        self._d = d = X.shape[1]
        self.beta_hat = np.linalg.pinv(self.X.T @ self.X) @self.X.T @ self.y
        return self
    
    def predict(self, X_new): #för att använda OLS
        pass

    def variance(self):
        return SSE / (self.n - self.d -1)
    
    def standard_deviation(self):
        pass

    def root_mean_squared_error(self):
        pass
