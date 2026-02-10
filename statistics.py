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
        #ta emot X, y
        # lägg till intercept 
        # sätt n och d 
        # spara X och y 
        # beräkna beta_hat med normalekvationen 
        # returnera modellen
        X = np.asarray(X)
        # tänk på vad som händer om X har fler än 2 dim
        y = np.asarray(y)
        # len(y) == X.shape[0] → annars är modellen meningslös

        ones = np.ones((X.shape[1], 1))

        return self
    
    def predict(self, X_new): #för att använda OLS
        pass

    def variance(self):
        return SSE / (self.n - self.d -1)
    
    def standard_deviation(self):
        pass

    def root_mean_squared_error(self):
        pass
