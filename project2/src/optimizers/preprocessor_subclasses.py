import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


class Preprocessor:

    def __init__(self):
        pass

    def scaler(self, X, y=None):
        scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(X)
        X_scaled[:,0] = 1
        if y:
            y_scaled = scaler.fit_transform(y)
            return X_scaled, y_scaled 
        return X_scaled
    
    def split(self, X, y, train_size=0.8):
        return train_test_split(X, y, train_size=train_size)

class PolynomialPreprocessor(Preprocessor):

    def __init__(self, *data, degree):
        self._degree = degree
        if len(data) == 2:
            self._x = data[0]
            self._y = data[1]
            self._X = self._create_X_2d()
        else:
            self._x = data[0]
            self._X = self._create_X_1d()
    
    def _create_X_2d(self):
        num_betas = (self._degree+1)*(self._degree+2)//2
        X = np.zeros((self._x.size, num_betas))
        idx = 0
        for i in range(self._degree+1):
            for j in range(self._degree+1): 
                if i+j <= self._degree:
                    entry = (self._x**i * self._y**j)
                    X[:, idx] = entry.ravel() # makes us able to send in both matrix and vector
                    idx += 1
        return X

    def _create_X_1d(self):
        X = np.column_stack([self._x**i for i in range(self._degree+1)])
        return X

    def get_X(self):
        return self._X


