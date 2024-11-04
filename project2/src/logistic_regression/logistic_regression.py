import numpy as np
from activation_functions import sigmoid
from sklearn.metrics import accuracy_score

class OwnLogisticRegression: 
    def __init__(self, input_size:int):
        self._input_size = input_size
        self._set_parameters()
    
    def _set_parameters(self):
        self._weights = np.random.randn(self._input_size, 1)
        self._bias = np.random.randn()
    
    def get_parameters(self):
        return self._weights, self._bias
    
    def _get_gradients(self, x:np.ndarray, target:np.ndarray):
        predict = self.feed_forward(x)
        target = target.reshape(-1, 1)
        return x.T @ (predict - target), np.sum(predict - target).item()
    
    def feed_forward(self, x:np.ndarray):
        predict = x @ self._weights + self._bias
        predict_normalized = sigmoid(predict)
        return predict_normalized
    
    def train(self, x:np.ndarray, target:np.ndarray, epochs:int, batch_size:int, lr:float=0.01, lmd:float=0.0, x_val:np.ndarray=None, target_val:np.ndarray=None):
        n_samples = x.shape[0]
        n_batches = n_samples//batch_size
        t0 = 5
        t1 = 50

        loss_train, loss_val = [], []
        target = target.reshape(-1, 1)
        if x_val is not None:
            target_val = target_val.reshape(-1, 1)

        for e in range(epochs):
            shuffled_indicies = np.random.permutation(n_samples) 
            for m in range(n_batches):
                indices = shuffled_indicies[m * batch_size : (m+1) * batch_size]
                x_i = x[indices]
                target_i = target[indices] 
                W_g, b_g = self._get_gradients(x_i, target_i)
                self._weights -= self._lrs(e*n_batches+m, t0, t1) * (W_g + lmd * self._weights)/batch_size
                self._bias -= self._lrs(e*n_batches+m, t0, t1) * b_g

            acc_train = accuracy_score(target, self.feed_forward(x)>=0.5)
            loss_train.append(acc_train) 
            if x_val is not None:
                acc_val = accuracy_score(target_val, self.feed_forward(x_val)>=0.5)
                loss_val.append(acc_val)

        return loss_train, loss_val
    
    def _lrs(self, t, t0, t1):
        return t0/(t+t1)

