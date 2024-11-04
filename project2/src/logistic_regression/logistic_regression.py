import numpy as np
from activation_functions import sigmoid
from sklearn.metrics import accuracy_score

class OwnLogisticRegression: 
    """
    A custom implementation of Logistic Regression for binary classification using gradient descent.
    
    Attributes
    ----------
    _input_size : int
        The number of input features.
    _weights : np.ndarray
        The weight parameters for the logistic regression model.
    _bias : float
        The bias parameter for the logistic regression model.

    Methods
    -------
    __init__(input_size: int)
        Initializes the logistic regression model with a specified input size and random parameters.
    
    _set_parameters()
        Initializes the weights and bias with random values.
    
    get_parameters()
        Returns the current weights and bias.
    
    _get_gradients(x: np.ndarray, target: np.ndarray)
        Computes gradients of the weights and bias for a given batch of data.
    
    feed_forward(x: np.ndarray)
        Performs the forward pass, applying the logistic regression model and sigmoid activation to the input data.
    
    train(x: np.ndarray, target: np.ndarray, epochs: int, batch_size: int, lr: float=0.01, lmd: float=0.0, x_val: np.ndarray=None, target_val: np.ndarray=None)
        Trains the model using mini-batch gradient descent for a specified number of epochs and batch size. 
        Optionally includes validation data to monitor performance.
        
        Parameters:
            x : np.ndarray
                Training input data.
            target : np.ndarray
                Training target labels.
            epochs : int
                Number of training epochs.
            batch_size : int
                Size of each mini-batch.
            lr : float, optional
                Initial learning rate for gradient descent (default is 0.01).
            lmd : float, optional
                Regularization parameter for L2 regularization (default is 0.0).
            x_val : np.ndarray, optional
                Validation input data (default is None).
            target_val : np.ndarray, optional
                Validation target labels (default is None).
                
        Returns:
            loss_train : list
                Training accuracy after each epoch.
            loss_val : list, optional
                Validation accuracy after each epoch, if validation data is provided.
    
    _lrs(t, t0, t1)
        Implements a time-based learning rate scheduler.
        
        Parameters:
            t : int
                Current time step in training.
            t0 : float
                Initial learning rate multiplier.
            t1 : float
                Scaling factor for learning rate decay.
        
        Returns:
            float
                The adjusted learning rate.
    """
    def __init__(self, input_size:int):
        self._input_size = input_size
        self._set_parameters()
    
    def _set_parameters(self) -> None:
        self._weights = np.random.randn(self._input_size, 1)
        self._bias = np.random.randn()
    
    def get_parameters(self) -> tuple:
        return self._weights, self._bias
    
    def _get_gradients(self, x:np.ndarray, target:np.ndarray) -> tuple:
        predict = self.feed_forward(x)
        target = target.reshape(-1, 1)
        return x.T @ (predict - target), np.sum(predict - target).item()
    
    def feed_forward(self, x:np.ndarray) -> np.ndarray:
        predict = x @ self._weights + self._bias
        predict_normalized = sigmoid(predict)
        return predict_normalized
    
    def train(self, x:np.ndarray, target:np.ndarray, epochs:int, batch_size:int, 
              lr:float=0.01, lmd:float=0.0, x_val:np.ndarray=None, target_val:np.ndarray=None) -> tuple:
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
    
    def _lrs(self, t, t0, t1) -> float:
        return t0/(t+t1)

