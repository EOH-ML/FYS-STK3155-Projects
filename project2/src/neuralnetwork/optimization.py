import numpy as np

class Optimization:
    """
    Base class for optimization algorithms used to train neural networks.

    This class provides a general structure for implementing optimization algorithms 
    that update the weights and biases of a neural network. Subclasses must implement 
    the `update` method to define the specific optimization logic.

    Attributes
    ----------
    _learning_rate : float
        Learning rate for the optimization algorithm.
    _nn : NeuralNetwork
        The neural network being optimized.
    _batch_size : int
        Size of the training batch, set through `set_batch_size`.
    
    Methods
    -------
    set_batch_size(batch_size: int)
        Sets the batch size for the optimizer.
    """
    def __init__(self, learning_rate, nn):
        self._learning_rate = learning_rate 
        self._nn = nn
    
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
    
class GradientDescent(Optimization):
    """
    Implements standard gradient descent optimization with optional momentum.

    This class performs gradient descent, which updates the network weights 
    by moving them in the direction that minimizes the loss. Momentum can be 
    added to reduce oscillations and improve convergence speed.

    Attributes
    ----------
    _momentum : float
        The momentum factor to smooth updates and speed up convergence.
    _changes : list[tuple[np.ndarray, np.ndarray]]
        Stores previous weight and bias changes to apply momentum.

    Methods
    -------
    update(gradients)
        Updates the weights and biases using gradient descent with momentum.
    """
    def __init__(self, learning_rate:float, nn:object, momentum:float=0.0) -> None:
        super().__init__(learning_rate, nn)
        self._momentum = momentum
        self._changes = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
    
    def update(self, gradients):
        layers = self._nn.layers
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(layers, gradients)):
            change_W = self._learning_rate * W_g + self._momentum * self._changes[i][0]
            change_b = self._learning_rate * b_g + self._momentum * self._changes[i][1]
            self._changes[i] = (change_W, change_b)
            new_W = W - change_W
            new_b = b - change_b
            self._nn.update_layer((new_W, new_b), i)

class AdaGrad(Optimization):
    """
    Implements the AdaGrad optimization algorithm.

    AdaGrad adapts the learning rate for each parameter based on the sum 
    of squared gradients, allowing larger steps for infrequently updated parameters.
    This can be useful in cases where features have varying frequencies.

    Attributes
    ----------
    _r : list[tuple[np.ndarray, np.ndarray]]
        Cumulative sum of squared gradients for weights and biases.
    _delta : float
        Small constant added to prevent division by zero.

    Methods
    -------
    update(gradients)
        Updates the weights and biases using AdaGrad.
    """
    def __init__(self, learning_rate:float, nn:object, delta:float=1e-8) -> None:
        super().__init__(learning_rate, nn)
        self._r = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        self._delta = delta
    
    def update(self, gradients) -> None:
        layers = self._nn.layers
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(layers, gradients)):
            W_g /= self._batch_size
            b_g /= self._batch_size
            self._r[i] = (W_g*W_g, b_g*b_g)
            new_W = W-(self._learning_rate/(self._delta + np.sqrt(self._r[i][0]))) * W_g
            new_b = b-(self._learning_rate/(self._delta + np.sqrt(self._r[i][1]))) * b_g
            self._nn.update_layer((new_W, new_b), i)

class RMSProp(Optimization):
    """
    Implements the RMSProp optimization algorithm.

    RMSProp maintains an exponentially decaying average of squared gradients 
    to adapt the learning rate. This helps in dealing with non-stationary 
    objectives and improves convergence on noisy data.

    Attributes
    ----------
    _rho : float
        Decay rate for moving average of squared gradients.
    _r : list[tuple[np.ndarray, np.ndarray]]
        Exponentially weighted sum of squared gradients for weights and biases.
    _delta : float
        Small constant to avoid division by zero.

    Methods
    -------
    update(gradients)
        Updates the weights and biases using RMSProp.
    """
    def __init__(self, learning_rate:float, nn:object, rho:float, delta:float=1e-7) -> None:
        super().__init__(learning_rate, nn)
        self._rho = rho
        self._r = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        self._delta = delta

    def update(self, gradients) -> None:
        layers = self._nn.layers
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(layers, gradients)):
            W_g /= self._batch_size
            b_g /= self._batch_size
            self._r[i] = (self._rho*self._r[i][0] + (1-self._rho)* W_g*W_g, self._rho*self._r[i][1] + (1-self._rho)* b_g*b_g)
            new_W = W-(self._learning_rate/(self._delta + np.sqrt(self._r[i][0]))) * W_g
            new_b = b-(self._learning_rate/(self._delta + np.sqrt(self._r[i][1]))) * b_g
            self._nn.update_layer((new_W, new_b), i)

class Adam(Optimization):
    """
    Implements the Adam optimization algorithm.

    Adam combines the benefits of RMSProp and momentum by computing adaptive 
    learning rates for each parameter and using momentum to smooth updates.
    It also uses bias correction to prevent initial step size bias.

    Attributes
    ----------
    _rho1 : float
        Decay rate for the first moment estimate.
    _rho2 : float
        Decay rate for the second moment estimate.
    _r : list[tuple[np.ndarray, np.ndarray]]
        Exponentially weighted sum of squared gradients.
    _s : list[tuple[np.ndarray, np.ndarray]]
        Exponentially weighted sum of gradients.
    _delta : float
        Small constant to avoid division by zero.
    _t : int
        Timestep used to adjust for bias in moment estimates.

    Methods
    -------
    update(gradients)
        Updates the weights and biases using Adam.
    """
    def __init__(self, learning_rate:float, nn:object, rho1:float, rho2:float, delta:float=1e-8) -> None:
        super().__init__(learning_rate, nn)
        self._rho1 = rho1
        self._rho2 = rho2
        self._r = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        self._s = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        self._delta = delta
        self._t = 1
    
    def update(self, gradients) -> None:
        layers = self._nn.layers
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(layers, gradients)):
            W_g /= self._batch_size
            b_g /= self._batch_size

            self._s[i] = (self._rho1*self._s[i][0] + (1-self._rho1)* W_g, 
                          self._rho1*self._r[i][1] + (1-self._rho1)* b_g)
            self._r[i] = (self._rho2*self._r[i][0] + (1-self._rho2)* W_g*W_g, 
                          self._rho2*self._r[i][1] + (1-self._rho2)* b_g*b_g)

            s_W, s_b = self._s[i][0]/(1-self._rho1**self._t), \
                        self._s[i][1]/(1-self._rho1**self._t)
            r_W, r_b = self._r[i][0]/(1-self._rho2**self._t), \
                        self._r[i][1]/(1-self._rho2**self._t)

            new_W = W - (self._learning_rate*s_W)/(np.sqrt(r_W)+self._delta)
            new_b = b - (self._learning_rate*s_b)/(np.sqrt(r_b)+self._delta)

            self._nn.update_layer((new_W, new_b), i)