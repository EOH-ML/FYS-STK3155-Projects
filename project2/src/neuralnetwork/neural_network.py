from optimization import Optimization, GradientDescent, AdaGrad, RMSProp, Adam
import numpy as np
from activation_functions import softmax, sigmoid, activation_functions_derived
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from plotting import Plotting

class NeuralNetwork:
    """
    A class for building, training, and evaluating a simple feedforward neural network.

    This class allows the construction of a neural network with a customizable number of layers, 
    layer sizes, activation functions, and a choice of loss function. It includes methods for 
    feedforward, backpropagation, and optimization to support training, as well as utilities 
    for evaluating loss over training and validation sets.

    Attributes
    ----------
    _input_size : int
        The number of input features.
    _layer_sizes : list[int]
        A list specifying the size of each layer in the network.
    _activation_funcs : list[Callable]
        A list of activation functions applied to each layer.
    _layers : list[tuple[np.ndarray, np.ndarray]]
        Weights and biases for each layer, initialized during instantiation.
    _activation_funcs_der : list[Callable]
        Derivatives of the activation functions, used during backpropagation.
    _loss : str
        The loss function to be used ('mse', 'cross entropy', or 'binary cross entropy').
    _epoch_values : list[int]
        A record of epoch numbers, used for tracking training progress.
    _loss_val_values : list[float]
        Validation loss values recorded at each epoch.
    _loss_train_values : list[float]
        Training loss values recorded at each epoch.

    Methods
    -------
    layers -> list
        Returns the list of layers in the network.
    update_layer(new_layer: list, index: int) -> None
        Updates a specific layer in the network with new weights and biases.
    _loss_func_der(predict: np.ndarray, target: np.ndarray) -> np.ndarray
        Computes the derivative of the loss function with respect to predictions.
    _loss_func(predict: np.ndarray, target: np.ndarray) -> float
        Computes the loss between predictions and target values.
    _train_feed_forward(x: np.ndarray) -> tuple[np.ndarray, list, list]
        Performs a forward pass during training, storing intermediate values for backpropagation.
    train(data: np.ndarray, target: np.ndarray, epochs: int, batch_size: int, optimizer: Optimization, data_val: np.ndarray=None, target_val: np.ndarray=None, lmd: float=0.0) -> None
        Trains the neural network using the specified optimizer, batch size, and regularization.
    _get_gradients(data: np.ndarray, target: np.ndarray, lmd: float) -> list
        Computes the gradients for each layer during backpropagation.
    feed_forward(x: np.ndarray) -> np.ndarray
        Performs a forward pass and outputs the network's prediction for given input.
    show_loss_function(filename: str=None, show: bool=False, title: str='', box_string: str=None) -> None
        Plots and optionally saves the loss curves for training and validation sets.
    """

    def __init__(self, input_size:int, layer_sizes:list[int], activation_funcs:list, loss:str) -> None:
        if len(layer_sizes) == 0:
            raise ValueError('Layer_sizes must contain at least one value.')

        if len(layer_sizes) != len(activation_funcs):
            raise ValueError(f"Number of elements in layer_sizes and activation_funcs are inconsistend: len(layer_sizes)={len(layer_sizes)}, len(activation_funcs)={len(activation_funcs)}")
        
        if loss not in ['cross entropy', 'binary cross entropy', 'mse']:
            raise ValueError(f"The provided loss function is not supported: {loss}. Try 'mse', 'cross entropy' or 'binary cross entropy'.")
        
        if (activation_funcs[-1] == softmax) ^ (loss == 'cross entropy'):
            raise ValueError("Softmax in the output layer is mutually dependent on cross entropy as loss function.")

        self._input_size = input_size
        self._layer_sizes = layer_sizes
        self._activation_funcs = activation_funcs
        self._layers = self._create_layers()
        self._activation_funcs_der = activation_functions_derived(activation_funcs)
        self._loss = loss
        self._epoch_values = []
        self._loss_val_values = []
        self._loss_train_values = []
    
    def _create_layers(self) -> list:
        layers = []
        i_size = self._input_size
        for layer_size in self._layer_sizes:
            W = np.random.randn(i_size, layer_size)
            b = np.random.randn(layer_size)
            layers.append((W, b))
            i_size = layer_size
        return layers

    @property 
    def layers(self) -> list:
        return self._layers

    def update_layer(self, new_layer:list, index:int) -> None: 
        self._layers[index] = new_layer
    
    def _loss_func_der(self, predict:np.ndarray, target:np.ndarray) -> np.ndarray: 
        if self._loss == 'mse':
            return 2/predict.size*(predict-target)
        elif self._loss == 'cross entropy':
            """
            Due to the simple expression for dC/dz in the training of the network, 
            this is appropriate for cross entropy in combination with softmax.
            """
            return predict - target
    
    def _loss_func(self, predict:np.ndarray, target:np.ndarray) -> float: 
        if self._loss == 'mse':
            return np.mean((predict-target)**2)
        elif self._loss == 'cross entropy':
            one_hot_predictions = np.zeros_like(target)
            for i, prediction in enumerate(predict):
                one_hot_predictions[i, np.argmax(prediction)] = 1
            return accuracy_score(one_hot_predictions, target)

    def _train_feed_forward(self, x:np.ndarray) -> tuple[np.ndarray, list, list]:
        layer_inputs = []
        zs = []
        a = x
        for (W, b), activation_func in zip(self._layers, self._activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return a, layer_inputs, zs
    
    def train(self, data:np.ndarray, target:np.ndarray,  epochs:int, batch_size:int, optimizer:Optimization, data_val:np.ndarray=None, target_val:np.ndarray=None, lmd:float=0.0) -> None:
        if data.shape[0] != target.shape[0]:
            raise ValueError(f"Prediction and target must have the same shape. Prediction shape: {data.shape}, target shape: {target.shape}")
        n_samples = data.shape[0]
        n_batches = n_samples//batch_size
        optimizer.set_batch_size(batch_size)

        for e in range(epochs):
            shuffled_indicies = np.random.permutation(n_samples) 
            for m in range(n_batches):
                indices = shuffled_indicies[m * batch_size : (m+1) * batch_size] # [2, 4, 1, 6, 5, 3, 0] [2, 4] [1, 6]
                X_i = data[indices]
                true_i = target[indices] 

                gradients = self._get_gradients(X_i, true_i, lmd)
                optimizer.update(gradients)
            # if e % 1 == 0 and data_val is not None:
            if data_val is not None:
                predict_val = self.feed_forward(data_val)
                predict_train = self.feed_forward(data)
                loss_val = self._loss_func(predict_val, target_val)
                loss_train = self._loss_func(predict_train, target)
                # print(f'Epoch: {e}, loss validation set: {loss_val}')
                self._loss_val_values.append(loss_val)
                self._loss_train_values.append(loss_train)
                self._epoch_values.append(e)
    
    def _get_gradients(self, data:np.ndarray, target:np.ndarray, lmd:float) -> list:
        predict, layer_inputs, zs = self._train_feed_forward(data)

        if predict.shape != target.shape:
            raise ValueError(f"Prediction and target must have the same shape. Prediction shape: {predict.shape}, target shape: {target.shape}")

        gradient_per_layer = []
        for i in reversed(range(len(self._layers))):
            layer_input, z, activation_func_der = layer_inputs[i], zs[i], self._activation_funcs_der[i]

            if i == len(self._layers) - 1:
                dC_da = self._loss_func_der(predict, target)

            else:
                (W, _) = self._layers[i+1]
                dC_da = np.dot(dC_dz, W.T) # dette trenger en forklaring

            dC_dz = dC_da * activation_func_der(z)
            dC_dW = np.dot(layer_input.T, dC_dz) + lmd * self._layers[i][0]# dette trenger en forklaring
            dC_db = np.sum(dC_dz, axis=0)
            gradient_per_layer.append((dC_dW, dC_db))
        return reversed(gradient_per_layer)

    def feed_forward(self, x:np.ndarray) -> np.ndarray:
        a = x
        for (W, b), activation_func in zip(self._layers, self._activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def show_loss_function(self, filename:str=None, show:bool=False, title:str='', box_string:str=None) -> None:
        Plotting().plot_1d((self._loss_val_values, 'Validation set'),
                           (self._loss_train_values, 'Train set'),
                           x_values=self._epoch_values,
                           x_label='Number of epochs',
                           title=title,
                           y_label=f'Loss from {self._loss}',
                           filename=filename,
                           box_string=box_string
                            )
        if show: plt.show()