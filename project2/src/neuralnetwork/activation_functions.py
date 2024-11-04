import numpy as np
from scipy.special import softmax as smx

def ReLU(z):
    return np.maximum(0, z)

def leakyReLU(z, negative_slope=0.01):
    return np.maximum(0, z) + negative_slope*np.minimum(0, z) 

def leakyReLU_der(z, negative_slope=0.01):
    return (z > 0) + (z < 0) * negative_slope

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def softmax(z):
    return smx(z, axis=1)

def softmax_der(z):
    """
    Due to the simple expression for dC/dz in the training of the network, 
    this is appropriate for cross entropy in combination with softmax.
    """
    return 1

def ReLU_der(z):
    return z > 0

def sigmoid_der(z):
    return sigmoid(z) * (1-sigmoid(z))

def linear(z):
    return z

def linear_der(z):
    z_der = np.ones(z.shape)
    return z_der

def activation_functions_derived(activation_funcs):
    activation_funcs_der = []
    for func in activation_funcs:
        if func == ReLU:
            activation_funcs_der.append(ReLU_der)
        elif func == sigmoid:
            activation_funcs_der.append(sigmoid_der)
        elif func == softmax:
            activation_funcs_der.append(softmax_der)
        elif func == linear:
            activation_funcs_der.append(linear_der)
        elif func == leakyReLU:
            activation_funcs_der.append(leakyReLU_der)
    return activation_funcs_der
