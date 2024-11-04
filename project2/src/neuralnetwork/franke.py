import numpy as np
from imageio import imread

def franke_function(x, y):
    """
    Computes the Franke function value at the given coordinates (x, y).

    Parameters:
    -----------
    x : ndarray
        The x-coordinates of the input data.
    y : ndarray
        The y-coordinates of the input data.

    Returns:
    --------
    ndarray
        The computed values of the Franke function at the given coordinates (x, y).
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_data_franke(n, noise):
    """
    Generates noisy data from the Franke function.

    Parameters:
    -----------
    n : int
        The number of grid points along each axis (n x n grid).
    noise : float
        The standard deviation of the Gaussian noise to be added to the function values.

    Returns:
    --------
    tuple of ndarrays
        (x, y, z) where:
        - x is the grid of x-coordinates.
        - y is the grid of y-coordinates.
        - z is the noisy Franke function values at the (x, y) points.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = franke_function(x, y) + noise * np.random.randn(n, n)
    return x, y, z

