import numpy as np

def MSE(predictions, data):
    """
    Calculates the mean squared error

    Parameters
    ----------
    predictions : 1d-array
        predicted value(s)
    
    data : 1d-array
        true data
    
    Returns
    -------
    mse : int
        the mean squared error

    """
    n = np.size(predictions) 
    mse = np.sum((data - predictions)**2)/n
    return mse

def R2(predictions, data):
    """
    Calculates the R2 score

    Parameters
    ----------
    predictions : 1d-array
        predicted value(s)
    
    data : 1d-array
        true data

    Returns
    -------
    r2_score : int
        the r2 score 
    """
    ss_res = np.sum((data - predictions)**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r2_score = 1 - (ss_res/ss_tot)
    return r2_score