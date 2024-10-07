import numpy as np

def create_design_matrix(x, y, degree, nxm, intercept):
    """
    Return the desing matrix of a two dimensional polynomial up to a given degree. 
    
    Parameters
    ----------
    x : 1d/2d-array 
        x-values, can either be meshgridded or flat-meshgridded

    y : 1d/2d-array
        y-values, can either be meshgridded or flat-meshgridded
    
    degree : int
        Defines the polynomial degree of the function we want to use a model. 

    nxm : int
        Number of rows in the design matrix, number of z-values

    intercept : Boolean
        True assumes we want to calculate intercept without the design matrix (There so be no ones column)
        False assumes we want to calculate intercept with the design matrix (There is a ones column)

    Returns
    -------
    X : 2d-array (Matrix)
        The design matrix
    
    """
    num_betas = int((degree+1)*(degree+2)/2)
    X = np.zeros((nxm, num_betas))

    idx = 0
    for i in range(degree+1):
        for j in range(degree+1): 
            if i+j <= degree:
                entry = (x**i * y**j)
                X[:, idx] = entry.ravel() # makes us able to send in both matrix and vector
                idx += 1
    if intercept:
        return X[:, 1:]
    
    return X

def scaling(X, intercept):
    """
    Scales the design matrix. 
    Takes the mean of each column and substract it from each entry of the associated column. 

    Parameters
    ----------
    X : 2d-array (Matrix)
        The design matrix

    intercept : Boolean
        true assumes we want to calculate intercept without the design matrix (there is no ones column)
        false assumes we want to calculate intercept with the design matrix (there is a ones column)
    
    Returns
    -------
    X_scaled : 2d-array (Matrix) 
        A scaled design matrix

    mean : 1d-array    
        Mean of every column given as a vector (The first entry is 0 for later convenience)

    """ 

    mean = np.mean(X, axis=0)
    X_scaled = X - mean
    
    if intercept: 
        mean = np.concatenate(([0], mean), axis=0)
    else:
        mean[0] = 0
        X_scaled[:, 0] = 1
        
    return X_scaled, mean

def calculate_intercept(X, z, betas):
    """
    Calculates the intercept for a given model, when it is fitted without a column of 1s. Excecutes when intercept_bool = True.

    Parameters
    ----------
    X : 2d-array (Matrix)
        The design matrix

    z : 1d-array (Flat vector)
        The training data for z

    betas : 1d-array
        The coefficients for a given model

    Returns
    -------
    intercept : int
        The intercept for a given model 
    """
    intercept = np.mean(z) - np.mean(X,axis=0).T @ betas
    return intercept