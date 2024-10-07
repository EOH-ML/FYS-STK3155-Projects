import numpy as np
from sklearn.linear_model import  Lasso
from preprocessing import calculate_intercept

def analytic_regression_model(z, X, reg_type, lmd, intercept, svd=True):
    """
    Finds the optimal coefficients for a Ridge or OLS regression models. 
    
    Parameters
    ----------
    z : 1d-array 
        Training data, the data points that we try to approximate

    X : 2d-array (Matrix)
        The design matrix 

    reg_type : str
        Either OLS or Ridge, depending on what regression type is preferred 

    lmd : float
        Only used for ridge regression, the regularization term.

    intercept : Boolean
        true assumes we want to calculate intercept without the design matrix (there is no 1s column)
        false assumes we want to calculate intercept with the design matrix (there is a 1s column)
    
    svd : Boolean
        determines wether SVD is used to calculate the analytic expression

    Returns
    -------
    betas : 1d-array 
        The predicted coefficents/weights for the polynomial. Return the right amount of coefficients, given an intercept_bool.
    
    """

    if reg_type == "OLS":
        if not svd:
            betas = np.linalg.inv(X.T @ X) @ X.T @ z.ravel() 
        else:
            U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
            Sigma_inv = np.diag(1/Sigma)

            # Calculate OLS coefficients using SVD
            betas = Vt.T @ Sigma_inv @ U.T @ z 
        
    elif reg_type == "Ridge":
        if not svd:
            I = np.identity(X.shape[1])
            betas = np.linalg.inv(X.T @ X + lmd*I) @ X.T @ z.ravel() 

        else:
        # Perform SVD

            U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
            # Ridge regression formula using SVD

            Sigma_ridge = Sigma / (Sigma**2 + lmd)
            betas = Vt.T @ np.diag(Sigma_ridge) @ U.T @ z
    
    if intercept: 
        beta_0 = calculate_intercept(X, z, betas)
        betas = np.concatenate(([beta_0], betas), axis=0)

    return betas

def lasso_regression(z, X, lmd, intercept):
    """
    Finds the optimal coefficients for Lasso regression. 

    Parameters
    ----------
    z : 1d-array 
        Training data, the data points that we try to approximate

    X : 2d-array (Matrix)
        The design matrix
    
    lmd : float
        Only used for ridge regression, the regularization term.

    intercept : Boolean
        True assumes we want to calculate intercept without the design matrix (there is no ones column)
        False assumes we want to calculate intercept with the design matrix (there is a ones column)
    
    Returns
    -------
    betas : 1d-array
        The predicted coefficents/weights for the polynomial. Return the right amount of coefficients, given an intercept_bool.
    """
    lasso = Lasso(alpha=lmd, fit_intercept=intercept) # default True
     
    lasso.fit(X, z)
    betas = lasso.coef_

    if intercept:
        return np.concatenate(([lasso.intercept_], betas), axis=0)
    
    return betas