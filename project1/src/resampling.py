import numpy as np
from sklearn.utils import shuffle, resample
from preprocessing import create_design_matrix, scaling
from regression_models import analytic_regression_model, lasso_regression
from statistic import MSE, R2

def k_crossval(x_train, y_train, z_train, k):
    """
    Returns a list of tuples with k crossvalidated validation and training sets for x, y, z

    Parameters
    ----------
    x : 1d-array
        raveled x-points
    y : 1d-array
        raveled y-points
    z : 1d-array
        raveled z-points

    k : int
        number of folds (>1)
    
    Returns
    -------
    train_sets : list
        list of k tuples x, y, z for each train_set
    val_sets : list
        list of k tuples x, y, z for each val_set

    """ 
    
    assert k > 1, f"Error! k must be > 1 (was {k})"
    assert x_train.size == y_train.size == z_train.size, f"Error! x, y, z not of same size ({x_train.size}, {y_train.size}, {z_train.size})"

    x_shuffle, y_shuffle, z_shuffle = shuffle(x_train, y_train, z_train, random_state=42) 

    n = x_shuffle.size
    val_size = n // k 

    train_sets = []
    val_sets = []

    for i in range(k):
        start_idx = i * val_size
        end_idx = (i + 1) * val_size  

        val_indices = np.arange(start_idx, end_idx)
        train_indices = np.delete(np.arange(n), val_indices) 

        x_train_shuffle, y_train_shuffle, z_train_shuffle = x_shuffle[train_indices], y_shuffle[train_indices], z_shuffle[train_indices]
        x_val_shuffle, y_val_shuffle, z_val_shuffle = x_shuffle[val_indices], y_shuffle[val_indices], z_shuffle[val_indices]

        train_sets.append((x_train_shuffle, y_train_shuffle, z_train_shuffle)) 
        val_sets.append((x_val_shuffle, y_val_shuffle, z_val_shuffle))

    return train_sets, val_sets  


def kfold_errors(x, y, z, K, model, degree, lmb, intercept, svd=True):
    """
    Returns an average mse for validation and training set, given a model, degree, lambda and intercept,
    using the k-fold cross validation method.

    Parameters
    ----------
    x : 2d-array
        meshed array of x-points

    y : 2d-array
        meshed array of y-points

    z : 2d-array
        meshed array of z-points

    K : int
        number of folds (>1)

    model : str
        name of model (typically 'OLS', 'Ridge', 'Lasso')

    degree : int
        complexity of model

    lmb : float
        lambda regularization term for Ridge/Lasso
        
    intercept : Boolean
        true assumes we want to calculate intercept without the design matrix (there is no ones column)
        false assumes we want to calculate intercept with the design matrix (there is a ones column) 
    
    Returns
    -------
    avg_mse_val : float
        average mse for k folds validation set
    
    avg_mse_train : float
        average mse for k folds training set
        
    avg_r2_score : float
        average r2 score for k folds training set
    """

    mse_val_results, mse_train_results = [], []
    r2_score = []

    x_flat, y_flat, z_flat = x.ravel(), y.ravel(), z.ravel()
    
    train_sets, val_sets = k_crossval(x_flat, y_flat, z_flat, K)
    
    for k in range(K):
        x_train, y_train, z_train = train_sets[k]
        x_val, y_val, z_val = val_sets[k]
    
        X_train = create_design_matrix(x_train, y_train, degree, len(x_train), intercept)
        X_val = create_design_matrix(x_val, y_val, degree, len(x_val), intercept)
    
        X_train_scaled, mean_X_train = scaling(X_train, intercept)
    
        mean_z_train = np.mean(z_train)
        z_train_scaled = z_train - mean_z_train
        z_val_scaled = z_val - mean_z_train

        if model == 'OLS' or model == 'Ridge':
            betas = analytic_regression_model(z_train_scaled, X_train_scaled, model, lmb, intercept, svd)
        elif model == 'Lasso':
            betas = lasso_regression(z_train_scaled, X_train_scaled, lmb, intercept)   
    
        if intercept:
            ones_train = np.ones((X_train_scaled.shape[0], 1))  # Create a column of ones with the same number of rows as X
            ones_val = np.ones((X_val.shape[0], 1))  # Create a column of ones with the same number of rows as X
            X_train_scaled = np.hstack([ones_train, X_train_scaled])
            X_val = np.hstack([ones_val, X_val])
        
        X_val_scaled = X_val - mean_X_train
        
        z_pred_train = (X_train_scaled @ betas) + mean_z_train
        z_pred_val = (X_val_scaled @ betas) + mean_z_train
    
        mse_val_results.append(MSE(z_val, z_pred_val))
        mse_train_results.append(MSE(z_train, z_pred_train))
        r2_score.append(R2(z_pred_val, z_val))
    
    avg_mse_val = np.mean(mse_val_results)
    avg_mse_train = np.mean(mse_train_results)
    avg_r2_score = np.mean(r2_score)
    
    return avg_mse_val, avg_mse_train, avg_r2_score

def bootstrap(X_train, X_test, z_train, z_test, B, reg_type, lmd, intercept):
    """
    Performs bootstrap resampling to estimate the bias, variance, and mean squared error (MSE) of a regression model.
    
    Parameters:
    -----------
    X_train : ndarray
        The feature matrix for the training data (n_samples_train x n_features).
    X_test : ndarray
        The feature matrix for the test data (n_samples_test x n_features).
    z_train : ndarray
        The target values for the training data (n_samples_train,).
    z_test : ndarray
        The true target values for the test data (n_samples_test,).
    B : int
        The number of bootstrap resamples to perform.
    reg_type : str
        The type of regression model to use. Must be one of 'OLS', 'Ridge', or 'Lasso'.
    lmd : float
        The regularization parameter (used for Ridge and Lasso regression).
    intercept : bool
        If True, includes an intercept term in the regression model.

    Returns:
    --------
    bias : float
        The squared bias of the predictions.
    variance : float
        The variance of the predictions.
    mse : float
        The mean squared error (MSE) of the predictions.
    betas_estimated : ndarray
        The mean of the estimated regression coefficients across all bootstrap samples.
    beta_matrix : ndarray
        The estimated regression coefficients for each bootstrap sample (n_features x B).
    """

    # all predicted z values
    z_pred = np.zeros((len(z_test), B))

    # the collected betas
    if intercept: 
        beta_matrix = np.zeros((X_train.shape[1] + 1, B))
    else:
        beta_matrix = np.zeros((X_train.shape[1], B))

    # scaling
    z_train_mean = np.mean(z_train)
    z_train_scaled = z_train - z_train_mean
    X_train_scaled, X_train_mean = scaling(X_train, intercept)
    if intercept:
        ones_val = np.ones((X_test.shape[0], 1))  # Create a column of ones with the same number of rows as X
        X_test = np.hstack([ones_val, X_test])
    X_test_scaled = X_test - X_train_mean

    # the bootstrap
    for b in range(B):
        X_train_, z_train_ = resample(X_train_scaled, z_train_scaled)
        if reg_type in ["OLS", "Ridge"]:
            betas = analytic_regression_model(z_train_, X_train_, reg_type, lmd, intercept)
        elif reg_type == "Lasso":
            betas = lasso_regression(z_train_, X_train_, lmd, intercept)
        z_pred[:, b] = (X_test_scaled@betas) + z_train_mean
        beta_matrix[:, b] = betas

    # bias, variance, error
    mean_z_pred = np.mean(z_pred, axis=1)  # Mean prediction across all bootstrap samples
    bias = np.mean((mean_z_pred - z_test) ** 2)  # Bias calculation
    variance = np.mean(np.var(z_pred, axis=1))  # Variance of predictions
    mse = np.mean((z_pred - z_test.reshape(-1, 1))**2)

    # all the estimated betas
    betas_estimated = np.mean(beta_matrix, axis=1)

    return bias, variance, mse, betas_estimated, beta_matrix


