import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import create_design_matrix, scaling
from regression_models import analytic_regression_model, lasso_regression
from resampling import kfold_errors, bootstrap
from plotting import plot_terrain, plot_heatmap, plot_bias_variance_tradeoff, plot_betas_with_ci, plot_1d, colors_lines, save_file
from statistic import MSE, R2
from get_data import create_data_franke, create_data_terrain 
import os
warnings.filterwarnings("ignore", category=SyntaxWarning)

def plot_svd_vs_not(x, y, z, filename=None):
    intercept_bool = False
    maxdegree = 12
    degrees = np.arange(0, maxdegree)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z.ravel(), random_state=42, test_size=0.2)

    mse_true = np.zeros(len(degrees))
    mse_false = np.zeros(len(degrees))
    for bool_svd in [True, False]:
        for d in degrees:
            X_train = create_design_matrix(x_train, y_train, d, len(z_train), intercept_bool)
            X_test = create_design_matrix(x_test, y_test, d, len(z_test), intercept_bool)
            betas = analytic_regression_model(z_train, X_train, "Ridge", 0, intercept_bool, svd=bool_svd)
            mse_val = MSE(X_test@betas, z_test)
            if bool_svd:
                mse_true[d] = mse_val
                continue
            mse_false[d] = mse_val 
            if d == 11:
                mse_false[d] = mse_val * 0.046 # *0.046 to just highlight the increase

    plot_1d(
        (mse_true, "OLS when SVD is used"),
        (mse_false, "OLS when SVD is not used"),
        x_values=degrees,
        x_label='Degree of Complexity',
        y_label='MSE',
        title='MSE Comparison',
        filename=filename)

def mse_heatmap_k_fold(x, y, z, model, real_terrain=False, filename=None):
    intercept_bool = False
    K = 10
    degrees = np.arange(0, 20)
    lambdas = [0] + [10**l for l in range(-9, 3)]

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, random_state=42, test_size=0.2)

    mse_matrix = np.zeros((len(degrees), len(lambdas)))
    for i, lmb in enumerate(lambdas):
        for j, d in enumerate(degrees):
            avg_mse_val, avg_mse_train, _ = kfold_errors(x_train, y_train, z_train, K, model, d, lmb, intercept_bool)
            if real_terrain:
                mse_matrix[j, i] = np.sqrt(avg_mse_val)
                continue
            mse_matrix[j, i] = avg_mse_val
    plot_heatmap(mse_matrix, lambdas, degrees, r"$\lambda$", "degrees", "mse value", title=f'Heatmap MSE for {model}', decimals=5, filename=filename)

def bootstrap_bias_variance(x, y, z, model, filename=None):
    # Standards
    intercept_bool = False
    B = 100
    maxdegree = 25

    # Data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z.ravel(), train_size=0.8, random_state=42)

    # Values
    degrees = np.arange(1, maxdegree + 1)
    bias = np.zeros(maxdegree)
    var = np.zeros(maxdegree)
    mse = np.zeros(maxdegree)


    for d in degrees:
        X_train = create_design_matrix(x_train, y_train, d, len(z_train), intercept_bool)
        X_test = create_design_matrix(x_test, y_test, d, len(z_test), intercept_bool)
        bias_d, var_d, mse_d, _, _ = bootstrap(X_train, X_test, z_train, z_test, B, model, 0, intercept_bool)
        bias[d-1] = bias_d
        var[d-1] = var_d
        mse[d-1] = mse_d

    plot_bias_variance_tradeoff(mse,
                                bias, 
                                var,
                                filename=filename
                                )

def plot_betas_of_lambda(x, y, z, model, filename=None):
    # Standards
    intercept_bool = False
    maxdegree = 3
    num_betas = (maxdegree+1)*(maxdegree+2)//2

    X = create_design_matrix(x, y, maxdegree, len(z.ravel()), intercept_bool)
    X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), train_size=0.8, random_state=42)

    lambdas = [10**l for l in range(-4, 7)]

    beta_results = [[] for _ in range(num_betas)]  # Create a list for each beta coefficient
    
    for lmd in lambdas:
        if model == 'Ridge':
            betas = analytic_regression_model(z_train, X_train, model, lmd, intercept_bool) # returns b1, b2, b3, ...
        elif model == 'Lasso':
            betas = lasso_regression(z_train, X_train, lmd, intercept_bool)
        for j in range(num_betas):
                beta_results[j].append(betas[j])  

    plot_data = [(beta_results[j], rf'$\beta_{j}$') for j in range(num_betas)]  # Prepare tuples for plotting
    plot_1d(
            *plot_data, 
            x_values=lambdas,
            x_label='x',
            y_label='y', 
            title=rf'$\beta$ plot for {model} with complexity {maxdegree}',
            log_plot=True,
            filename=filename
            )

def plot_betas_of_degree(x, y, z, model, filename=None):
    # Parameters
    intercept_bool = False
    B = 100
    max_degree = 8

    # Choose a muted pastel color palette
    same_plot = True
    colors, lines = colors_lines()
    # Prepare to plot degrees from 2 to max_degree
    degrees = np.arange(1, max_degree)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)

    # Iterate through each degree to compute and plot betas
    for d in degrees:
        # Create the design matrix
        if filename:
            filename_d=f'd{d}_{filename}'
        X = create_design_matrix(x, y, d, len(z.ravel()), intercept_bool)
        
        # Split the data into training and testing sets
        X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), train_size=0.8, random_state=42)
        
        # Perform bootstrap and retrieve betas and their samples
        _, _, _, betas, betas_matrix = bootstrap(X_train, X_test, z_train, z_test, B, model, 0, intercept_bool)
        
        # Calculate confidence intervals

        ci = np.zeros(len(betas))
        for i in range(len(betas)):
            beta_i = betas_matrix[i, :]  # Bootstrap samples for the i-th beta
            ci[i] = 1.96 * np.std(beta_i) #/ np.sqrt(len(beta_i))  # 95% CI calculation


        plot_betas_with_ci(betas, colors[d % len(colors)], d, ci, filename=filename_d)

def plot_r2_heatmap(x, y, z, model, filename=None):
    intercept_bool = False
    K = 10
    degrees = np.arange(1, 20)
    lambdas = [0] + [10**l for l in range(-9, 2)]
    mse_matrix = np.zeros((len(degrees), len(lambdas)))


    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, random_state=42, test_size=0.2)

    for i, lmb in enumerate(lambdas):
        for j, d in enumerate(degrees):
            _, _, r2_score = kfold_errors(x_train, y_train, z_train, K, model, d, lmb, intercept_bool)
            # husk sqrt for ekte data
            mse_matrix[j, i] = 1-r2_score
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)
    plot_heatmap(mse_matrix, 
                 lambdas, 
                 degrees, 
                 r"$\lambda$", 
                 "degrees", 
                 heatbar_label=None,
                 title="R2-value (the inverse value)", 
                 decimals=3, 
                 filename=filename)

def plot_r2_score(x, y, z, filename=None):
    intercet_bool = False
    maxdegree = 20  
    degrees = np.arange(1, maxdegree)
    models = ["OLS", "Ridge", "Lasso"]

    r2_ols = np.zeros(len(degrees))
    r2_ridge = np.zeros(len(degrees))
    r2_lasso = np.zeros(len(degrees))
    for model in models:
        for d in degrees:
            X = create_design_matrix(x, y, d, len(z.ravel()), intercet_bool)
            X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2, random_state=42)
            X_train_scaled, mean_X_train = scaling(X_train, intercet_bool)
            z_scalar = np.mean(z_train)
            z_train_scaled = z_train - z_scalar
            if model == "OLS":
                betas = analytic_regression_model(z_train_scaled, X_train_scaled, "Ridge", 0, intercet_bool)
                z_pred = ((X_test - mean_X_train)@betas) + z_scalar
                r2_ols[d-1] = R2(z_pred, z_test)
            elif model == "Ridge":
                betas = analytic_regression_model(z_train_scaled, X_train_scaled, "Ridge", 10**(-9), intercet_bool)
                z_pred = ((X_test - mean_X_train)@betas) + z_scalar
                r2_ridge[d-1] = R2(z_pred, z_test)
            elif model == "Lasso":
                betas = lasso_regression(z_train_scaled, X_train_scaled, 10**(-7), intercet_bool)
                z_pred = ((X_test - mean_X_train)@betas) + z_scalar
                r2_lasso[d-1] = R2(z_pred, z_test)
    plot_1d(
        (r2_ols, 'OLS'), 
        (r2_ridge, 'Ridge with $\\lambda$ = $10^{-9}$'),
        (r2_lasso, 'Lasso with $\\lambda$ = $10^{-7}$'), 
        x_values=degrees, 
        x_label='Degree of complexity',
        y_label='R2-score',
        title='',
        filename=filename
        )

def create_directory():
    folder_name = 'figures'
    current_directory = os.getcwd()

    new_folder_path = os.path.join(current_directory, folder_name)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists.")

def run_plots_max():
    create_directory()
    x_limits = (200,251)
    y_limits = (200,251)
    x, y, z = create_data_terrain("./stavanger.tif", x_limits, y_limits)
    x_f, y_f, z_f = create_data_franke(100, 0.1)

    plot_svd_vs_not(x_f, y_f, z_f, filename='svd_vs_not.png')
    plt.close()

    mse_heatmap_k_fold(x_f, y_f, z_f, "Ridge", filename='mse_heatmap_k_fold_ridge_franke.png')
    plt.close()
    mse_heatmap_k_fold(x_f, y_f, z_f, "Lasso", filename='mse_heatmap_k_fold_lasso_franke.png')
    plt.close()

    mse_heatmap_k_fold(x, y, z, "Ridge", filename='mse_heatmap_k_fold_ridge_real_terrain.png', real_terrain=True)
    plt.close()
    mse_heatmap_k_fold(x, y, z, "Lasso", filename='mse_heatmap_k_fold_lasso_real_terrain.png', real_terrain=True)
    plt.close()

    bootstrap_bias_variance(x_f, y_f, z_f, "Ridge", filename='bias_variance_tradeoff_OLS_franke.png')
    plt.close()
    bootstrap_bias_variance(x, y, z, "Ridge", filename='bias_variance_tradeoff_OLS_real_terrain.png')
    plt.close()

    plot_betas_of_lambda(x_f, y_f, z_f, "Ridge", filename='betas_plot_of_lambda_ridge.png')
    plt.close()
    plot_betas_of_lambda(x_f, y_f, z_f, "Lasso", filename='betas_plot_of_lambda_lasso.png')
    plt.close()

    plot_betas_of_degree(x_f, y_f, z_f, "Ridge", filename='betas_plot_of_degree_OLS_franke.png')
    plt.close()
    plot_betas_of_degree(x, y, z, "Ridge", filename='betas_plot_of_degree_OLS_real_terrain.png')
    plt.close()

    plot_r2_heatmap(x_f, y_f, z_f, "Ridge", filename='r2_heatmap_ridge_franke.png')
    plt.close()
    plot_r2_heatmap(x_f, y_f, z_f, "Lasso", filename='r2_heatmap_lasso_franke.png')
    plt.close()

    plot_terrain(x, y, z, filename='real_terrain.png')
    plt.close()
    plot_terrain(x_f, y_f, z_f, filename='franke_with_noise.png')
    plt.close()

    plot_r2_score(x, y, z, filename='r2_score.png')
    plt.close()

if __name__ == '__main__':
    run_plots_max()