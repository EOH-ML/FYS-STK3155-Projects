import numpy as np
from optimization_algorithms import Optimization
from preprocessor_subclasses import Preprocessor, PolynomialPreprocessor
from cost_functions import ols, ridge, lasso
import matplotlib.pyplot as plt
import os
from plotting import Plotting

def create_folder_in_current_directory(folder_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    new_folder_path = os.path.join(current_directory, folder_name)
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully at: {new_folder_path}")
    else:
        print(f"Folder '{folder_name}' already exists at: {new_folder_path}")

def mse(true, pred):
    return np.mean((true - pred)**2)

def GD_eta_momentum(X, y_true, cost, alpha, title, filename):
    momentums = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    etas = [0.00001, 0.0001, 0.001, 0.005, 0.01]

    heatmap = np.zeros((len(momentums), len(etas)))

    for i, momentum in enumerate(momentums):
        for j, eta in enumerate(etas):
            optimization = Optimization(X, y_true, init_guess, batch_size=1, epochs=1, alpha=alpha)
            y_pred_SGD = X @ optimization.gradient_descent_plain(cost=cost, eta=eta, momentum=momentum, max_iterations=100)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=etas,
                       y_data=momentums,
                       x_label=r'$\eta$',
                       y_label='Momentum',
                       title=title,
                       decimals=4,
                       filename=filename)
    plt.close()


def SGD_epochs_batch_size(X, y_true, cost, alpha, title, filename, momentum):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    epochs_sizes = [1, 10, 20, 30, 40, 50]

    heatmap = np.zeros((len(batch_sizes), len(epochs_sizes)))

    for i, batch in enumerate(batch_sizes):
        for j, epoch in enumerate(epochs_sizes):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=epoch, alpha=alpha)
            y_pred_SGD = X @ optimization.SGD_momentum(cost = cost,
                                                        t0 = 5,
                                                        t1 = 50,
                                                        momentum = momentum)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epochs_sizes,
                       y_data=batch_sizes,
                       x_label='epoch sizes',
                       y_label='batch sizes',
                       title=title,
                       decimals=4,
                       filename=filename)
    plt.close()

def SGD_momentum_batch_size(X, y_true, cost, alpha, title, filename):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    momentum_sizes = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    heatmap = np.zeros((len(batch_sizes), len(momentum_sizes)))

    for i, batch in enumerate(batch_sizes):
        for j, momentum in enumerate(momentum_sizes):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=10, alpha=alpha)
            y_pred_SGD = X @ optimization.SGD_momentum(cost = cost,
                                                        t0 = 5,
                                                        t1 = 50,
                                                        momentum = momentum)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=momentum_sizes,
                       y_data=batch_sizes,
                       x_label='momentum',
                       y_label='batch sizes',
                       title=title,
                       decimals=4,
                       filename=filename)
    plt.close()

def adaGrad_epochs_batch_size(X, y_true, cost, alpha, title, filename, epsilon):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    epochs_sizes = [1, 10, 20, 30, 40, 50]

    heatmap = np.zeros((len(batch_sizes), len(epochs_sizes)))

    for i, batch in enumerate(batch_sizes):
        for j, epoch in enumerate(epochs_sizes):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=epoch, alpha=alpha)
            y_pred_SGD = X @ optimization.adaGrad(cost=cost, epsilon=epsilon)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epochs_sizes,
                       y_data=batch_sizes,
                       x_label='epoch sizes',
                       y_label='batch sizes',
                       title=title,
                       decimals=6,
                       filename=filename)
    plt.close()

def adaGrad_batch_epsilon_size(X, y_true, cost, alpha, title, filename):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    epsilons = [0.1, 0.4, 0.8, 0.9, 0.95, 0.999]

    heatmap = np.zeros((len(batch_sizes), len(epsilons)))

    for i, batch in enumerate(batch_sizes):
        for j, epsilon in enumerate(epsilons):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=20, alpha=alpha)
            y_pred_SGD = X @ optimization.adaGrad(cost=cost, epsilon=epsilon)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epsilons,
                       y_data=batch_sizes,
                       x_label='epsilon',
                       y_label='batch sizes',
                       title=title,
                       decimals=6,
                       filename=filename)
    plt.close()

def adaGrad_alpha_epsilon(X, y_true, cost, title, filename):
    alphas = [10**l for l in range(-4, 1)]
    epsilons = [0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1]

    heatmap = np.zeros((len(alphas), len(epsilons)))

    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            optimization = Optimization(X, y_true, init_guess, batch_size=4, epochs=20, alpha=alpha)
            y_pred_SGD = X @ optimization.adaGrad(cost=cost, epsilon=epsilon)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epsilons,
                       y_data=alphas,
                       x_label='epsilon',
                       y_label='alpha',
                       title=title,
                       decimals=5,
                       filename=filename)
    plt.close()

def RMSProp_batch_epoch_size(X, y_true, cost, title, filename):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    epochs_sizes = [1, 10, 20, 30, 40, 50]

    heatmap = np.zeros((len(batch_sizes), len(epochs_sizes)))

    for i, batch in enumerate(batch_sizes):
        for j, epoch in enumerate(epochs_sizes):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=epoch, alpha=0)
            y_pred_SGD = X @ optimization.RMSProp(cost=cost, epsilon=0.99, rho=0.9)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epochs_sizes,
                       y_data=batch_sizes,
                       x_label='epoch sizes',
                       y_label='batch sizes',
                       title=title,
                       decimals=4,
                       filename=filename)
    plt.close()

def RMSProp_epsilon_rho(X, y_true, cost, title, filename, alpha):
    rhos = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    epsilons = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    heatmap = np.zeros((len(rhos), len(epsilons)))

    for i, rho in enumerate(rhos):
        for j, epsilon in enumerate(epsilons):
            optimization = Optimization(X, y_true, init_guess, batch_size=8, epochs=10, alpha=alpha)
            y_pred_SGD = X @ optimization.RMSProp(cost=cost, epsilon=epsilon, rho=rho)

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epsilons,
                       y_data=rhos,
                       x_label='epsilon',
                       y_label='rho',
                       title=title,
                       decimals=5,
                       filename=filename)
    plt.close()

def ADAM_batch_epoch_size(X, y_true, epsilon, cost, title, filename):
    batch_sizes = [2, 4, 8, 16, 32, 64]
    epochs_sizes = [1, 10, 20, 30, 40, 50]

    heatmap = np.zeros((len(batch_sizes), len(epochs_sizes)))

    for i, batch in enumerate(batch_sizes):
        for j, epoch in enumerate(epochs_sizes):
            optimization = Optimization(X, y_true, init_guess, batch_size=batch, epochs=epoch, alpha=0)
            y_pred_SGD = X @ optimization.Adam(cost=cost, epsilon=epsilon, rho1=0.9, rho2=0.999) # Parameters from Goodfellow

            heatmap[i,j] = mse(y_pred_SGD, y_true)

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=epochs_sizes,
                       y_data=batch_sizes,
                       x_label='epoch sizes',
                       y_label='batch sizes',
                       title=title,
                       decimals=4,
                       filename=filename)
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    create_folder_in_current_directory('../../figures')
    create_folder_in_current_directory('../../figures/figures_GD')
    filepath = '../../figures/figures_GD'

    print("Working on optimizers...")
    
    n = 100
    x = np.linspace(0, 1, n)

    pre = PolynomialPreprocessor(x, degree=2)
    X = pre.get_X()

    y_true = X @ np.array([1, 3, 2])
    X = pre.scaler(X)

    y_pred_analytic = X @ np.linalg.inv(X.T @ X) @ X.T @ y_true
    assert np.isclose(mse(y_true, y_pred_analytic), 0)

    init_guess = np.random.randn(3)
    alphas = [10**l for l in range(-4, 1)]


    # Heatmap for OLS, given momentum and eta, plain GD 
    GD_eta_momentum(X=X, y_true=y_true, cost=ols, alpha=0,
                    title=r'MSE as function of momentum and $\eta$, with plain GD, given OLS, iterations=100',
                    filename=f'{filepath}/MSE_momentum_eta_OLS_GD_iterations100.png')

    # HEATMAP for OLS, given epochs and batch sizes, SGD
    SGD_epochs_batch_size(X=X, y_true=y_true, cost=ols, alpha=0, momentum=0, 
                          title=f'MSE as function of epoch and batch size, given OLS',
                          filename=f'{filepath}/MSE_epoch_batch_OLS_SGD.png')

    # Heatmap for OLS, given momentum and batch sizes, for epoch = 10
    SGD_momentum_batch_size(X=X, y_true=y_true, cost=ols, alpha=0, 
                    title=f'MSE as function of momentum and batch size, given OLS and epoch = 10',
                    filename=f'{filepath}/MSE_momentum_batch_OLS_SGD_epoch10.png')
    
    # Heatmap for OLS, given epochs and batch sizes, for epsilon = 0.99
    adaGrad_epochs_batch_size(X=X, y_true=y_true, cost=ols, alpha=0, epsilon=0.99,
                              title='MSE as function of epoch and batch size for adaGrad, epsilon = 0.99',
                              filename=f'{filepath}/MSE_epoch_batch_OLS_adaGrad_epsilon99e-2.png')  

    # Heatmap for OLS, given epsilon and batch sizes, for epoch = 20, adaGrad
    adaGrad_batch_epsilon_size(X=X, y_true=y_true, cost=ols, alpha=0,
                               title='MSE as function of epsilon and batch size for adaGrad with OLS, epoch = 20',
                               filename=f'{filepath}/MSE_batch_epsilon_OLS_adaGrad_epoch20.png')

    
    # Heatmap for OLS, given epochs and batch sizes, for RMSProp, epsilon = 0.99, rho = 0.9
    RMSProp_batch_epoch_size(X=X, y_true=y_true, cost=ols, 
                             title='MSE as function of epoch and batch size for RMSProp with OLS, epsilon = 0.99, rho = 0.9',
                             filename=f'{filepath}/MSE_batch_epoch_OLS_RMSProp_epsilon99e-2_rho9e-1.png')

    # Heatmap for OLS, given epsilon and rho, for RMSProp, epoch = 10, batch = 8
    RMSProp_epsilon_rho(X=X, y_true=y_true, cost=ols, alpha=0,
                        title='MSE as function of epsilon and rho for RMSProp with OLS, epoch = 10, batch = 8',
                        filename=f'{filepath}/MSE_epsilon_rho_OLS_RMSProp_epoch10_batch8.png')

    # Heatmap for OLS, given epochs and batch sizes, for Adam, epsilon = 0.001, rho1 = 0.9 and rho2 = 0.999
    ADAM_batch_epoch_size(X=X, y_true=y_true, cost=ols, epsilon=0.001,
                          title='MSE as function of epoch and batch size for Adam with OLS, epsilon = 0.001, rho1 = 0.9 and rho2 = 0.999',
                            filename=f'{filepath}/MSE_epoch_batch_OLS_Adam_epsilon_1e-3_rho1_9e-1_rho2_999e-4.png')
    
    # Heatmap for OLS, given epochs and batch sizes, for Adam, epsilon = 0.4, rho1 = 0.9 and rho2 = 0.999
    ADAM_batch_epoch_size(X=X, y_true=y_true, cost=ols, epsilon=0.4,
                          title='MSE as function of epoch and batch size for Adam with OLS, epsilon = 0.4, rho1 = 0.9 and rho2 = 0.999',
                            filename=f'{filepath}/MSE_epoch_batch_OLS_Adam_epsilon_4e-1_rho1_9e-1_rho2_999e-4.png')

     print("Hold tight. Almost there now...")
    
    # Heatmap for Ridge, given epochs/momentum and batch sizes for various alphas, and also GD for eta and momentum
    for alpha in alphas:
        GD_eta_momentum(X=X, y_true=y_true, cost=ridge, alpha=alpha,
                title=rf'MSE as function of momentum and $\eta$, with plain GD, given Ridge with $\alpha$={alpha}, iterations=100',
                filename=f'{filepath}/MSE_momentum_eta_OLS_GD_iterations100_alpha1e{int(np.log10(alpha))}.png')

        SGD_epochs_batch_size(X=X, y_true=y_true, cost=ridge, alpha=alpha, momentum=0,
                    title=rf'MSE as function of epoch and batch size, given Ridge with $\alpha$ = {alpha}',
                    filename=f'{filepath}/MSE_epoch_batch_ridge_e{np.log10(alpha)}.png')

        SGD_momentum_batch_size(X=X, y_true=y_true, cost=ridge, alpha=alpha, 
                    title=rf'MSE as function of momentum and batch size, given Ridge with $\alpha$ = {alpha} and epoch = 10',
                    filename=f'{filepath}/MSE_momentum_batch_ridge_e{np.log10(alpha)}.png')
        
    # Heatmap for Ridge, given alpha and epsilon, batch = 4, epoch = 20, adaGrad
    adaGrad_alpha_epsilon(X=X, y_true=y_true, cost=ridge, 
                          title='MSE as function of alpha and epsilon for adaGrad, batch = 4 and epoch = 20',
                          filename=f'{filepath}/MSE_alpha_epsilon_adaGrad_ridge_batch4_epoch20.png')

    print("Done with optimizers.")

