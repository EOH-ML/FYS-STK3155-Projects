import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from plotting import Plotting
from neural_network import NeuralNetwork
from franke import create_data_franke
from activation_functions import sigmoid, ReLU, leakyReLU, linear
from optimization import GradientDescent
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def create_folder_in_current_directory(folder_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    new_folder_path = os.path.join(current_directory, folder_name)
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully at: {new_folder_path}")
    else:
        print(f"Folder '{folder_name}' already exists at: {new_folder_path}")

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(9*x - 2)**2 / 4.0 - (9*y - 2)**2 / 4.0)
    term2 = 0.75*np.exp(-(9*x + 1)**2 / 49.0 - (9*y + 1)/10.0)
    term3 = 0.5*np.exp(-(9*x - 7)**2 / 4.0 - (9*y - 3)**2 / 4.0)
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

def get_franke(n, split=True):
    # Generer gridet
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    X = np.c_[x_flat, y_flat]
    z = FrankeFunction(x_flat, y_flat).reshape(-1, 1)

    if split:
        # Del dataene i trenings- og testsett
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2, random_state=42)
        return X_train, X_test, z_train, z_test
    else:
        # Returner hele datasettet uten splitting
        return x_grid, y_grid, X, z

def pred_plot_franke(filepath:str=None): 
    n = 100  
    X_train, X_test, z_train, z_test = get_franke(n)
    
    x_grid, y_grid, X_full, z_full = get_franke(n, split=False)
    
    scaler_input = StandardScaler()
    X_train_scaled = scaler_input.fit_transform(X_train)
    X_test_scaled = scaler_input.transform(X_test)
    X_full_scaled = scaler_input.transform(X_full)
    
    scaler_output = StandardScaler()
    z_train_scaled = scaler_output.fit_transform(z_train)
    z_test_scaled = scaler_output.transform(z_test)
    
    nn = NeuralNetwork(input_size=2,
                       layer_sizes=[10, 10, 1], 
                       activation_funcs=[sigmoid, sigmoid, linear],
                       loss='mse')
    optimizer = GradientDescent(learning_rate=0.01, momentum=0.0, nn=nn)
    nn.train(X_train_scaled, z_train_scaled, epochs=30, batch_size=4, optimizer=optimizer, lmd=1e-7)
    
    predict_scaled = nn.feed_forward(X_full_scaled)
    predict = scaler_output.inverse_transform(predict_scaled)
    
    z_pred = predict.reshape(n, n)
    
    plt.figure(figsize=(5, 5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    ax = plt.axes(projection='3d')
    
    plot = ax.plot_surface(x_grid, y_grid, z_pred, cmap=plt.cm.terrain, linewidth=0, antialiased=True)
    
    ax.xaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)
    ax.yaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)  
    ax.zaxis._axinfo["grid"].update(color='lightgray', linewidth=0.5)  

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.zaxis.set_tick_params(labelsize=16)
    
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    ax.set_xlabel(r'\textbf{x}', fontsize=18)
    ax.set_ylabel(r'\textbf{y}', fontsize=18)
    ax.set_zlabel(r'\textbf{$f$}', fontsize=18)
    
    if filepath:
        plt.savefig(f'{filepath}/franke_nn_3d.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def franke_activation_fns_learning_rates(filepath:str=None):
    
    n_splits = 5
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(n=100)

    input_size = 2
    layer_sizes = [10, 10, 1]
    activation_functions = [sigmoid, ReLU, leakyReLU]
    learning_rates = [0.02, 0.01, 0.005, 0.001]

    errors_sigmoid = np.zeros(len(learning_rates))
    errors_ReLU = np.zeros(len(learning_rates))
    errors_LeakyReLU = np.zeros(len(learning_rates))

    kf = KFold(n_splits=n_splits)
    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for activation_function in activation_functions:
            for jdx, learning_rate in enumerate(learning_rates):
                nn = NeuralNetwork(input_size=input_size, 
                                    layer_sizes=layer_sizes, 
                                    activation_funcs=[activation_function] * (len(layer_sizes)-1) + [linear],
                                    loss='mse')
                optimizer = GradientDescent(learning_rate=learning_rate, momentum=0.0, nn=nn)
                nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=4, optimizer=optimizer)
                predict = nn.feed_forward(X_train_scaled[val_index])
                mse_error = np.mean((predict - z_train[val_index])**2)

                print(f"MSE: {mse_error:.6f}, Activation: {activation_function.__name__}, Learning Rate: {learning_rate}")

                if activation_function == sigmoid:
                    errors_sigmoid[jdx] += mse_error
                elif activation_function == ReLU:
                    errors_ReLU[jdx] += mse_error
                elif activation_function == leakyReLU:
                    errors_LeakyReLU[jdx] += mse_error

        print(f'Run {i + 1} completed')

    errors_sigmoid /= n_splits
    errors_ReLU /= n_splits
    errors_LeakyReLU /= n_splits

    # Prepare data for plotting
    x_labels = [activ_func.__name__ for activ_func in activation_functions]
    y_values = [errors_sigmoid, errors_ReLU, errors_LeakyReLU]
    group_labels = [f'LR={lr}' for lr in learning_rates]

    # Plotting
    Plotting().plot_grouped_bar(x_labels=x_labels, 
                                y_values=y_values, 
                                group_labels=group_labels, 
                                x_label='Activation Function', 
                                y_label='Mean Squared Error', 
                                title='MSE per Activation Function and Learning Rate',
                                filename=f'{filepath}/histogram_act_funcs_learning_rates.png',
                                box_string=f'10 neurons, folds={n_splits}, epochs=30, batch-size=4, SGD, loss=MSE'
                                )
    plt.close()


def franke_number_of_layers(filepath:str=None, show_text:bool=False):
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(n=100) 

    n_splits = 5
    input_size = 2
    layer_sizes = []
    number_of_layers = []
    for i in range(1, 5):
        layer_sizes.append([10] * i + [1])
        number_of_layers.append(i + 1)
    n_layer_sizes = len(layer_sizes)
    activation_functions = [sigmoid, ReLU, leakyReLU]

    errors_sigmoid = np.zeros(n_layer_sizes)
    errors_ReLU = np.zeros(n_layer_sizes)
    errors_LeakyReLU = np.zeros(n_layer_sizes)

    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for activation_function in activation_functions:
            for j, layers in enumerate(layer_sizes):
                nn = NeuralNetwork(input_size=input_size, 
                                    layer_sizes=layers, 
                                    activation_funcs=[activation_function] * (len(layers)-1) + [linear],
                                    loss='mse')
                optimizer = GradientDescent(learning_rate=0.01, momentum=0.0, nn=nn)
                nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=4, optimizer=optimizer)
                predict = nn.feed_forward(X_train_scaled[val_index])
                if activation_function == sigmoid:
                    errors_sigmoid[j] += np.sum((predict - z_train[val_index])**2)/predict.size
                elif activation_function == ReLU:
                    errors_ReLU[j] += np.sum((predict - z_train[val_index])**2)/predict.size
                elif activation_function == leakyReLU:
                    errors_LeakyReLU[j] += np.sum((predict - z_train[val_index])**2)/predict.size

        print(f'Run {i + 1} completed')
    errors_sigmoid /= n_splits 
    errors_ReLU /= n_splits 
    errors_LeakyReLU /= n_splits
    
    Plotting().plot_1d((errors_sigmoid, 'Sigmoid in the hidden layers'),
                        (errors_ReLU, 'ReLU in the hidden layers'),
                        (errors_LeakyReLU, 'LeakyReLU in the hidden layers'),
                        x_values=number_of_layers,
                        x_label='number of layers',
                        y_label='mse',
                        title=f'MSE for n layers',
                        filename=f'{filepath}/mse_n_hidden_layers_franke.png',
                        box_string=f'10 neurons, folds={n_splits}, lr=0.01, epochs=30, batch-size=4, SGD, loss=MSE'
                        )
    plt.close()

def franke_number_of_neurons(filepath:str=None, show_text:bool=False):
    n_splits = 5
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(n=100) 

    input_size = 2
    layer_sizes = []
    for i in range(1, 102, 5):
        layer_sizes.append([i, i, 1])
    n_layer_sizes = len(layer_sizes)
    activation_functions = [sigmoid, ReLU, leakyReLU]

    number_of_parameters = np.zeros(n_layer_sizes)
    errors_sigmoid = np.zeros(n_layer_sizes)
    errors_ReLU = np.zeros(n_layer_sizes)
    errors_LeakyReLU = np.zeros(n_layer_sizes)

    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for activation_function in activation_functions:
            for j, layers in enumerate(layer_sizes):
                number_of_parameters[j] = get_number_of_parameters(input_size, layers) 
                nn = NeuralNetwork(input_size=input_size, 
                                    layer_sizes=layers, 
                                    activation_funcs=[activation_function] * (len(layers)-1) + [linear],
                                    loss='mse')
                optimizer = GradientDescent(learning_rate=0.01, momentum=0.0, nn=nn)
                nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=4, optimizer=optimizer)
                predict = nn.feed_forward(X_train_scaled[val_index])
                if activation_function == sigmoid:
                    errors_sigmoid[j] += np.sum((predict - z_train[val_index])**2)/predict.size
                elif activation_function == ReLU:
                    errors_ReLU[j] += np.sum((predict - z_train[val_index])**2)/predict.size
                elif activation_function == leakyReLU:
                    errors_LeakyReLU[j] += np.sum((predict - z_train[val_index])**2)/predict.size

        print(f'Run {i + 1} completed')
    errors_sigmoid /= n_splits 
    errors_ReLU /= n_splits
    errors_LeakyReLU /= n_splits
    
    Plotting().plot_1d((errors_sigmoid, 'Sigmoid in the hidden layers'),
                        (errors_ReLU, 'ReLU in the hidden layers'),
                        (errors_LeakyReLU, 'LeakyReLU in the hidden layers'),
                        x_values=number_of_parameters,
                        x_label='number of parameters',
                        y_label='mse',
                        title='MSE for number of parameters, adjusted by number of neurons',
                        filename=f'{filepath}/mse_n_neurons_franke.png',
                        box_string=f'1-102 neurons per layer, 2 layers, folds={n_splits}, lr=0.01, epochs=30, batch size=4, SGD, loss=MSE'
                        )
    plt.close()

def mse_batch_size(filepath:str=None, show_text:bool=False):
    n_splits = 5
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(n=100) 

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    learning_rates = [0.001, 0.005, 0.010, 0.015, 0.020]
    n_batch_sizes = len(batch_sizes)

    errors_0_001 = np.zeros(n_batch_sizes)
    errors_0_005 = np.zeros(n_batch_sizes)
    errors_0_010 = np.zeros(n_batch_sizes) 
    errors_0_015 = np.zeros(n_batch_sizes)
    errors_0_020 = np.zeros(n_batch_sizes)

    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for learning_rate in learning_rates:
            for i, batch_size in enumerate(batch_sizes):
                    nn = NeuralNetwork(input_size=2, 
                                        layer_sizes=[10, 10, 1], 
                                        activation_funcs=[sigmoid, sigmoid, linear],
                                        loss='mse')
                    optimizer = GradientDescent(learning_rate=learning_rate, momentum=0.0, nn=nn)
                    nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=batch_size, optimizer=optimizer)
                    predict = nn.feed_forward(X_train_scaled[val_index])
                    mse = np.sum((predict - z_train[val_index])**2)/predict.size
                    if learning_rate == 0.001:
                        errors_0_001[i] += mse
                    elif learning_rate == 0.005:
                        errors_0_005[i] += mse
                    elif learning_rate == 0.010:
                        errors_0_010[i] += mse
                    elif learning_rate == 0.015:
                        errors_0_015[i] += mse
                    elif learning_rate == 0.020:
                        errors_0_020[i] += mse

            print(f'Done with learning rate: {learning_rate}')

    errors_0_001 /= n_splits
    errors_0_005 /= n_splits
    errors_0_010 /= n_splits
    errors_0_015 /= n_splits
    errors_0_020 /= n_splits
     
    Plotting().plot_1d((errors_0_001, 'Learning rate 0.001'),
                        (errors_0_005, 'Learning rate 0.005'),
                        (errors_0_010, 'Learning rate 0.010'),
                        (errors_0_015, 'Learning rate 0.015'),
                        (errors_0_020, 'Learning rate 0.020'),
                        x_values=batch_sizes,
                        x_label='batch size',
                        y_label='mse',
                        title='MSE as function of batch size',
                        filename=f'{filepath}/mse_learning_rate_batch_size_sigmoid.png',
                        box_string=f'sigmoid, SGD, epochs=30, 2 hidden layers, 10 neurons, folds={n_splits}, loss=MSE'
                        )
    plt.close()

def mse_batch_size_fixed_iterations(filepath:str=None, show_text:bool=False):
    n_splits = 5
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(n=100) 

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    learning_rates = [0.001, 0.005, 0.010, 0.015, 0.020]

    nr_iterations = 10000
    samples = X_train_scaled.shape[0]

    n_batch_sizes = len(batch_sizes)

    errors_0_001 = np.zeros(n_batch_sizes)
    errors_0_005 = np.zeros(n_batch_sizes)
    errors_0_010 = np.zeros(n_batch_sizes) 
    errors_0_015 = np.zeros(n_batch_sizes)
    errors_0_020 = np.zeros(n_batch_sizes)

    time_0_001 = np.zeros(n_batch_sizes)
    time_0_005 = np.zeros(n_batch_sizes)
    time_0_010 = np.zeros(n_batch_sizes)
    time_0_015 = np.zeros(n_batch_sizes)
    time_0_020 = np.zeros(n_batch_sizes)

    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for learning_rate in learning_rates:
            for i, batch_size in enumerate(batch_sizes):
                    nn = NeuralNetwork(input_size=2, 
                                        layer_sizes=[10, 10, 1], 
                                        activation_funcs=[sigmoid, sigmoid, linear],
                                        loss='mse')
                    optimizer = GradientDescent(learning_rate=learning_rate, momentum=0.0, nn=nn)
                    epochs = max(1, nr_iterations * batch_size // samples)
                    start_time = time.time()
                    nn.train(X_train_scaled[train_index], z_train[train_index], epochs=epochs, batch_size=batch_size, optimizer=optimizer)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    predict = nn.feed_forward(X_train_scaled[val_index])
                    mse = np.sum((predict - z_train[val_index])**2)/predict.size
                    if learning_rate == 0.001:
                        errors_0_001[i] += mse
                        time_0_001[i] += elapsed_time 
                    elif learning_rate == 0.005:
                        errors_0_005[i] += mse
                        time_0_005[i] += elapsed_time 
                    elif learning_rate == 0.010:
                        errors_0_010[i] += mse
                        time_0_010[i] += elapsed_time 
                    elif learning_rate == 0.015:
                        errors_0_015[i] += mse
                        time_0_015[i] += elapsed_time 
                    elif learning_rate == 0.020:
                        errors_0_020[i] += mse
                        time_0_020[i] += elapsed_time 

            print(f'Done with learning rate: {learning_rate}')

    errors_0_001 /= n_splits
    errors_0_005 /= n_splits
    errors_0_010 /= n_splits
    errors_0_015 /= n_splits
    errors_0_020 /= n_splits

    time_0_001 /= n_splits 
    time_0_005 /= n_splits
    time_0_010 /= n_splits 
    time_0_015 /= n_splits 
    time_0_020 /= n_splits 
     
    Plotting().plot_1d((errors_0_001, 'Learning rate 0.001'),
                        (errors_0_005, 'Learning rate 0.005'),
                        (errors_0_010, 'Learning rate 0.010'),
                        (errors_0_015, 'Learning rate 0.015'),
                        (errors_0_020, 'Learning rate 0.020'),
                        x_values=batch_sizes,
                        x_label='batch size',
                        y_label='mse',
                        log2_scale=True,
                        title=f'MSE as function of batch size Franke, iterations = {nr_iterations}',
                        filename=f'{filepath}/mse_learning_rate_batch_size_sigmoid_fixed_iterations.png',
                        box_string=f'sigmoid, log2 plot, SGD, 2 hidden layers, 10 neurons, folds={n_splits}, loss=MSE'
                        )
    plt.close()

    Plotting().plot_1d((time_0_001, 'Learning rate 0.001'),
                        (time_0_005, 'Learning rate 0.005'),
                        (time_0_010, 'Learning rate 0.010'),
                        (time_0_015, 'Learning rate 0.015'),
                        (time_0_020, 'Learning rate 0.020'),
                        x_values=batch_sizes,
                        x_label='batch size',
                        y_label='time taken to train network (in seconds)',
                        title=f'Time for training network for Frankes func, iterations = {nr_iterations}',
                        log2_scale=True,
                        box_string=f'log2 plot, SGD, sigmoid, 2 hidden layers, 10 neurons, loss=mse, folds={n_splits}',
                        filename=f'{filepath}/time_given_batchsize_fixed_iterations_frankes.png'
                        )
    plt.close()

def get_number_of_parameters(input_size, layer_size):
    s = sum(layer_size)
    sizes = [input_size] + layer_size
    for i in range(len(sizes) - 1):
        s += sizes[i] * sizes[i+1] 
    return s

def lambda_learningrate_mse_r2(filepath:str=None):
    mse_plot = True
    r2_plot = True
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(100)
    lambdas = [0] + [10**l for l in range(-7, 2)]
    learning_rates = [0.001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1]
    epochs = 30
    n_splits = 5

    heatmap_mse = np.zeros((len(lambdas), len(learning_rates)))
    heatmap_r2 = np.zeros((len(lambdas), len(learning_rates)))

    kf = KFold(n_splits=n_splits)

    for k,(train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for i, lmd in enumerate(lambdas):
            for j, lr in enumerate(learning_rates):
                nn = NeuralNetwork(input_size=2, 
                                layer_sizes=[10, 10, 1], 
                                activation_funcs=[sigmoid, sigmoid, linear],
                                loss='mse',
                )
                optimizer = GradientDescent(learning_rate=lr, momentum=0.0, nn=nn)
                nn.train(X_train_scaled[train_index], z_train[train_index], epochs=epochs, batch_size=4, optimizer=optimizer, lmd=lmd)
                predict = nn.feed_forward(X_train_scaled[val_index])
                mse = 1/predict.size*np.sum((predict-z_train[val_index])**2)

                ss_res = np.sum((z_train[val_index] - predict)**2)
                ss_tot = np.sum((z_train[val_index] - np.mean(z_train[val_index]))**2)
                r2_score = 1-(ss_res/ss_tot)

                heatmap_mse[i, j] += mse
                heatmap_r2[i, j] += r2_score
        print(f'Working on heatmap Franke function, fold = {k + 1}/{n_splits}')
    heatmap_mse /= n_splits
    heatmap_r2 /= n_splits
    if mse_plot:
        Plotting().heatmap(heatmap_matrix=heatmap_mse,
                            x_data=learning_rates,
                            y_data=lambdas,
                            x_label=r'$\eta$',
                            y_label=r'$\lambda$',
                            title='MSE heatmap',
                            decimals=5,
                            filename=f'{filepath}/mse_heatmap_lambda_lr_franke.png',
                            box_string=f'10 neurons, 2 layers, folds={n_splits}, epochs=30, batch size=4, sigmoid, SGD, loss=MSE'
                            )
        plt.close()
    if r2_plot:
        Plotting().heatmap(heatmap_matrix=heatmap_r2,
                            x_data=learning_rates,
                            y_data=lambdas,
                            x_label=r'$\eta$',
                            y_label=r'$\lambda$',
                            title='R2 Heatmap',
                            decimals=5,
                            min_patch=False,
                            filename=f'{filepath}/r2_heatmap_lambda_lr_franke.png',
                            box_string=f'10 neurons, 2 layers, folds={n_splits}, epochs=30, batch size=4, sigmoid, SGD, loss=MSE'
                            )
        plt.close()

def train_val_franke(filepath:str=None, show_text:bool=False):
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(100)

    activation_funcs = [sigmoid, leakyReLU, ReLU]

    for act in activation_funcs:
        nn = NeuralNetwork(input_size=2, 
                        layer_sizes=[10, 10, 1],
                        activation_funcs= [act]*2 + [linear],
                        loss='mse',
        )
        optimizer = GradientDescent(learning_rate=0.01, momentum=0.0, nn=nn)
        nn.train(X_train_scaled, z_train, epochs=100, batch_size=4, optimizer=optimizer, data_val=X_test_scaled, target_val=z_test)
        nn.show_loss_function(show=False, 
                              filename=f'{filepath}/mse_val_vs_train_{act.__name__}.png', 
                              title=f'Val vs train set with activation func {act.__name__}',
                              box_string=f'10 neurons, 2 layers, epochs=100, batch size=4, sigmoid, SGD, loss=MSE'
                                )
        plt.close()

def bias_initialization_sigmoid(filepath:str=False, show_text:bool=False):
    
    # Load data
    X_train_scaled, X_test_scaled, z_train, z_val = get_franke(n=100)
    
    # Configuration
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    epochs = 30
    learning_rate = 0.01
    bias_values = {
        'bias_minus1': -1,
        'bias_random': 'random',  # Special case: leave random initialization as-is
        'bias_point1': 0.1,
        'bias_zero': 0,
        'bias_three': 3
    }
    
    # Storage for results
    results = {}

    def train_with_bias(bias_value):
        """Trains the network with specified bias initialization and returns the loss."""
        loss_accumulated = np.zeros(epochs)
        
        for (train_index, val_index) in kf.split(X_train_scaled):
            nn = NeuralNetwork(input_size=2, 
                               layer_sizes=[10, 10, 1], 
                               activation_funcs=[sigmoid, sigmoid, linear],
                               loss='mse')
            optimizer = GradientDescent(learning_rate=learning_rate, momentum=0.0, nn=nn)
            
            # Initialize biases if not random
            if bias_value != 'random':
                for i, (W, _) in enumerate(nn.layers):
                    nn.update_layer((W, bias_value), i)
            
            # Train and accumulate loss
            nn.train(X_train_scaled[train_index], z_train[train_index], epochs=epochs, batch_size=4, optimizer=optimizer, 
                     data_val=X_train_scaled[val_index], target_val=z_train[val_index])
            loss_accumulated += nn._loss_val_values
        
        return loss_accumulated / n_splits
    
    # Train for each bias configuration
    for label, bias_value in bias_values.items():
        results[label] = train_with_bias(bias_value)
    
    # Plot results
    Plotting().plot_1d(
        (results['bias_minus1'], 'Biases initialized as -1.0'),
        (results['bias_random'], r'Biases initialized $\sim \mathcal{N}(0,1)$'),
        (results['bias_point1'], 'Biases initialized as 0.1'),
        (results['bias_zero'], 'Biases initialized as 0.0'),
        (results['bias_three'], 'Biases initialized as 3.0'),
        x_values=range(epochs),
        title="Bias initialization on Franke's function using SGD",
        x_label='Epochs',
        y_label='MSE',
        filename=f'{filepath}/bias_initialization_franke.png',
        box_string=f'10 neurons, 2 hidden layers, folds={n_splits}, lr=0.01, epochs=30, SGD, loss=MSE'
    )

    plt.close()

if __name__=="__main__":
    create_folder_in_current_directory('../../figures')
    create_folder_in_current_directory('../../figures/Franke_nn')
    filepath = '../../figures/Franke_nn'
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Working on Neural Network for Franke's Function...")
    
    # Lambda learning rate MSE and R2
    lambda_learningrate_mse_r2(filepath=filepath)
    print("Lambda learning rate MSE and R2 analysis completed.")

    # Train and validate on the Franke function
    train_val_franke(filepath=filepath)
    print("Training and validation on the Franke function completed.")

    # Bias initialization with sigmoid activation
    bias_initialization_sigmoid(filepath=filepath)
    print("Bias initialization with sigmoid activation completed.")

    # Franke function activation functions and learning rates
    franke_activation_fns_learning_rates(filepath=filepath)
    print("Franke activation functions and learning rates analysis completed.")

    # Franke function number of layers
    franke_number_of_layers(filepath=filepath)
    print("Franke function number of layers analysis completed.")

    # Franke function number of neurons
    franke_number_of_neurons(filepath=filepath)
    print("Franke function number of neurons analysis completed.")

    # MSE batch size analysis
    mse_batch_size(filepath=filepath)
    print("MSE batch size analysis completed.")

    # MSE batch size with fixed iterations
    mse_batch_size_fixed_iterations(filepath=filepath)
    print("MSE batch size with fixed iterations analysis completed.")

    # Prediction plot for Franke function
    pred_plot_franke(filepath=filepath)
    print("Prediction plot for Franke function completed.")

    print("Done with Neural Network for Franke's Function.")
