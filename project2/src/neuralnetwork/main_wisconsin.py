import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimization import GradientDescent, RMSProp, AdaGrad, Adam
from activation_functions import sigmoid, softmax, ReLU, leakyReLU
from neural_network import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from plotting import Plotting

def create_folder_in_current_directory(folder_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    new_folder_path = os.path.join(current_directory, folder_name)
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully at: {new_folder_path}")
    else:
        print(f"Folder '{folder_name}' already exists at: {new_folder_path}")

def load_scaled_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    y = one_hot(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def one_hot(values):
    targets = np.zeros((len(values), 2))
    for i, t in enumerate(values):
        targets[i, t] = 1
    return targets

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def bias_initialization(filepath:str=None):
    # Load data
    X_train_scaled, X_test_scaled, y_train, y_val = load_scaled_breast_cancer()
    
    # Configuration
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    epochs = 30
    learning_rate = 0.001
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
            nn = NeuralNetwork(input_size=30, 
                               layer_sizes=[12, 12, 2], 
                               activation_funcs=[sigmoid, sigmoid, softmax],
                               loss='cross entropy')
            optimizer = RMSProp(learning_rate=learning_rate, rho=0.9, nn=nn)
            
            # Initialize biases if not random
            if bias_value != 'random':
                for i, (W, _) in enumerate(nn.layers):
                    nn.update_layer((W, bias_value), i)
            
            # Train and accumulate loss
            nn.train(X_train_scaled[train_index], y_train[train_index], epochs=epochs, batch_size=4, optimizer=optimizer, 
                     data_val=X_train_scaled[val_index], target_val=y_train[val_index])
            loss_accumulated += nn._loss_val_values
        
        return loss_accumulated / n_splits
    
    # Train for each bias configuration
    for i, (label, bias_value) in enumerate(bias_values.items()):
        results[label] = train_with_bias(bias_value)
        print(f'Working on bias initialization {label} Wisconsin breast cancer, method {i+1} out of 5')
    
    # Plot results
    Plotting().plot_1d(
        (results['bias_minus1'], 'Biases initialized as -1.0'),
        (results['bias_random'], r'Biases initialized $\sim \mathcal{N}(0,1)$'),
        (results['bias_point1'], 'Biases initialized as 0.1'),
        (results['bias_zero'], 'Biases initialized as 0.0'),
        (results['bias_three'], 'Biases initialized as 3.0'),
        x_values=range(epochs),
        title="Bias initialization on breast cancer data set using RMSProp",
        x_label='Epochs',
        y_label='Accuracy',
        filename=f'{filepath}/bias_initialization_wisconsin.png',
        box_string=f'12 neurons, 2 hidden layers, folds={n_splits}, lr=0.001, rho=0.9 epochs=30, loss=cross entropy'
    )

    plt.close()


def lambda_learningrate_rho_accuracy(lr_or_rho:list, x_label:str, filepath:str=None):
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()
    lambdas = [0] + [10**l for l in range(-7, 2)]
    epochs = 30
    n_splits = 5

    heatmap_accuracy = np.zeros((len(lambdas), len(lr_or_rho)))

    kf = KFold(n_splits=n_splits)

    for k,(train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for i, lmd in enumerate(lambdas):
            for j, lr_rho in enumerate(lr_or_rho):
                nn = NeuralNetwork(input_size=30, 
                                layer_sizes=[12, 12, 2], 
                                activation_funcs=[sigmoid, sigmoid, softmax],
                                loss='cross entropy',
                )
                if x_label == 'rho':
                    lr = 0.1
                    rho = lr_rho
                if x_label == 'eta':
                    rho = 0.9
                    lr = lr_rho
                optimizer = RMSProp(learning_rate=lr, rho=rho, nn=nn)
                nn.train(X_train_scaled[train_index], y_train[train_index], epochs=epochs, batch_size=4, optimizer=optimizer, lmd=lmd)
                predict = nn.feed_forward(X_train_scaled[val_index])

                acc = accuracy(predict, y_train[val_index])
                heatmap_accuracy[i, j] += acc
        print(f'Working on regularization and {x_label} for Wisconsin breast cancer, fold: {k+1}/{n_splits}')

    heatmap_accuracy /= n_splits
    Plotting().heatmap(heatmap_matrix=heatmap_accuracy,
                        x_data=lr_or_rho,
                        y_data=lambdas,
                        x_label=rf'$\{x_label}$',
                        y_label=r'$\lambda$',
                        title='Accuracy heatmap',
                        decimals=5,
                        min_patch=False,
                        filename=f'{filepath}/accuracy_heatmap_lambda_{x_label}_franke.png',
                        box_string=f'12 neurons, 2 layers, folds={n_splits}, epochs=30, batch size=4, sigmoid, RMSprop, loss=cross entropy'
                        )
    plt.close()

def cancer_lr_vs_rhos_RMSProp(filepath:str=None):
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    rhos = [0.75, 0.8, 0.85, 0.9, 0.95]

    results = []
    n_splits = 5
    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for lr in learning_rates:
            for r in rhos:
                nn = NeuralNetwork(
                    input_size=30, 
                    layer_sizes=[32, 32, 2], 
                    activation_funcs=[sigmoid, sigmoid, softmax],
                    loss='cross entropy'
                )
                
                optimizer = RMSProp(lr, nn=nn, rho=r)
                
                nn.train(data=X_train_scaled[train_index], target=y_train[train_index], 
                         epochs=30, 
                         batch_size=4, 
                         optimizer=optimizer)
                predictions = nn.feed_forward(X_train_scaled[val_index])
                acc = accuracy(predictions, y_train[val_index])
                
                results.append({
                    "learning_rate": lr,
                    "rho": r,
                    "accuracy": acc,
                })
        print(f'Working on lr and rho for Wisconsin breast cancer, fold: {i+1}/{n_splits}')

    # Using pandas to convert to a DataFrame
    df = pd.DataFrame(results)
    # Grouping by unique combinations of learning rate and rho, 
    # calculating the mean accuracy and converting back to standard DataFrame
    df_mean = df.groupby(['learning_rate', 'rho']).mean().reset_index()
    # Creating pivot table for heatmap
    pivot_table = df_mean.pivot(values='accuracy', index='rho', columns='learning_rate')
    # Extracting data for heatmap 
    heatmap_matrix = pivot_table.values
    x_data = pivot_table.columns.tolist() # Learning rates
    y_data = pivot_table.index.tolist() # Rhos

    Plotting().heatmap(
        heatmap_matrix=heatmap_matrix,
        x_data=x_data,
        y_data=y_data,
        x_label="Learning Rate",
        y_label="Rho",
        title="Heatmap of Accuracy for Different Learning Rates and Rho Values",
        decimals=4,
        min_patch=False, # highlighting the maximum value
        filename=f'{filepath}/heatmap_acc_rho_lr.png',
        is_minimal=False,
        box_string=f'RMSProp, folds={n_splits}, loss=cross entropy, epochs=30, batch-size=4, sigmoid, 2 hidden layers, 32 neurons'
    )
    plt.close()

def cancer_networksizes_and_lrs(filepath:str=None):
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    rho = 0.9
    layer_sizes = [[10, 2], [64, 32, 2], [16, 8, 2], [128, 64, 32, 16, 2], [32, 32, 2]]
    activation_funcs=[[sigmoid, softmax], [sigmoid, sigmoid, softmax], [sigmoid, sigmoid, softmax], 
                    [sigmoid, sigmoid, sigmoid, sigmoid, softmax], [sigmoid, sigmoid, softmax]]


    results = []
    n_splits = 5
    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        for lr in learning_rates:
            for idx, layers in enumerate(layer_sizes):
                nn = NeuralNetwork(
                    input_size=30, 
                    layer_sizes=layers, 
                    activation_funcs=activation_funcs[idx],
                    loss='cross entropy'
                )
                
                optimizer = RMSProp(lr, nn=nn, rho=rho)
                
                # Train the model and capture metrics
                nn.train(data=X_train_scaled[train_index], target=y_train[train_index], epochs=30, batch_size=4, optimizer=optimizer)
                predictions = nn.feed_forward(X_train_scaled[val_index])
                acc = accuracy(predictions, y_train[val_index])
                
                results.append({
                    "learning_rate": lr,
                    "total_nodes": sum(layers),
                    "accuracy": acc,
                })
                
        print(f'Working on lr and network size for Wisconsin breast cancer, fold: {i+1}/{n_splits}')

    df = pd.DataFrame(results)
    df_mean = df.groupby(["learning_rate", "total_nodes"]).mean().reset_index()
    pivot_table = df_mean.pivot_table(values='accuracy', index='total_nodes', columns='learning_rate')
    heatmap_matrix = pivot_table.values
    x_data = pivot_table.columns.tolist()
    y_data = pivot_table.index.tolist()

    Plotting().heatmap(
        heatmap_matrix=heatmap_matrix,
        x_data=x_data,
        y_data=y_data,
        x_label="Learning Rate",
        y_label="Total number of nodes",
        title="Heatmap of Accuracy for Different Learning Rates and Network sizes",
        decimals=4,
        min_patch=False,
        filename=f'{filepath}/heatmap_lr_network_size.png',
        is_minimal=False, 
        box_string=f'RMSProp, folds={n_splits}, rho=0.9, loss=cross entropy, epochs=30, batch-size=4, sigmoid'
    )
    plt.close()

def wisconsin_number_of_neurons(filepath:str=None):
    n_splits = 5
    X_train_scaled, X_test_scaled, z_train, z_test = load_scaled_breast_cancer()

    optimizers = ['SGD', 'SGD_Momentum', 'adaGrad', 'RMSProp', 'Adam']

    input_size = 30
    layer_sizes = []
    for i in range(1, 111, 5):
        layer_sizes.append([i, i, 2])
    n_layer_sizes = len(layer_sizes)
    activation_functions = [sigmoid, ReLU, leakyReLU]

    number_of_parameters = np.zeros(n_layer_sizes)
    errors_sigmoid = np.zeros(n_layer_sizes)
    errors_ReLU = np.zeros(n_layer_sizes)
    errors_LeakyReLU = np.zeros(n_layer_sizes)

    kf = KFold(n_splits=n_splits)
    for optim in optimizers:
        for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
            for activation_function in activation_functions:
                for j, layers in enumerate(layer_sizes):
                    number_of_parameters[j] = get_number_of_parameters(input_size, layers) 
                    nn = NeuralNetwork(input_size=input_size, 
                                        layer_sizes=layers, 
                                        activation_funcs=[activation_function] * (len(layers)-1) + [softmax],
                                        loss='cross entropy')
                    if optim == 'SGD' and optim in optimizers:
                        optimizer = GradientDescent(learning_rate=0.001, momentum=0.0, nn=nn) 
                    elif optim == 'SGD_Momentum' and optim in optimizers:
                        optimizer = GradientDescent(learning_rate=0.001, momentum=0.8, nn=nn) 
                    elif optim == 'adaGrad' and optim in optimizers:
                        optimizer = AdaGrad(learning_rate=0.001, nn=nn)
                    elif optim == 'RMSProp' and optim in optimizers:
                        optimizer = RMSProp(learning_rate=0.001, nn=nn, rho=0.9)
                    elif optim == 'Adam' and optim in optimizers:
                        optimizer = Adam(learning_rate=0.001, nn=nn, rho1=0.9, rho2=0.999)
                    else:
                        raise TypeError(f'{optim} not a valid optimzer')

                    nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=4, optimizer=optimizer)
                    predict = nn.feed_forward(X_train_scaled[val_index])
                    if activation_function == sigmoid:
                        errors_sigmoid[j] += accuracy(predictions=predict, targets=z_train[val_index])
                    elif activation_function == ReLU:
                        errors_ReLU[j] += accuracy(predictions=predict, targets=z_train[val_index]) 
                    elif activation_function == leakyReLU:
                        errors_LeakyReLU[j] += accuracy(predictions=predict, targets=z_train[val_index])
            print(f'Working on number of neurons for wisconsin breast cancer, fold = {i + 1}/{n_splits}')
            
        errors_sigmoid /= n_splits 
        errors_ReLU /= n_splits 
        errors_LeakyReLU /= n_splits
        
        Plotting().plot_1d((errors_sigmoid, 'Sigmoid in the hidden layers'),
                            (errors_ReLU, 'ReLU in the hidden layers'),
                            (errors_LeakyReLU, 'LeakyReLU in the hidden layers'),
                            x_values=number_of_parameters,
                            x_label='number of parameters',
                            y_label='Accuracy',
                            title=f'Accuracy for {optim}, n neurons in the hidden layers',
                            filename=f'{filepath}/{optim}_n_neurons_BC.png',
                            box_string=f'2 hidden layers, folds={n_splits}, lr=0.001, epochs=30, loss=cross entropy, standard Goodfellow hyperparameters'
                            )

        number_of_parameters = np.zeros(n_layer_sizes)
        errors_sigmoid = np.zeros(n_layer_sizes)
        errors_ReLU = np.zeros(n_layer_sizes)
        errors_LeakyReLU = np.zeros(n_layer_sizes)
        print(f'{optim} done')

        plt.close()

def wisonsin_batch_size_learning_rate(filepath:str=None):
    n_splits = 5
    layer_sizes = [32, 32, 2]
    X_train_scaled, X_test_scaled, z_train, z_test = load_scaled_breast_cancer()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    learning_rates = [0.001, 0.005, 0.010, 0.015, 0.020]

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
            for j, batch_size in enumerate(batch_sizes):
                    nn = NeuralNetwork(input_size=30, 
                                        layer_sizes=layer_sizes, 
                                        activation_funcs=[sigmoid, sigmoid, softmax],
                                        loss='cross entropy')
                    optimizer = RMSProp(learning_rate=learning_rate, rho=0.9, nn=nn)
                    start_time = time.time()
                    nn.train(X_train_scaled[train_index], z_train[train_index], epochs=30, batch_size=batch_size, optimizer=optimizer)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    predict = nn.feed_forward(X_train_scaled[val_index])
                    if learning_rate == 0.001:
                        errors_0_001[j] += accuracy(predictions=predict, targets=z_train[val_index])  
                        time_0_001[j] += elapsed_time 
                    elif learning_rate == 0.005:
                        errors_0_005[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_005[j] += elapsed_time 
                    elif learning_rate == 0.010:
                        errors_0_010[j] += accuracy(predictions=predict, targets=z_train[val_index]) 
                        time_0_010[j] += elapsed_time 
                    elif learning_rate == 0.015:
                        errors_0_015[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_015[j] += elapsed_time 
                    elif learning_rate == 0.020:
                        errors_0_020[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_020[j] += elapsed_time 

        print(f'Working on batch size for wisconsin breast cancer, fold = {i + 1}/{n_splits}')

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
                        y_label='accuracy',
                        title='Accuracy as function of batch size for Wisconsin breast cancer',
                        log2_scale=True,
                        box_string=f'log2 plot, RMSProp, rho=0.9, 3 layers, 32 neurons / layer, loss=cross entropy, folds={n_splits}',
                        filename=f'{filepath}/accuracy_given_batchsize_wisconsin.png'
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
                        title='Time taken to train network given batch size for Wisconsin breast cancer',
                        log2_scale=True,
                        box_string=f'log2 plot, RMSProp, rho=0.9, 3 layers, 32 neurons / layer, loss=cross entropy, folds={n_splits}',
                        filename=f'{filepath}/time_given_batchsize_wisconsin.png'
                        )
    plt.close()

def wisonsin_batch_size_with_same_nr_iterations(filepath:str=None):
    n_splits = 5
    layer_sizes = [32, 32, 2]
    X_train_scaled, X_test_scaled, z_train, z_test = load_scaled_breast_cancer()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    learning_rates = [0.001, 0.005, 0.010, 0.015, 0.020]

    nr_iterations = 1000
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
            for j, batch_size in enumerate(batch_sizes):
                    nn = NeuralNetwork(input_size=30, 
                                        layer_sizes=layer_sizes, 
                                        activation_funcs=[sigmoid, sigmoid, softmax],
                                        loss='cross entropy')
                    optimizer = RMSProp(learning_rate=learning_rate, rho=0.9, nn=nn)
                    epochs = nr_iterations * batch_size // samples
                    start_time = time.time()
                    nn.train(X_train_scaled[train_index], z_train[train_index], epochs=epochs, batch_size=batch_size, optimizer=optimizer)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    predict = nn.feed_forward(X_train_scaled[val_index])
                    if learning_rate == 0.001:
                        errors_0_001[j] += accuracy(predictions=predict, targets=z_train[val_index])  
                        time_0_001[j] += elapsed_time 
                    elif learning_rate == 0.005:
                        errors_0_005[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_005[j] += elapsed_time 
                    elif learning_rate == 0.010:
                        errors_0_010[j] += accuracy(predictions=predict, targets=z_train[val_index]) 
                        time_0_010[j] += elapsed_time 
                    elif learning_rate == 0.015:
                        errors_0_015[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_015[j] += elapsed_time 
                    elif learning_rate == 0.020:
                        errors_0_020[j] += accuracy(predictions=predict, targets=z_train[val_index])
                        time_0_020[j] += elapsed_time 

        print(f'Working on batch size for wisconsin breast cancer, fold = {i + 1}/{n_splits}')

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
                        y_label='accuracy',
                        title=f'Accuracy given batch size for Wisconsin dataset, nr_iterations = {nr_iterations}',
                        log2_scale=True,
                        box_string=f'iterations ={nr_iterations}, log2 plot, RMSProp, rho=0.9, 3 layers, 32 neurons / layer, loss=cross entropy, folds={n_splits}',
                        filename=f'{filepath}/accuracy_given_batchsize_fixed_iterations_wisconsin.png'
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
                        title=f'Time taken to train network given batch size for Wisconsin dataset, nr_iterations = {nr_iterations}',
                        log2_scale=True,
                        box_string=f'iterations ={nr_iterations}, log2 plot, RMSProp, rho=0.9, 3 layers, 32 neurons / layer, loss=cross entropy, folds={n_splits}',
                        filename=f'{filepath}/time_given_batchsize_fixed_iterations_wisconsin.png'
                        )
    plt.close()

def get_number_of_parameters(input_size, layer_size):
    s = sum(layer_size)
    sizes = [input_size] + layer_size
    for i in range(len(sizes) - 1):
        s += sizes[i] * sizes[i+1] 
    return s

if __name__ == "__main__":
    create_folder_in_current_directory('../../figures')
    create_folder_in_current_directory('../../figures/wisconsin_nn')
    filepath = '../../figures/wisconsin_nn'
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Working on Neural Network for Wisconsin Breast Cancer dataset...")
    
    # Initialize biases
    bias_initialization(filepath=filepath)
    print("Bias initialization completed.")

    # Define learning rates and rho values
    learning_rates = [0.001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1]
    rho = [0.999, 0.99, 0.9, 0.85, 0.8]

    # Run lambda learning rate accuracy function with learning rates
    lambda_learningrate_rho_accuracy(filepath=filepath, lr_or_rho=learning_rates, x_label='eta')
    print("Lambda learning rate accuracy with learning rates completed.")

    # Run lambda learning rate accuracy function with rho values
    lambda_learningrate_rho_accuracy(filepath=filepath, lr_or_rho=rho, x_label='rho')
    print("Lambda learning rate accuracy with rho values completed.")

    # Wisconsin batch size learning rate
    wisonsin_batch_size_learning_rate(filepath=filepath)
    print("Wisconsin batch size learning rate analysis completed.")

    # Wisconsin number of neurons
    wisconsin_number_of_neurons(filepath=filepath)
    print("Wisconsin number of neurons analysis completed.")

    # Cancer network sizes and learning rates
    cancer_networksizes_and_lrs(filepath=filepath)
    print("Cancer network sizes and learning rates analysis completed.")

    # Cancer learning rate vs rhos with RMSProp
    cancer_lr_vs_rhos_RMSProp(filepath=filepath)
    print("Cancer learning rate vs rhos with RMSProp completed.")

    # Wisconsin batch size with same number of iterations
    wisonsin_batch_size_with_same_nr_iterations(filepath=filepath)
    print("Wisconsin batch size with the same number of iterations completed.")

    print("Done with Neural Network for Wisconsin Breast Cancer dataset...")
