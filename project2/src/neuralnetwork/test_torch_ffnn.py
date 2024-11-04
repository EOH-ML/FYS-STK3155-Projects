import os
import numpy as np
import matplotlib.pyplot as plt
from plotting import Plotting
from neural_network import NeuralNetwork
from franke import create_data_franke
from activation_functions import sigmoid, linear, softmax
from optimization import GradientDescent, RMSProp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from main_wisconsin import load_scaled_breast_cancer
from main_franke import get_franke
import numpy as np
import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_folder_in_current_directory(folder_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    new_folder_path = os.path.join(current_directory, folder_name)
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully at: {new_folder_path}")
    else:
        print(f"Folder '{folder_name}' already exists at: {new_folder_path}")

def franke_custom_vs_torch(filepath:str = None, verbose:bool = False):

    epochs = 5
    learning_rate = 0.01
    layer_sizes=[10, 10, 1]
    activation_funcs = [sigmoid, sigmoid, linear]
    batch_size = 4
    
    X_train_scaled, X_test_scaled, z_train, z_test = get_franke(100)

    #### Initializing our own FFNN
    custom_nn = NeuralNetwork(
        input_size=2, 
        layer_sizes=layer_sizes,
        activation_funcs=activation_funcs,
        loss='mse'
    )

    optimizer = GradientDescent(learning_rate=learning_rate, momentum=0.0, nn=custom_nn)

    custom_nn.train(
        data=X_train_scaled, 
        target=z_train, 
        epochs=epochs,
        batch_size=batch_size, 
        optimizer=optimizer, 
        data_val=X_test_scaled, 
        target_val=z_test
    )

    # Getting the training and test/val losses
    custom_train_loss = custom_nn._loss_train_values
    custom_val_loss = custom_nn._loss_val_values

    #### Implementing PyTorch, as identical as possible

    class TorchModel(torch_nn.Module):
        def __init__(self):
            super(TorchModel, self).__init__()
            self.layer1 = torch_nn.Linear(2, layer_sizes[1])
            self.layer2 = torch_nn.Linear(layer_sizes[1], layer_sizes[2])
            self.output = torch_nn.Linear(layer_sizes[2], 1)
            self.sigmoid = torch_nn.Sigmoid()
        
        def forward(self, x):
            x = self.sigmoid(self.layer1(x))
            x = self.sigmoid(self.layer2(x))
            return self.output(x)

    torch_model = TorchModel()

    # Need to initialize weights ~N(0, 1), like we do in our own network
    def initialize_weights(m):
        if isinstance(m, torch_nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)

    torch_model.apply(initialize_weights)

    # Converting to pytorch tensors
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
    z_train_torch = torch.tensor(z_train, dtype=torch.float32).view(-1, 1)
    z_test_torch = torch.tensor(z_test, dtype=torch.float32).view(-1, 1)

    # Making data loader with batch size 4
    train_dataset = TensorDataset(X_train_torch, z_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setting loss function and optimizer
    criterion = torch_nn.MSELoss()
    torch_optimizer = optim.SGD(torch_model.parameters(), lr=learning_rate)

    torch_train_loss = []
    torch_val_loss = []

    for epoch in range(epochs):
        # Training
        torch_model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_z in train_loader:
            torch_optimizer.zero_grad()
            output = torch_model(batch_X)
            loss = criterion(output, batch_z)
            loss.backward()
            torch_optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        # Avg loss
        epoch_train_loss /= len(train_loader.dataset)
        torch_train_loss.append(epoch_train_loss)

        # Validation/test phase
        torch_model.eval()
        with torch.no_grad():
            output_val = torch_model(X_test_torch)
            loss_val = criterion(output_val, z_test_torch).item()
            torch_val_loss.append(loss_val)

        # Print nåværende epoch's loss hvis ønskelig
        if verbose:
            print(f"Epoch {epoch + 1}: "
                  f"Our FFNN Train Loss = {custom_train_loss[epoch]:.6f}, "
                  f"Our FFNN Test Loss = {custom_val_loss[epoch]:.6f}, "
                  f"PyTorch Train Loss = {torch_train_loss[-1]:.6f}, "
                  f"PyTorch Test Loss = {torch_val_loss[-1]:.6f}")

    ############################
    # Plotting Resultater
    ############################

    # Initialiser Plotting-objektet
    plotting = Plotting()

    # Definer x-verdiene (epoker)
    epochs_range = range(1, epochs + 1)

    # Forbered data series som tupler (y-verdier, label)
    data_series = [
        (custom_train_loss, 'Our FFNN Train Loss'),
        (custom_val_loss, 'Our FFNN Test Loss'),
        (torch_train_loss, 'PyTorch Train Loss'),
        (torch_val_loss, 'PyTorch Test MSE')
    ]

    # Definer andre parametere
    x_label = 'Epochs'
    y_label = 'MSE'
    title = 'Training and Validation MSE for Custom NN and PyTorch Model'
    box_string = '10 nevroner, 2 lag, 30 epoker, batch size=4, sigmoid, SGD, loss=MSE'

    # Kall plot_1d funksjonen
    plotting.plot_1d(
        *data_series,
        x_values=epochs_range,
        x_label=x_label,
        y_label=y_label,
        title=title,
        filename=filepath,
        box_string=box_string
    )

    plt.close()


def wisconsin_custom_vs_torch(filepath:str = None, verbose:bool = False):

    epochs = 4
    learning_rate = 0.01
    layer_sizes=[32, 32, 2]
    activation_funcs = [sigmoid, sigmoid, softmax]
    batch_size = 2
    
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()

    #### Initializing our own FFNN
    custom_nn = NeuralNetwork(input_size=30, 
                              layer_sizes=layer_sizes, 
                              activation_funcs=activation_funcs,
                              loss='cross entropy')
    
    optimizer = RMSProp(learning_rate=learning_rate, rho=0.9, nn=custom_nn)

    custom_nn.train(
        data=X_train_scaled, 
        target=y_train, 
        epochs=epochs,
        batch_size=batch_size, 
        optimizer=optimizer, 
        data_val=X_test_scaled, 
        target_val=y_test
    )

    # storing our train and test values
    custom_train_accuracies = custom_nn._loss_train_values
    custom_test_accuracies = custom_nn._loss_val_values

    #### Initializing a Pytorch model
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    class TorchModel(torch_nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(TorchModel, self).__init__()
            self.layer1 = torch_nn.Linear(input_size, hidden_size1)
            self.layer2 = torch_nn.Linear(hidden_size1, hidden_size2)
            self.output = torch_nn.Linear(hidden_size2, output_size)
            self.sigmoid = torch_nn.Sigmoid()
    
        def forward(self, x):
            x = self.sigmoid(self.layer1(x))
            x = self.sigmoid(self.layer2(x))
            return self.output(x)  # In torch CrossEntropyLoss is using Softmax by default internally

    # using same parameters a
    torch_model = TorchModel(input_size=30, hidden_size1=layer_sizes[0], hidden_size2=layer_sizes[1], output_size=layer_sizes[2])

    # Initializing weights ~N(0, 1)
    def initialize_weights(m):
        if isinstance(m, torch_nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
    
    torch_model.apply(initialize_weights)

    # Converting data to torch tensors
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train_labels, dtype=torch.long)  # Bruk klasseindekser
    X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test_labels, dtype=torch.long)

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setting loss function and optimizer like in our custom model
    criterion = torch_nn.CrossEntropyLoss() 
    torch_optimizer = optim.RMSprop(torch_model.parameters(), lr=learning_rate, alpha=0.9)
    #torch_optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)


    torch_train_accuracies = []
    torch_test_accuracies = []

    # Training loop
    for epoch in range(epochs):

        torch_model.train()
        correct_train = 0
        for batch_X, batch_y in train_loader:
            torch_optimizer.zero_grad()
            output = torch_model(batch_X)
    
            # Calculating loss
            loss = criterion(output, batch_y)
            loss.backward()
            torch_optimizer.step()
    
            # Caluculate accuracy for batch
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == batch_y).sum().item()
    
        # Accuracy for epoch
        train_accuracy = correct_train / len(y_train_torch)
        torch_train_accuracies.append(train_accuracy)
    
        # Testing loop
        torch_model.eval()
        correct_test = 0
        with torch.no_grad():
            output_val = torch_model(X_test_torch)
            _, predicted_val = torch.max(output_val, 1)
            correct_test = (predicted_val == y_test_torch).sum().item()
    
            # Test accuracy
            test_accuracy = correct_test / len(y_test_torch)
            torch_test_accuracies.append(test_accuracy)
    
        # If verbose: print results
        if verbose:
            print(f"Epoch {epoch + 1}: "
                  f"Our FFNN Train Accuracy = {custom_train_accuracies[epoch]:.6f}, "
                  f"Our FFNN Test Accuracy = {custom_test_accuracies[epoch]:.6f}, "
                  f"PyTorch Train Accuracy = {torch_train_accuracies[-1]:.6f}, "
                  f"PyTorch Test Accuracy = {torch_test_accuracies[-1]:.6f}")

    ############################
    # Plotting Resultater
    ############################

    # Initialiser Plotting-objektet
    plotting = Plotting()

    # Definer x-verdiene (epoker)
    epochs_range = range(1, epochs + 1)

    # Forbered data series som tupler (y-verdier, label)
    data_series = [
        (custom_train_accuracies, 'Our FFNN Train Accuracy'),
        (custom_test_accuracies, 'Our FFNN Test Accuracy'),
        (torch_train_accuracies, 'PyTorch Train Accuracy'),
        (torch_test_accuracies, 'PyTorch Test Accuracy')
    ]

    # Definer andre parametere
    x_label = 'Epochs'
    y_label = 'Accuracy'
    title = 'Training and Testing Accuracy for Custom NN and PyTorch Model'
    box_string = f'{layer_sizes[0]} neurons, {len(layer_sizes) - 1} hidden layers, {epochs} epochs, batch size={batch_size}, {activation_funcs[0].__name__}, RMSProp, loss=Cross Entropy'

    # Kall plot_1d funksjonen
    plotting.plot_1d(
        *data_series,
        x_values=epochs_range,
        x_label=x_label,
        y_label=y_label,
        title=title,
        filename=filepath,
        box_string=box_string
    )

    plt.close()

if __name__ == '__main__':
    create_folder_in_current_directory('../../figures')
    create_folder_in_current_directory('../../figures/torch_testing')
    filepath = '../../figures/torch_testing'

    print("\nWorking on testing with PyTorch")
    
    franke_custom_vs_torch(filepath='franke_custom_vs_torch.png', verbose=True)
    wisconsin_custom_vs_torch(filepath='wisconsin_custom_vs_torch.png', verbose=True)

    print("\nDone with testing PyTorch.")
