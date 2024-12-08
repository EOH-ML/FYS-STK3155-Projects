import torch
import torch.nn as nn
import torch.nn as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from csv_reader import CSVReader

class SleepScoringTrainerEval:
    """
    A class for training, validating, and testing sleep scoring models with PyTorch.

    This class handles the entire lifecycle of a sleep scoring model, from training and 
    validation to test evaluation. It uses PyTorch for model training, evaluation, and 
    saving the best performing model. The class integrates with a CSVReader utility to 
    track and log model performance metrics, and it also saves model parameters and 
    optimizer details in a text file for future reference.

    Attributes:
        _train_files (list): List of training data files.
        _test_files (list): List of testing data files.
        _val_files (list): List of validation data files.
        _model (nn.Module): The PyTorch model to be trained and evaluated.
        _criterion (nn.Module): The loss function used during training.
        _optimizer (torch.optim.Optimizer): Optimizer to update model weights based on the loss.
        _csv_reader (CSVReader): Instance of CSVReader for logging performance metrics.
        _model_name (str): A unique identifier for the model, used when saving model 
                           states and parameter files.
        _best_acc (float): Tracks the best validation accuracy achieved during training.
        _epochs_no_improve (int): Counter for epochs without improvement to facilitate 
                                  early stopping.

    Methods:
        train(train_dataloader, val_dataloader=None, epochs=10, verbose=False) -> None:
            Train the model for a specified number of epochs. Optionally validates 
            after each epoch and applies early stopping if no improvement is seen 
            after a set number of epochs.

        _evaluate_validation(val_dataloader, n_epochs_stop, verbose) -> bool:
            Evaluate the model on the validation set to determine accuracy. 
            Saves the model state if it achieves the best accuracy so far. 
            Returns True if early stopping should be triggered, else False.

        evaluate(test_dataloader, conf_matrix_gui=False, class_report=False, load_state=True) -> np.ndarray:
            Evaluate the model on the test set. Loads the saved best model state 
            if requested. Optionally prints a classification report, logs metrics 
            to the CSV file, displays a confusion matrix heatmap, and returns 
            the confusion matrix.

        _write_to_file() -> None:
            Writes the model architecture, optimizer details, loss function, and 
            training/testing/validation file information to a text file for record-keeping.
    """
    def __init__(self, model, criterion, optimizer, train_files, test_files, val_files):
        self._train_files = train_files
        self._test_files = test_files
        self._val_files = val_files
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._csv_reader = CSVReader('../models/preferred_models.csv')
        self._model_name = f'{self._model.__class__.__name__}_{self._csv_reader.get_line()}'
    
    def train(self, train_dataloader, val_dataloader=None, epochs=10, verbose=False):
        self._best_acc = 0.0
        n_epochs_stop = 3

        for e in range(epochs):
            self._model.train()
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data

                self._optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
        
                running_loss += loss.item()

                if i % 2000 == 1999 and verbose:    # print every 2000 mini-batches
                    print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            if val_dataloader:
                stop = self._evaluate_validation(val_dataloader, n_epochs_stop=n_epochs_stop, verbose=verbose)
            if stop:
                return
        
    def _evaluate_validation(self, val_dataloader, n_epochs_stop, verbose):
        self._model.eval()
        correcto = 0
        total = 0
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                outputs = self._model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correcto += (predicted==labels).sum().item()

        val_acc = 100 * correcto / total
        if verbose:
            print(f'Validation Accuracy: {val_acc:2f}%')
        
        if val_acc > self._best_acc:
            self._best_acc = val_acc
            torch.save(self._model.state_dict(), f'../models/{self._model_name}.pth')
            print('Best model saved')
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1

        if self._epochs_no_improve == n_epochs_stop:
            print('Early stopping, best model is saved')
            return True
        return False

    def evaluate(self, test_dataloader, conf_matrix_gui=False, class_report=False, load_state=True):
        classes = (0, 1, 2, 3)
        stages = ['Wake', 'NREM', 'IS', 'REM']
        correcto = 0
        total = 0
        if load_state:
            self._model.load_state_dict(torch.load(f'../models/{self._model_name}.pth', weights_only=True))
        self._model.eval()

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        num_classes = len(classes)
        confusion_matrix = np.zeros((num_classes, num_classes))
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                outputs = self._model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correcto += (predicted==labels).sum().item()
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                    confusion_matrix[label, prediction] += 1
                    y_true.append(label)
                    y_pred.append(prediction)

        cp = classification_report(y_true=y_true, y_pred=y_pred, target_names=stages)
        if class_report:
            print('Classification report:')
            print(cp)
        
        if load_state:
            self._csv_reader.add_cp(model_name=self._model_name, cp=cp)
            self._write_to_file()
        
        print(f'Accuracy of the network on the test data: {100 * correcto // total} %')

        # Confuison matrix
        if conf_matrix_gui:
            sns.heatmap(confusion_matrix.astype(int), annot=True, cmap='Blues', fmt='d', xticklabels=stages, yticklabels=stages) 
            plt.xlabel('Predicted')
            plt.ylabel('True')

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {stages[classname]:5s} is {accuracy:.1f} %')
        
        plt.show()
        return confusion_matrix
    
    def _write_to_file(self):
        """
        Writes model, optimizer, loss function, and hyperparameter details to a text file.
        
        :param model: PyTorch model (e.g., nn.Module)
        :param optimizer: Optimizer used for training
        :param criterion: Loss function
        :param hyperparameters: Dictionary of additional hyperparameters
        :param file_path: Path to save the .txt file
        """
        with open(f'../parameters/{self._model_name}.txt', 'w') as f:
            # Write model architecture
            f.write("Model Architecture:\n")
            f.write(str(self._model))
            f.write("\n\n")

            # Write optimizer details
            f.write("Optimizer Details:\n")
            optimizer_hyperparams = self._optimizer.state_dict()["param_groups"][0]
            f.write(f"Optimizer Type: {type(self._optimizer).__name__}\n")
            f.write("Hyperparameters:\n")
            for key in ["lr", "betas", "eps", "weight_decay", "amsgrad"]:
                if key in optimizer_hyperparams:
                    f.write(f"  {key}: {optimizer_hyperparams[key]}\n")
            f.write("\n")

            # Write loss function details
            f.write("Loss Function:\n")
            f.write(f"{type(self._criterion).__name__}\n")
            f.write("\n")

            # Write model parameters
            f.write("Model Parameters:\n")
            for name, param in self._model.named_parameters():
                f.write(f"{name}: {list(param.shape)}\n")
            f.write("\n")

            f.write("Train mice:\n")
            f.write(f'{self._train_files}\n')

            f.write("Test mice:\n")
            f.write(f'{self._test_files}\n')

            f.write("Val mice:\n")
            f.write(f'{self._val_files}')

"""
All classes below are made for testing different structures of Neural Networks.
"""

class SleepScoringCNN_1E(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 16, kernel_size=2)           # Reduce spatial dimensions

        self._fc1 = nn.Linear(128, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = F.relu(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(F.relu(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x, 1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x
    
class SleepScoringCNN_1O(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.AvgPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._fc1 = nn.Linear(64, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = F.relu(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(F.relu(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x
    
class SleepScoringCNN_OE(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.AvgPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._fc1 = nn.Linear(64, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = F.relu(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(F.relu(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x

class SleepScoringCNN_2O(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable weights for each input channel
        # self.channel_weights = nn.Parameter(torch.ones(2))

        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 12, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                                   # Halve spatial dimensions
        self._conv2 = nn.Conv2d(12, 4, kernel_size=2)                      # Reduce spatial dimensions

        self._fc1 = nn.Linear(40, 32)                                    # Reduced fully connected layer
        self._fc2 = nn.Linear(32, 16)                                    # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                                     # Output 4 classes

    def forward(self, x):
        # Apply channel weights
        # weighted_x = x * self.channel_weights.view(1, -1, 1, 1)

        x = F.relu(self._conv1(x))                              # No change in spatial dimensions
        x = self._pool(x)                                               # Halve dimensions
        x = self._pool(F.relu(self._conv2(x)))                          # Further reduce dimensions
        x = torch.flatten(x, 1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x

class SleepScoringCNN_3O_kodd(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable weights for each input channel
        # self.channel_weights = nn.Parameter(torch.ones(2))

        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.LazyConv2d(2, 2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.AdaptiveAvgPool2d(2)                                   # Halve spatial dimensions
        self._conv2 = nn.LazyConv2d(12, 2, padding=2)                      # Reduce spatial dimensions

        self._fc1 = nn.Linear(48, 32)                                    # Reduced fully connected layer
        self._fc2 = nn.Linear(32, 16)                                    # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                                     # Output 4 classes

    def forward(self, x):
        # Apply channel weights
        # weighted_x = x * self.channel_weights.view(1, -1, 1, 1)

        x = F.hardshrink(self._conv1(x))                              # No change in spatial dimensions
        x = self._pool(x)                                               # Halve dimensions
        x = self._pool(F.silu(self._conv2(x)))                          # Further reduce dimensions
        x = torch.flatten(x, 1)
        x = F.gelu(self._fc1(x))
        x = F.mish(self._fc2(x))
        x = self._fc3(x)
        return x
    
class BiLSTMModel1H(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        """
        Bidirectional LSTM model for sleep stage classification.

        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of classes (sleep stages).
            dropout (float): Dropout rate for regularization.
        """
        super(BiLSTMModel1H, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirection

    def forward(self, x):
        """
        Forward pass through the BiLSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        lstm_out, _ = self.lstm(x)  # LSTM output and hidden states
        last_hidden_state = lstm_out[:, -1, :]  # Use the last time step
        out = self.fc(last_hidden_state)
        return out

class BiLSTMModel1E(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTMModel1E, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

class NNModel1O(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(121*6, 64)  
        self.relu = nn.ReLU()                      
        self.fc2 = nn.Linear(64, 11)
        self.fc3 = nn.Linear(11, 4) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class NNModel1E(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(75, 64)  
        self.sigmoid = nn.Sigmoid()                      
        self.fc2 = nn.Linear(64, 11)
        self.fc3 = nn.Linear(11, 4) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

class CNN_Model_1(nn.Module):
    def __init__(self, act):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = act
        self._fc1 = nn.Linear(80, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class NN_Model_1(nn.Module):
    def __init__(self, act, window_size=15):
        super().__init__()
        
        self.fc1 = nn.Linear(window_size*5, 64)  
        self.act = act
        self.fc2 = nn.Linear(64, 11)
        self.fc3 = nn.Linear(11, 4) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class CNN_Model_1K(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=1, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=1)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(176, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_2K(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(80, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_3K(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=1)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(80, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_1W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(8, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_11W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(24, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_21W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(40, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_31W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(64, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_41W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(80, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_51W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(104, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_61W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(120, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_71W(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(2, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(144, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x

class CNN_Model_41PR(nn.Module):
    def __init__(self):
        super().__init__()
        # Padding ensures output dimensions match input dimensions
        self._conv1 = nn.Conv2d(3, 8, kernel_size=2, padding=2, stride=1)  # Maintain spatial dimensions
        self._pool = nn.MaxPool2d(2, 2)                         # Halve spatial dimensions
        self._conv2 = nn.Conv2d(8, 8, kernel_size=2)           # Reduce spatial dimensions

        self._act = F.relu
        self._fc1 = nn.Linear(80, 64)                  # Reduced fully connected layer
        self._fc2 = nn.Linear(64, 16)                          # Compact hidden layer
        self._fc3 = nn.Linear(16, 4)                           # Output 4 classes

    def forward(self, x):
        x = self._act(self._conv1(x))                             # No change in spatial dimensions
        x = self._pool(x)                                      # Halve dimensions
        x = self._pool(self._act(self._conv2(x)))                 # Further reduce dimensions
        x = torch.flatten(x,  1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return x