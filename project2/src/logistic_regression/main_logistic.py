from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from plotting import Plotting
from logistic_regression import OwnLogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def accuracy_train_val(filepath:str=None):
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()
    epochs = 40
    n_splits = 10
    batch_size = 4
    kf = KFold(n_splits=n_splits)
    loss_train_sum, loss_val_sum = np.zeros(epochs), np.zeros(epochs)

    print('ACCURACY TRAIN VS VAL WISCONSIN DATA SET; LOGISTIC REGRESSION')

    for i, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        logreg = OwnLogisticRegression(30)
        loss_train, loss_val = logreg.train(X_train_scaled[train_index], y_train[train_index], 
                                             epochs=epochs, 
                                             batch_size=batch_size, 
                                             lmd=0.0, 
                                             x_val=X_train_scaled[val_index], 
                                             target_val=y_train[val_index])
        loss_train_sum += loss_train
        loss_val_sum += loss_val
        print(f'Working on Logistic Regression accuracy train vs val, fold:{i+1}/{n_splits}')
    loss_train_sum /= n_splits
    loss_val_sum /= n_splits
    Plotting().plot_1d((loss_train, 'loss train'),
                     (loss_val, 'loss val'),
                     x_values=list(range(epochs)),
                     x_label='epochs',
                     y_label='accuracy',
                     title='accuracy score',
                     filename=f'{filepath}/logreg_val_vs_train.png',
                     box_string=f'Epochs={epochs}, folds={n_splits}, batch-size={batch_size}, no penalty'
                     )
    plt.close()

def accuracy_lambda_batch_size(filepath:str=None, n_best:int=3):
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_train_scaled, y_train)
    epochs = 40
    n_splits = 5
    batch_sizes = [2**i for i in range(8)]
    lambdas = [10**l for l in range(-8, 0)]

    print('ACCURACY REGULARIZATION AND BATCH SIZE WISCONSIN DATA SET; LOGISTIC REGRESSION')

    heatmap = np.zeros((len(batch_sizes), len(lambdas)))
    logreg_matrix = np.empty_like(heatmap, dtype=OwnLogisticRegression)

    for j, lmd in enumerate(lambdas):
        for k, b in enumerate(batch_sizes):
            logreg = OwnLogisticRegression(30)
            logreg.train(X_train_scaled, y_train, 
                                                epochs=epochs, 
                                                batch_size=b, 
                                                lmd=lmd, 
                                                    )
            y_pred = logreg.feed_forward(X_val_scaled) >= 0.5
            acc = accuracy_score(y_pred=y_pred, y_true=y_val)
            
            heatmap[j, k] += acc
            logreg_matrix[j, k] = logreg

    Plotting().heatmap(heatmap_matrix=heatmap,
                       x_data=lambdas,
                       y_data=batch_sizes,
                       x_label=r'$\lambda$',
                       y_label='batch size',
                       title='',
                        decimals=3,
                        min_patch=False,
                        box_string=f'Epochs={epochs}, folds={n_splits}',
                        filename=f'{filepath}/heatmap_accuracy_batch_size.png'
                       )
    top_n_indices = np.argpartition(heatmap.ravel(), -n_best)[-n_best:]

    plt.close()
    return logreg_matrix.ravel()[top_n_indices], X_test_scaled, y_test

def validate_scikit():
    X_train_scaled, X_test_scaled, y_train, y_test = load_scaled_breast_cancer()
    logreg = OwnLogisticRegression(input_size=30)
    logreg.train(X_train_scaled, target=y_train, epochs=30, batch_size=8) 
    pred = (logreg.feed_forward(X_test_scaled)>=0.5)
    accuracy = accuracy_score(pred, y_test)
    print("Own Test Set Accuracy:", accuracy)

    model = SGDClassifier(loss='log_loss')
    model.fit(X_train_scaled, y_train.ravel())

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_pred, y_test)
    print("SCIKIT Test Set Accuracy:", accuracy)
    print('\n\n\n')

def best_test_accuracy(filepath:str=None):
    best_logreg_models, X_test_scaled, y_test = accuracy_lambda_batch_size(filepath=filepath, n_best=5)
    
    accuracies = []
    model_labels = []

    # Calculate accuracy for each model and store in lists
    for i, model in enumerate(best_logreg_models):
        y_pred = model.feed_forward(X_test_scaled) >= 0.5
        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        accuracies.append(acc)
        model_labels.append(f'{i+1}')

    Plotting().plot_bar(
        bar_labels=model_labels,
        y_values=accuracies,
        y_lims=(0.85, 1),
        x_label='Model',
        y_label='Accuracy',
        title='Accuracy of best models',
        filename=f'{filepath}/bars_best_models.png',
        is_minimal=False
    )

if __name__=="__main__":
    # np.random.seed(42)
    create_folder_in_current_directory('../../figures')
    create_folder_in_current_directory('../../figures/logreg_wisconsin')
    filepath = '../../figures/logreg_wisconsin'
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\nWorking on logistic regression testing")

    # Logistic regression: accuracy on training and validation sets
    accuracy_train_val(filepath=filepath)
    print("Logistic regression: accuracy on training and validation sets completed.")

    # Logistic regression: best test accuracy
    best_test_accuracy(filepath=filepath)
    print("Logistic regression: best test accuracy analysis completed.")

    # Logistic regression: validation using Scikit-Learn
    validate_scikit()
    print("Logistic regression: validation using Scikit-Learn completed.")
