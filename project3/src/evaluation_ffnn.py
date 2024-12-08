from sleep_scoring import SleepScoringTrainerEval
from make_containers import CNNContainer, FFNNContainer
from data_sampler import MakeDataSet 
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sleep_scoring import CNN_Model_1, NN_Model_1, NNModel1O
import matplotlib.pyplot as plt
import seaborn as sns

def get_train_test_val():
    train_files=['trial_11_mouse_evm6.csv', 'trial_12_mouse_evm6.csv', 'trial_13_mouse_evm1.csv',
                 'trial_14_mouse_evm1.csv', 'trial_16_mouse_evm2.csv', 'trial_15_mouse_evm2.csv', 
                 'trial_19_mouse_evm4.csv', 'trial_20_mouse_evm4.csv', 'trial_21_mouse_evm4.csv',
                 'trial_24_mouse_evm1.csv', 'trial_25_mouse_evm1.csv', 'trial_26_mouse_evm2.csv',
                 'trial_27_mouse_evm2.csv']
    test_files=['trial_8_mouse_b2wtm2.csv', 'trial_18_mouse_evf3.csv']
    val_files=['trial_5_mouse_b2aqm2.csv', 'trial_2_mouse_b1aqm1.csv']
    return train_files, test_files, val_files

def model_1_cnn(act):
    train_files, test_files, val_files = get_train_test_val()

    c = CNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=41,
                    # power_ratio=True
                    )
    train, test, val = c.get_data()

    all_labels = [t[1] for t in train]

    train_dataset = MakeDataSet(train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 

    val_dataset = MakeDataSet(val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True) 

    cnn = CNN_Model_1(act)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, criterion=criterion, optimizer=optimizer, 
                                      train_files=train_files, test_files=test_files, val_files=val_files)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    confusion_matrix = trainer.evaluate(test_dataloader=test_dataloader, conf_matrix_gui=True, class_report=True)
    return confusion_matrix

def model_NN(act, name=False, window_size=15):
    train_files, test_files, val_files = get_train_test_val()

    c = FFNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=window_size,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    train, test, val = c.get_data()
    all_labels = [t[1] for t in train]

    train_dataset = MakeDataSet(train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 

    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 

    val_dataset = MakeDataSet(val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True) 

    ffnn = NN_Model_1(act, window_size)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(ffnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=ffnn, 
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      train_files=train_files,
                                      test_files=test_files,
                                      val_files=val_files)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    cm = trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=True)
    if name:
        return trainer._model_name
    return cm


def plot_bar_from_cm(matrix_groups, labels, x_label, filename):
    # Transpose the structure to group matrices by index (0, 1, 2 across groups)
    transposed_matrices = list(zip(*matrix_groups))

    average_accuracies = []
    all_accuracies = []

    # Calculate accuracies for each index across all groups
    for index, matrices_at_index in enumerate(transposed_matrices):
        index_accuracies = []

        for matrix in matrices_at_index:
            matrix = np.array(matrix)
            if matrix.shape != (4, 4):
                raise ValueError(f"Confusion matrix does not have shape (4, 4): {matrix.shape}")

            # Calculate accuracy
            total = np.sum(matrix)
            correct = np.trace(matrix)  # Sum of diagonal elements
            accuracy = correct / total if total > 0 else 0
            index_accuracies.append(accuracy)

        all_accuracies.append(index_accuracies)
        average_accuracies.append(np.mean(index_accuracies))

    # Plot the bar chart
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.figure(figsize=(6, 6))

    x_positions = np.arange(len(labels))
    bar_width = 0.2  # Thinner bars

    # Plot the average accuracies as bars
    plt.bar(x_positions, average_accuracies, width=bar_width, alpha=0.9, color='gray', edgecolor='black', label='Average Accuracy')

    # Overlay individual accuracies slightly offset to the side
    colors = ['blue', 'green', 'purple', 'orange']
    test_files = ['trial_1_b1aqm2', 'trial_8_b2wtm2', 'trial_4_b1wtm2']
    for i, (index_accuracies, x_pos) in enumerate(zip(all_accuracies, x_positions)):
        for j, accuracy in enumerate(index_accuracies):
            offset = (j - 1) * bar_width / 2  # Offset bars slightly to avoid overlap
            plt.bar(x_pos + offset, accuracy, width=bar_width / 2, alpha=0.6, color=colors[j], label=test_files[j] if i == 0 else "")

    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)  # Accuracies range from 0 to 1
    plt.xticks(x_positions, labels)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=1, frameon=True)

    # Annotate bars with average values
    for i, v in enumerate(average_accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=16)

    plt.savefig(f'../figures/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_state_act_func(test_file):
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=15,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()

    relu_model = NN_Model_1(F.relu) 
    relu_model.load_state_dict(torch.load("../models/NN_Model_1_6.pth", weights_only=True))
    trainer_relu = SleepScoringTrainerEval(model=relu_model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_relu = trainer_relu.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    
    sigmoid_model = NN_Model_1(F.sigmoid) 
    sigmoid_model.load_state_dict(torch.load("../models/NN_Model_1_7.pth", weights_only=True))
    trainer_sigmoid = SleepScoringTrainerEval(model=sigmoid_model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_sigmoid = trainer_sigmoid.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    lrelu_model = NN_Model_1(F.leaky_relu) 
    lrelu_model.load_state_dict(torch.load("../models/NN_Model_1_8.pth", weights_only=True))
    trainer_lrelu = SleepScoringTrainerEval(model=lrelu_model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_lrelu = trainer_lrelu.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    return cm_relu, cm_sigmoid, cm_lrelu

def test_state_optmizers(test_file):
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=15,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()

    adam = NN_Model_1(F.relu) 
    adam.load_state_dict(torch.load("../models/NN_Model_1_6.pth", weights_only=True))
    trainer_adam = SleepScoringTrainerEval(model=adam, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_adam = trainer_adam.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)

    rmsprop = NN_Model_1(F.relu) 
    rmsprop.load_state_dict(torch.load("../models/NN_Model_1_9.pth", weights_only=True))
    trainer_rmsprop = SleepScoringTrainerEval(model=rmsprop, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_rmsprop = trainer_rmsprop.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    return cm_adam, cm_rmsprop

def act_funcs():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    cms = []
    for i in range(3):
        cms.append(test_state_act_func(test_files[i]))
    plot_bar_from_cm(cms, labels=['ReLU', 'sigmoid', 'LeakyReLU'], x_label='Activation Functions', filename='bar_plot_act_func')

def optimizers():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    cms = []
    for i in range(3):
        cms.append(test_state_optmizers(test_files[i]))
    plot_bar_from_cm(cms, labels=['Adam', 'RmsProp'], x_label='Optimizer', filename='bar_plot_optimizer')

def windowsizes():
    model_names = []
    sizes = [121, 141, 151]
    for i in sizes:
        name = model_NN(act=F.relu, window_size=i)
        model_names.append(name)

def colors_lines():
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']  # Blue, Green, Red, Orange, Purple, Brown
    line_styles = ['-', '--', '-.', ':']
    return colors, line_styles

def plot_1d(*fs, x_values, x_label, y_label, title, filename=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.figure(figsize=(6, 6))
    colors, line_styles = colors_lines()

    if title:
        plt.title(rf'{title}')

    else:
        for i, f in enumerate(fs):
            plt.plot(x_values, f[0], label=rf'{f[1]}', linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
    

    plt.xlabel(rf'{x_label}')
    plt.ylabel(rf'{y_label}')

    plt.legend()
    plt.savefig(f'../figures/1d_plot_f1_score_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_window_model(w_size, i, test_file):
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=w_size,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = NN_Model_1(F.relu, window_size=w_size) 
    model.load_state_dict(torch.load(f"../models/NN_Model_1_{10+i}.pth", weights_only=True))
    trainer_relu = SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer_relu.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    return cm

def all_window_sizes(test_file):
    wake, nrem, isleep, rem, avg_f1 = [], [], [], [], []
    sizes = [1, 5, 15, 31, 45, 71, 101, 121, 141, 151]
    for i, w in enumerate(sizes):
        cm = test_window_model(w, i, test_file)
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP

        # Calculate F1-scores only
        f1 = np.where((TP+FP==0)&(TP+FN==0), 0,  # Handle zero-division case
                    np.where(TP+FP==0, 0,
                    np.where(TP+FN==0, 0,
                    2*TP/(2*TP + FP + FN))))
        wake.append(f1[0])
        nrem.append(f1[1])
        isleep.append(f1[2])
        rem.append(f1[3])

        f1_avg = np.mean(f1)
        avg_f1.append(f1_avg)

    plot_1d((wake, 'Wake'),
            (nrem, 'NREM'),
            (isleep, 'IS'),
            (rem, 'REM'),
            (avg_f1, 'Avg F1'),
            x_values=sizes,
            x_label='Window size',
            y_label='F1-Score',
            title=None,
            filename=test_file
            )

def different_mice_windows():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    for test_file in test_files:
        all_window_sizes(test_file)

def confusion_matrix():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_files[1]],
                    val_files=[],
                    window_size=121,
                    features=['sigma_power', 'emg_power', 'delta_theta_ratio', 'sigma_beta_ratio', 'theta_sigma_ratio', 'emg_delta_ratio']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = NNModel1O() 
    model.load_state_dict(torch.load(f"../models/NNModel1O_51.pth", weights_only=True))
    trainer= SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=True, load_state=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm.astype(int), annot=True, cmap='Blues', fmt='d', 
                xticklabels=('Wake', 'NREM', 'IS', 'REM'), yticklabels=('Wake', 'NREM', 'IS', 'REM'),
                cbar=None) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'../figures/feature_engineered_confusion.png', dpi=300, bbox_inches='tight')

def confusion_matrix_not_engineered():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_files[1]],
                    val_files=[],
                    window_size=151,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = NN_Model_1(F.relu, window_size=151) 
    model.load_state_dict(torch.load(f"../models/NN_Model_1_19.pth", weights_only=True))
    trainer= SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=True, load_state=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm.astype(int), annot=True, cmap='Blues', fmt='d', 
                xticklabels=('Wake', 'NREM', 'IS', 'REM'), yticklabels=('Wake', 'NREM', 'IS', 'REM'),
                cbar=None) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'../figures/confusion_matrix_model_19.png', dpi=300, bbox_inches='tight')

def test_state_feature_engineering(test_file):
    c_no_ratios = FFNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=151,
                    features=['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']
                    )
    _, test, _ = c_no_ratios.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()

    model = NN_Model_1(F.relu, window_size=151) 
    model.load_state_dict(torch.load("../models/NN_Model_1_19.pth", weights_only=True))
    trainer_adam = SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm_adam = trainer_adam.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=True, load_state=False)
    c = FFNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=121,
                    features=['sigma_power', 'emg_power', 'delta_theta_ratio', 'sigma_beta_ratio', 'theta_sigma_ratio', 'emg_delta_ratio']
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = NNModel1O() 
    model.load_state_dict(torch.load(f"../models/NNModel1O_51.pth", weights_only=True))
    trainer= SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer.evaluate(test_dataloader=test_dataloader, conf_matrix_gui=False, class_report=True, load_state=False)
    return cm_adam, cm

def feature_engingeering():
    test_files = ['trial_8_mouse_b2wtm2.csv']
    cms = []
    for i in range(1):
        cms.append(test_state_feature_engineering(test_files[i]))
    plot_bar_from_cm(cms, labels=['Model 1', 'Model 2'], x_label='Activation Functions', filename='bar_plot_feature_engineering')

if __name__ == "__main__":
    act_funcs()
    optimizers()
    different_mice_windows()
    confusion_matrix()
    plt.close()
    feature_engingeering()
    confusion_matrix_not_engineered()
