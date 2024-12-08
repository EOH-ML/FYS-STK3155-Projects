from sleep_scoring import SleepScoringTrainerEval
from make_containers import CNNContainer, FFNNContainer
from data_sampler import MakeDataSet 
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sleep_scoring import CNN_Model_1K, CNN_Model_2K, CNN_Model_3K, CNN_Model_41PR, CNN_Model_1W, CNN_Model_11W, CNN_Model_21W, CNN_Model_31W, CNN_Model_41W, CNN_Model_51W, CNN_Model_61W, CNN_Model_71W
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

def train_cnn(cnn_model, window_size=41):
    train_files, test_files, val_files = get_train_test_val()

    c = CNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=window_size,
                    power_ratio=True
                    )
    train, test, val = c.get_data()

    all_labels = [t[1] for t in train]

    train_dataset = MakeDataSet(train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 

    val_dataset = MakeDataSet(val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True) 

    cnn = cnn_model

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, criterion=criterion, optimizer=optimizer, 
                                      train_files=train_files, test_files=test_files, val_files=val_files)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    confusion_matrix = trainer.evaluate(test_dataloader=test_dataloader, conf_matrix_gui=True, class_report=True)
    return confusion_matrix

def test_cnn(test_file, model, modelname):
    c = CNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=41,
                    # power_ratio=True
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 
    criterion = nn.CrossEntropyLoss()

    
    model.load_state_dict(torch.load(f"../models/{modelname}.pth", weights_only=True))
    trainer= SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
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

def confusion_matrix():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    c = CNNContainer(
                    train_files=[],
                    test_files=[test_files[1]],
                    val_files=[],
                    window_size=41,
                    # power_ratio=True
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = CNN_Model_41W() 
    model.load_state_dict(torch.load(f"../models/CNN_Model_41W_27.pth", weights_only=True))
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
    plt.savefig(f'../figures/conf_matrix_cnn_41_window_b2wtm2.png', dpi=300, bbox_inches='tight')

def confusion_matrix_power_ratios():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    c = CNNContainer(
                    train_files=[],
                    test_files=[test_files[1]],
                    val_files=[],
                    window_size=41,
                    power_ratio=True
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model = CNN_Model_41PR() 
    model.load_state_dict(torch.load(f"../models/CNN_Model_41PR_31.pth", weights_only=True))
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
    plt.savefig(f'../figures/conf_matrix_cnn_ratios_41_window_b2wtm2.png', dpi=300, bbox_inches='tight')
    
def kernel_sizes():
    test_files = ['trial_1_mouse_b1aqm2.csv', 'trial_8_mouse_b2wtm2.csv', 'trial_4_mouse_b1wtm2.csv']
    models = [CNN_Model_1K(), CNN_Model_2K(), CNN_Model_3K()]
    cms = []
    for i in range(3):
        cm_1 = test_cnn(test_files[i], model=models[0], modelname=f'CNN_Model_{1}K_{20}')
        cm_2 = test_cnn(test_files[i], model=models[1], modelname=f'CNN_Model_{2}K_{21}')
        cm_3 = test_cnn(test_files[i], model=models[2], modelname=f'CNN_Model_{3}K_{22}')
        cms.append((cm_1, cm_2, cm_3))
    plot_bar_from_cm(cms, labels=['1', '2', '3'], x_label='Size of kernel', filename='bar_plot_kernel_size')

def test_window_model(w_size, test_file, model_name, model):
    c = CNNContainer(
                    train_files=[],
                    test_files=[test_file],
                    val_files=[],
                    window_size=w_size,
                    # power_ratio=True
                    )
    _, test, _ = c.get_data()
    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f"../models/{model_name}.pth", weights_only=True))
    trainer= SleepScoringTrainerEval(model=model, 
                                      criterion=criterion, 
                                      optimizer=None, 
                                      train_files=None,
                                      test_files=None,
                                      val_files=None)
    cm = trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=False, class_report=False, load_state=False)
    return cm

def all_window_sizes(test_file):
    wake, nrem, isleep, rem, avg_f1 = [], [], [], [], []
    sizes = [3, 11, 21, 31, 41, 51, 61, 71]
    models = [CNN_Model_1W(), CNN_Model_11W(), CNN_Model_21W(), CNN_Model_31W(), CNN_Model_41W(), CNN_Model_51W(), CNN_Model_61W(), CNN_Model_71W()]
    model_names = [
                    'CNN_Model_1W_23', 
                    'CNN_Model_11W_24',
                    'CNN_Model_21W_25', 
                    'CNN_Model_31W_26', 
                    'CNN_Model_41W_27',
                    'CNN_Model_51W_28', 
                    'CNN_Model_61W_29',
                    'CNN_Model_71W_30'
                    ]
    for i, w in enumerate(sizes):
        cm = test_window_model(w_size=w, test_file=test_file, model=models[i], model_name=model_names[i])
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
    plt.savefig(f'../figures/1d_plot_f1_score_cnn_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    confusion_matrix()
    plt.close()
    confusion_matrix_power_ratios()
    plt.close()
    different_mice_windows()
    plt.close()