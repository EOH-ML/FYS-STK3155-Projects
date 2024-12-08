from load_mice import MouseLoader
from preprocessor import Preprocessor
from sleep_scoring import SleepScoringTrainerEval
from make_containers import CNNContainer, RNNSequences, FFNNContainer
from data_sampler import MakeDataSet 
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
This file was made for preliminary testing, and is not made to be ran. This shows our initial workflow on how to 
make and test different models.
"""


# Håkons RNN_modell
from sleep_scoring import BiLSTMModel1H
def model1H():
    train_files=['trial_11_mouse_evm6.csv', 'trial_12_mouse_evm6.csv', 'trial_13_mouse_evm1.csv',
                 'trial_14_mouse_evm1.csv', 'trial_16_mouse_evm2.csv', 'trial_15_mouse_evm2.csv', 
                 'trial_19_mouse_evm4.csv', 'trial_20_mouse_evm4.csv', 'trial_21_mouse_evm4.csv',
                 'trial_24_mouse_evm1.csv', 'trial_25_mouse_evm1.csv', 'trial_26_mouse_evm2.csv',
                 'trial_27_mouse_evm2.csv']
    test_files=['trial_8_mouse_b2wtm2.csv', 'trial_18_mouse_evf3.csv', 'trial_9_mouse_evf3.csv', 'trial_10_mouse_evf3.csv']
    val_files=['trial_5_mouse_b2aqm2.csv', 'trial_2_mouse_b1aqm1.csv', 'trial_1_mouse_b1aqm2.csv', 'trial_3_mouse_b1wtm1.csv']

    c = RNNSequences(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    sequence_length=10,
                    overlap=2,
                    power_ratio=True
                    )
    
    train_dataloader, test_dataloader, val_dataloader = c.get_loaders(batch_size=4)

    input_size = 9  # Antall funksjoner per tidssteg
    hidden_size = 64
    num_layers = 4
    output_size = 5  # Antall klasser
    dropout = 0.5

    rnn_model = BiLSTMModel1H(input_size, hidden_size, num_layers, output_size, dropout)

    optimizer = optim.Adam(rnn_model.parameters(), lr=0.002)

    trainer = SleepScoringTrainerEval(model=rnn_model, criterion=nn.CrossEntropyLoss(), optimizer=optimizer, 
                                      train_files=train_files, test_files=test_files, val_files=val_files)
    trainer.train(train_dataloader, val_dataloader, epochs=20, verbose=True)
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)


    # MODEL 1 - lagd 22.11
from sleep_scoring import SleepScoringCNN_1E
def model1e():
    c = CNNContainer(
                    train_files=['trial_11_mouse_evm6.csv', 'trial_10_mouse_evf3.csv', 'trial_15_mouse_evm2.csv'],
                    test_files=['trial_13_mouse_evm1.csv'],
                    val_files=['trial_14_mouse_evm1.csv'],
                    window_size=31,
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

    cnn = SleepScoringCNN_1E()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, criterion=criterion, optimizer=optimizer)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)

    # MODEL 2 - Oscar - lagd 26.11
from sleep_scoring import SleepScoringCNN_2O
def model2O():
    c = CNNContainer(
                    train_files=['trial_11_mouse_evm6.csv', 'trial_10_mouse_evf3.csv', 'trial_15_mouse_evm2.csv', 'trial_9_mouse_evf3.csv', 'trial_1_mouse_b1aqm2.csv', 'trial_18_mouse_evf3.csv'],
                    test_files=['trial_13_mouse_evm1.csv'],
                    val_files=['trial_5_mouse_b2aqm2.csv'],
                    window_size=31,
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

    cnn = SleepScoringCNN_2O()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, criterion=criterion, optimizer=optimizer)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)

    # MODEL 3 Oscar - lagd 26.11
from sleep_scoring import SleepScoringCNN_3O_kodd
def model3O():
    train_files=['trial_11_mouse_evm6.csv', 'trial_12_mouse_evm6.csv', 'trial_13_mouse_evm1.csv',
                 'trial_14_mouse_evm1.csv', 'trial_16_mouse_evm2.csv', 'trial_15_mouse_evm2.csv', 
                 'trial_19_mouse_evm4.csv', 'trial_20_mouse_evm4.csv', 'trial_21_mouse_evm4.csv',
                 'trial_24_mouse_evm1.csv', 'trial_25_mouse_evm1.csv', 'trial_26_mouse_evm2.csv',
                 'trial_27_mouse_evm2.csv']
    test_files=['trial_8_mouse_b2wtm2.csv', 'trial_18_mouse_evf3.csv']
    val_files=['trial_5_mouse_b2aqm2.csv', 'trial_2_mouse_b1aqm1.csv']

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

    cnn = SleepScoringCNN_3O_kodd()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.Adam(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, criterion=criterion, optimizer=optimizer, 
                                      train_files=train_files, test_files=test_files, val_files=val_files)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)

from sleep_scoring import SleepScoringCNN_OE
def modelOE():
    train_files = ['trial_11_mouse_evm6.csv', 'trial_10_mouse_evf3.csv', 'trial_15_mouse_evm2.csv']
    test_files=['trial_13_mouse_evm1.csv']
    val_files=['trial_14_mouse_evm1.csv']
    c = CNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=31,
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

    cnn = SleepScoringCNN_OE()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = optim.RMSprop(cnn.parameters())
    
    trainer = SleepScoringTrainerEval(model=cnn, 
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      train_files=train_files,
                                      test_files=test_files,
                                      val_files=val_files)
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=30, verbose=True)
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)

from sleep_scoring import BiLSTMModel1E
def modelERNN1():
    train_files = ['trial_11_mouse_evm6.csv', 'trial_10_mouse_evf3.csv', 'trial_15_mouse_evm2.csv']
    test_files=['trial_13_mouse_evm1.csv']
    val_files=['trial_14_mouse_evm1.csv']

    ml= MouseLoader()
    train = ml.load_mouse_data('trial_11_mouse_evm6.csv')
    test = ml.load_mouse_data('trial_10_mouse_evf3.csv')

    train_prep = Preprocessor(train)
    train_prep.remove_unknowns()
    train_prep.standard_scale()
    train_data = train_prep.get_mouse_data()

    test_prep = Preprocessor(test)
    test_prep.remove_unknowns()
    test_prep.standard_scale()
    test_data = test_prep.get_mouse_data()

    look_back = 60 # choose sequence length
    x_train, y_train, x_test, y_test = load_data(train_data, look_back)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    input_dim = 6
    hidden_dim = 32
    num_layers = 2 
    output_dim = 1

    model = BiLSTMModel1E(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1  

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()
        
        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        optimiser.step()
        # Update parameters
    
    y_test_pred = model(x_test)
    print(accuracy_score(y_pred=y_test_pred, y_true=y_test))

def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
from sleep_scoring import NNModel1O
def modelNNO():
    train_files=['trial_11_mouse_evm6.csv', 'trial_12_mouse_evm6.csv', 'trial_13_mouse_evm1.csv',
                 'trial_14_mouse_evm1.csv', 'trial_16_mouse_evm2.csv', 'trial_15_mouse_evm2.csv', 
                 'trial_19_mouse_evm4.csv', 'trial_20_mouse_evm4.csv', 'trial_21_mouse_evm4.csv',
                 'trial_24_mouse_evm1.csv', 'trial_25_mouse_evm1.csv', 'trial_26_mouse_evm2.csv',
                 'trial_27_mouse_evm2.csv']
    test_files=['trial_8_mouse_b2wtm2.csv', 'trial_18_mouse_evf3.csv']
    val_files=['trial_5_mouse_b2aqm2.csv', 'trial_2_mouse_b1aqm1.csv']

    c = FFNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=15,
                    features=['sigma_power', 'emg_power', 'delta_theta_ratio', 'sigma_beta_ratio', 'theta_sigma_ratio', 'emg_delta_ratio']
                    )
    train, test, val = c.get_data()
    print(train[0])
    all_labels = [t[1] for t in train]

    train_dataset = MakeDataSet(train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 

    test_dataset = MakeDataSet(test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) 

    val_dataset = MakeDataSet(val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True) 

    ffnn = NNModel1O()

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
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)
     
from sleep_scoring import NNModel1E
def modelNNE():
    train_files=['trial_11_mouse_evm6.csv', 'trial_12_mouse_evm6.csv', 'trial_13_mouse_evm1.csv',
                 'trial_14_mouse_evm1.csv', 'trial_16_mouse_evm2.csv', 'trial_15_mouse_evm2.csv', 
                 'trial_19_mouse_evm4.csv', 'trial_20_mouse_evm4.csv', 'trial_21_mouse_evm4.csv',
                 'trial_24_mouse_evm1.csv', 'trial_25_mouse_evm1.csv', 'trial_26_mouse_evm2.csv',
                 'trial_27_mouse_evm2.csv']
    test_files=['trial_8_mouse_b2wtm2.csv', 'trial_18_mouse_evf3.csv']
    val_files=['trial_5_mouse_b2aqm2.csv', 'trial_2_mouse_b1aqm1.csv']

    c = FFNNContainer(
                    train_files=train_files,
                    test_files=test_files,
                    val_files=val_files,
                    window_size=15,
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

    ffnn = NNModel1E()

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
    trainer.evaluate(test_dataloader=test_dataloader,conf_matrix_gui=True, class_report=True)
