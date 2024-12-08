from load_mice import MouseLoader
from preprocessor import Preprocessor
import torch
from data_sampler import MakeDataSet
from torch.utils.data import DataLoader
import numpy as np

class CNNContainer:
    """
    A container class for preparing EEG/EMG data for CNN-based sleep stage classification.

    This class loads, processes, and segments time-series data into fixed-size windows. It supports 
    the inclusion of additional power ratio features for enhanced model inputs. The processed data 
    (training, testing, and validation sets) are stored as lists of tuples, where each tuple consists 
    of a data batch (window) and its corresponding sleep stage label.

    Attributes:
        _train_container (list): A list of (data_window, label) tuples for the training dataset.
        _test_container (list): A list of (data_window, label) tuples for the testing dataset.
        _val_container (list): A list of (data_window, label) tuples for the validation dataset.
        _window_size (int): The size of the time window (in samples) to segment the data into. 
                            Must be an odd number to allow for a center data point label alignment.
        _power_ratio (bool): A flag indicating whether to compute and include power ratio features 
                             along with the base EEG/EMG power features.

    Methods:
        get_data() -> (list, list, list):
            Returns the preprocessed and windowed training, testing, and validation data.

    """
    def __init__(self, train_files=[], test_files=[], val_files=[], window_size=5, power_ratio=False):
        self._train_container = []
        self._test_container = []
        self._val_container = []

        if window_size % 2 != 1:
            raise ValueError('Window size must be an odd number!')
        self._window_size = window_size
        self._power_ratio = power_ratio

        self._load(train_files, self._train_container)
        self._load(test_files, self._test_container)
        self._load(val_files, self._val_container)
    
    def _load(self, files, container):
        mouseloader = MouseLoader()
        power_ratios = ['beta_delta_ratio', 'delta_beta_ratio', 'beta_theta_ratio', 'theta_beta_ratio']
        for f in files:
            df = mouseloader.load_mouse_data(f)
            df_ecog = df[['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'sleep_episode']]
            df_emg = df[['emg_power', 'sleep_episode']]
            prep_ecog, prep_emg = Preprocessor(df_ecog), Preprocessor(df_emg)

            prep_ecog.remove_unknowns()
            prep_emg.remove_unknowns()

            if self._power_ratio:
                prep_ecog.power_ratios(power_ratios)

            prep_ecog.log_transform(power_ratio=self._power_ratio)
            prep_emg.log_transform()

            prep_ecog.standard_scale()
            prep_emg.standard_scale()

            df_ecog = prep_ecog.get_mouse_data()
            df_emg = prep_emg.get_mouse_data()

            if self._power_ratio:
                df_power_ratios = df_ecog[power_ratios]
                df_power_ratios = torch.tensor(df_power_ratios[power_ratios].values).float()

            df_ecog = df_ecog[['delta_power', 'theta_power', 'sigma_power', 'beta_power']]

            for i in range(1, 4): # Mulig dette må være variabelt
                df_emg[f'emg_power_{i}'] = df_emg['emg_power']

            ecog = torch.tensor(df_ecog[['delta_power', 'theta_power', 'sigma_power', 'beta_power']].values).float()
            emg = torch.tensor(df_emg[['emg_power', 'emg_power_1', 'emg_power_2', 'emg_power_3']].values).float()
            labels = torch.tensor(df_emg[['sleep_episode']].values)

            combined = torch.stack([ecog.T, emg.T], dim=0)
            if self._power_ratio:
                combined = torch.stack([ecog.T, emg.T, df_power_ratios.T], dim=0)
            self._make_frames(combined, labels, container)
    
    def _make_frames(self, data, labels, container):
        for i in range(0, data[0].size(1)-self._window_size):
            batch = data[:, :, i:self._window_size+i]
            label = labels[i+self._window_size//2].item()
            container.append((batch, label))

    def get_data(self):
        return self._train_container, self._test_container, self._val_container
    
class RNNSequences:
    """
    A class for preparing time-series data as sequences suitable for RNN-based models.

    This class reads EEG/EMG data files, preprocesses them, and slices the data into sequences
    of a specified length with optional overlap. The sequences can optionally include power ratio 
    features, and are returned as tuples of (sequence_data, label). Sequences and labels can 
    then be easily loaded into PyTorch DataLoaders for model training, validation, and testing.

    Attributes:
        _sequence_length (int): The length of each time-series sequence (number of timesteps).
        _overlap (int): The number of overlapping timesteps between consecutive sequences.
        _power_ratio (bool): Whether to compute and include additional power ratio features 
                             in the data.
        _train_container (list): A list of (sequence, label) tuples for the training set.
        _test_container (list): A list of (sequence, label) tuples for the testing set.
        _val_container (list): A list of (sequence, label) tuples for the validation set.

    Methods:
        get_loaders(batch_size=32) -> (DataLoader, DataLoader, DataLoader):
            Returns PyTorch DataLoaders for the training, testing, and validation sets.
    """

    def __init__(self, train_files=[], test_files=[], val_files=[], sequence_length=10, overlap=5, power_ratio=True):
        self._sequence_length = sequence_length
        self._overlap = overlap
        self._power_ratio = power_ratio

        self._train_container = self._load(train_files)
        self._test_container = self._load(test_files)
        self._val_container = self._load(val_files)

    def _load(self, files):
        mouseloader = MouseLoader()
        power_ratios = ['beta_delta_ratio', 'delta_beta_ratio', 'beta_theta_ratio', 'theta_beta_ratio']
        all_sequences, all_labels = [], []

        for f in files:
            df = mouseloader.load_mouse_data(f)
            df_ecog = df[['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'sleep_episode']]
            df_emg = df[['emg_power', 'sleep_episode']]
            prep_ecog, prep_emg = Preprocessor(df_ecog), Preprocessor(df_emg)

            prep_ecog.remove_unknowns()
            prep_emg.remove_unknowns()

            if self._power_ratio:
                prep_ecog.power_ratios(power_ratios)

            prep_ecog.log_transform(power_ratio=self._power_ratio)
            prep_emg.log_transform()

            prep_ecog.standard_scale()
            prep_emg.standard_scale()

            df_ecog = prep_ecog.get_mouse_data()
            df_emg = prep_emg.get_mouse_data()

            # Create sequences
            features_ecog = df_ecog.iloc[:, :-1].values
            features_emg = df_emg.iloc[:, :-1].values
            labels = df_ecog.iloc[:, -1].values  # Assuming labels are the same for ECoG and EMG

            step = self._sequence_length - self._overlap
            for i in range(0, len(features_ecog) - self._sequence_length + 1, step):
                ecog_seq = features_ecog[i:i + self._sequence_length]
                emg_seq = features_emg[i:i + self._sequence_length]
                seq = np.concatenate((ecog_seq, emg_seq), axis=1).astype(np.float32)  # Convert to float32
                # print(f"Sequence shape: {seq.shape}")
                target = labels[i + self._sequence_length // 2]  # Middle timestep label
                all_sequences.append(seq)
                all_labels.append(target)

            return list(zip(all_sequences, all_labels))  # Return sequences and labels as a list
        
    def get_loaders(self, batch_size=32):
        """Return DataLoaders for training, testing, and validation."""
        train_loader = DataLoader(self._train_container, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self._test_container, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(self._val_container, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader


class FFNNContainer:
    """
    A container class for preparing data for a Feed-Forward Neural Network (FFNN).

    This class loads EEG/EMG data, preprocesses it to extract specified features, and 
    segments the data into fixed-size sliding windows. Each window is flattened into 
    a single feature vector, and the corresponding central label is extracted for 
    classification tasks such as sleep stage prediction.

    Attributes:
        _train_container (list): A list of (feature_vector, label) tuples for the training set.
        _test_container (list): A list of (feature_vector, label) tuples for the testing set.
        _val_container (list): A list of (feature_vector, label) tuples for the validation set.
        _features (list): A list of feature column names to include in the feature vector.
        _window_size (int): The number of time steps (rows) to include in each sliding window.

    Methods:
        get_data() -> (list, list, list):
            Returns the prepared training, testing, and validation data.

    """
    def __init__(self, train_files=[], test_files=[], val_files=[], window_size = 5, features=[]):
        self._train_container = []
        self._test_container = []
        self._val_container = []
        self._features = features

        if window_size % 2 != 1:
            raise ValueError('Window size must be an odd number!')
        self._window_size = window_size

        self._load(train_files, self._train_container)
        self._load(test_files, self._test_container)
        self._load(val_files, self._val_container)
    
    def _load(self, files, container):
        mouseloader = MouseLoader()
        for f in files:
            df = mouseloader.load_mouse_data(f)
            df = df[['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power', 'sleep_episode']]
            prep_df = Preprocessor(df)

            prep_df.remove_unknowns()

            prep_df.power_ratios(all_ratios=True)

            prep_df.standard_scale()

            data = prep_df.get_mouse_data()

            # features = torch.tensor(data[['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power']].values).float()
            features = torch.tensor(data[[col for col in data if col in self._features]].values).float()
            labels = torch.tensor(data[['sleep_episode']].values)

            self._make_frames(features, labels, container)

    def _make_frames(self, data, labels, container):
        num_samples = data.size(0) 
        for i in range(num_samples - self._window_size + 1):  
            batch = data[i:i + self._window_size]  
            flattened_batch = batch.flatten()  

            label = labels[i + self._window_size // 2].item()  
            container.append((flattened_batch, label))

    def get_data(self):
        return self._train_container, self._test_container, self._val_container