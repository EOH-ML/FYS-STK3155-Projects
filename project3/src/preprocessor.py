import numpy as np

class Preprocessor:
    """
    Preprocessor class for cleaning, transforming, and scaling data.

    Attributes:
        _train_data (pd.DataFrame): Training data to preprocess.
        _test_data (pd.DataFrame): Test data to preprocess.
        _val_data (pd.DataFrame): Validation data to preprocess.
        _delta (float): A small constant to avoid log transformations on zero values.

    Methods:
        remove_unknowns():
            Removes rows where the 'sleep_episode' column is zero and adjusts indexing to start from zero.
        power_ratios():
            Computes and adds various power ratios to the dataset, such as theta/delta and beta/sigma.
        log_transform():
            Applies a log transformation to power-related columns to stabilize variance.
        standard_scale():
            Standardizes power-related columns by scaling to zero mean and unit variance.
        get_mouse_data():
            Returns the preprocessed training, test, and validation datasets.
    """
    def __init__(self, data, delta=1):
        self._delta = delta
        self._data = data
    
    def remove_unknowns(self):
        known_mask = self._data['sleep_episode'] != 0

        # Using the mask on the data
        self._data = self._data.loc[known_mask]

        # Subtracting 1, because pytorch starts indexing at 0
        self._data.loc[:, 'sleep_episode'] -= 1
    
    def power_ratios(self, ratio_columns=[], all_ratios=False):
        self._data = self._power_ratios(self._data, ratio_columns, all_ratios)
    
    def _power_ratios(self, df, ratio_columns, all_ratios):
        df = df.copy()
        df['theta_delta_ratio'] = df['theta_power'] / df['delta_power']
        df['sigma_delta_ratio'] = df['sigma_power'] / df['delta_power']
        df['beta_delta_ratio'] = df['beta_power'] / df['delta_power']
        df['theta_sigma_ratio'] = df['theta_power'] / df['sigma_power']
        df['delta_sigma_ratio'] = df['delta_power'] / df['sigma_power']
        df['beta_sigma_ratio'] = df['beta_power'] / df['sigma_power']
        df['theta_beta_ratio'] = df['theta_power'] / df['beta_power']
        df['sigma_beta_ratio'] = df['sigma_power'] / df['beta_power']
        df['delta_beta_ratio'] = df['delta_power'] / df['beta_power']
        df['delta_theta_ratio'] = df['delta_power'] / df['theta_power']
        df['sigma_theta_ratio'] = df['sigma_power'] / df['theta_power']
        df['beta_theta_ratio'] = df['beta_power'] / df['theta_power']

        if all_ratios:
            df['emg_delta_ratio'] = df['emg_power'] / df['delta_power']
            return df

        columns = [col for col in self._data if 'power' in col] + [col for col in ratio_columns]  + ['sleep_episode']
        return df[columns]
        
    def log_transform(self, power_ratio=False):
        power_columns = [col for col in self._data.columns if 'power' in col]
        min_value = abs(np.min(self._data[power_columns]))
        self._data.loc[:,power_columns] = np.log(self._data[power_columns] + min_value + self._delta)

        if power_ratio: 
            ratio_columns = [col for col in self._data.columns if 'ratio' in col]
            min_value = abs(np.min(self._data[ratio_columns]))
            self._data.loc[:,ratio_columns] = np.log(self._data[ratio_columns] + min_value + self._delta)
    
    def standard_scale(self):
        power_columns = [col for col in self._data.columns if 'power' in col]
        self._data.loc[:,power_columns] = (self._data[power_columns] - self._data[power_columns].mean()) / self._data[power_columns].std()

        ratio_columns = [col for col in self._data.columns if col.endswith('ratio')]
        self._data.loc[:,ratio_columns] = (self._data[ratio_columns] - self._data[ratio_columns].mean()) / self._data[ratio_columns].std()

    def get_mouse_data(self):
        return self._data