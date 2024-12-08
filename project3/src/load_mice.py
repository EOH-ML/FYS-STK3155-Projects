import os
import pandas as pd

class MouseLoader:
    """
    MouseLoader class for loading mouse sleep study datasets.

    Methods:
        load_mouse_data():
            Returns the mosue data from the file. 
    """

    def load_mouse_data(self, file):
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, '..', 'data', file)
        try:
            data = pd.read_csv(file_path)[['delta_power', 'theta_power', 'sigma_power', 'beta_power', 'emg_power', 'sleep_episode']]
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"No data found in file: {file_path}")
        except Exception as e:
            print(f"An error occurred while loading {file_path}: {e}")
        return data
    