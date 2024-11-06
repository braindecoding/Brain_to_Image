import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch

root_dir = "Datasets/MindBigData - The Visual MNIST of Brain Digits/2022Data"
# ## TRAIN
train_input_file = "train_MindBigDataVisualMnist2021-Muse2v0.17.csv"
train_output_file = "train_MindBigDataVisualMnist2021-Muse2v0.17.pkl"
## TEST
test_input_file = "test_MindBigDataVisualMnist2021-Muse2v0.17.csv"
test_output_file = "test_MindBigDataVisualMnist2021-Muse2v0.17.pkl"
keys_to_import = ['EEGdata_TP9','EEGdata_TP10','EEGdata_AF7','EEGdata_AF8']
label_field = 'digit_label'
sample_rate = 256  #Hz
# Define notch frequencies and widths
notch_freqs = [50, 60] #, 60]  # Line noise frequencies (50 Hz and harmonics)
notch_widths = [2, 2] #, 2]  # Notch widths (in Hz)

# Define butterworth filter parameters
butter_order = 2 # 4
lowcut = 12 # 0.4  # Low-cutoff frequency (Hz)
highcut = 75 # 110  # High-cutoff frequency (Hz)

# test_data = pd.read_pickle(f"{root_dir}/{test_output_file}")   # processed_
train_data = pd.read_pickle(f"{root_dir}/{train_output_file}")
train_data.info()

for key in keys_to_import:
    train_data[f"{key}_Freq"] = pd.NA
    train_data[f"{key}_PSD"] = pd.NA

for key in keys_to_import:
    train_data[f"{key}_Freq"] = pd.NA
    train_data[f"{key}_PSD"] = pd.NA
#train_data.info()

train_data_copy = train_data.copy()
for idx, row in tqdm(train_data.iterrows()):
    filtered = hf.apply_notch_filter(row[keys_to_import],sample_rate,notch_freqs=notch_freqs,notch_widths=notch_widths)
    filtered = hf.apply_butterworth_filter(filtered,sample_rate,lowcut,highcut,order=butter_order)
    window_length = 1 * sample_rate    # 2 seconds
    for key in keys_to_import:
        freq, PSD = welch(filtered[key], fs=sample_rate, nperseg=window_length)
        filtered[f"{key}_Freq"] = freq
        filtered[f"{key}_PSD"] = PSD
    filtered[label_field] = row[label_field]
    train_data_copy.loc[idx] = filtered

train_data_copy.info()

## CHANGE FILE NAME FOR TEST / TRAIN
with open(f"{root_dir}/processed_{test_output_file}", 'wb') as f:
    pickle.dump(train_data_copy, f)
  
