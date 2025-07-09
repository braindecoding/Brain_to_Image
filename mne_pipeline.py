"""
Title: mne_pipeline

Purpose:
    process fileter eeg signals using MNE to remove artifacts from eye movements, muscular effects

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""


import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from mne.preprocessing import ICA

import dataset_formats
from dataset_formats import keys_MNIST_EP

import warnings
warnings.filterwarnings('ignore')

dataset = "MNIST_EP"
root_dir = f"Datasets/MindBigData MNIST of Brain Digits/{dataset}"

# Use the filtered data from previous step
input_file = f"filtered_train_MindBigData2022_{dataset}.pkl"
output_file = f"train_MindBigData2022_{dataset}.pkl"

label = 'label'  # Changed from 'digit_label' to 'label'
## MNIST_MU sf = 220, 440 samples , MNIST_EP sf = 128, 256 samples , MNIST_IN sf = 128, 256 samples
if "_EP" in dataset or "_IN" in dataset:
    sample_rate = 128  #Hz
else:
    sample_rate = 220  #Hz
# Define notch frequencies and widths
notch_freqs = [ 50] #, 60]  # Line noise frequencies (50 Hz and harmonics)
notch_widths = [1] #, 2]  # Notch widths (in Hz)
# Define filter parameters
lowcut = 0.4 # 0.4  # Low-cutoff frequency (Hz)
highcut = 60 # 110  # High-cutoff frequency (Hz)
class_labels = [0,1,2,3,4,5,6,7,8,9]
keys_ = ['T7','P7','T8','P8']
keys_ext = ['EEGdata_T7','EEGdata_P7','EEGdata_T8','EEGdata_P8']

montage = mne.channels.make_standard_montage('easycap-M1')

print(f"Loading filtered data from: {root_dir}/{input_file}")
all_data_array = pd.read_pickle(f"{root_dir}/{input_file}")
print(f"Loaded data shape: {all_data_array.shape}")
print(f"Data columns: {list(all_data_array.columns)}")

# Filter out invalid labels if any
all_data_array = all_data_array[all_data_array[label] != -1]
print(f"Data shape after filtering invalid labels: {all_data_array.shape}")

processed_data = []
keys_ = []
# Extract only the actual EEG data columns (not Freq or PSD)
eeg_columns = [col for col in all_data_array.columns if col.startswith('EEGdata_') and not col.endswith('_Freq') and not col.endswith('_PSD')]
for col in eeg_columns:
    channel_name = col.replace('EEGdata_', '')
    keys_.append(channel_name)

print(f"Available EEG channels: {keys_}")
print(f"EEG data columns: {eeg_columns}")
n_channels = len(keys_)
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=keys_, sfreq=sample_rate, ch_types=ch_types)
passed_idx = []
rejected_idx = []
verbose = False

print("Starting MNE artifact removal processing...")
print("Using 100μV peak-to-peak threshold (as per paper)")

for index, row in tqdm(all_data_array.iterrows(), desc="Processing MNE pipeline"):
        try:
            # Extract EEG data for each channel
            data_list = []
            for col in eeg_columns:
                channel_data = row[col]
                if isinstance(channel_data, np.ndarray):
                    data_list.append(channel_data.astype(np.float64))
                elif isinstance(channel_data, list):
                    data_list.append(np.array(channel_data, dtype=np.float64))
                else:
                    # Skip this row if data format is unexpected
                    raise ValueError(f"Unexpected data type for {col}: {type(channel_data)}")

            # Stack arrays to create (n_channels, n_samples) array
            data = np.stack(data_list, axis=0)

            # Verify data shape
            if data.shape[1] != 256:
                raise ValueError(f"Expected 256 samples, got {data.shape[1]}")

            # Create MNE Raw object
            raw = mne.io.RawArray(data, info, verbose=False)

            # Set montage (skip if channel not found in montage)
            try:
                raw.set_montage(montage, verbose=False)
            except:
                # If montage fails, continue without it
                pass

            # Apply additional filtering (data is already filtered, but MNE can do more)
            # Skip filtering if signal is too short to avoid distortion
            if data.shape[1] >= 1057:  # Minimum length for filter
                raw.filter(l_freq=lowcut, h_freq=highcut, verbose=False)

            raw.set_eeg_reference(ref_channels='average', ch_type='eeg', projection=False, verbose=False)

            # Create fixed length Epochs (2 seconds = 256 samples at 128Hz)
            epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True, verbose=False)

            # Apply 100μV peak-to-peak threshold as described in paper
            # "A maximum 100μV peak-to-peak threshold was set based on previous studies"

            # Get epoch data for peak-to-peak calculation
            epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)

            # Calculate peak-to-peak for each epoch and channel
            peak_to_peak = np.max(epochs_data, axis=2) - np.min(epochs_data, axis=2)

            # 100μV peak-to-peak threshold as per paper
            # Paper uses stratified sampling per class to achieve 38,509 total samples
            # This suggests ~3,851 samples per class (38,509/10 classes)

            # Use 100μV peak-to-peak threshold as per paper
            # "A maximum 100μV peak-to-peak threshold was set based on previous studies"
            threshold_uv = 100.0  # 100 microvolts peak-to-peak threshold

            # Find epochs where ANY channel exceeds peak-to-peak threshold
            bad_epochs_mask = np.any(peak_to_peak > threshold_uv, axis=1)

            # Keep only good epochs
            good_epochs_indices = np.where(~bad_epochs_mask)[0]

            if len(good_epochs_indices) > 0:
                epochs_clean = epochs[good_epochs_indices]
            else:
                epochs_clean = epochs[[]]  # Empty epochs object if all rejected

            if len(epochs_clean) > 0:
                passed_idx.append(index)
            else:
                rejected_idx.append(index)

        except Exception as e:
            if len(rejected_idx) < 10:  # Only print first 10 errors to avoid spam
                print(f"Error processing row {index}: {e}")
            rejected_idx.append(index)
            continue

print(f"\nMNE processing completed!")
print(f"Total passed samples: {len(passed_idx)}")
print(f"Total rejected samples: {len(rejected_idx)}")
print(f"Overall pass rate: {len(passed_idx)/(len(passed_idx)+len(rejected_idx))*100:.2f}%")
print(f"Threshold used: 100μV peak-to-peak")

mne_epoch_rejection = {'passed':passed_idx,'reject':rejected_idx}
print(f"{root_dir}/mne_epoch_rejection_idx.pkl")
with open(f"{root_dir}/mne_epoch_rejection_idx.pkl", 'wb') as f:
    pickle.dump(mne_epoch_rejection, f)
