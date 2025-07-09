import csv
import os
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from keras.utils import to_categorical
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper_functions as hf
from dataset_formats import (MDB2022_MNIST_EP_params,
                            MDB2022_MNIST_IN_params,
                            MDB2022_MNIST_MU_params,
                            keys_MNIST_EP, keys_MNIST_IN,
                            keys_MNIST_MU)

class_labels = [0,1,2,3,4,5,6,7,8,9]
keys_ = ['T7','P7','T8','P8']  # Core channels for final processing
dataset = "MNIST_EP"
root_dir = f"Datasets/MindBigData MNIST of Brain Digits/{dataset}"
label = 'label'  # Label field name

# Use filtered data from previous step
input_file = f"filtered_train_MindBigData2022_{dataset}.pkl"
output_file = f"train_MindBigData2022_{dataset}.pkl"

print(f"** Loading filtered data: {root_dir}/{input_file}")
df = pd.read_pickle(f"{root_dir}/{input_file}")
df = df[df[label] != -1]
print(f"Loaded data shape: {df.shape}")
df.info()

## Get all passed EEG signals from MNE pipeline
print(f"** Loading MNE results: {root_dir}/mne_epoch_rejection_idx.pkl")
epoched_indexs = pd.read_pickle(f"{root_dir}/mne_epoch_rejection_idx.pkl")
print(f"MNE passed samples: {len(epoched_indexs['passed'])}")

# Filter to only include samples that passed MNE pipeline
df = df.loc[epoched_indexs['passed']]
print(f"Data shape after MNE filtering: {df.shape}")

## Use all data (fraction = 1)
fraction = 1
sampled_df = df.copy()
print(f"Using {len(sampled_df)} samples for CAR processing")
df = None

df_copy = sampled_df.copy()

# Get EEG data columns (exclude Freq and PSD columns)
eeg_columns = [col for col in sampled_df.columns if col.startswith('EEGdata_') and not col.endswith('_Freq') and not col.endswith('_PSD')]
print(f"EEG columns for CAR: {eeg_columns}")

# Add correlation columns
for col in eeg_columns:
    df_copy[f"{col}_corr"] = None
df_copy['erp'] = None

print("Starting CAR (Common Average Reference) processing...")
for idx, row in tqdm(sampled_df.iterrows(), desc="CAR processing"):
    # Extract EEG data for all channels
    eeg_data = {}
    for col in eeg_columns:
        eeg_data[col] = row[col]

    # Stack all EEG channels into array
    arr = np.stack([eeg_data[col] for col in eeg_columns], axis=0)

    # Calculate Common Average Reference (CAR)
    car = np.mean(arr, axis=0)

    # Baseline correction (using first 16 samples as baseline, ~125ms at 128Hz)
    baseline = np.mean(car[:16])
    car_corrected = car - baseline

    # Calculate correlation for each channel with CAR
    for col in eeg_columns:
        ## Method 2 (used in project):
        ## Find correlation between each signal and the CAR
        correlation = np.corrcoef(eeg_data[col], car_corrected)[0, 1]
        df_copy.at[idx, f"{col}_corr"] = correlation

    # Store the corrected CAR
    df_copy.at[idx, 'erp'] = car_corrected

print("CAR processing completed!")
df_copy.info()

# Calculate correlation means for core channels (T7, P7, T8, P8)
core_corr_keys = ['EEGdata_T7_corr', 'EEGdata_P7_corr', 'EEGdata_T8_corr', 'EEGdata_P8_corr']
df_copy['corr_mean_core'] = df_copy[core_corr_keys].mean(axis=1)

# Calculate correlation mean for all channels
all_corr_keys = [f"{col}_corr" for col in eeg_columns]
df_copy['corr_mean_all'] = df_copy[all_corr_keys].mean(axis=1)

# Filter based on correlation threshold
# Method 2 CAR: use 0.9 threshold (as per paper methodology)
factor = 0.9  # Paper uses ~0.9 correlation for method 2
print(f"Applying correlation filter with threshold: {factor}")

# Filter samples with correlation > threshold
high_corr_samples = df_copy[df_copy['corr_mean_core'] > factor]
print(f"Samples with correlation > {factor}: {len(high_corr_samples)}")

# Group by class and sample
sampled_df = high_corr_samples.groupby(label).apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)
print(f"Final samples after correlation filtering: {len(sampled_df)}")

## Apply sliding window and prepare training data
print("Applying sliding window to core channels...")

feature_data = []
label_data = []

for class_label in class_labels:
    class_df = sampled_df[sampled_df[label] == class_label]
    print(f"Processing class {class_label}: {len(class_df)} samples")

    for idx, row in tqdm(class_df.iterrows(), desc=f"Class {class_label}"):
        # Use core channels for final training data
        for key in keys_:  # ['T7', 'P7', 'T8', 'P8']
            eeg_key = f"EEGdata_{key}"
            if eeg_key in row:
                # Apply sliding window (32 samples, overlap 4)
                w_data = hf.sliding_window_eeg(row[eeg_key])
                feature_data.append(np.array(w_data))
                label_data.append(to_categorical(int(class_label), num_classes=len(class_labels)))

# Convert to numpy arrays
train_data = np.array(feature_data)
labels = np.array(label_data).astype(np.uint8)

print(f"Final training data shape: {train_data.shape}")
print(f"Final labels shape: {labels.shape}")

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=42)

print(f"Train set: {x_train.shape}")
print(f"Test set: {x_test.shape}")

# Save the processed data
output_filename = f"data_4ch_epoch_filtered_324_0-85_{output_file}"
print(f"Saving final training data to: {root_dir}/{output_filename}")

data_out = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train': y_train,
    'y_test': y_test
}

with open(f"{root_dir}/{output_filename}", 'wb') as f:
    pickle.dump(data_out, f)

print("CAR pipeline completed successfully!")
