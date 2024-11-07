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
from scipy.integrate import simps
from scipy.signal import butter, filtfilt, iirnotch, welch
from tqdm import tqdm

from Brain_to_Image import batch_csv as batch
from Brain_to_Image import helper_functions as hf
from Brain_to_Image.dataset_formats import (MDB2022_MNIST_EP_params,
                                            MDB2022_MNIST_IN_params,
                                            MDB2022_MNIST_MU_params,
                                            keys_MNIST_EP, keys_MNIST_IN,
                                            keys_MNIST_MU)


dataset = "MNIST_EP"
root_dir = f"Datasets/MindBigData MNIST of Brain Digits/{dataset}"
if True:
    # ## TRAIN
    input_file = f"train_MindBigData2022_{dataset}.csv"
    output_file = f"train_MindBigData2022_{dataset}.pkl"
else:
    ## TEST
    input_file = f"test_MindBigData2022_{dataset}.csv"
    output_file = f"test_MindBigData2022_{dataset}.pkl"

print(f"** reading file {root_dir}/filtered_bp_corr_{output_file}") #filtered_{output_file}")
    df = pd.read_pickle(f"{root_dir}/filtered_bp_corr_{output_file}") #filtered_{output_file}")
    df = df[df[label]!=-1]
    df.info()

## get all passed eeg signals from MNE pipeline.
print(f"** reading file {root_dir}/mne_epoch_rejection_idx.pkl")
epoched_indexs = pd.read_pickle(f"{root_dir}/mne_epoch_rejection_idx.pkl")
df = df.loc[epoched_indexs['passed']]

## sample if required. fraction = 1 to take all data
fraction = 1
sampled_indexes = df.groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
sampled_df = df.loc[sampled_indexes]
#sampled_df.info()
df = None

df_copy = sampled_df.copy()

for key in keys_MNIST_EP:
    df_copy[f"{key}_corr"] = pd.NA
df_copy[f"erp"] = pd.NA

for idx, row in tqdm(sampled_df.iterrows()):
    corr_data = row[keys_MNIST_EP]
    arr = np.stack(corr_data.values)
    # Calculate ERP
    car = np.mean(arr, axis=0)
    # Baseline correction (using first 125 ms as baseline)
    baseline = np.mean(car[:16])
    car_corrected = car - baseline

    for key in keys_MNIST_EP:
        #car_subtracted = corr_data[key] - car_corrected
        #correlation = np.corrcoef(car_subtracted, car_corrected)[0, 1]
        correlation = np.corrcoef(corr_data[key], car_corrected)[0, 1]
        row[f"{key}_corr"] = correlation
    #corr_data[label] = row[label]
    row['erp'] = car_corrected

    df_copy.loc[idx] = row

df_copy.info()

corr_keys_ = ['EEGdata_T7_corr','EEGdata_P7_corr','EEGdata_T8_corr','EEGdata_P8_corr']
df_copy['corr_mean_core'] = df_copy[corr_keys_].mean(axis=1)
corr_keys_ = [f"{key}_corr" for key in keys_MNIST_EP]
df_copy['corr_mean_all'] = df_copy[corr_keys_].mean(axis=1)

fraction = 1
sampled_indexes = df_copy[df_copy['corr_mean_core'] > 0.2].groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
sampled_df = df_copy.loc[sampled_indexes]
