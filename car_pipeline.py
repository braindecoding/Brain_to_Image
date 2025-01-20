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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Brain_to_Image import batch_csv as batch
from Brain_to_Image import helper_functions as hf
from Brain_to_Image.dataset_formats import (MDB2022_MNIST_EP_params,
                                            MDB2022_MNIST_IN_params,
                                            MDB2022_MNIST_MU_params,
                                            keys_MNIST_EP, keys_MNIST_IN,
                                            keys_MNIST_MU)

class_labels = [0,1,2,3,4,5,6,7,8,9]
keys_ = ['EEGdata_T7','EEGdata_P7','EEGdata_T8','EEGdata_P8']
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
        ## CAR common avaerage reference
        ## Correlation = similarity between signal or subtracted signal and CAR
        ##
        ## method not used in project
        ## subtract the CAR from each signal and then find corelation with result and CAR
        # un comment for method 1
        #car_subtracted = corr_data[key] - car_corrected
        #correlation = np.corrcoef(car_subtracted, car_corrected)[0, 1]
        ## Alternative method 2
        ## method used in project
        ## find correaltion between each signal and the CAR.
        # comment out when using method 1
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

## chnage factor depending upon CAR method used. typically 0.9 or above for method 2 CAR subtraction, 0.2 for method 1 for CAR correlation
factor = 0.925 # 0.2
sampled_indexes = df_copy[df_copy['corr_mean_core'] > factor].groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
sampled_df = df_copy.loc[sampled_indexes]

## split data to test train

feature_data = []
label_data = []
for class_label in class_labels:
    class_df = sampled_df[sampled_df[label]==class_label]
    for idx, row in tqdm(class_df.iterrows()):
        for key in keys_:
            w_data = hf.sliding_window_eeg(row[key])
            feature_data.append(np.array(w_data))
            label_data.append(to_categorical(int(class_label),num_classes=len(class_labels)))

train_data = np.array(feature_data)
labels = np.array(label_data).astype(np.uint8)

print(train_data.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=42)

## use prefix list to save filenames for different test sceanrios
prefix = ["data_4ch_90_32_4_","fil_corr_","data_4ch_epoch_filtered_324_0-2","data_4ch_epoch_filtered_324_0-85"]
print(f"writing {root_dir}/{prefix[3]}{output_file}")
data_out = {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test} #{'x_test':train_data,'y_test':labels}
with open(f"{root_dir}/{prefix[3]}{output_file}", 'wb') as f:
    pickle.dump(data_out, f)
