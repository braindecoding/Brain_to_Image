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

from Brain_to_Image.dataset_formats import (MDB2022_MNIST_EP_params,
                                            keys_MNIST_EP)

from asrpy import ASR
import warnings
warnings.filterwarnings('ignore')

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

label = 'digit_label'
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

prefix = [""]
ix = 0
print(f"{root_dir}/{prefix[ix]}{output_file}")
all_data_array = pd.read_pickle(f"{root_dir}/{prefix[ix]}{output_file}")
all_data_array = all_data_array[all_data_array[label]!=-1]

processed_data = []
keys_ = []
for key in keys_MNIST_EP: #[8:12]:    # [8:12]
    keys_.append(key.split("_")[1])
#keys_.append("STI014")
n_channels = len(keys_)
ch_types = ['eeg'] * (n_channels) # - 1)
#ch_types.append('stim')
info = mne.create_info(ch_names=keys_, sfreq=sample_rate, ch_types=ch_types)
passed_idx = []
rejected_idx = []
verbose = False
for index, row in tqdm(all_data_array.iterrows()):
    #data = row[keys_MNIST_EP].values.tolist()
    
    data = np.array(row[keys_MNIST_EP].values.tolist(), dtype=object)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, verbose=False)
    # could move filter to after epoch reject check
    raw.filter(l_freq=lowcut, h_freq=highcut,verbose=False)
    raw.set_eeg_reference(ref_channels='average',ch_type='eeg',projection=False,verbose=False)
    ## Create fixed length Epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True,verbose=False)
    epochs_clean = epochs.drop_bad(reject={'eeg': 100e-0},verbose=False)
    if epochs_clean:
        # could filter here
        #epochs_clean.filter(l_freq=lowcut, h_freq=highcut)
        passed_idx.append(index)
        # clean_df = epochs_clean.to_data_frame()
        # clean_df[label] = row[label]
        # processed_data.append(clean_df)
        # for epoch in epochs_clean:
        #     clean_df = pd.DataFrame(epoch, columns=keys_)
        #     clean_df[label] = row[label]
        #     processed_data.append(clean_df)
    else:
        rejected_idx.append(index)

mne_epoch_rejection = {'passed':passed_idx,'reject':rejected_idx}
print(f"{root_dir}/mne_epoch_rejection_idx.pkl")
with open(f"{root_dir}/mne_epoch_rejection_idx.pkl", 'wb') as f:
    pickle.dump(mne_epoch_rejection, f)
