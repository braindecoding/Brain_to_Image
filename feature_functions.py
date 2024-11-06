import antropy as ant
import numpy as np
import scipy.stats as st
import yasa as yas

from Brain_to_Image import helper_functions as hf


## inferential statistical features
def eeg_median(row,signal_label):
    return np.median(row[signal_label])

def eeg_mean(row,signal_label):
    return row[signal_label].mean()

def eeg_std(row,signal_label):
    return row[signal_label].std()

def eeg_var(row,signal_label):
    return row[signal_label].var()

def eeg_range(row,signal_label):
    return row[signal_label].ptp()

def eeg_min(row,signal_label):
    return row[signal_label].min()

def eeg_max(row,signal_label):
    return row[signal_label].max()

def eeg_quantile(row,signal_label,quantile):
    return np.quantile(row[signal_label],quantile)

def eeg_iqr(row,signal_label,upper,lower):
    return (np.quantile(row[signal_label],upper) - np.quantile(row[signal_label],lower))

def eeg_skew(row,signal_label):
    return st.skew(row[signal_label])

def eeg_kurtosis(row,signal_label):
    return st.kurtosis(row[signal_label])

## Band power features as defined by YASA, package ref https://raphaelvallat.com/yasa
def eeg_rel_band_powers(row,signal_label):
    band_powers = yas.bandpower_from_psd(row[f"{signal_label}_PSD"],row[f"{signal_label}_Freq"],ch_names=[signal_label],relative=True)
    return [band_powers.loc[0]['Delta'],
            band_powers.loc[0]['Theta'],
            band_powers.loc[0]['Alpha'],
            band_powers.loc[0]['Sigma'],
            band_powers.loc[0]['Beta'],
            band_powers.loc[0]['Gamma'],
            band_powers.loc[0]['TotalAbsPow']]

## Entropy features
## from scipy.stats
def eeg_diff_entropy(row,signal_label):
    return st.differential_entropy(row[signal_label])
## entropy features from Antropy, package defined by https://raphaelvallat.com/antropy
def eeg_app_entropy(row,signal_label):
    return ant.app_entropy(row[signal_label])

def eeg_sample_entropy(row,signal_label):
    return ant.sample_entropy(row[signal_label])

def eeg_spectral_entropy(row,signal_label,sample_rate):
    return ant.spectral_entropy(row[signal_label],sf=sample_rate,method='welch',normalize=True)

def eeg_svd_entropy(row,signal_label):
    return ant.svd_entropy(row[signal_label],normalize=True)

def eeg_zerocross(row,signal_label):
    return ant.num_zerocross(hf.normalize_time_series(row[signal_label]),normalize=False)
