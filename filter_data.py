import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.signal import welch
import helper_functions as hf

root_dir = "Datasets/MindBigData MNIST of Brain Digits/MNIST_EP"
# ## TRAIN
train_input_file = "train_MindBigData2022_MNIST_EP.pkl"
train_output_file = "train_MindBigData2022_MNIST_EP.pkl"
## TEST
test_input_file = "test_MindBigData2022_MNIST_EP.pkl"
test_output_file = "test_MindBigData2022_MNIST_EP.pkl"

# Available channels from the dataset
available_channels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']
keys_to_import = available_channels
label_field = 'label'
sample_rate = 256  #Hz
# Define notch frequencies and widths
notch_freqs = [50, 60] #, 60]  # Line noise frequencies (50 Hz and harmonics)
notch_widths = [2, 2] #, 2]  # Notch widths (in Hz)

# Define butterworth filter parameters
butter_order = 2 # 4
lowcut = 12 # 0.4  # Low-cutoff frequency (Hz)
highcut = 75 # 110  # High-cutoff frequency (Hz)

# Load the pickle data
print("Loading train data...")
train_data = pd.read_pickle(f"{root_dir}/{train_output_file}")
print("Train data loaded successfully!")
print(f"Data shape: {train_data.shape}")
print(f"Available channels: {keys_to_import}")

# Convert HF dataset format to expected format
print("Converting data format...")
def convert_hf_to_expected_format(df):
    """Convert Hugging Face dataset format to expected EEG format"""
    converted_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting format"):
        new_row = {label_field: row[label_field]}

        # Extract EEG data for each channel
        for channel in keys_to_import:
            # Extract the 256 samples for this channel from the flattened format
            channel_data = []
            for i in range(256):  # 256 samples per channel
                col_name = f"{channel}-{i}"
                if col_name in row:
                    channel_data.append(row[col_name])

            if len(channel_data) == 256:
                new_row[f"EEGdata_{channel}"] = channel_data
            else:
                print(f"Warning: Channel {channel} has {len(channel_data)} samples instead of 256")

        converted_data.append(new_row)

    return pd.DataFrame(converted_data)

# Convert the data format
converted_train_data = convert_hf_to_expected_format(train_data)
print("Data conversion completed!")
print(f"Converted data shape: {converted_train_data.shape}")
print(f"Converted data columns: {list(converted_train_data.columns)}")

# Add columns for frequency and PSD data
for key in keys_to_import:
    converted_train_data[f"EEGdata_{key}_Freq"] = None
    converted_train_data[f"EEGdata_{key}_PSD"] = None

print("Starting filtering process...")
train_data_copy = converted_train_data.copy()

for idx, row in tqdm(converted_train_data.iterrows(), desc="Filtering data"):
    # Create a pandas Series with EEG data for filtering
    eeg_data = {}
    for key in keys_to_import:
        eeg_data[f"EEGdata_{key}"] = row[f"EEGdata_{key}"]

    eeg_series = pd.Series(eeg_data)

    # Apply filters
    try:
        filtered = hf.apply_notch_filter(eeg_series, sample_rate, notch_freqs=notch_freqs, notch_widths=notch_widths)
        filtered = hf.apply_butterworth_filter(filtered, sample_rate, lowcut, highcut, order=butter_order)

        # Calculate PSD using Welch method
        window_length = min(128, sample_rate)  # Use smaller window to avoid issues
        for key in keys_to_import:
            if len(filtered[f"EEGdata_{key}"]) >= window_length:
                freq, PSD = welch(filtered[f"EEGdata_{key}"], fs=sample_rate, nperseg=window_length)
                train_data_copy.at[idx, f"EEGdata_{key}_Freq"] = freq.tolist()
                train_data_copy.at[idx, f"EEGdata_{key}_PSD"] = PSD.tolist()

            # Update the filtered signal
            train_data_copy.at[idx, f"EEGdata_{key}"] = filtered[f"EEGdata_{key}"]

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

print("Filtering completed!")
print(f"Processed data shape: {train_data_copy.shape}")

# Save the processed data
output_filename = f"filtered_{train_output_file}"
print(f"Saving processed data to {root_dir}/{output_filename}")
with open(f"{root_dir}/{output_filename}", 'wb') as f:
    pickle.dump(train_data_copy, f)

print("Processing completed successfully!")
  
