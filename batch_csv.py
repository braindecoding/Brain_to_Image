import csv
import os
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycatch22 as catch22
from scipy.signal import butter, filtfilt, iirnotch, welch
from tqdm import tqdm

from Brain_to_Image import helper_functions as hf
from Brain_to_Image import dataset_formats

# Define the function to convert list of single float to int
def list_float_to_int(lst):
    return int(lst[0])

def process_chunk(chunk):
    # Perform your data processing here
    # This is just an example - modify as needed
    #processed = chunk.apply(lambda x: x * 2 if np.issubdtype(x.dtype, np.number) else x)
    reshaped_data = []
    for _, row in chunk.iterrows():
        flat_values = row.values.flatten()
        new_row ={}
        start_idx = 0
        # if start_idx == 0:
        #     print(flat_values)
        for col, values in dataset_formats.Muse2_v017_params.items():
            end_idx = start_idx + values['size']
            if values['size'] == 1:
                new_row[col] = flat_values[start_idx:end_idx]
            else:
                new_row[col] = flat_values[start_idx:end_idx]

            start_idx = end_idx

        reshaped_data.append(new_row)

    df = pd.DataFrame(reshaped_data)
    df.drop(dataset_formats.keys_to_drop, axis=1, inplace=True)
    df['digit_label'] = df['digit_label'].apply(list_float_to_int)

    return df

def process_chunk_MBD(chunk,params):
    # Perform your data processing here
    # This is just an example - modify as needed
    #processed = chunk.apply(lambda x: x * 2 if np.issubdtype(x.dtype, np.number) else x)
    reshaped_data = []
    for _, row in chunk.iterrows():
        flat_values = row.values.flatten()
        new_row ={}
        start_idx = 0
        # if start_idx == 0:
        #     print(flat_values)
        for col, values in params.items():
            end_idx = start_idx + values['size']
            if values['size'] == 1:
                new_row[col] = flat_values[start_idx:end_idx]
            else:
                new_row[col] = flat_values[start_idx:end_idx]

            start_idx = end_idx

        reshaped_data.append(new_row)

    df = pd.DataFrame(reshaped_data)
    #df.drop(dataset_formats.keys_to_drop, axis=1, inplace=True)
    df['digit_label'] = df['digit_label'].apply(list_float_to_int)

    return df


def batch_process_csv_pandas(input_file, output_file, chunksize=10000, MBD=None):
    # Initialize an empty list to store processed chunks
    processed_data = []

    # Read and process the CSV file in chunks
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize)):
        if MBD:
            processed_chunk = process_chunk_MBD(chunk,MBD)
        else:
            processed_chunk = process_chunk(chunk)

        processed_data.append(processed_chunk)

    # Combine all processed chunks
    final_data = pd.concat(processed_data, ignore_index=True)

    # Save the processed data
    if output_file.endswith('.pkl'):
        with open(output_file, 'wb') as f:
            pickle.dump(final_data, f)
    elif output_file.endswith('.npy'):
        np.save(output_file, final_data.to_numpy())
    else:
        raise ValueError("Unsupported output file format. Use .pkl or .npy")

    return final_data

## Usage
#input_file = 'large_data.csv'
#output_file = 'processed_data.pkl'  # or 'processed_data.npy'
#result = batch_process_csv(input_file, output_file)

# Read a CSV file and process each row using the parse_text method.
def batch_process_csv_file(file_path, row_range=None):
    """     Read a CSV file and process each row using the parse_text method.

    Args:
        file_path (str): The path to the CSV file.
        row_range (int or tuple, optional): The number of rows to process or a range of rows.
            If an integer is provided, it specifies the number of rows to process.
            If a tuple is provided, it specifies the start and end rows (inclusive) to process.
            If None, all rows will be processed.

    Returns:
        list: A list of dictionaries, where each dictionary represents a processed row.
    """
    processed_rows = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        row_count = 0

        for row in csv_reader:
            # Check if the row is within the specified range
            if row_range is None and row_count == 0:
                continue
            if row_range is None or (
                (isinstance(row_range, int) and row_count < row_range) or
                (isinstance(row_range, tuple) and row_range[0] <= row_count < row_range[1])
            ):
                # Join the row elements into a comma-separated string
                text = ','.join(row)

                # Process the text using the parse_text method
                parsed_row = hf.parse_text(text)

                # Append the parsed row to the list
                processed_rows.append(parsed_row)

            row_count += 1

            # Break the loop if the specified number of rows has been processed
            if isinstance(row_range, int) and row_count >= row_range:
                break
            if isinstance(row_range, tuple) and row_count >= row_range[1]:
                break

    return processed_rows