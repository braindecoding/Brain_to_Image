from datasets import load_dataset
import pandas as pd
import pickle
import os

# Create output directory if it doesn't exist
root_dir = "Datasets/MindBigData MNIST of Brain Digits/MNIST_EP"
os.makedirs(root_dir, exist_ok=True)

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
ds = load_dataset("DavidVivancos/MindBigData2022_MNIST_EP")

# Convert train split to pandas DataFrame
print("Converting train split to pandas DataFrame...")
train_df = ds['train'].to_pandas()

# Convert test split to pandas DataFrame
print("Converting test split to pandas DataFrame...")
test_df = ds['test'].to_pandas()

# Save train data as pickle
train_output_file = "train_MindBigData2022_MNIST_EP.pkl"
print(f"Saving train data to {root_dir}/{train_output_file}")
with open(f"{root_dir}/{train_output_file}", 'wb') as f:
    pickle.dump(train_df, f)

# Save test data as pickle
test_output_file = "test_MindBigData2022_MNIST_EP.pkl"
print(f"Saving test data to {root_dir}/{test_output_file}")
with open(f"{root_dir}/{test_output_file}", 'wb') as f:
    pickle.dump(test_df, f)

print("\nTrain DataFrame info:")
train_df.info()
print(f"\nTrain data shape: {train_df.shape}")

print("\nTest DataFrame info:")
test_df.info()
print(f"\nTest data shape: {test_df.shape}")

print("\nFirst few rows of train data:")
print(train_df.head())
