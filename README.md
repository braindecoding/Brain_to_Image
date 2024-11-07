# braintoimage

## Datasets
All data used for training has been based the that provided by https://vivancos.com/ https://mindbigdata.com/opendb/index.html
In particular the data from https://huggingface.co/datasets/DavidVivancos/MindBigData2022, using dataset https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_EP

Use raw_to_pickle to extract the eeg signal data from the raw files and create a pandas data table to make it easier to process.

## Filtering data

use filter_data to apply notch and bandwidth filters to data

## Use MNE to remove artifacts

use mne_pipeline to apply artifact removal

## Apply common average reference

use car_pipeline to apply common average reference, filter on correltation, apply sliding window and save train/test split data for training models.
