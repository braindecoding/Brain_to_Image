# Brain to Image

This repository is a collection of scripts and code used in my Master's project. 

### Abstract 
This study explores the integration of Electroencephalography (EEG) data and Generative Adversarial Networks (GANs) to generate visual representations of perceived images. Utilizing the Mind Big Data (MBD) dataset and Emotiv EPOC® device, EEG signals were captured and pre-processed to remove noise and artifacts. A Convolutional Neural Network (CNN) was employed to classify and encode these signals into latent space vectors. These vectors, along with class conditioning labels, were fed into an Auxiliary Classifier GAN (ACGAN) to generate images. The study faced challenges such as signal complexity, noise, and limited training data. Despite these, the ACGAN model, particularly with a modulation layer and concatenation with embedding, demonstrated the ability to produce images with a Structural Similarity Index (SSIM) score of approximately 0.3, indicating a 65% similarity to the ground truth. The results validate the hypothesis that EEG signals can be used to generate representative images, although further improvements in data preprocessing and classification accuracy are needed. Future work could explore reinforcement learning to enhance the model's performance.

## Paper published BIOSIGNALS 2025, 18th International conference on Bio-Inspired system and signal porcessing
https://biosignals.scitevents.org/

## Datasets
All data used for training has been based the that provided by https://vivancos.com/ https://mindbigdata.com/opendb/index.html
In particular the data from https://huggingface.co/datasets/DavidVivancos/MindBigData2022, using dataset https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_EP

Use raw_to_pickle to extract the eeg signal data from the raw files and create a pandas data table to make it easier to process.

## Filtering data

use filter_data to apply notch and bandwidth filters to data

## Use MNE to remove artifacts

use mne_pipeline to apply artifact removal, mne pipeline can also be used to apply filtering and display and inspect eeg data. https://mne.tools/stable/index.html

## Apply common average reference

use car_pipeline to apply common average reference (CAR), filter on correlation, apply sliding window and save train/test split data for training models.
The project paper uses method 2 where; 
  method 1 for CAR which is where the baselined car is subtracted from each signal and then the resultand signal is correlated with the CAR and selection is made > ~0.2 correlation
  method 2 for CAR which is where the correlation between the baselined car and each signal is taken and the selection is made for > ~0.9 correlation
in notebooks the pipeline_data not book shows how this is done with some comments.

## Training models

apply filtering, artifact removal and CAR, save test/train split as pickle file. Use the train scripts to laod the data and train GAN.
