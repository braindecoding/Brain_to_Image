from keras.layers import (BatchNormalization, Conv2D, Dense,
                            Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt

"""
Title: EEG Classifier model design

Purpose:
    Testing design and build of EEG Classifier, Functional blocks for training
    CNN Classifier model. Call from training script.

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    build CNN model

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""
# define the CNN model for classification
def convolutional_encoder_model(channels, observations, num_classes, verbose=False):
    model = Sequential([
    BatchNormalization(input_shape=(channels, observations, 1), name="EEG_BN1"),
    Conv2D(128, (1, 4), activation='relu', padding='same', name="EEG_series_Conv2D"),
    Conv2D(64, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D"),
    MaxPooling2D((1, 2), name="EEG_feature_pool1"),
    Conv2D(64, (4, 25), activation='relu', padding='same', name="EEG_feature_Conv2D1"),  # Removed data_format and added padding
    MaxPooling2D((1, 2), name="EEG_feature_pool2"),
    Conv2D(128, (1, 2), activation='relu', padding='same', name="EEG_feature_Conv2D2"),  # Adjusted kernel size
    Flatten(name="EEG_feature_flatten"),
    BatchNormalization(name="EEG_feature_BN1"),
    Dense(512, activation='relu', name="EEG_feature_FC512"),
    Dropout(0.1, name="EEG_feature_drop1"),
    Dense(256, activation='relu', name="EEG_feature_FC256"),
    Dropout(0.1, name="EEG_feature_drop2"),
    Dense(128, activation='relu', name="EEG_feature_FC128"),   ## extract and use this as latent space for input to GAN
    Dropout(0.1, name="EEG_feature_drop3"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

if __name__ == '__main__':
    classifier = convolutional_encoder_model(9, 32, 10, verbose=True)