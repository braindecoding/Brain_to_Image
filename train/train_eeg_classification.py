import os
import pickle
import sys

# Force CPU usage only and set data format
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Set data format to channels_last (NHWC)
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eegclassifier import convolutional_encoder_model


"""
Title: Training script for EEG Classification

Purpose:
    Testing  build and training of EEG classifier, Functional blocks for training
    classification model

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    Run the script as is, uses the online MNIST dataset to train the GAN

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""



print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
print(os.getcwd())

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def save_model(model,name,path):
    check_path(path)
    filename = os.path.join(path,name)
    with open(filename,"wb") as f:
        pickle.dump(model,f)

    print("Model {} saved to {}".format(name,path))



run_id = "eeg_classifier_car_pipeline"
dataset = "MNIST_EP"
root_dir = f"../Datasets/MindBigData MNIST of Brain Digits/{dataset}"
data_file = "data_4ch_epoch_filtered_324_0-85_train_MindBigData2022_MNIST_EP.pkl"
model_save_dir = os.path.join("data", "models")
batch_size, num_epochs = 128, 50  # Reduced epochs for initial testing

print(f"Reading data file {root_dir}/{data_file}")
eeg_data = pickle.load(open(f"{root_dir}/{data_file}", 'rb'))

# Load training data from CAR pipeline output
x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_test'], eeg_data['y_test']

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Ensure data is in correct format (samples, height, width, channels)
# Current shape should be (samples, 9, 32, 1) which is already correct for NHWC
print(f"Data format: {K.image_data_format()}")
print(f"Input shape for model: {x_train.shape[1:]}")

# Create classifier model
# Input shape: (9, 32, 1) - 9 windows, 32 time points, 1 channel
print(f"Creating model with input shape: {x_train.shape[1:]}")
classifier = convolutional_encoder_model(x_train.shape[1], x_train.shape[2], 10)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# location for the trained model file
saved_model_file = os.path.join(model_save_dir, str(run_id) + '_final' + '.h5')

# location for the intermediate model files
filepath = os.path.join(model_save_dir, str(run_id) + "-model-improvement-{epoch:02d}-{val_accuracy:.2f}.h5")  #{epoch:02d}-{val_accuracy:.2f}

# call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=False, save_best_only=True, mode='max')
variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

sgd = optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adm = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, decay=1e-6)

classifier.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
classifier.summary()
history = classifier.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[callback_checkpoint], verbose=True)
save_model(history.history, f"history_{str(run_id)}_final.pkl",model_save_dir)
#classifier.load_weights(saved_model_file)
classifier.save(saved_model_file)


accuracy = classifier.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)
print(accuracy)