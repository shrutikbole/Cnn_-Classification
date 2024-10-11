import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import config 
from tensorflow import keras
from keras import layers, models, Sequential

data_train_path = config.DATA_TRAIN_PATH
data_test_path = config.DATA_TEST_PATH
data_val_path = config.DATA_VAL_PATH

img_width = config.IMG_WIDTH
img_height = config.IMG_HIEGHT


def preproess():

    data = {}

    data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle = True,
    image_size = (img_width, img_height),
    batch_size = 32,
    validation_split = False)
    data["data_train"] = data_train

    data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle = False,
    image_size = (img_width, img_height),
    batch_size = 32,
    validation_split = False)
    data["data_test"] = data_test

    data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle = False,
    image_size = (img_width, img_height),
    batch_size = 32,
    validation_split = False)
    data["data_val"] = data_val

    return data



def model(data_train, data_val, epochs: int):
    model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dropout(0,2),
    layers.Dense(128),
    layers.Dense(len(data_train.class_names))
    
                      ])
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics= ['accuracy'])
    history = model.fit(data_train, validation_data = data_val, epochs = epochs, batch_size =32, verbose = 1)
    model.save('train_model.keras')
    return history


def validation(history, epochs:int):
    epochs_range = range(epochs)
    plt.figure(figsize= (8, 4))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'],  label = 'Validation Accuracy')
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history.history['loss'], label = 'Training Loss')
    plt.plot(epochs_range, history.history['val_loss'],  label = 'Validation Loss')
    plt.title('Loss')


# from module.train import preproess, model

data = preproess()
mod = model(data["data_train"], data["data_val"], 25)