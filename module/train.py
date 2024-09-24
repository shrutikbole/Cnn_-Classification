import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import config as config
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