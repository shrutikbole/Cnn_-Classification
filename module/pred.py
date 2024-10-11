import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import config as config


def input_preprocess(image_path):
    image = tf.keras.utils.load_img(image_path, target_size = (config.IMG_HIEGHT, config.IMG_HIEGHT))
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr, axis=0)
    return img_bat

def load_trained_model():
    model = load_model(config.MODEL_PATH)
    return model

def model_predict(model, data):
    prediction = model.predict(data)
    return prediction


