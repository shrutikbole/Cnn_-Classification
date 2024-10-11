# from module.train import preproess, model
import tensorflow as tf
from module.pred import load_trained_model, input_preprocess, model_predict
import config as config
import numpy as np

# data = preproess()
# mod = model(data["data_train"], data["data_val"], 25)

data_cat = config.DATA_CAT

model = load_trained_model()

input_data = input_preprocess("images.jpg")

predict = model_predict(model, input_data)

score = tf.nn.softmax(predict)
print(f"The category of the fruit/vegitable is {data_cat[np.argmax(score)]}")
