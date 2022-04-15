from src.models.model_tf import Model_TF
from src.common.constant import PATH
import tensorflow as tf


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(8, return_sequences=True))
# model.add(tf.keras.layers.LSTM(4))
# model.add(tf.keras.layers.Dense(1))
# model.build((None, 3, 1))
# print(model.summary())


model_file = PATH.model_tf
model = Model_TF()
model.load(model_file)
print(model.model.  summary())