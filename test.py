import tensorflow as tf
import numpy as np
from src.common.constant import PATH
from src.common.yaml_util import read_yaml_file
# arr = np.random.rand(10, 20, 30)
# # LSTM_no_ret = tf.keras.layers.LSTM(16)
# # LSTM_ret = tf.keras.layers.LSTM(16, return_sequences=True)
# # dense = tf.keras.layers.Dense(8)
# print(arr.shape)
# arr1 = tf.keras.layers.LSTM(16)(arr)
# print(arr1.shape)
# arr2 = tf.keras.layers.Dense(8)(arr1)
# print(arr2.shape)
# arr3 = tf.keras.layers.Dense(8)(arr2)
# print(arr3.shape)
# arr4 = tf.keras.layers.Dense(8)(arr3)
# print(arr4.shape)
# arr5 = tf.keras.layers.LSTM(16)(arr4)
# print(arr5.shape)
EXPLANATION_TEXT = read_yaml_file(PATH.config)
EXPLANATION_TEXT = EXPLANATION_TEXT['explanation_text']
print(EXPLANATION_TEXT)