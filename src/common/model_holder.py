import tensorflow as tf

model_dict = {
    'LSTM' : tf.keras.Sequential([
        tf.keras.layers.LSTM(8, return_sequences=True, activation='relu'),
        tf.keras.layers.LSTM(4, activation='relu'),
        tf.keras.layers.Dense(1)
    ]),
    'dense' : tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=8, activation='relu'),
        tf.keras.layers.Dense(units=4, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
}
