from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler_dict = {
    'Standard' : StandardScaler(),
    'Robust' : RobustScaler(),
    'MinMax' : MinMaxScaler()
}

tf_losses_dict = {
    'Huber' : tf.losses.Huber(),
    'MAE' :tf.losses.MeanAbsoluteError(),
    'MSE' : tf.losses.MeanSquaredError()
}

tf_optimizer_dict = {
    'SGD' : tf.optimizers.SGD(),
    'Adam' : tf.optimizers.Adam(),
    'RMSProp' : tf.optimizers.RMSprop()
}

tf_metrics_dict = {
    'MAE' : tf.metrics.MeanAbsoluteError(),
    'MSE' : tf.metrics.MeanSquaredError()
}

tf_monitor_dict = {
    'MAE' : 'mean_absolute_error',
    'MSE' : 'mean_squared_error',
    'Huber' : 'mean_absolute_error'
}