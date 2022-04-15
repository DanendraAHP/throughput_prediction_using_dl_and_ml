from src.common.model_holder import model_dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import pandas as pd

class Model_TF():
    def __init__(self):
        self.model = tf.keras.Sequential()
    
    def build(self, layers):
        """
        build tf model with corresponding layers
        input :
            layers : dictionary of layer with the key is their type
                     and the value is their units
        output :
            tf.keras.sequential
        """
        depth = 0
        for layer, unit in layers.items():
            if layer == 'LSTM':
                if depth == 0:
                    self.model.add(tf.keras.layers.LSTM(unit))
                else :
                    self.model.add(tf.keras.layers.LSTM(unit, return_sequences=True))
            else:
                self.model.add(tf.keras.layers.Dense(unit))
            depth+=1
        self.model.add(tf.keras.layers.Dense(1))
        
    def compile_and_fit(self, data, epochs, verbose, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                                        patience=patience,
                                                        mode='min')

        self.model.compile(loss=tf.losses.Huber(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanSquaredError()])

        history = self.model.fit(data.X_tr, data.y_tr, epochs=epochs,
                        callbacks=[early_stopping],verbose=verbose)
        return history
        
    def infer(self, data):
        """
        khusus univariate
        """
        predicted_list = []
        init = data.X_test[0].reshape(1, data.X_test.shape[1], data.X_test.shape[2])
        for i in range(len(data.X_test)):
            predicted = self.model.predict(init)
            init = init[0][1:]
            init = np.append(init, predicted)
            init = init.reshape(1, data.X_test.shape[1], data.X_test.shape[2])
            predicted_list.append(predicted[0][0])
        return np.array(predicted_list)
        
    def evaluate(self, data, scaled):
        y_pred = self.model.predict(data.X_test)
        if scaled:
            #original
            ori_predict = data.transformer.inverse_transform(y_pred)
            ori_y = data.transformer.inverse_transform(data.y_test.reshape(-1,1))
            
            #calculate mae
            norm_mae = mean_absolute_error(y_pred[:, 0], data.y_test)
            ori_mae = mean_absolute_error(ori_predict, ori_y)
            
            #calculate mse
            norm_mse = mean_squared_error(y_pred[:,0], data.y_test)
            ori_mse = mean_squared_error(ori_predict, ori_y)
            
            #calculate r-squared
            norm_r_squared = r2_score(y_pred[:,0], data.y_test)
            ori_r_squared = r2_score(ori_predict, ori_y)
            
            eval_holder = {
                'norm_mae' : [norm_mae],
                'ori_mae' : [ori_mae],
                'norm_mse' : [norm_mse],
                'ori_mse' : [ori_mse],
                'norm_r_squared' : [norm_r_squared],
                'ori_r_squared' : [ori_r_squared]
            }
        else :
            mae = mean_absolute_error(y_pred[:, 0], data.y_test)
            mse = mean_squared_error(y_pred[:,0], data.y_test)
            r_squared = r2_score(y_pred[:,0], data.y_test)
            eval_holder = {
                'mae' : [mae],
                'mse' : [mse],
                'r_squared' : [r_squared]
            }
        return eval_holder
    
    def visualize(self, data, scaled):
        y_pred = self.model.predict(data.X_test)
        if scaled:
            #original
            ori_predict = list(data.transformer.inverse_transform(y_pred.reshape(-1,1))[:, 0])
            ori_y = list(data.transformer.inverse_transform(data.y_test.reshape(-1,1))[:, 0])
            return pd.DataFrame({
                'y_original':ori_y,
                'y_predicted':ori_predict
            })
        else: 
            return pd.DataFrame({
                'y_original':list(data.y_test[:, 0]),
                'y_predicted':list(y_pred[:, 0])
            })
        