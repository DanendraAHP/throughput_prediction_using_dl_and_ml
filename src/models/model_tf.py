from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.common.constant import PATH
import tensorflow as tf
import pandas as pd
from src.common.hyperparameter_holder import tf_optimizer_dict, tf_losses_dict, tf_metrics_dict, tf_monitor_dict
import streamlit as st

class Model_TF():
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.path = PATH.model_tf
    
    def create(self, data, layers, units, epochs, loss, optimizer, metrics, lr, monitor, callback, patience):
        """
        build tf model with corresponding layers
        input :
            layers : dictionary of layer with the key is their type
                     and the value is their units
        output :
            tf.keras.sequential
        """
        for i in range(len(layers)):
            if layers[i] == 'LSTM':
                if i<len(layers)-1:
                    if layers[i+1]=='LSTM':
                        print(i, layers[i], 'return seq')
                        self.model.add(tf.keras.layers.LSTM(units[i], return_sequences=True))
                    else :
                        print(i, layers[i], 'no return seq')
                        self.model.add(tf.keras.layers.LSTM(units[i]))
                else:
                    print(i, layers[i], 'no return seq')
                    self.model.add(tf.keras.layers.LSTM(units[i]))
            else:
                self.model.add(tf.keras.layers.Dense(units[i]))
                if i==len(layers)-1:
                    self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1))
        self.compile_and_fit(data, epochs, loss, optimizer, metrics, lr, monitor, callback, patience)
        self.save()

    def compile_and_fit(self, data, epochs, loss, optimizer, metrics, lr, monitor, callback, patience):
        #set the optimizer and their lr
        optimizer = tf_optimizer_dict[optimizer]
        optimizer.learning_rate = lr
        self.model.compile(
            loss = tf_losses_dict[loss],
            optimizer = optimizer,
            metrics = [tf_metrics_dict[metrics]]
        )
        if callback:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor = monitor,
                patience = patience,
                mode='min'
            )
            history = self.model.fit(data.X_tr, data.y_tr, epochs=epochs, callbacks=[early_stopping],verbose=0)
            print(self.model.summary())
        else :
            history = self.model.fit(data.X_tr, data.y_tr, epochs=epochs,verbose=0)
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
            #norm_mae = mean_absolute_error(y_pred[:, 0], data.y_test)
            ori_mae = mean_absolute_error(ori_predict, ori_y)
            
            #calculate mse
            #norm_mse = mean_squared_error(y_pred[:,0], data.y_test)
            ori_mse = mean_squared_error(ori_predict, ori_y)
            
            #calculate r-squared
            #norm_r_squared = r2_score(y_pred[:,0], data.y_test)
            ori_r_squared = r2_score(ori_predict, ori_y)
            eval_holder = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'Score' : [ori_mae, ori_mse, ori_r_squared]
            }
        else :
            mae = mean_absolute_error(y_pred[:, 0], data.y_test)
            mse = mean_squared_error(y_pred[:,0], data.y_test)
            r_squared = r2_score(y_pred[:,0], data.y_test)
            eval_holder = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'Score' : [mae, mse, r_squared]
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

    def save(self):
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
    