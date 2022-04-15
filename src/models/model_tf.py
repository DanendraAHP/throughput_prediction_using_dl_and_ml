from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.common.constant import PATH
import tensorflow as tf
import pandas as pd
import streamlit as st

class Model_TF():
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.path = PATH.model_tf
    
    def create(self, data, layers, units, epoch):
        """
        build tf model with corresponding layers
        input :
            layers : dictionary of layer with the key is their type
                     and the value is their units
        output :
            tf.keras.sequential
        """
        depth=0
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
            depth+=1
        self.model.add(tf.keras.layers.Dense(1))
        self.compile_and_fit(data, epoch)
        #self.model.build((None, data.X_tr.shape[1], data.X_tr.shape[2]))
        # print('-----------MODEL SUMMARY------------------')
        # print(self.model.summary())
        self.save(self.path)

    def compile_and_fit(self, data, epochs, verbose=0, patience=5):
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
        st.write(y_pred)
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

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
    