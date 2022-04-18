from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
from statsmodels.tsa.arima_model import ARIMAResults
from src.common.constant import PATH
import streamlit as st

class Model_Stat():
    def __init__(self, data):
        self.data = data
        self.train_data = np.append(data.X_tr, data.X_test)
        self.train_len = data.X_tr.shape[0]
        self.file = PATH.model_stat

    def train(self, model, p, d, q, P=0, D=0, Q=0, s=24):
        if model == 'ARIMA':
            self.model = ARIMA(self.train_data,order=(p, d, q))
        else:
            self.model = SARIMAX(self.train_data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s)
                        )
        self.model = self.model.fit()
        self.save()

    def evaluate(self, data, scaled):
        y_pred = self.model.fittedvalues[data.X_tr.shape[0]:]
        if scaled:
            #original
            ori_predict = data.transformer.inverse_transform(y_pred.reshape(-1,1))
            ori_y = data.transformer.inverse_transform(data.y_test.reshape(-1,1))
            #calculate mae
            ori_mae = mean_absolute_error(ori_predict, ori_y)
            #calculate mse
            ori_mse = mean_squared_error(ori_predict, ori_y)
            #calculate r-squared
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
        y_pred = self.model.fittedvalues[data.X_tr.shape[0]:]
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
        self.model.save(self.file)

    def load(self):
        self.model = ARIMAResults.load(self.file)