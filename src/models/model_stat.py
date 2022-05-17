from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
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

    def evaluate(self, scaled):
        y_pred = self.model.fittedvalues[self.data.X_tr.shape[0]:]
        if scaled:
            #original
            ori_predict = self.data.transformer.inverse_transform(y_pred)
            ori_y = self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))
            #calculate mae
            mae = mean_absolute_error(ori_y, ori_predict)
            #calculate mse
            mse = mean_squared_error(ori_y, ori_predict)
            #calculate r-squared
            r_squared = r2_score(ori_y, ori_predict)
            #calculate MAPE
            mape = mean_absolute_percentage_error(ori_y, ori_predict)
            #calculate msle
            msle = mean_squared_log_error(ori_y, ori_predict)
        else :
            #calculate mae
            mae = mean_absolute_error(self.data.y_test, y_pred[:, 0])
            #calculate mse
            mse = mean_squared_error(self.data.y_test, y_pred[:, 0])
            #calculate r squaraed
            r_squared = r2_score(self.data.y_test, y_pred[:, 0])
            #calculate MAPE
            mape = mean_absolute_percentage_error(self.data.y_test, y_pred[:, 0])
            #calculate msle
            msle = mean_squared_log_error(self.data.y_test, y_pred[:, 0])
        eval_holder = {
            'Metrics' : ['MAE', 'MSE', 'R_Squared', 'MAPE', 'MSLE'],
            'Score' : [mae, mse, r_squared, mape, msle]
        }
        return eval_holder
    
    def visualize(self, scaled):
        y_pred = self.model.fittedvalues[self.data.X_tr.shape[0]:]
        if scaled:
            #original
            ori_predict = list(self.data.transformer.inverse_transform(y_pred.reshape(-1,1))[:, 0])
            ori_y = list(self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))[:, 0])
            return pd.DataFrame({
                'y_original':ori_y,
                'y_predicted':ori_predict
            })
        else: 
            return pd.DataFrame({
                'y_original':list(self.data.y_test[:, 0]),
                'y_predicted':list(y_pred[:, 0])
            })

    def save(self):
        self.model.save(self.file)

    def load(self):
        self.model = ARIMAResults.load(self.file)
    
    def infer(self, forecast, scaled):
        # Forecast
        predicted_list = self.model.forecast(forecast, alpha=0.05)  # 95% conf
        #make dataframe
        if scaled:
            #original
            ori_predict = self.data.transformer.inverse_transform(predicted_list.reshape(-1,1))[:, 0]
            ori_y = self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))[:, 0]
        else: 
            ori_predict = predicted_list
            ori_y = self.data.y_test[:, 0]
        #add nan for original y
        an_array = np.empty((ori_predict.shape[0]))
        an_array[:] = np.NaN
        vis_y = np.append(ori_y, an_array)
        #add nan for forecasted y
        an_array = np.empty((ori_y.shape[0]))
        an_array[:] = np.NaN
        vis_forecast = np.append(an_array, ori_predict)
        return pd.DataFrame({
            'Original y':vis_y,
            'Forecast':vis_forecast
        })
