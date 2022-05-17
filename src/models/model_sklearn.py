from src.common.model_holder import sklearn_model
import pickle
from src.common.constant import PATH
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import pandas as pd
import streamlit as st
import numpy as np

class Model_SKLearn():
    def __init__(self, model_name, data):
        self.model = sklearn_model[model_name]
        self.file = PATH.model_sklearn
        self.data = data

    def fit_and_save(self, **params):
        self.model.set_params(**params)
        if len(self.data.X_tr.shape)>2:
            self.data.X_tr = self.data.X_tr.reshape(self.data.X_tr.shape[0], (self.data.X_tr.shape[1]*self.data.X_tr.shape[2]))
        self.model.fit(self.data.X_tr, self.data.y_tr)
        pickle.dump(self.model, open(self.file, 'wb'))

    def load(self):
        self.model = pickle.load(open(self.file, 'rb'))

    def visualize(self, scaled):
        if len(self.data.X_test.shape)>2:
            self.data.X_test = self.data.X_test.reshape(self.data.X_test.shape[0], (self.data.X_test.shape[1]*self.data.X_test.shape[2]))
        y_pred = self.model.predict(self.data.X_test)
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
                'y_predicted':list(y_pred)
            })
    def evaluate(self, scaled):
        if len(self.data.X_test.shape)>2:
            self.data.X_test = self.data.X_test.reshape(self.data.X_test.shape[0], (self.data.X_test.shape[1]*self.data.X_test.shape[2]))
        y_pred = self.model.predict(self.data.X_test)
        if scaled:
            #original
            ori_predict = self.data.transformer.inverse_transform(y_pred.reshape(-1, 1))
            ori_y = self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))           
            #calculate mae
            mae = mean_absolute_error(ori_y, ori_predict)
            #calculate mse
            mse = mean_squared_error(ori_predict, ori_y)
            #calculate r-squared
            r_squared = r2_score(ori_predict, ori_y)
            #calculate MAPE
            mape = mean_absolute_percentage_error(ori_y, ori_predict)
            #calculate msle
            msle = mean_squared_log_error(ori_y, ori_predict)
        else :
            #calculate mae
            mae = mean_absolute_error(self.data.y_test, y_pred)
            #calculate mse
            mse = mean_squared_error(self.data.y_test, y_pred)
            #calculate r squaraed
            r_squared = r2_score(self.data.y_test, y_pred)
            #calculate MAPE
            mape = mean_absolute_percentage_error(self.data.y_test, y_pred)
            #calculate msle
            msle = mean_squared_log_error(self.data.y_test, y_pred)
        eval_holder = {
            'Metrics' : ['MAE', 'MSE', 'R_Squared', 'MAPE', 'MSLE'],
            'Score' : [mae, mse, r_squared, mape, msle]
        }
        return eval_holder

    def infer(self, forecast, scaled):
        """
        khusus univariate
        """
        predicted_list = []
        if len(self.data.X_test.shape)>2:
            self.data.X_test = self.data.X_test.reshape(self.data.X_test.shape[0], (self.data.X_test.shape[1]*self.data.X_test.shape[2]))
        init = self.data.X_test[-1].reshape(1,-1)
        for i in range(forecast):
            predicted = self.model.predict(init)
            init = init[0][1:]
            init = np.append(init, predicted)
            init = init.reshape(1, (init.shape[0]))
            predicted_list.append(predicted[0])
        predicted_list = np.array(predicted_list)
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