import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.data import Dataset
import streamlit as st
from src.pages.filter_features import feature_sidebar
from src.pages.filter_scaling import scaling_sidebar
from src.pages.model_selection import model_page
from src.models.model_tf import Model_TF
from src.models.model_sklearn import Model_SKLearn
from src.models.model_stat import Model_Stat
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from throughput_prediction_using_dl_and_ml.src.models.model_stat import Model_Stat

# For ARIMA (order: dl_avg, ul_avg, dl_peak, ul_peak)
p_ar = {'dl_avg':1, 'ul_avg':1, 'dl_peak':1, 'ul_peak':3}
d_ar = {'dl_avg':0, 'ul_avg':1, 'dl_peak':0, 'ul_peak':1}
q_ar = {'dl_avg':1, 'ul_avg':1, 'dl_peak':1, 'ul_peak':3}
# For SARIMAX (order: dl_avg, ul_avg, dl_peak, ul_peak)
p_sar = {'dl_avg':1, 'ul_avg':1, 'dl_peak':1, 'ul_peak':3}
d_sar = {'dl_avg':0, 'ul_avg':1, 'dl_peak':0, 'ul_peak':1}
q_sar = {'dl_avg':1, 'ul_avg':1, 'dl_peak':1, 'ul_peak':3}
P = {'dl_avg':2, 'ul_avg':2, 'dl_peak':2, 'ul_peak':2}
D = {'dl_avg':0, 'ul_avg':0, 'dl_peak':0, 'ul_peak':0}
Q = {'dl_avg':1, 'ul_avg':1, 'dl_peak':1, 'ul_peak':1}

target_list = ['dl_avg', 'ul_avg', 'dl_peak', 'ul_peak']
target_dict = {j: i for i, j in enumerate(target_list)}


def demo_page():
    #st.markdown("### Upload a csv file for analysis.") 
    #st.write("\n")

    # Code to read a single file 
    #uploaded_file = st.sidebar.file_uploader("Choose a file", type = ['csv'])
    df = pd.read_csv('resources/csv/102248_warujaya_mbts.csv')
    #variate, target = st.columns(2)
    #choose the variate type of data
    variate = st.selectbox(
        "What kind of data you want to use?",
        ['Univariable', 'Multivariable']
    )
    target = st.selectbox(
        "What column you want to predict",
        df.columns
    )
    
    data = Dataset(df, target, variate)
    if data.target_num :
        #initial state
        scale = True 
        drop_low = True
        scaling_option = 'Robust'
        percentage = 0.75
        data(variate, 
            scale=scale, 
            scale_kind=scaling_option, 
            train_perc=percentage, 
            drop_low=drop_low)
        
        
        #must check if the data multivariable or not
        i = target_dict[target]
        if variate=='Multivariable':
            # RF
            model_rf = Model_SKLearn('RF', data).fit_and_save(n_estimators= 1400, 
                                                            min_samples_split= 2, 
                                                            min_samples_leaf= 1, 
                                                            max_features= 'auto', 
                                                            max_depth= 100, 
                                                            bootstrap= True, 
                                                            random_state = 42)
            visualize_df = model_rf.visualize(scaled=scale)
            rf_pred = visualize_df['y_predicted'].values
            rf_pred = data.transformer.inverse_transform(rf_pred.reshape(-1, 1))
            # SVR
            model_svr = Model_SKLearn('SVR', data).fit_and_save(kernel='linear')
            visualize_df = model_svr.visualize(scaled=scale)
            svr_pred = visualize_df['y_predicted'].values
            svr_pred = data.transformer.inverse_transform(svr_pred.reshape(-1, 1))
            #LSTM
            model_lstm = Model_TF()
            model_lstm.create(data, ['LSTM', 'LSTM'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
            visualize_df = model_lstm.visualize(data, scaled=True)
            lstm_pred = visualize_df['y_predicted'].values
            lstm_pred = data.transformer.inverse_transform(lstm_pred.reshape(-1, 1))
            #FNN
            model_dense = Model_TF()
            model_dense.create(data, ['Dense', 'Dense'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
            visualize_df = model_dense.visualize(data, scaled=True)
            dense_pred = visualize_df['y_predicted'].values
            dense_pred = data.transformer.inverse_transform(dense_pred.reshape(-1, 1))

            # Evaluation
            y_test = data.transformer.inverse_transform(data.y_test.reshape(-1,1)) 
            rf_mae = round(mean_absolute_error(y_test, rf_pred),4)
            rf_mse = round(mean_squared_error(y_test, rf_pred),4)
            rf_msle = round(mean_squared_log_error(y_test, rf_pred),4)
            rf_mape = round(mean_absolute_percentage_error(y_test, rf_pred),4)
            rf_r2 = round(r2_score(y_test, rf_pred),4)
            svr_mae = round(mean_absolute_error(y_test, svr_pred),4)
            svr_mse = round(mean_squared_error(y_test, svr_pred),4)
            svr_msle = round(mean_squared_log_error(y_test, svr_pred),4)
            svr_mape = round(mean_absolute_percentage_error(y_test, svr_pred),4)
            svr_r2 = round(r2_score(y_test, svr_pred),4)
            lstm_mae = round(mean_absolute_error(y_test, lstm_pred),4)
            lstm_mse = round(mean_squared_error(y_test, lstm_pred),4)
            lstm_msle = round(mean_squared_log_error(y_test, lstm_pred),4)
            lstm_mape = round(mean_absolute_percentage_error(y_test, lstm_pred),4)
            lstm_r2 = round(r2_score(y_test, lstm_pred),4)
            dense_mae = round(mean_absolute_error(y_test, dense_pred),4)
            dense_mse = round(mean_squared_error(y_test, dense_pred),4)
            dense_msle = round(mean_squared_log_error(y_test, dense_pred),4)
            dense_mape = round(mean_absolute_percentage_error(y_test, dense_pred),4)
            dense_r2 = round(r2_score(y_test, dense_pred),4)
            eval_df = {
                'Metrics' : ['MAE', 'MSE', 'MSLE', 'MAPE', 'R_Squared'],
                'RF' : [rf_mae, rf_mse, rf_msle, rf_mape, rf_r2],
                'SVR' : [svr_mae, svr_mse, svr_msle, svr_mape, svr_r2],
                'LSTM' : [lstm_mae, lstm_mse, lstm_msle, lstm_mape, lstm_r2],
                'Dense' : [dense_mae, dense_mse, dense_msle, dense_mape, dense_r2]
            }
            # Plot RF
            fig = plt.figure(figsize=(20, 10))
            rows = 4
            columns = 1
            fig.add_subplot(rows, columns, 1)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(rf_pred, color='red', label='RF Prediction')
            plt.title("RF Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')
            plt.legend(loc='best')
            # Plot SVR
            fig.add_subplot(rows, columns, 2)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(svr_pred, color='red', label='SVR Prediction')
            plt.title("SVR Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')                     
            plt.legend(loc='best')
            # Plot LSTM
            fig.add_subplot(rows, columns, 3)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(lstm_pred, color='red', label='LSTM Prediction')
            plt.title("LSTM Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')                     
            plt.legend(loc='best')
            # Plot Dense
            fig.add_subplot(rows, columns, 4)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(dense_pred, color='red', label='Dense Prediction')
            plt.title("Dense Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')                    
            plt.legend(loc='best')

            plt.legend()
            st.pyplot(fig)
            #plt.show()
            st.dataframe(eval_df)
        else:
            # ARIMA
            model_ar = Model_Stat(data).train(model='ARIMA', p=p_ar[target], d=d_ar[target], q=q_ar[target])
            visualize_df = model_ar.visualize(scaled=scale)
            ar_pred = visualize_df['y_predicted'].values
            ar_pred = data.transformer.inverse_transform(ar_pred.reshape(-1, 1))

            #SARIMAX
            model_sar = Model_Stat(data).train(model='SARIMAX', p=p_ar[target], d=d_ar[target], q=q_ar[target], P=P[target], D=D[target], Q=Q[target])
            visualize_df = model_sar.visualize(scaled=scale)
            sar_pred = visualize_df['y_predicted'].values
            sar_pred = data.transformer.inverse_transform(sar_pred.reshape(-1, 1))

            #LSTM
            model_lstm = Model_TF()
            model_lstm.create(data, ['LSTM', 'LSTM'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
            visualize_df = model_lstm.visualize(data, scaled=True)
            lstm_pred = visualize_df['y_predicted'].values
            lstm_pred = data.transformer.inverse_transform(lstm_pred.reshape(-1, 1))
            #FNN
            model_dense = Model_TF()
            model_dense.create(data, ['Dense', 'Dense'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
            visualize_df = model_dense.visualize(data, scaled=True)
            dense_pred = visualize_df['y_predicted'].values
            dense_pred = data.transformer.inverse_transform(dense_pred.reshape(-1, 1))

            # Evaluation
            y_test = data.transformer.inverse_transform(data.y_test.reshape(-1,1))[:, 0]
            ar_mae = round(mean_absolute_error(y_test, ar_pred),4)
            ar_mse = round(mean_squared_error(y_test, ar_pred),4)
            ar_msle = round(mean_squared_log_error(y_test, ar_pred),4)
            ar_mape = round(mean_absolute_percentage_error(y_test, ar_pred),4)
            ar_r2 = round(r2_score(y_test, ar_pred),4)
            sar_mae = round(mean_absolute_error(y_test, sar_pred),4)
            sar_mse = round(mean_squared_error(y_test, sar_pred),4)
            sar_msle = round(mean_squared_log_error(y_test, sar_pred),4)
            sar_mape = round(mean_absolute_percentage_error(y_test, sar_pred),4)
            sar_r2 = round(r2_score(y_test, sar_pred),4)
            lstm_mae = round(mean_absolute_error(y_test, lstm_pred),4)
            lstm_mse = round(mean_squared_error(y_test, lstm_pred),4)
            lstm_msle = round(mean_squared_log_error(y_test, lstm_pred),4)
            lstm_mape = round(mean_absolute_percentage_error(y_test, lstm_pred),4)
            lstm_r2 = round(r2_score(y_test, lstm_pred),4)
            dense_mae = round(mean_absolute_error(y_test, dense_pred),4)
            dense_mse = round(mean_squared_error(y_test, dense_pred),4)
            dense_msle = round(mean_squared_log_error(y_test, dense_pred),4)
            dense_mape = round(mean_absolute_percentage_error(y_test, dense_pred),4)
            dense_r2 = round(r2_score(y_test, dense_pred),4)
            eval_df = {
                'Metrics' : ['MAE', 'MSE', 'MSLE', 'MAPE', 'R_Squared'],
                'ARIMA' : [ar_mae, ar_mse, ar_msle, ar_mape, ar_r2],
                'SARIMAX' : [sar_mae, sar_mse, sar_msle, sar_mape, sar_r2],
                'LSTM' : [lstm_mae, lstm_mse, lstm_msle, lstm_mape, lstm_r2],
                'Dense' : [dense_mae, dense_mse, dense_msle, dense_mape, dense_r2]
            }
            # Plot ARIMA
            fig = plt.figure(figsize=(20, 10))
            rows = 4
            columns = 1
            fig.add_subplot(rows, columns, 1)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(ar_pred, color='red', label='ARIMA Prediction')
            plt.title("ARIMA Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (Scaled)')
            plt.legend(loc='best')
            # Plot SARIMAX
            fig.add_subplot(rows, columns, 2)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(sar_pred, color='red', label='SARIMAX Prediction')
            plt.title("SARIMAX Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')
            plt.legend(loc='best')
            # Plot LSTM
            fig.add_subplot(rows, columns, 3)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(lstm_pred, color='red', label='LSTM Prediction')
            plt.title("LSTM Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')                     
            plt.legend(loc='best')
            # Plot Dense
            fig.add_subplot(rows, columns, 4)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(dense_pred, color='red', label='Dense Prediction')
            plt.title("Dense Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (kbps)')                    
            plt.legend(loc='best')

            plt.legend()
            st.pyplot(fig)
            st.dataframe(eval_df)
            #plt.show()
        
    else:
        st.error("Target is not numeric columns")
    

