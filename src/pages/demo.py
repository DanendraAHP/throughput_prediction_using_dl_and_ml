import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.data import Dataset
import streamlit as st
from src.pages.filter_features import feature_sidebar
from src.pages.filter_scaling import scaling_sidebar
from src.pages.model_selection import model_page
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# For ARIMA (order: dl_avg, ul_avg, dl_peak, ul_peak)
AR = [1, 1, 1, 3]
MA = [0, 0, 0, 0]
# For SARIMAX (order: dl_avg, ul_avg, dl_peak, ul_peak)
p = [1,1,1,3]
P = [0,0,2,2]
Q = [0,0,1,0]
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
        
        X_train = data.X_tr
        y_train = data.y_tr
        X_test = data.X_test
        y_test = data.y_test
        
        #must check if the data multivariable or not
        i = target_dict[target]
        if variate=='Multivariable':
            # RF
            model_rf = RandomForestRegressor(n_estimators= 1400, 
                                    min_samples_split= 2, 
                                    min_samples_leaf= 1, 
                                    max_features= 'auto', 
                                    max_depth= 100, 
                                    bootstrap= True, 
                                    random_state = 42)
            X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1]*X_train.shape[2]))
            X_test = X_test.reshape(X_test.shape[0], (X_test.shape[1]*X_test.shape[2]))
            model_rf.fit(X_train, y_train)
            rf_pred = model_rf.predict(X_test)
            # SVR
            model_svr = SVR(kernel='linear')
            model_svr.fit(X_train, y_train)
            svr_pred = model_svr.predict(X_test)
            # Evaluation
            y_test = y_test#.reset_index(drop=True)
            rf_mse = round(mean_squared_error(y_test, rf_pred),4)
            rf_mae = round(mean_absolute_error(y_test, rf_pred),4)
            rf_r2 = round(r2_score(y_test, rf_pred),4)
            svr_mse = round(mean_squared_error(y_test, svr_pred),4)
            svr_mae = round(mean_absolute_error(y_test, svr_pred),4)
            svr_r2 = round(r2_score(y_test, svr_pred),4)
            eval_df = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'RF' : [rf_mae, rf_mse, rf_r2],
                'SVR' : [svr_mae, svr_mse, svr_r2]
            }
            # Plot RF
            fig = plt.figure(figsize=(20, 10))
            rows = 2
            columns = 1
            fig.add_subplot(rows, columns, 1)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(rf_pred, color='red', label='RF Prediction')
            plt.title("RF Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (Scaled)')
            #plt.text(12, 2, "MAE: {}\nMSE: {}\nR^2 Score: {}".format(rf_mae, rf_mse, rf_r2))
            plt.legend(loc='best')
            # Plot SVR
            fig.add_subplot(rows, columns, 2)
            plt.plot(y_test, color='blue', label='Test Data')
            plt.plot(svr_pred, color='red', label='SVR Prediction')
            plt.title("SVR Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (Scaled)')
            #plt.text(12, 2, "MAE: {}\nMSE: {}\nR^2 Score: {}".format(svr_mae, svr_mse, svr_r2))                      
            plt.legend(loc='best')

            plt.legend()
            st.pyplot(fig)
            #plt.show()
            st.dataframe(eval_df)
        else:
            # ARIMA
            X = df[target].values
            size = int(len(X) * 0.75)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            model_arima = ARIMA(history, order=(AR[i],0,MA[i]))
            model_arima = model_arima.fit()
            output_arima = model_arima.fittedvalues[:len(test)]
            # SARIMAX
            model_sarimax = SARIMAX(history,
                            order=(p[i], 0, 0),
                            seasonal_order=(P[i], 0, Q[i], 24),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_sarimax = model_sarimax.fit()
            output_sarimax = model_sarimax.get_forecast(steps=180).predicted_mean
            # Evaluation
            arima_mse = round(mean_squared_error(test, output_arima),4)
            arima_mae = round(mean_absolute_error(test, output_arima),4)
            arima_r2 = round(r2_score(test, output_arima),4)
            sarimax_mse = round(mean_squared_error(test, output_sarimax),4)
            sarimax_mae = round(mean_absolute_error(test, output_sarimax),4)
            sarimax_r2 = round(r2_score(test, output_sarimax),4)
            eval_df = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'ARIMA' : [arima_mae, arima_mse, arima_r2],
                'SARIMAX' : [sarimax_mae, sarimax_mse, sarimax_r2]
            }
            # Plot ARIMA
            fig = plt.figure(figsize=(20, 10))
            rows = 2
            columns = 1
            fig.add_subplot(rows, columns, 1)
            plt.plot(test, color='blue', label='Test Data')
            plt.plot(output_arima, color='red', label='ARIMA Prediction')
            plt.title("ARIMA Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (Scaled)')
            #plt.text(225, np.min(test), "MAE: {}\nMSE: {}\nR^2 Score: {}".format(arima_mae, arima_mse, arima_r2))
            plt.legend(loc='best')
            # Plor SARIMAX
            fig.add_subplot(rows, columns, 2)
            plt.plot(test, color='blue', label='Test Data')
            plt.plot(output_sarimax, color='red', label='SARIMAX Prediction')
            plt.title("SARIMAX Predictions for {}".format(target))
            plt.xlabel('Hours')
            plt.ylabel('Throughput (Scaled)')
            #plt.text(225, np.min(test), "MAE: {}\nMSE: {}\nR^2 Score: {}".format(sarimax_mae, sarimax_mse, sarimax_r2))                      
            plt.legend(loc='best')

            plt.legend()
            st.pyplot(fig)
            st.dataframe(eval_df)
            #plt.show()
        
    else:
        st.error("Target is not numeric columns")
    

