import pandas as pd
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

# For ARIMA (order: dl_avg, ul_avg, dl_peak, ul_peak)
AR = [1, 1, 1, 3]
MA = [0, 0, 0, 0]
# For SARIMAX (order: dl_avg, ul_avg, dl_peak, ul_peak)
p = [1,1,1,3]
P = [0,0,2,2]
Q = [0,0,1,0]
target_list = ['dl_avg', 'ul_avg', 'dl_peak', 'ul_peak']
target_dict = {j: i for i, j in target_list.enumerate()}


def demo_page():
    #st.markdown("### Upload a csv file for analysis.") 
    #st.write("\n")

    # Code to read a single file 
    #uploaded_file = st.sidebar.file_uploader("Choose a file", type = ['csv'])
    df = pd.read_csv('throughput_prediction_using_dl_and_ml\data')
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
        i = target_dict(target)
        if variate=='Multivariable':
            # RF dan SVR
            rf_model = RandomForestRegressor(n_estimators= 1400, 
                                    min_samples_split= 2, 
                                    min_samples_leaf= 1, 
                                    max_features= 'auto', 
                                    max_depth= 100, 
                                    bootstrap= True, 
                                    random_state = 42)
            svr_model = SVR(kernel='linear')
        else:
            # ARIMA dan SARIMAX
            arima_model = ARIMA(data, order=(AR[i],0,MA[i]))
            sarimax_model = SARIMAX(data,
                            order=(p[i], 0, 0),
                            seasonal_order=(P[i], 0, Q[i], 24),
                            enforce_stationarity=False,
                            enforce_invertibility=False)

        
    else:
        st.error("Target is not numeric columns")
    

