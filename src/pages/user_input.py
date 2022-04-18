import pandas as pd
import os
from src.utils.data import Dataset
import streamlit as st
from src.pages.filter_features import feature_sidebar
from src.pages.filter_scaling import scaling_sidebar
from src.pages.model_selection import model_page

def user_page():
    # Code to read a single file 
    uploaded_file = st.sidebar.file_uploader("Choose a file", type = ['csv'])
    if uploaded_file is None:
        st.info('First please upload your file in the sidebar menu')
    else:
        df = pd.read_csv(uploaded_file)
        variate, target = st.columns(2)
        #choose the variate type of data
        st.header('Data Preprocessing')
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
            scale = False
            drop_low = False
            scaling_option = 'Robust'
            #train test split percentage
            percentage = st.sidebar.select_slider(
                    "Train to Val ratio?",
                    [i*0.1 for i in range(1,10)],
                    value = 0.8
            )
            #must check if the data multivariable or not
            if variate=='Multivariable':
                #choose if want to use feature selection
                if st.sidebar.checkbox('Do you want to use feature selection?'):
                    feature_sidebar(data)
                #choose if want to drop low variance column
                if st.sidebar.checkbox('Do you want to drop low variance in the data?'):
                    drop_low = True
            #choose if want to scale the data
            if st.sidebar.checkbox('Do you want to scale the data?'):
                scaling_option = scaling_sidebar()
                scale = True
            
            #choose the data timelag
            time_lag = st.number_input('The timelag you want to use?', value=1)
            verify = st.checkbox("Verify your choice?")
            
            if verify:
                data(variate, time_lag=time_lag, scale=scale, scale_kind=scaling_option, train_perc=percentage, drop_low=drop_low)
                model_page(data, scale)
        else:
            st.error("Target is not numeric columns")
    

