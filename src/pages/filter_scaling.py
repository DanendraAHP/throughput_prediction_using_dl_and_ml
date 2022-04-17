from src.common.hyperparameter_holder import scaler_dict
import streamlit as st

def scaling_sidebar():
    scaling_option = st.sidebar.selectbox(
        'Choose the scaler you want to use',
        list(scaler_dict.keys())
    )
    return scaling_option