import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from src.utils.feature_selection import feature_selection
from sklearn.feature_selection import SelectFromModel, f_regression
import numpy as np

def feature_sidebar(data):
    select = st.sidebar.selectbox(
        'Type of feature selection you want to use?',
        ('Correlation', 'SelectKBest', 'RFE', 'PCA', 'SelectFromModel')
    )
    n_features = st.sidebar.number_input(
        'Number of features after selection', value=5
    )
    n_features = int(n_features)
    if select == 'Correlation':
        data.X = feature_selection(data.X, data.y, how=select, n_features=n_features, threshold=None)
    elif select == 'SelectKBest':
        data.X = feature_selection(data.X, data.y, how=select, k=n_features, score_func=f_regression)
    elif select == 'RFE':
        estimator = RandomForestRegressor(max_depth=5, random_state=42)
        data.X = feature_selection(data.X, data.y, how=select, estimator=estimator, n_features_to_select=n_features, step=0.5)
    elif select == 'PCA':
        data.X = feature_selection(data.X, data.y, how=select, n_components=n_features, random_state=42)
    elif select == 'SelectFromModel':
        estimator = RandomForestRegressor(max_depth=5, random_state=42)
        data.X = feature_selection(data.X, data.y, how=select, estimator=estimator, threshold=-np.inf, max_features=n_features)
    data.columns = data.X.columns