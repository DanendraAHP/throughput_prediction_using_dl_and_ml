import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from src.common.scaler_holder import scaler_dict

def low_variance(data, threshold=(.8 * (1 - .8))):
    """
    Remove features from the data where the variance is low
    Input :
        data : data we want to clean
        threshold : p(1-p) where p is percentage where the features have more than p% of the same values.
    Output :
        dataframe with no more low variance features
    """
    select = VarianceThreshold(threshold=(.8 * (1 - .8)))
    select.fit(data)
    features = select.get_feature_names_out(input_features=data.columns)
    data = select.transform(data)
    return pd.DataFrame(data, columns=features)

def windowing_dataset(X, y, timelag, kind):
    """
    Function to windowing our dataset
    Input :
        X : dataframe 
        y : dataframe
        timelag : how many timelag to window the data
        kind : univariable or multivariable
    Out :
        seq_x : np array that hold windowed X data
        seq_y : np array that hold windowed y data
    """
    seq_x = []
    seq_y = []        
    for i in range(0,len(X)-timelag):
        seq_x.append(X.iloc[i:i+timelag].to_numpy())
        if kind == 'Univariable':
            seq_y.append(X.iloc[i+timelag])
        else:
            seq_y.append(y.iloc[i+timelag])
    seq_x = np.array(seq_x)
    seq_y = np.array(seq_y)
    if len(seq_x.shape) <3:
        seq_x = np.reshape(seq_x, (seq_x.shape[0], 1, seq_x.shape[1]))  
    return seq_x, seq_y

def scale_data(data, column, kind):
    """
    Input :
        data : dataframe 
        column : list of column
        if univariable please use [column] when inputting to function
        kind = scaler type
    Output :
        scaled dataframe 
    """
    #default scaler to use
    if kind not in scaler_dict:
        scaler = scaler_dict['robust']
    else:
        scaler = scaler_dict[kind]
    if len(data.shape)>1:
        transformer = scaler.fit(data)
        data = transformer.transform(data)
    else :
        transformer = scaler.fit(data.to_numpy().reshape(-1, 1))
        data = transformer.transform(data.to_numpy().reshape(-1, 1))
    return pd.DataFrame(data, columns = column), transformer

def split_data(X, y, train_perc, timestep):
    """
    split train dan testing
    Input :
        X = 
        y = 
        train_perc = persentase data train
    Output :
        X_train
        y_train
        X_test
        y_test
    """
    train_size = int(len(X)*train_perc)
    X_tr = X[:train_size].reset_index().drop(columns='index')
    y_tr = y[:train_size].reset_index().drop(columns='index')
    X_test = X[train_size-timestep:].reset_index().drop(columns='index')
    y_test = y[train_size-timestep:].reset_index().drop(columns='index')
    return X_tr, y_tr, X_test, y_test