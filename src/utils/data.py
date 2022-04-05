import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from src.utils.preprocess import scale_data, low_variance, windowing_dataset, split_data

class Dataset():
    def __init__(self, df, target, variate):
        self.df = df
        self.df = self.df.fillna(method="ffill")
        self.target = target
        #check targetnya numeric atau bukan
        self.target_num = is_numeric_dtype(self.df[self.target])
        self.numeric_cols()
        if variate == "Univariable":
            self.X = self.df[self.target]
            self.columns = [self.X.name]
        elif variate == "Multivariable":
            self.X = self.df.drop([self.target]+self.other_cols, axis=1)
            self.columns = self.X.columns
        self.y = self.df[self.target]
       
    def numeric_cols(self):
        """
        check columns is a numeric column or not
        """
        num_cols = []
        other_cols = []
        for col in self.df.columns:
            if is_numeric_dtype(self.df[col]):
                num_cols.append(col)
            else:
                other_cols.append(col)
        self.num_cols = num_cols
        self.other_cols = other_cols
                
        
    def __call__(self, variate, time_lag=1, scale=False, scale_kind='Robust', train_perc=0.8, drop_low=False):
        if scale:
            self.X, _ = scale_data(self.X, self.columns, scale_kind)
            self.y, self.transformer = scale_data(self.y, [self.y.name], scale_kind)
        if drop_low:
            self.X = low_variance(self.X)
        self.X_tr, self.y_tr, self.X_test, self.y_test = split_data(self.X, self.y, train_perc, time_lag)
        self.X_tr, self.y_tr = windowing_dataset(self.X_tr, self.y_tr, time_lag, variate)
        self.X_test, self.y_test = windowing_dataset(self.X_test, self.y_test, time_lag, variate)
