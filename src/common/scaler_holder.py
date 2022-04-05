from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler_dict = {
    'Standard' : StandardScaler(),
    'Robust' : RobustScaler(),
    'MinMax' : MinMaxScaler()
}