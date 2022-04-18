from src.common.model_holder import sklearn_model
import pickle
from src.common.constant import PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class Model_SKLearn():
    def __init__(self, model_name):
        self.model = sklearn_model[model_name]
        self.file = PATH.model_sklearn

    def fit_and_save(self, data, **params):
        self.model.set_params(**params)
        if len(data.X_tr.shape)>2:
            data.X_tr = data.X_tr.reshape(data.X_tr.shape[0], (data.X_tr.shape[1]*data.X_tr.shape[2]))
        self.model.fit(data.X_tr, data.y_tr)
        pickle.dump(self.model, open(self.file, 'wb'))

    def load(self):
        self.model = pickle.load(open(self.file, 'rb'))

    def visualize(self, data, scaled):
        if len(data.X_test.shape)>2:
            data.X_test = data.X_test.reshape(data.X_test.shape[0], (data.X_test.shape[1]*data.X_test.shape[2]))
        y_pred = self.model.predict(data.X_test)
        if scaled:
            #original
            ori_predict = list(data.transformer.inverse_transform(y_pred.reshape(-1,1))[:, 0])
            ori_y = list(data.transformer.inverse_transform(data.y_test.reshape(-1,1))[:, 0])
            return pd.DataFrame({
                'y_original':ori_y,
                'y_predicted':ori_predict
            })
        else: 
            return pd.DataFrame({
                'y_original':list(data.y_test[:, 0]),
                'y_predicted':list(y_pred)
            })
    def evaluate(self, data, scaled):
        if len(data.X_test.shape)>2:
            data.X_test = data.X_test.reshape(data.X_test.shape[0], (data.X_test.shape[1]*data.X_test.shape[2]))
        y_pred = self.model.predict(data.X_test)
        if scaled:
            #original
            ori_predict = data.transformer.inverse_transform(y_pred.reshape(-1, 1))
            ori_y = data.transformer.inverse_transform(data.y_test.reshape(-1,1))           
            #calculate mae
            #norm_mae = mean_absolute_error(y_pred[:, 0], data.y_test)
            ori_mae = mean_absolute_error(ori_predict, ori_y)
            #calculate mse
            #norm_mse = mean_squared_error(y_pred[:,0], data.y_test)
            ori_mse = mean_squared_error(ori_predict, ori_y)
            #calculate r-squared
            #norm_r_squared = r2_score(y_pred[:,0], data.y_test)
            ori_r_squared = r2_score(ori_predict, ori_y)
            eval_holder = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'Score' : [ori_mae, ori_mse, ori_r_squared]
            }
        else :
            mae = mean_absolute_error(y_pred, data.y_test)
            mse = mean_squared_error(y_pred, data.y_test)
            r_squared = r2_score(y_pred, data.y_test)
            eval_holder = {
                'Metrics' : ['MAE', 'MSE', 'R_Squared'],
                'Score' : [mae, mse, r_squared]
            }
        return eval_holder