from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
from src.common.constant import PATH
import tensorflow as tf
import pandas as pd
from src.common.hyperparameter_holder import tf_optimizer_dict, tf_losses_dict, tf_metrics_dict, tf_monitor_dict
#set the random seed
from numpy.random import seed
seed(42)
tf.random.set_seed(42)

#initializer
recurrent_init = tf.keras.initializers.Orthogonal(seed=42)
kernel_init = tf.keras.initializers.GlorotUniform(seed=42)
class Model_TF():
    def __init__(self, data):
        self.model = tf.keras.Sequential()
        self.path = PATH.model_tf
        self.data = data
    
    def create(self, layers, units, epochs, loss, optimizer, metrics, lr, monitor, callback, patience):
        """
        build tf model with corresponding layers
        input :
            layers : dictionary of layer with the key is their type
                     and the value is their units
        output :
            tf.keras.sequential
        """
        for i in range(len(layers)):
            if layers[i] == 'LSTM':
                if i<len(layers)-1:
                    if layers[i+1]=='LSTM':
                        #print(i, layers[i], 'return seq')
                        self.model.add(tf.keras.layers.LSTM(units[i], return_sequences=True, kernel_initializer=kernel_init, recurrent_initializer = recurrent_init, activation = 'relu'))
                    else :
                        #print(i, layers[i], 'no return seq')
                        self.model.add(tf.keras.layers.LSTM(units[i], kernel_initializer=kernel_init, recurrent_initializer = recurrent_init, activation = 'relu'))
                else:
                    #print(i, layers[i], 'no return seq')
                    self.model.add(tf.keras.layers.LSTM(units[i], kernel_initializer=kernel_init, recurrent_initializer = recurrent_init, activation = 'relu'))
            else:
                self.model.add(tf.keras.layers.Dense(units[i], kernel_initializer=kernel_init, activation = 'relu'))
                if i==len(layers)-1:
                    self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1))
        self.compile_and_fit(epochs, loss, optimizer, metrics, lr, monitor, callback, patience)
        self.save()

    def compile_and_fit(self, epochs, loss, optimizer, metrics, lr, monitor, callback, patience):
        #set the optimizer and their lr
        optimizer = tf_optimizer_dict[optimizer]
        optimizer.learning_rate = lr
        self.model.compile(
            loss = tf_losses_dict[loss],
            optimizer = optimizer,
            metrics = [tf_metrics_dict[metrics]]
        )
        if callback:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor = monitor,
                patience = patience,
                mode='min'
            )
            history = self.model.fit(self.data.X_tr, self.data.y_tr, epochs=epochs, callbacks=[early_stopping],verbose=0)
            #print(self.model.summary())
        else :
            history = self.model.fit(self.data.X_tr, self.data.y_tr, epochs=epochs,verbose=0)
        return history
        
    def infer(self, forecast, scaled):
        """
        khusus univariate
        """
        predicted_list = []
        init = self.data.X_test[-1].reshape(1, self.data.X_test.shape[1], self.data.X_test.shape[2])
        for i in range(forecast):
            predicted = self.model.predict(init)
            init = init[0][1:]
            init = np.append(init, predicted)
            init = init.reshape(1, self.data.X_test.shape[1], self.data.X_test.shape[2])
            predicted_list.append(predicted[0][0])
        predicted_list = np.array(predicted_list)
        #make dataframe
        if scaled:
            #original
            ori_predict = self.data.transformer.inverse_transform(predicted_list.reshape(-1,1))[:, 0]
            ori_y = self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))[:, 0]
        else: 
            ori_predict = predicted_list
            ori_y = self.data.y_test[:, 0]
        #add nan for original y
        an_array = np.empty((ori_predict.shape[0]))
        an_array[:] = np.NaN
        vis_y = np.append(ori_y, an_array)
        #add nan for forecasted y
        an_array = np.empty((ori_y.shape[0]))
        an_array[:] = np.NaN
        vis_forecast = np.append(an_array, ori_predict)
        return pd.DataFrame({
            'Original y':vis_y,
            'Forecast':vis_forecast
        })

        
    def evaluate(self, scaled):
        y_pred = self.model.predict(self.data.X_test)
        if scaled:
            #original
            ori_predict = self.data.transformer.inverse_transform(y_pred)
            ori_y = self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))
            #calculate mae
            mae = mean_absolute_error(ori_y, ori_predict)
            #calculate mse
            mse = mean_squared_error(ori_y, ori_predict)
            #calculate r-squared
            r_squared = r2_score(ori_y, ori_predict)
            #calculate MAPE
            mape = mean_absolute_percentage_error(ori_y, ori_predict)
            #calculate msle
            if (ori_predict<0).any() or (ori_y<0).any():
                msle = 0
            else:
                msle = mean_squared_log_error(ori_y, ori_predict)
        else :
            #calculate mae
            mae = mean_absolute_error(self.data.y_test, y_pred[:, 0])
            #calculate mse
            mse = mean_squared_error(self.data.y_test, y_pred[:, 0])
            #calculate r squaraed
            r_squared = r2_score(self.data.y_test, y_pred[:, 0])
            #calculate MAPE
            mape = mean_absolute_percentage_error(self.data.y_test, y_pred[:, 0])
            #calculate msle
            msle = mean_squared_log_error(self.data.y_test, y_pred[:, 0])
        eval_holder = {
            'Metrics' : ['MAE', 'MSE', 'R_Squared', 'MAPE', 'MSLE'],
            'Score' : [round(mae,3), round(mse,3), round(r_squared,3), round(mape,3), round(msle,3)]
        }
        return eval_holder
    
    def visualize(self, scaled):
        y_pred = self.model.predict(self.data.X_test)
        if scaled:
            #original
            ori_predict = list(self.data.transformer.inverse_transform(y_pred.reshape(-1,1))[:, 0])
            ori_y = list(self.data.transformer.inverse_transform(self.data.y_test.reshape(-1,1))[:, 0])
            return pd.DataFrame({
                'y_original':ori_y,
                'y_predicted':ori_predict
            })
        else: 
            return pd.DataFrame({
                'y_original':list(self.data.y_test[:, 0]),
                'y_predicted':list(y_pred[:, 0])
            })

    def save(self):
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
    