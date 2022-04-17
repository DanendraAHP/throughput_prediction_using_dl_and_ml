import streamlit as st
from tensorflow import keras
from src.models.create_model_tf import get_config, add_layer, remove_layer, clear_layer,check_layer
from src.common.hyperparameter_holder import tf_losses_dict, tf_metrics_dict, tf_monitor_dict, tf_optimizer_dict
from src.common.constant import PATH
from src.models.model_tf import Model_TF
import pandas as pd
from src.models.model_sklearn import Model_SKLearn
from src.common.model_holder import sklearn_model
import tensorflow as tf


def model_page(data, scaled):
    method_select = st.sidebar.selectbox(
        "Select what method to use?",
        ['Machine Learning', 'Deep Learning', 'Statistic', 'Compare All Model']
    )
    if method_select == 'Machine Learning':
        model_sklearn_page(data)
    elif method_select == 'Deep Learning':
        model_tf_page(data)
    elif method_select == 'Compare All Model':
        visualize_df, eval_df = compare_all_page(data, scaled)
    elif method_select == 'Statistic' :
        st.write('Not yet implemented')
        # model_select = st.sidebar.selectbox(
        #     "Select what model to use?",
        #     ['ARIMA', 'SARIMAX']
        # )
    eval = st.button('evaluate the model')
    if eval:
        #for compare all model
        if method_select == 'Compare All Model':
            #evaluation and visualization
            with st.container():
                st.header("Model Performance")
                #for graphic
                st.line_chart(visualize_df)
                #for eval metric
                st.dataframe(pd.DataFrame(eval_df))
        else:
            #for tf model
            if method_select == 'Deep Learning':
                model = Model_TF()
                model.load()
            #for sklearn model
            elif method_select == 'Machine Learning':
                model = Model_SKLearn('Random Forest')
                model.load()
            #for statsmodel
            #not implemented yet
            #evaluation and visualization
            with st.container():
                st.header("Model Performance")
                #for graphic
                visualize_df = model.visualize(data, scaled)
                st.line_chart(visualize_df)
                #for eval metric
                eval_metric = model.evaluate(data, scaled)
                st.dataframe(pd.DataFrame(eval_metric))

def compare_all_page(data, scaled):
    with st.spinner("Training the model"):
        #holder
        all_vis_df = pd.DataFrame()
        all_eval_df = pd.DataFrame()
        #create LSTM model
        model = Model_TF()
        model.create(data, ['LSTM', 'LSTM'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
        print('---------LSTM-------------')
        visualize_df = model.visualize(data, scaled)
        eval_metric = model.evaluate(data, scaled)
        all_vis_df['y_original'] = visualize_df['y_original']
        all_vis_df['LSTM Prediction'] = visualize_df['y_predicted']
        all_eval_df['Metrics'] = eval_metric['Metrics']
        all_eval_df['LSTM Score'] = eval_metric['Score']
        #create FNN model
        model = Model_TF()
        model.create(data, ['Dense', 'Dense'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
        print('---------Dense-------------')
        visualize_df = model.visualize(data, scaled)
        eval_metric = model.evaluate(data, scaled)
        all_vis_df['Dense Prediction'] = visualize_df['y_predicted']
        all_eval_df['Dense Score'] = eval_metric['Score']
        #Create RF Model
        model = Model_SKLearn("Random Forest")
        model.fit_and_save(data, n_estimators=10, max_depth=5,min_samples_split=2)
        visualize_df = model.visualize(data, scaled)
        eval_metric = model.evaluate(data, scaled)
        all_vis_df['RF Prediction'] = visualize_df['y_predicted']
        all_eval_df['RF Score'] = eval_metric['Score']
        #Create SVR Model
        model = Model_SKLearn("Support Vector Regression")
        model.fit_and_save(data, kernel='rbf', C=1, epsilon=0.1)
        visualize_df = model.visualize(data, scaled)
        eval_metric = model.evaluate(data, scaled)
        all_vis_df['SVR Prediction'] = visualize_df['y_predicted']
        all_eval_df['SVR Score'] = eval_metric['Score']
    st.success("All model has been trained")
    return all_vis_df, all_eval_df

def model_sklearn_page(data):
    model_select = st.sidebar.selectbox(
            "Select what model to use?",
            sklearn_model.keys()
    )
    model = Model_SKLearn(model_select)
    if model_select == 'Support Vector Regression':
        with st.container():
            st.header('Model Hyperparameter')
            kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly'])
            c = st.number_input('Regularization', value=1)
            epsilon = st.number_input('Epsilon', value=0.1)
        if st.button('Finish Model Creation'):
            with st.spinner('Wait for the model to be trained'):
                model.fit_and_save(data, kernel=kernel, C=c, epsilon=epsilon)
            st.success('Model has been created')
    else:
        with st.container():
            st.header('Model Hyperparameter')
            n_estimator = int(st.number_input('Number of estimator', value=10, format='%d'))
            max_depth = int(st.number_input('Maximum depth', value=5, format='%d'))
            min_samples_split = int(st.number_input('Minumum samples split', value=2, format='%d'))
        if st.button('Finish Model Creation'):
            with st.spinner('Wait for the model to be trained'):
                model.fit_and_save(data, n_estimators=n_estimator, max_depth=max_depth,min_samples_split=min_samples_split)
            st.success('Model has been created')

def model_tf_page(data):
    config_file = PATH.config
    #model creation
    with st.container():
        st.header("Model Architecture")
        create_col1, create_col2 = st.columns([2, 1])
        with create_col1 :
            layer_type_submit = st.selectbox('layer type', ['LSTM','Dense'])
            layer_unit_submit = st.number_input('Number of hidden unit', value=1)
        with create_col2 :
            layer_submit_button = st.button('Add to model')
            layer_remove_button = st.button('Remove last added layer')
            layer_clear_button = st.button('Clear the model')
    if layer_submit_button:
        add_layer(config_file, layer_type_submit, layer_unit_submit)
        is_good = check_layer(config_file)
        if not is_good:
            st.warning("The model architecture is false, clearing the last added layer ")
            remove_layer(config_file)
    if layer_remove_button:
        remove_layer(config_file)
    if layer_clear_button:
        clear_layer(config_file)
    model_tf_layer, model_tf_units = get_config(config_file)
    #model hyperparameter
    with st.container():
        st.header("Model Hyperparameter")
        epochs = int(st.number_input('Training Epochs', value=10, format='%d'))
        optimizer = st.selectbox('What optimizer you want to use?', tf_optimizer_dict.keys())
        lr = st.number_input('The learning value for optimizer', value=0.001, format='%f')
        loss = st.selectbox('What loss function you want to use?', tf_losses_dict.keys())
        metric = st.selectbox('What metrics you want to monitor?', tf_metrics_dict.keys())
        early_stop = st.selectbox('Use early stopping', ['Yes', 'No'])
        monitor = tf_monitor_dict[loss]
        #will be rewrite if there are early stop
        patience = 0
        callback = False
        if early_stop == 'Yes':
            patience = int(st.number_input('How much you want to wait before early stopping?', value=5, format='%d'))
            callback = True
    #submit model
    model_submit_button = st.button('Finish the model creation and train it')
    if model_submit_button:
        with st.spinner('Wait for the model to be trained'):
            model = Model_TF()
            model.create(data, model_tf_layer, model_tf_units, epochs, loss, optimizer, metric, lr, monitor, callback, patience)
            clear_layer(config_file)
        st.success('Model saved')
        