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
from src.common.yaml_util import read_yaml_file

EXPLANATION_TEXT = read_yaml_file(PATH.config)
EXPLANATION_TEXT = EXPLANATION_TEXT['explanation_text']

def model_page(data, scaled):
    st.header("Model Creation Page")
    method_select = st.sidebar.selectbox(
        "Select what method to use?",
        ['Machine Learning', 'Deep Learning', 'Statistic', 'Compare All Model']
    )
    if method_select == 'Machine Learning':
        model_sklearn_page(data)
    elif method_select == 'Deep Learning':
        model_tf_page(data)
    elif method_select == 'Compare All Model':
        st.info(EXPLANATION_TEXT['compare_all_model'])
        if st.button("Compare all model"):
            compare_all_page(data, scaled)
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
                visualize_df = pd.read_csv(PATH.visualize_df)
                eval_df = pd.read_csv(PATH.eval_df)
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
        visualize_df = model.visualize(data, scaled)
        eval_metric = model.evaluate(data, scaled)
        all_vis_df['y_original'] = visualize_df['y_original']
        all_vis_df['LSTM Prediction'] = visualize_df['y_predicted']
        all_eval_df['Metrics'] = eval_metric['Metrics']
        all_eval_df['LSTM Score'] = eval_metric['Score']
        #create FNN model
        model = Model_TF()
        model.create(data, ['Dense', 'Dense'], [8,4], 1000, 'Huber', 'Adam', 'MSE', 0.001, "mean_squared_error", True, 5)
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
        all_vis_df.to_csv(PATH.visualize_df, index=False)
        all_eval_df.to_csv(PATH.eval_df, index=False)

def model_sklearn_page(data):
    model_select = st.sidebar.selectbox(
            "Select what model to use?",
            sklearn_model.keys()
    )
    model = Model_SKLearn(model_select)
    if model_select == 'Support Vector Regression':
        with st.container():
            #explanation
            st.subheader('What is Support Vector Regression')
            st.write(EXPLANATION_TEXT['SVR_paragraph_1'])
            st.image(PATH.SVR_img)
            st.write(EXPLANATION_TEXT['SVR_paragraph_2'])
            #model hyperparameter
            st.subheader('Model Hyperparameter')
            kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly'])
            kernel_explanation = st.expander("See kernel explanation")
            kernel_explanation.write(EXPLANATION_TEXT['SVR_kernel'])
            degree = st.number_input('Polynom Degree', value=3)
            degree_explanation = st.expander("See degree explanation")
            degree_explanation.write(EXPLANATION_TEXT['SVR_degree'])
            c = st.number_input('Regularization', value=1)
            c_explanation = st.expander("See C explanation")
            c_explanation.write(EXPLANATION_TEXT['SVR_C'])
            epsilon = st.number_input('Epsilon', value=0.1)
            epsilon_explanation = st.expander('See epsilon explanation')
            epsilon_explanation.write(EXPLANATION_TEXT['SVR_epsilon'])
        if st.button('Finish Model Creation'):
            with st.spinner('Wait for the model to be trained'):
                model.fit_and_save(data, kernel=kernel, C=c, epsilon=epsilon, degree=degree)
            st.success('Model has been created')
    else:
        with st.container():
            #model explanation
            st.subheader('What is Random Forest')
            st.image(PATH.RF_img)
            st.write(EXPLANATION_TEXT['RF'])
            #model hyperparameter
            st.subheader('Model Hyperparameter')
            n_estimator = int(st.number_input('Number of estimator', value=10, format='%d'))
            n_estimator_explanation = st.expander("See number of estimator explanation")
            n_estimator_explanation.write(EXPLANATION_TEXT['RF_n_estimator'])
            max_depth = int(st.number_input('Maximum depth', value=5, format='%d'))
            max_depth_explanation = st.expander("See maximum depth explanation")
            max_depth_explanation.write(EXPLANATION_TEXT['RF_max_depth'])
            min_samples_split = int(st.number_input('Minimum samples split', value=2, format='%d'))
            min_samples_split_explanation = st.expander("See minimum samples split explanation")
            min_samples_split_explanation.write(EXPLANATION_TEXT['RF_min_samples_split'])
        if st.button('Finish Model Creation'):
            with st.spinner('Wait for the model to be trained'):
                model.fit_and_save(data, n_estimators=n_estimator, max_depth=max_depth,min_samples_split=min_samples_split)
            st.success('Model has been created')

def model_tf_page(data):
    config_file = PATH.config
    #model creation
    with st.container():
        st.subheader("Model Architecture")
        create_col1, create_col2 = st.columns([2, 1])
        with create_col1 :
            layer_type_submit = st.selectbox('Layer Type', ['LSTM','Dense'])
            layer_unit_submit = st.number_input('Number of Hidden Unit', value=1)
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
        st.subheader("Model Hyperparameter")
        epochs = int(st.number_input('Training Epochs', value=10, format='%d'))
        hyp_col1, hyp_col2 = st.columns(2)
        with hyp_col1:
            st.subheader("Model Optimizer")
            optimizer = st.selectbox('What optimizer you want to use?', tf_optimizer_dict.keys())
            optimizer_explanation = st.expander('See Optimizer Explanation')
            optimizer_explanation.write(EXPLANATION_TEXT['tf_optimizer'])
            lr = st.number_input('The learning value for optimizer', value=0.001, format='%f')
            lr_explanation = st.expander('See learning rate explanation')
            lr_explanation.write(EXPLANATION_TEXT['tf_lr'])
        with hyp_col2:
            st.subheader("Model Monitoring")
            loss = st.selectbox('What loss function you want to use?', tf_losses_dict.keys())
            loss_explanation = st.expander('See loss function explanation')
            loss_explanation.write(EXPLANATION_TEXT['tf_loss'])
            metric = st.selectbox('What metrics you want to monitor?', tf_metrics_dict.keys())
            metric_explanation = st.expander('Seen metrics to monitor explanation')
            metric_explanation.write(EXPLANATION_TEXT['tf_metric'])
        early_stop = st.selectbox('Use early stopping', ['Yes', 'No'])
        early_stop_explanation = st.expander("See early stopping explanation")
        early_stop_explanation.write(EXPLANATION_TEXT['tf_early_stop'])
        monitor = tf_monitor_dict[loss]
        #will be rewrite if there are early stop
        patience = 0
        callback = False
        if early_stop == 'Yes':
            patience = int(st.number_input('How much epoch you want to wait before early stopping?', value=5, format='%d'))
            patience_explanation = st.expander("See patience explanation")
            patience_explanation.write(EXPLANATION_TEXT['tf_patience'])
            callback = True
    #submit model
    model_submit_button = st.button('Finish the model creation and train it')
    if model_submit_button:
        with st.spinner('Wait for the model to be trained'):
            model = Model_TF()
            model.create(data, model_tf_layer, model_tf_units, epochs, loss, optimizer, metric, lr, monitor, callback, patience)
            clear_layer(config_file)
        st.success('Model Created')

        