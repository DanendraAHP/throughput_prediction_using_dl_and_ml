import streamlit as st
from tensorflow import keras
from src.models.create_model_tf import get_config, add_layer, remove_layer, clear_layer,check_layer
from src.common.constant import PATH
from src.models.model_tf import Model_TF
import pandas as pd

def model_page(data, scaled):
    method_select = st.sidebar.selectbox(
        "Select what method to use?",
        ['Machine Learning', 'Deep Learning', 'Statistic']
    )
    if method_select == 'Machine Learning':
        st.write('Not yet implemented')
        # model_select = st.sidebar.selectbox(
        #     "Select what model to use?",
        #     ['Support Vector Regression', 'Random Forest']
        # )
    elif method_select == 'Deep Learning':
        model_tf_page(data)
        model_file = PATH.model_tf
    else :
        st.write('Not yet implemented')
        # model_select = st.sidebar.selectbox(
        #     "Select what model to use?",
        #     ['ARIMA', 'SARIMAX']
        # )
    eval = st.checkbox('evaluate the model')
    if eval:
        model = Model_TF()
        model.load(model_file)
        eval_metric = model.evaluate(data, scaled)
        st.dataframe(pd.DataFrame(eval_metric))
    visualize = st.checkbox('visualize the result')
    if visualize:
        visualize_df = model.visualize(data, scaled)
        st.line_chart(visualize_df)
    
def model_tf_page(data):
    config_file = PATH.config
    layer_type, layer_unit  = st.columns(2)
    layer_submit, layer_remove, layer_clear = st.columns(3)
    with layer_type:
        layer_type_submit = st.selectbox('layer type', ['LSTM','Dense'])
    with layer_unit:
        layer_unit_submit = st.number_input('Number of hidden unit', value=1)
    with layer_submit:
        layer_submit_button = st.button('Add to model')
    with layer_remove:
        layer_remove_button = st.button('Remove last added layer')
    with layer_clear:
        layer_clear_button = st.button('Clear the model')
    epochs = int(st.number_input('Training Epochs'))
    model_submit_button = st.button('Finish the model and train it')
    if layer_submit_button:
        add_layer(config_file, layer_type_submit, layer_unit_submit)
        is_good = check_layer(config_file)
        if not is_good:
            remove_layer(config_file)
    if layer_remove_button:
        remove_layer(config_file)
    if layer_clear_button:
        clear_layer(config_file)
    model_tf_layer, model_tf_units = get_config(config_file)
    if model_submit_button:
        st.write("The model is created with this architecture")
        st.write(model_tf_layer)
        st.write(model_tf_units)
        model = Model_TF()
        model.create(data, model_tf_layer, model_tf_units, epochs)
        clear_layer(config_file)
        return model