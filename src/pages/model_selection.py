import streamlit as st
from src.models.create_model_tf import get_config, write_to_yaml, add_layer, remove_layer, clear_layer,check_layer
from src.models.model_tf import Model_TF
import pandas as pd

def model_page(data, scaled):
    method_select = st.sidebar.selectbox(
        "Select what method to use?",
        ['Machine Learning', 'Deep Learning', 'Statistic']
    )
    if method_select == 'Machine Learning':
        pass
        # model_select = st.sidebar.selectbox(
        #     "Select what model to use?",
        #     ['Support Vector Regression', 'Random Forest']
        # )
    elif method_select == 'Deep Learning':
        model_tf_page()
    else :
        pass
        # model_select = st.sidebar.selectbox(
        #     "Select what model to use?",
        #     ['ARIMA', 'SARIMAX']
        # )

    verify = st.checkbox('Build the model')
    if verify:
        model.compile_and_fit(data=data, epochs=epochs, verbose=0, patience=5)
    eval = st.checkbox('evaluate the model')
    if eval:
        eval_metric = model.evaluate(data, scaled)
        st.dataframe(pd.DataFrame(eval_metric))
    visualize = st.checkbox('visualize the result')
    if visualize:
        visualize_df = model.visualize(data, scaled)
        st.line_chart(visualize_df)
    
def model_tf_page():
    layer_type, layer_unit  = st.columns(2)
    layer_submit, layer_remove, layer_clear, model_submit = st.columns(4)
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
    with model_submit:
        model_submit_button = st.button('Submit the model')
    if layer_submit_button:
        add_layer(layer_type_submit, layer_unit_submit)
        is_good = check_layer()
        if not is_good:
            remove_layer()
    if layer_remove_button:
        remove_layer()
    if layer_clear_button:
        clear_layer()
    model_tf_layer, model_tf_units = get_config('config.yaml')
    st.write(model_tf_layer)
    st.write(model_tf_units)

