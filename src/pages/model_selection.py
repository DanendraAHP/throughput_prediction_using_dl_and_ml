import streamlit as st
from src.common.model_holder import model_dict
from src.models.model_tf import Model_TF
import pandas as pd

def model_page(data, scaled):
    model_select = st.sidebar.selectbox(
        "Select what model to use?",
        model_dict.keys()
    )
    
    model = Model_TF(model_select)
    epochs = st.number_input('how many epochs?', value=5)
    verify = st.checkbox('verify the model training parameter')
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
    