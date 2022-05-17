import streamlit as st
import pandas as pd
from src.common.constant import PATH

def visualize_page(model=None, scaled=None, compared_all = False):
    if compared_all:
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
        #evaluation and visualization
        with st.container():
            st.header("Model Performance")
            #for graphic
            visualize_df = model.visualize(scaled)
            st.line_chart(visualize_df)
            #for eval metric
            eval_metric = model.evaluate(scaled)
            st.dataframe(pd.DataFrame(eval_metric))