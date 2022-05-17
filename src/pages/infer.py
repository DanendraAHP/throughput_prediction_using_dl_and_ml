import streamlit as st
import pandas as pd
from src.common.constant import PATH

def infer_page(model=None, scaled=None):
    #evaluation and visualization
    with st.container():
        st.header("Model Inference")
        forecast = st.number_input(
                "How many hours to forecast",
                value = 24
        )
        #for graphic
        infer_df = model.infer(forecast, scaled)
        st.line_chart(infer_df)