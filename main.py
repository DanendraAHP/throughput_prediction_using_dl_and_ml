import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pages.user_input import user_page
from src.pages.demo import demo_page

##############################sidebar##############################
select_page = st.sidebar.selectbox(
    "App Navigation",
    ("Home", "Example", "Create your own")
)

##############################main page##############################
if select_page == "Home":
    st.title("About This App")
elif select_page=="Create your own":
    user_page()
else:
    demo_page() #st.markdown("Still in development")
