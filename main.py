import streamlit as st
from src.pages.user_input import user_page
from src.common.constant import PATH
from src.common.yaml_util import read_yaml_file
from src.pages.demo import demo_page


EXPLANATION_TEXT = read_yaml_file(PATH.config)
EXPLANATION_TEXT = EXPLANATION_TEXT['explanation_text']
##############################sidebar##############################
select_page = st.sidebar.selectbox(
    "App Navigation",
    ("Home", "Example", "Create your own")
)
##############################main page##############################
if select_page == "Home":
    st.title("About This App")
    st.info(EXPLANATION_TEXT['home'])
elif select_page=="Create your own":
    user_page()
else:
    demo_page() #st.markdown("Still in development")
